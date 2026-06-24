import streamlit as st
from PIL import Image
import numpy as np
import io


def pil_la_bytes(imagine_pil, format="PNG"):
    buf = io.BytesIO()
    imagine_pil.save(buf, format=format)
    return buf.getvalue()


@st.cache_resource
def incarca_model_ddcolor():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    return pipeline(
        Tasks.image_colorization,
        model="iic/cv_ddcolor_image-colorization"
    )


def recoloreaza_ddcolor(imagine_pil):
    import cv2

    pipeline_ddcolor = incarca_model_ddcolor()

    rezultat = pipeline_ddcolor(imagine_pil)

    # Rezultatul e un dict cu valorile numpy BGR
    if isinstance(rezultat, dict):
        for v in rezultat.values():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                rgb = cv2.cvtColor(v, cv2.COLOR_BGR2RGB) if v.shape[2] == 3 else v
                return Image.fromarray(rgb)

    raise RuntimeError(f"Format rezultat neasteptat: {type(rezultat)}")


def afiseaza_recolorare():

    st.markdown("""
    <div class="card" style="margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="font-size:28px;">🎨</div>
            <div>
                <div style="font-size:16px; font-weight:600; color:#2C1810;">Recolorare fotografii</div>
                <div style="font-size:12px; color:#9A7A5A; margin-top:2px;">
                    DDColor — model state-of-the-art pentru recolorarea fotografiilor alb-negru și sepia
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Imaginea poate veni de la restaurare (prin session_state) sau incarcata manual
    # O mutam intr-o cheie persistenta ca sa nu dispara la rerun
    if "imagine_pentru_recolorare" in st.session_state:
        st.session_state._recolorare_imagine_activa = st.session_state.pop("imagine_pentru_recolorare")
        st.session_state.pop("imagine_recolorata", None)
        st.session_state._recolorare_img_id = id(st.session_state._recolorare_imagine_activa)

    col_stanga, col_dreapta = st.columns([1, 1], gap="large")

    with col_stanga:
        st.markdown('<div class="sectiune-titlu">📁 Încarcă fotografia</div>',
                    unsafe_allow_html=True)

        fisier = st.file_uploader(
            "Trage fotografia alb-negru sau sepia aici",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="upload_recolorare"
        )

        if fisier:
            imagine_originala = Image.open(fisier).convert("RGB")
            sursa = "upload"
        elif "_recolorare_imagine_activa" in st.session_state:
            imagine_originala = st.session_state._recolorare_imagine_activa
            sursa = "restaurare"
            st.info("📷 Folosind fotografia trimisă de la restaurare.")
        else:
            imagine_originala = None
            sursa = None

        if imagine_originala is None:
            st.markdown("""
            <div style="background:#FFFCF8; border:1.5px dashed #D9CDBF; border-radius:10px;
                        padding:32px; text-align:center; color:#9A7A5A;">
                <div style="font-size:32px; margin-bottom:10px;">🖼️</div>
                <div style="font-size:13px; color:#7A5A3A;">
                    Încarcă o fotografie alb-negru sau sepia<br>sau trimite-o de la pagina de Restaurare
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Reseteaza rezultatul cand se incarca o poza noua via upload
        if sursa == "upload":
            fisier_id = fisier.file_id
            if fisier_id != st.session_state.get("_recolorare_img_id"):
                st.session_state._recolorare_img_id = fisier_id
                st.session_state.pop("imagine_recolorata", None)
                st.session_state.pop("_recolorare_imagine_activa", None)

        st.image(imagine_originala, caption="Fotografie originală", use_column_width=True)

        w, h = imagine_originala.size
        st.markdown(f"""
        <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
            <span class="badge">📐 {w}×{h}px</span>
            <span class="badge">🎨 {'Alb-negru / Sepia' if sursa else 'RGB'}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            DDColor analizează structura imaginii și adaugă culori realiste folosind un model
            transformer antrenat pe milioane de fotografii. Funcționează cel mai bine pe fotografii
            alb-negru sau sepia cu contrast bun.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🎨 Recolorează fotografia", use_container_width=True, type="primary"):
            with st.spinner("Procesează imaginea..."):
                try:
                    rezultat = recoloreaza_ddcolor(imagine_originala)
                    st.session_state.imagine_recolorata = rezultat
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Eroare la recolorare: {e}")

    with col_dreapta:
        st.markdown('<div class="sectiune-titlu">🎨 Rezultat recolorare</div>',
                    unsafe_allow_html=True)

        if "imagine_recolorata" not in st.session_state:
            st.markdown("""
            <div style="background:#FFFCF8; border:1.5px dashed #D9CDBF; border-radius:10px;
                        padding:32px; text-align:center; color:#9A7A5A;">
                <div style="font-size:32px; margin-bottom:10px;">🎨</div>
                <div style="font-size:13px; color:#7A5A3A;">
                    Încarcă o fotografie și apasă<br>"Recolorează fotografia"
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            imagine_recolorata = st.session_state.imagine_recolorata
            st.image(imagine_recolorata, caption="Fotografie recolorată", use_column_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(imagine_originala, caption="Original", use_column_width=True)
            with col_b:
                st.image(imagine_recolorata, caption="Recolorat", use_column_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "⬇️ Descarcă recolorat (PNG)",
                    data=pil_la_bytes(imagine_recolorata, "PNG"),
                    file_name="recolorat.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_d2:
                st.download_button(
                    "⬇️ Descarcă recolorat (JPG)",
                    data=pil_la_bytes(imagine_recolorata, "JPEG"),
                    file_name="recolorat.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
