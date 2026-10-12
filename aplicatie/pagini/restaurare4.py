import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import io
import sys
import os
import cv2

# ============================================================
# INCARCA MODELELE
# ============================================================

@st.cache_resource
def incarca_restaurator(cale_model):
    cod_path = r"D:\LICENTA\scripts\google_colab_notebooks\restaurare"
    if cod_path not in sys.path:
        sys.path.insert(0, cod_path)
    from restaurator_imagini import RestauratorImagini
    return RestauratorImagini(cale_model, dimensiune_imagine=256)

@st.cache_resource
def incarca_lama():
    from simple_lama_inpainting import SimpleLama
    return SimpleLama()

@st.cache_resource
def incarca_realesrgan():
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )
    return upsampler

@st.cache_resource
def incarca_gfpgan():
    from gfpgan import GFPGANer
    restaurator_fete = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2,
    )
    return restaurator_fete

CALE_MODEL_RESTAURARE = r"D:\LICENTA\models\restaurator_versiuni\generator_restaurare_ultim.pth"


def pil_la_bytes(imagine_pil, format="PNG"):
    buf = io.BytesIO()
    imagine_pil.save(buf, format=format)
    return buf.getvalue()


# ============================================================
# FUNCTII DE PROCESARE
# ============================================================

def aplica_clahe(imagine_pil):
    """Imbunatateste contrastul local cu CLAHE."""
    arr = np.array(imagine_pil.convert('RGB'))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    rezultat = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rezultat)


def detecteaza_fete_dlib(imagine_pil, marja=40):
    """Returneaza masca PIL 'L' cu zonele de fata dilatate (alb=fata, negru=rest)."""
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        arr_gray = np.array(imagine_pil.convert('L'))
        fete = detector(arr_gray, 1)
        masca = Image.new('L', imagine_pil.size, 0)
        draw = ImageDraw.Draw(masca)
        w, h = imagine_pil.size
        for fata in fete:
            x1 = max(0, fata.left() - marja)
            y1 = max(0, fata.top() - marja)
            x2 = min(w, fata.right() + marja)
            y2 = min(h, fata.bottom() + marja)
            draw.ellipse([x1, y1, x2, y2], fill=255)
        return masca
    except Exception:
        return Image.new('L', imagine_pil.size, 0)


def detecteaza_zgarieturi(imagine_pil, prag_alb=180, prag_negru=30, kernel=3):
    """Detecteaza zgârieturi prin threshold pe luminanta + muchii Canny pentru linii fine."""
    arr = np.array(imagine_pil.convert('L'))
    masca_alb = (arr >= prag_alb).astype(np.uint8) * 255
    masca_negru = (arr <= prag_negru).astype(np.uint8) * 255
    masca_prag = cv2.bitwise_or(masca_alb, masca_negru)
    arr_blur = cv2.GaussianBlur(arr, (3, 3), 0)
    muchii = cv2.Canny(arr_blur, threshold1=60, threshold2=140)
    masca_linii = cv2.bitwise_and(muchii, masca_prag)
    masca = cv2.bitwise_or(masca_prag, masca_linii)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    masca = cv2.dilate(masca, element, iterations=2)
    return Image.fromarray(masca, mode='L')


def creeaza_masca_lama_automata(imagine_pil):
    """Zgârieturi detectate minus zonele cu fete (pentru a nu distorsiona fete)."""
    masca_zgarieturi = detecteaza_zgarieturi(imagine_pil)
    masca_fete = detecteaza_fete_dlib(imagine_pil)
    # scade zonele cu fete din masca de zgârieturi
    arr_zg = np.array(masca_zgarieturi)
    arr_fete = np.array(masca_fete)
    arr_final = np.where(arr_fete > 0, 0, arr_zg).astype(np.uint8)
    return Image.fromarray(arr_final, mode='L')


def aplica_smooth_selectiv(imagine_pil, intensitate=15, fereastra_varianta=15):
    """Smooth selectiv bazat pe varianta locala: zone plate (fundal, cer) => smooth,
    zone cu detalii (fata, cladiri, frunze) => neatinse. Functioneaza pe orice tip de fotografie."""
    arr = np.array(imagine_pil.convert('RGB'), dtype=np.float32)
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # varianta locala: masoara cat de mult variaza luminanta in jurul fiecarui pixel
    mean_local = cv2.boxFilter(gray, -1, (fereastra_varianta, fereastra_varianta))
    mean_sq_local = cv2.boxFilter(gray ** 2, -1, (fereastra_varianta, fereastra_varianta))
    varianta = np.clip(mean_sq_local - mean_local ** 2, 0, None)

    # normalizeaza varianta la [0, 1]: 0 = zona perfect plata, 1 = zona cu mult detaliu
    var_max = np.percentile(varianta, 95)
    if var_max < 1e-6:
        return imagine_pil
    harta_detalii = np.clip(varianta / var_max, 0, 1)

    # blur bilateral: pastreaza muchiile dar netezeste suprafetele uniforme
    d_bilateral = max(5, intensitate // 2 * 2 + 1)
    arr_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    smoothed_bgr = cv2.bilateralFilter(arr_bgr, d=d_bilateral,
                                       sigmaColor=intensitate * 3,
                                       sigmaSpace=intensitate)
    smoothed = cv2.cvtColor(smoothed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # blend: in zonele cu detalii pastreaza originalul, in zonele plate aplica smoothing
    harta = harta_detalii[:, :, np.newaxis]
    rezultat = arr * harta + smoothed * (1.0 - harta)
    return Image.fromarray(rezultat.clip(0, 255).astype(np.uint8))


def aplica_realesrgan(upsampler, imagine_pil):
    """Mareste rezolutia 2x cu Real-ESRGAN."""
    arr = np.array(imagine_pil.convert('RGB'))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    output, _ = upsampler.enhance(arr_bgr, outscale=2)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)


def aplica_gfpgan(restaurator_fete, imagine_pil):
    """Restaureaza fetele din imagine cu GFPGAN."""
    arr = np.array(imagine_pil.convert('RGB'))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, _, output = restaurator_fete.enhance(
        arr_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    if output is not None:
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)
    return imagine_pil


def aplica_lama(lama, imagine_pil, masca_pil):
    """Aplica LaMa inpainting pe zona mascata."""
    if masca_pil.size != imagine_pil.size:
        masca_pil = masca_pil.resize(imagine_pil.size, Image.BILINEAR)
    return lama(imagine_pil, masca_pil)


def blend_cu_original(original, rezultat, masca):
    """Combina originalul cu rezultatul folosind masca."""
    w, h = original.size
    if rezultat.size != (w, h):
        rezultat = rezultat.resize((w, h), Image.BICUBIC)
    if masca.size != (w, h):
        masca = masca.resize((w, h), Image.BILINEAR)
    arr_orig = np.array(original, dtype=np.float32)
    arr_rez = np.array(rezultat, dtype=np.float32)
    arr_masca = np.array(masca, dtype=np.float32) / 255.0
    arr_masca = arr_masca[..., None]
    final = arr_orig * (1 - arr_masca) + arr_rez * arr_masca
    return Image.fromarray(final.clip(0, 255).astype(np.uint8))


# ============================================================
# INTERFATA
# ============================================================

def afiseaza_restaurare():

    st.markdown("""
    <div style="background:#FFFCF8; border:0.5px solid #D9CDBF; border-radius:12px; 
                padding:20px; margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="font-size:28px;">✨</div>
            <div>
                <div style="font-size:16px; font-weight:600; color:#2C1810;">
                    Restaurare AI — Sistem Hibrid
                </div>
                <div style="font-size:12px; color:#9A7A5A; margin-top:2px;">
                    CLAHE + Real-ESRGAN + GFPGAN + LaMa + CycleGAN
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Incarca modelele cu fallback individual
    try:
        restaurator = incarca_restaurator(CALE_MODEL_RESTAURARE)
    except Exception as e:
        st.error(f"Nu pot incarca CycleGAN: {e}")
        st.stop()

    lama = None
    try:
        lama = incarca_lama()
    except Exception as e:
        st.warning(f"LaMa nu e disponibil: {e}")

    upsampler = None
    realesrgan_ok = False
    try:
        upsampler = incarca_realesrgan()
        realesrgan_ok = True
    except Exception as e:
        st.warning(f"Real-ESRGAN nu e disponibil: {e}")

    gfpgan_model = None
    gfpgan_ok = False
    try:
        gfpgan_model = incarca_gfpgan()
        gfpgan_ok = True
    except Exception as e:
        st.warning(f"GFPGAN nu e disponibil: {e}")

    # Upload
    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#5A3A2A; margin-bottom:10px; 
                padding-bottom:6px; border-bottom:0.5px solid #D9CDBF;">
        Incarca fotografia
    </div>
    """, unsafe_allow_html=True)

    # CSS pentru butonul disabled — pastreaza culoarea primara in loc de gri
    st.markdown("""
    <style>
    button[disabled] { opacity: 0.6 !important; background-color: #C4823A !important;
                       color: white !important; cursor: not-allowed !important; }
    </style>
    """, unsafe_allow_html=True)

    imagine_initiala = st.session_state.get("imagine_pentru_restaurare", None)
    fisier = st.file_uploader(
        "Trage fotografia degradata aici",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # reseteaza rezultatele cand se incarca o poza noua
    fisier_id = fisier.file_id if fisier else None
    if fisier_id != st.session_state.get("fisier_id_curent"):
        st.session_state.fisier_id_curent = fisier_id
        st.session_state.pop("imagine_restaurata", None)
        st.session_state.pop("imagine_restaurata_orig_size", None)
        st.session_state.pop("imagine_restaurata_masca", None)
        st.session_state.restaurare_in_procesare = False
        st.session_state.masca_in_procesare = False

    if fisier:
        imagine_originala = Image.open(fisier).convert("RGB")
    elif imagine_initiala:
        imagine_originala = imagine_initiala
        st.info("Folosind fotografia trimisa de la clasificare.")
    else:
        st.markdown("""
        <div style="background:#FFFCF8; border:1.5px dashed #D9CDBF; border-radius:10px; 
                    padding:32px; text-align:center; margin-bottom:20px;">
            <div style="font-size:32px; margin-bottom:10px;">📷</div>
            <div style="font-size:13px; color:#7A5A3A;">
                Incarca o fotografie pentru a incepe restaurarea
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#5A3A2A; margin-bottom:10px; 
                padding-bottom:6px; border-bottom:0.5px solid #D9CDBF;">
        Alege modul de restaurare
    </div>
    """, unsafe_allow_html=True)

    mod = st.radio("Mod", options=[
        "Automat — restaurare completa",
        "Cu masca — restaurez zone specifice",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # MOD 1: AUTOMAT
    # ============================================================
    if mod.startswith("Automat"):

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div style='font-size:12px; font-weight:600; color:#5A3A2A; margin-bottom:6px;'>Original</div>", unsafe_allow_html=True)
            st.image(imagine_originala, use_column_width=True)
        with col2:
            st.markdown("<div style='font-size:12px; font-weight:600; color:#5A3A2A; margin-bottom:6px;'>Restaurat</div>", unsafe_allow_html=True)
            zona_rezultat = st.empty()
            zona_rezultat.markdown("""
            <div style="background:#EDE0CF; border-radius:10px; height:200px; 
                        display:flex; align-items:center; justify-content:center; 
                        font-size:12px; color:#9A7A5A;">
                Apasa Restaureaza pentru a vedea rezultatul
            </div>
            """, unsafe_allow_html=True)

        lama_auto = lama is not None
        st.markdown(f"""
        <div style="background:#EDE0CF; border-left:3px solid #C4823A; border-radius:0 8px 8px 0;
                    padding:10px 14px; margin-bottom:14px; font-size:12px; color:#5A3A2A; line-height:1.8;">
            Cum functioneaza restaurarea automata:<br>
            1. CLAHE — imbunatateste contrastul local<br>
            1.5. Smooth selectiv — reduce granulaia din zone plate, pastreaza detaliile<br>
            2. LaMa — detecteaza si repara zgârieturile/petele, evitând fetele {'✓' if lama_auto else '(indisponibil)'}<br>
            3. Real-ESRGAN — mareste rezolutia 2x cu detalii reale<br>
            4. GFPGAN — restaureaza fetele (daca exista in fotografie)<br>
            5. CycleGAN — imbunatateste stilul global (modelul antrenat)<br><br>
            <strong>Atentie:</strong> GFPGAN poate moderniza aspectul fetelor istorice.
        </div>
        """, unsafe_allow_html=True)

        col_sl1, col_sl2 = st.columns(2)
        with col_sl1:
            intensitate_smooth = st.slider(
                "Smooth fundal (0 = dezactivat)",
                min_value=0, max_value=30, value=10, step=1,
                help="Reduce granulaia din zone plate (fundal, cer). Detaliile (fata, cladiri) raman neatinse."
            )
        with col_sl2:
            intensitate_cyclegan = st.slider(
                "Intensitate CycleGAN (0 = original, 100 = efect complet)",
                min_value=0, max_value=100, value=60, step=5,
                help="Reduce daca culorile apar suprasaturate sau aspectul e prea modificat."
            )

        in_procesare = st.session_state.get("restaurare_in_procesare", False)
        if not in_procesare:
            if st.button("Restaureaza automat complet", use_container_width=True, type="primary"):
                st.session_state.restaurare_in_procesare = True
                st.session_state.pop("imagine_restaurata", None)
                st.session_state.pop("imagine_restaurata_orig_size", None)
                st.rerun()
        else:
            st.button("Restaureaza automat complet", use_container_width=True,
                      type="primary", disabled=True)

        if st.session_state.get("restaurare_in_procesare", False):
            rezultat = imagine_originala.copy()

            with st.spinner("Pasul 1/5:  Imbunatatesc contrastul..."):
                rezultat = aplica_clahe(rezultat)

            if intensitate_smooth > 0:
                with st.spinner("Pasul 1.5/5: Netezesc zonele plate..."):
                    try:
                        rezultat = aplica_smooth_selectiv(rezultat, intensitate=intensitate_smooth)
                    except Exception as e:
                        st.warning(f"Smooth selectiv a esuat: {e}")

            if lama_auto:
                with st.spinner("Pasul 2/5: Detectez si repar zgârieturile..."):
                    try:
                        masca_auto = creeaza_masca_lama_automata(rezultat)
                        nr_pixeli = np.count_nonzero(np.array(masca_auto))
                        if nr_pixeli > 100:
                            rezultat = aplica_lama(lama, rezultat, masca_auto)
                        else:
                            st.info("LaMa: nu s-au detectat zgârieturi semnificative, pas omis.")
                    except Exception as e:
                        st.warning(f"LaMa a esuat: {e}")
            else:
                st.info("LaMa omis (indisponibil).")

            if realesrgan_ok and upsampler:
                with st.spinner("Pasul 3/5: Maresc rezolutia 2x..."):
                    try:
                        rezultat = aplica_realesrgan(upsampler, rezultat)
                    except Exception as e:
                        st.warning(f"Real-ESRGAN a esuat: {e}")
            else:
                st.info("Real-ESRGAN omis.")

            if gfpgan_ok and gfpgan_model:
                with st.spinner("Pasul 4/5: Retusez fetele..."):
                    try:
                        rezultat = aplica_gfpgan(gfpgan_model, rezultat)
                    except Exception as e:
                        st.warning(f"GFPGAN a esuat: {e}")
            else:
                st.info("GFPGAN omis.")

            with st.spinner("Pasul 5/5: Imbunatatesc stilul global..."):
                rezultat_cyclegan = restaurator.restaureaza_complet_hd(rezultat)
                if intensitate_cyclegan < 100:
                    alpha = intensitate_cyclegan / 100.0
                    ref = rezultat.resize(rezultat_cyclegan.size, Image.BICUBIC)
                    arr_ref = np.array(ref, dtype=np.float32)
                    arr_cyc = np.array(rezultat_cyclegan, dtype=np.float32)
                    rezultat_final = Image.fromarray(
                        (arr_ref * (1 - alpha) + arr_cyc * alpha).clip(0, 255).astype(np.uint8)
                    )
                else:
                    rezultat_final = rezultat_cyclegan

            w_orig, h_orig = imagine_originala.size
            rezultat_orig_size = rezultat_final.resize((w_orig, h_orig), Image.BICUBIC)

            st.session_state.imagine_restaurata = rezultat_final
            st.session_state.imagine_restaurata_orig_size = rezultat_orig_size
            st.session_state.restaurare_in_procesare = False

        if "imagine_restaurata" in st.session_state:
            imagine_restaurata = st.session_state.imagine_restaurata
            imagine_restaurata_orig = st.session_state.get(
                "imagine_restaurata_orig_size", imagine_restaurata
            )

            with col2:
                zona_rezultat.image(imagine_restaurata, use_column_width=True)

            w_r, h_r = imagine_restaurata.size
            w_o, h_o = imagine_originala.size
            st.markdown(f"""
            <div style="background:#FFFCF8; border:0.5px solid #D9CDBF; border-radius:8px; 
                        padding:10px 14px; margin:10px 0; font-size:11px; color:#7A5A3A;">
                Original: {w_o}x{h_o}px &nbsp;|&nbsp;
                Restaurat HD: {w_r}x{h_r}px
                &nbsp;(marire {round(w_r/w_o, 1)}x)
            </div>
            """, unsafe_allow_html=True)

            tip_comparatie = st.radio(
                "Tip comparatie",
                ["Side by side", "Slider interactiv"],
                horizontal=True,
                label_visibility="collapsed"
            )

            if tip_comparatie == "Side by side":
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(imagine_originala, use_column_width=True)
                with col_b:
                    st.image(imagine_restaurata_orig, use_column_width=True)
                st.markdown("""
                <div style="display:flex; justify-content:space-around; margin-top:-10px; margin-bottom:10px;">
                    <span style="font-size:13px; color:#7A5A3A; text-align:center; flex:1;">Original degradat</span>
                    <span style="font-size:13px; color:#7A5A3A; text-align:center; flex:1;">Restaurat</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                pozitie = st.slider("Trage pentru a compara", 0, 100, 50)
                latime, inaltime = imagine_originala.size
                taietura = int(latime * pozitie / 100)
                rez_resize = imagine_restaurata_orig.resize((latime, inaltime), Image.BICUBIC)
                comparatie = imagine_originala.copy()
                comparatie.paste(
                    rez_resize.crop((taietura, 0, latime, inaltime)),
                    (taietura, 0)
                )
                draw = ImageDraw.Draw(comparatie)
                draw.line([(taietura, 0), (taietura, inaltime)], fill="#C4823A", width=3)
                st.image(comparatie, use_column_width=True)
                st.markdown(f"""
                <div style="text-align:center; color:#7A5A3A; font-size:12px; margin-top:4px;">
                    Original | {pozitie}% | Restaurat
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "Descarca restaurat HD (PNG)",
                    data=pil_la_bytes(imagine_restaurata, "PNG"),
                    file_name="restaurat_hd.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_d2:
                st.download_button(
                    "Descarca restaurat dim. originala (JPG)",
                    data=pil_la_bytes(imagine_restaurata_orig, "JPEG"),
                    file_name="restaurat_orig.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#EDE0CF; border-left:3px solid #C4823A; border-radius:0 8px 8px 0;
                        padding:10px 14px; font-size:12px; color:#5A3A2A; margin-bottom:10px;">
                Fotografia restaurată este alb-negru sau sepia? O poți trimite direct la recolorare.
            </div>
            """, unsafe_allow_html=True)
            if st.button("🎨 Trimite la recolorare", use_container_width=True, type="primary"):
                st.session_state.imagine_pentru_recolorare = imagine_restaurata_orig
                st.session_state.pagina_curenta = "recolorare"
                st.rerun()

    # ============================================================
    # MOD 2: CU MASCA — LaMa + CycleGAN
    # ============================================================
    elif mod.startswith("Cu masca"):

        try:
            from streamlit_drawable_canvas import st_canvas
        except ImportError:
            st.error("Trebuie sa instalezi: pip install streamlit-drawable-canvas")
            return

        if lama is None:
            st.error("LaMa nu e disponibil. Verifica instalarea.")
            return

        st.markdown("""
        <div style="background:#EDE0CF; border-left:3px solid #C4823A; border-radius:0 8px 8px 0; 
                    padding:10px 14px; margin-bottom:14px; font-size:12px; color:#5A3A2A; line-height:1.6;">
            Deseneaza peste zona defecta. LaMa repara fizic defectul, CycleGAN rafineaza.
            Restul fotografiei ramane neatins.<br><br>
            <strong>Atentie:</strong> Evita sa maschezi fetele — LaMa poate sterge trasaturile.
            Fetele sunt restaurate automat in modul Automat cu GFPGAN.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            grosime = st.slider("Dimensiunea pensulei", 5, 80, 25)
        with col2:
            culoare_masca = st.color_picker("Culoare masca", "#FF4444")
        with col3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Sterge tot", use_container_width=True):
                st.session_state.canvas_version = st.session_state.get("canvas_version", 0) + 1

        latime_max = 650
        lw, lh = imagine_originala.size
        factor = latime_max / lw if lw > latime_max else 1
        dim_canvas = (int(lw * factor), int(lh * factor))
        imagine_afisare = imagine_originala.resize(dim_canvas, Image.BICUBIC)

        canvas_key = f"canvas_masca_{st.session_state.get('canvas_version', 0)}"
        rezultat_canvas = st_canvas(
            fill_color=f"{culoare_masca}44",
            stroke_width=grosime,
            stroke_color=culoare_masca,
            background_image=imagine_afisare,
            update_streamlit=True,
            height=dim_canvas[1],
            width=dim_canvas[0],
            drawing_mode="freedraw",
            key=canvas_key,
        )

        in_procesare_masca = st.session_state.get("masca_in_procesare", False)
        if st.button("Restaureaza zona marcata", use_container_width=True, type="primary",
                     disabled=in_procesare_masca):
            if rezultat_canvas.image_data is None:
                st.warning("Deseneaza mai intai zona de reparat!")
                return

            st.session_state.masca_in_procesare = True
            canvas_np = rezultat_canvas.image_data
            masca_np = (canvas_np[:, :, 3] > 50).astype(np.uint8) * 255
            masca_pil = Image.fromarray(masca_np, mode='L').resize(
                imagine_originala.size, Image.BILINEAR
            )

            with st.spinner("Pasul 1/2: Repara defectul fizic..."):
                dupa_lama = aplica_lama(lama, imagine_originala, masca_pil)
            with st.spinner("Pasul 2/2: Rafineaza zona..."):
                rezultat = restaurator.restaureaza_cu_masca(
                    dupa_lama, masca_pil, pixeli_tranzitie=15
                )
                st.session_state.imagine_restaurata_masca = blend_cu_original(
                    imagine_originala, rezultat, masca_pil
                )
            st.session_state.masca_in_procesare = False

        if "imagine_restaurata_masca" in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(imagine_originala, use_column_width=True)
            with col_b:
                st.image(st.session_state.imagine_restaurata_masca, use_column_width=True)
            st.markdown("""
            <div style="display:flex; justify-content:space-around; margin-top:-10px; margin-bottom:10px;">
                <span style="font-size:13px; color:#7A5A3A; text-align:center; flex:1;">Original</span>
                <span style="font-size:13px; color:#7A5A3A; text-align:center; flex:1;">Restaurat selectiv</span>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                "Descarca rezultatul",
                data=pil_la_bytes(st.session_state.imagine_restaurata_masca),
                file_name="restaurat_selectiv.png",
                mime="image/png",
                use_container_width=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎨 Trimite la recolorare", use_container_width=True,
                         type="primary", key="recolorare_masca"):
                st.session_state.imagine_pentru_recolorare = st.session_state.imagine_restaurata_masca
                st.session_state.pagina_curenta = "recolorare"
                st.rerun()

