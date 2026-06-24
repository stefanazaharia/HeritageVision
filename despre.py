import streamlit as st


def afiseaza_despre():
    
    st.markdown("""
    <div class="hero-box" style="padding:24px 28px;">
        <div class="hero-titlu-licenta">
            Lucrare de licență — HeritageVision<br>
            Facultatea de Matematică și Informatică, Universitatea Ovidius Constanța
        </div>
        <div class="hero-titlu" style="font-size:22px;">
            Clasificarea și restaurarea imaginilor istorice degradate<br>
            prin tehnici de învățare profundă
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sectiune-titlu">🧠 Arhitectura sistemului</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="card">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
                <div style="width:36px; height:36px; border-radius:8px; background:#EDE0CF; 
                            display:flex; align-items:center; justify-content:center; font-size:18px;">🔍</div>
                <div>
                    <div style="font-size:14px; font-weight:600; color:#2C1810;">Model 1 — Clasificare</div>
                    <div style="font-size:11px; color:#9A7A5A;">CNN cu Spatial Attention</div>
                </div>
            </div>
            <div style="font-size:12px; color:#7A5A3A; line-height:1.7;">
                <b>Arhitectură:</b> CNN cu 4 straturi convoluționale, BatchNorm, MaxPooling și 
                un modul Spatial Attention care permite rețelei să se concentreze pe 
                zonele relevante ale imaginii.<br><br>
                <b>Output:</b> 4 scoruri de probabilitate (Sigmoid) pentru defectele:
                Decolorat, Zgâriat, Pete, Rupt și Îndoit.<br><br>
                <b>Antrenament:</b> clasificare multi-label pe dataset propriu de fotografii istorice.
            </div>
            <div style="margin-top:12px; display:flex; flex-wrap:wrap; gap:6px;">
                <span class="badge">Conv2D ×4</span>
                <span class="badge">BatchNorm</span>
                <span class="badge">Spatial Attention</span>
                <span class="badge">Dropout 0.5</span>
                <span class="badge">Sigmoid</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
                <div style="width:36px; height:36px; border-radius:8px; background:#EDE0CF; 
                            display:flex; align-items:center; justify-content:center; font-size:18px;">✨</div>
                <div>
                    <div style="font-size:14px; font-weight:600; color:#2C1810;">Model 2 — Restaurare</div>
                    <div style="font-size:11px; color:#9A7A5A;">CycleGAN cu U-Net Generator</div>
                </div>
            </div>
            <div style="font-size:12px; color:#7A5A3A; line-height:1.7;">
                <b>Arhitectură:</b> CycleGAN cu Generator U-Net (skip connections) și 
                Discriminator PatchGAN 70×70. Permite antrenare pe date <em>nepereche</em> 
                prin cycle consistency loss.<br><br>
                <b>Date:</b> 14.367 fotografii istorice reale (8.545 curate + 5.822 degradate),
                antrenate pe Google Colab A100.<br><br>
                <b>Antrenament:</b> 200 epoci, ~10 min/epocă pe A100 GPU.
            </div>
            <div style="margin-top:12px; display:flex; flex-wrap:wrap; gap:6px;">
                <span class="badge">U-Net Generator</span>
                <span class="badge">PatchGAN</span>
                <span class="badge">Cycle Loss</span>
                <span class="badge">LSGAN</span>
                <span class="badge">Image Pool</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset - carduri HTML in loc de st.metric
    st.markdown('<div class="sectiune-titlu">📊 Despre dataset</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    statistici_dataset = [
        ("Total fotografii", "14.367"),
        ("Fotografii curate", "8.545"),
        ("Fotografii degradate", "5.822"),
        ("Epoci antrenament", "200"),
    ]
    for col, (label, valoare) in zip([col1, col2, col3, col4], statistici_dataset):
        with col:
            st.markdown(f"""
            <div style="background:#FFFCF8; border:0.5px solid #D9CDBF; 
                        border-radius:10px; padding:16px 20px; margin-bottom:8px;">
                <div style="font-size:12px; color:#7A5A3A; margin-bottom:6px;">{label}</div>
                <div style="font-size:24px; font-weight:600; color:#2C1810;">{valoare}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="margin-top:12px;">
        Datasetul conține fotografii istorice reale din arhive publice, cuprinzând 
        imagini din perioada 1860–1960. Fotografiile degradate prezintă defecte naturale 
        (decolorare, pete, rupturi, zgârieturi) — fără degradare sintetică artificială.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sectiune-titlu">🛠️ Tehnologii folosite</div>', 
                unsafe_allow_html=True)
    
    tehnologii = {
        "Deep Learning": ["PyTorch", "torchvision", "CycleGAN", "U-Net", "CNN"],
        "Procesare imagine": ["PIL / Pillow", "NumPy", "OpenCV"],
        "Infrastructură": ["Google Colab A100", "Google Drive 5TB", "CUDA"],
        "Interfață": ["Streamlit", "streamlit-drawable-canvas"],
    }
    
    cols = st.columns(4)
    for col, (categorie, lista) in zip(cols, tehnologii.items()):
        with col:
            st.markdown(f"""
            <div class="card" style="padding:14px;">
                <div style="font-size:12px; font-weight:600; color:#5A3A2A; 
                            margin-bottom:8px;">{categorie}</div>
                {''.join(f'<div style="font-size:11px; color:#7A5A3A; padding:3px 0; border-bottom:0.5px solid #EDE0CF;">{t}</div>' for t in lista)}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="sectiune-titlu">⚠️ Limitări și direcții viitoare</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div style="font-size:13px; font-weight:600; color:#8B2500; margin-bottom:8px;">
                Limitări actuale
            </div>
            <div style="font-size:12px; color:#7A5A3A; line-height:1.7;">
                • CycleGAN nu poate <em>elimina</em> defecte localizate (rupturi, pete mari) — 
                poate doar îmbunătăți calitatea globală<br><br>
                • Tenta de culoare generată poate varia față de culoarea originală a fotografiei<br><br>
                • Performanța depinde de similaritatea fotografiei cu datasetul de antrenament
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div style="font-size:13px; font-weight:600; color:#2C6B3A; margin-bottom:8px;">
                Direcții viitoare
            </div>
            <div style="font-size:12px; color:#7A5A3A; line-height:1.7;">
                • Integrarea unui model de inpainting dedicat (ex: LaMa) pentru rupturi și zone lipsă<br><br>
                • Colorizare automată bazată pe context istorico-geografic<br><br>
                • Detecție automată a tipului de defect pentru aplicare selectivă a restaurării
            </div>
        </div>
        """, unsafe_allow_html=True)