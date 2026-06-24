import streamlit as st


def afiseaza_home():
    
    # ---- HERO cu stiluri inline complete (fara clase CSS) ----
    st.markdown("""
    <div style="background:#2C1810; border-radius:12px; padding:32px 28px; margin-bottom:20px;">
        <div style="color:#9A7A5A; font-size:12px; font-style:italic; margin-bottom:10px; line-height:1.5;">
            HeritageVision: Clasificarea și restaurarea imaginilor istorice degradate 
            prin tehnici de învățare profundă
        </div>
        <div style="color:#F7EFE3; font-size:28px; font-weight:600; line-height:1.3; margin-bottom:10px;">
            Dă viață fotografiilor<br>
            istorice degradate
        </div>
        <div style="color:#C4A882; font-size:13px; line-height:1.7; max-width:540px;">
            HeritageVision folosește rețele neuronale profunde (CNN + CycleGAN) pentru a 
            clasifica automat tipul de degradare și a restaura fotografii istorice valoroase.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- STATISTICI ----
    col1, col2, col3, col4 = st.columns(4)
    statistici = [
        ("Fotografii în dataset", "14.367"),
        ("Epoci antrenament", "200"),
        ("Tipuri de defecte", "4"),
        ("Acuratețe clasificare", "~89%"),
    ]
    for col, (label, valoare) in zip([col1, col2, col3, col4], statistici):
        with col:
            st.markdown(f"""
            <div style="background:#FFFCF8; border:0.5px solid #D9CDBF; 
                        border-radius:10px; padding:16px 20px; margin-bottom:8px;">
                <div style="font-size:12px; color:#7A5A3A; margin-bottom:6px;">{label}</div>
                <div style="font-size:28px; font-weight:600; color:#2C1810;">{valoare}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---- CARDURI FUNCTII ----
    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#5A3A2A; margin-bottom:10px; 
                padding-bottom:6px; border-bottom:0.5px solid #D9CDBF;">
        Ce poți face cu HeritageVision
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background:#FFFCF8; border:0.5px solid #D9CDBF; border-radius:12px; 
                    padding:20px; margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <div style="width:38px; height:38px; border-radius:8px; background:#EDE0CF; 
                            display:flex; align-items:center; justify-content:center; font-size:18px;">🔍</div>
                <div>
                    <div style="font-size:14px; font-weight:600; color:#2C1810;">Clasificare defecte</div>
                    <div style="font-size:11px; color:#9A7A5A;">CNN cu Spatial Attention</div>
                </div>
            </div>
            <div style="font-size:12px; color:#7A5A3A; line-height:1.6;">
                Detectează automat tipul și severitatea degradării fotografiei tale: 
                decolorat, zgâriat, pete sau rupt și îndoit — cu procente de încredere.
            </div>
            <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:4px;">
                <span style="background:#EDE0CF; color:#5A3A2A; font-size:11px; padding:3px 10px; border-radius:20px;">Decolorat</span>
                <span style="background:#EDE0CF; color:#5A3A2A; font-size:11px; padding:3px 10px; border-radius:20px;">Zgâriat</span>
                <span style="background:#EDE0CF; color:#5A3A2A; font-size:11px; padding:3px 10px; border-radius:20px;">Pete</span>
                <span style="background:#EDE0CF; color:#5A3A2A; font-size:11px; padding:3px 10px; border-radius:20px;">Rupt și Îndoit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Încearcă clasificarea", use_container_width=True, key="btn_clas"):
            st.session_state.pagina_curenta = "clasificare"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background:#FFFCF8; border:0.5px solid #D9CDBF; border-radius:12px; 
                    padding:20px; margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <div style="width:38px; height:38px; border-radius:8px; background:#EDE0CF; 
                            display:flex; align-items:center; justify-content:center; font-size:18px;">✨</div>
                <div>
                    <div style="font-size:14px; font-weight:600; color:#2C1810;">Restaurare AI</div>
                    <div style="font-size:11px; color:#9A7A5A;">CycleGAN cu U-Net Generator</div>
                </div>
            </div>
            <div style="font-size:12px; color:#7A5A3A; line-height:1.6;">
                Restaurare globală sau selectivă cu control complet. Poți alege exact ce zone 
                să repare modelul — restul fotografiei rămâne neatins pentru autenticitate.
            </div>
            <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:4px;">
                <span style="background:rgba(196,130,58,0.15); color:#C4823A; font-size:11px; 
                             padding:3px 10px; border-radius:20px; border:0.5px solid rgba(196,130,58,0.4);">
                    Restaurare automată</span>
                <span style="background:rgba(196,130,58,0.15); color:#C4823A; font-size:11px; 
                             padding:3px 10px; border-radius:20px; border:0.5px solid rgba(196,130,58,0.4);">
                    Control cu mască</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✨ Încearcă restaurarea", use_container_width=True, key="btn_rest"):
            st.session_state.pagina_curenta = "restaurare"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---- TIMELINE EPOCI ----
    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#5A3A2A; margin-bottom:10px; 
                padding-bottom:6px; border-bottom:0.5px solid #D9CDBF;">
        ⏱️ Epoci fotografice istorice
    </div>
    <div style="background:#EDE0CF; border-left:3px solid #C4823A; border-radius:0 8px 8px 0; 
                padding:10px 14px; margin-bottom:14px; font-size:12px; color:#5A3A2A; line-height:1.6;">
        Fotografiile istorice variază ca proces și degradare în funcție de epoca în care 
        au fost realizate. HeritageVision a fost antrenat pe imagini din toate aceste perioade.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    epoci = [
        ("1839–1860", "#C4823A", "Dagherotip", 
         "Primele fotografii pe plăci de argint. Degradări tipice: oxidare, pete de umiditate."),
        ("1860–1900", "#8B5E3C", "Albumine", 
         "Hârtie albuminată, tonuri sepia. Degradări: decolorare, rupturi, pete gălbui."),
        ("1900–1930", "#6B4226", "Gelatino-bromură", 
         "Alb-negru clasic. Degradări: zgârieturi, decolorare, murdărie de suprafață."),
        ("1930–1960", "#4A2C1A", "Fotografii moderne", 
         "Primele fotografii color. Degradări: decolorare cromatică, pete chimice."),
    ]
    
    for col, (an, culoare, tip, desc) in zip([col1, col2, col3, col4], epoci):
        with col:
            st.markdown(f"""
            <div style="background:#FFFCF8; border:0.5px solid #D9CDBF; border-radius:8px; 
                        padding:12px; text-align:center;">
                <div style="width:10px; height:10px; border-radius:50%; background:{culoare}; 
                            margin:0 auto 6px;"></div>
                <div style="font-size:12px; font-weight:600; color:#2C1810;">{an}</div>
                <div style="font-size:11px; font-weight:600; color:#5A3A2A; margin:3px 0;">{tip}</div>
                <div style="font-size:10px; color:#9A7A5A; line-height:1.4; margin-top:2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---- SFATURI CONSERVARE ----
    st.markdown("""
    <div style="font-size:13px; font-weight:600; color:#5A3A2A; margin-bottom:10px; 
                padding-bottom:6px; border-bottom:0.5px solid #D9CDBF;">
        💡 Sfaturi pentru conservarea fotografiilor vechi
    </div>
    """, unsafe_allow_html=True)
    
    sfaturi = [
        ("🌡️", "Temperatură și umiditate", 
         "Păstrează fotografiile la 15–20°C și umiditate relativă 30–50%. Variațiile bruște accelerează degradarea suportului."),
        ("📦", "Materiale de arhivare", 
         "Folosește mape, plicuri și cutii certificate fără acid (acid-free). Evită materialele plastice PVC care emit gaze nocive."),
        ("💻", "Digitalizare preventivă", 
         "Scanează fotografiile importante la minim 600 DPI (1200 DPI pentru detalii fine). Salvează în format TIFF, nu doar JPG."),
        ("☀️", "Protecție față de lumină", 
         "Evită expunerea la lumina directă a soarelui. Radiațiile UV decolorează fotografiile ireversibil în câteva luni."),
        ("🧤", "Manipulare corectă", 
         "Mânuiește fotografiile cu mănuși din bumbac sau nitril. Grăsimile de pe piele lasă urme permanente pe gelatina fotografică."),
        ("🏠", "Condiții de depozitare", 
         "Evită podurile, subsolurile și garajele — umiditatea și temperatura variabilă sunt dușmanii fotografiilor vechi."),
    ]
    
    col1, col2, col3 = st.columns(3)
    coloane = [col1, col2, col3]
    
    for i, (icon, titlu, text) in enumerate(sfaturi):
        with coloane[i % 3]:
            st.markdown(f"""
            <div style="background:#FFFCF8; border:0.5px solid #D9CDBF; border-radius:8px; 
                        padding:12px; margin-bottom:10px;">
                <div style="font-size:18px; margin-bottom:6px;">{icon}</div>
                <div style="font-size:12px; font-weight:600; color:#2C1810; margin-bottom:4px;">{titlu}</div>
                <div style="font-size:11px; color:#7A5A3A; line-height:1.5;">{text}</div>
            </div>
            """, unsafe_allow_html=True)