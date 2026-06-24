import streamlit as st

st.set_page_config(
    page_title="HeritageVision",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS GLOBAL - Tema Vintage/Sepia
# ============================================================
st.markdown("""
<style>
    /* Fundal general */
    .stApp {
        background-color: #F7F3EE;
    }
    
    /* Ascunde sidebar si header default */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stHeader"] { background: transparent; }
    .stDeployButton { display: none; }
    
    /* Navigare */
    .nav-container {
        background: #2C1810;
        padding: 0 24px;
        display: flex;
        align-items: center;
        height: 52px;
        position: sticky;
        top: 0;
        z-index: 999;
        margin: -1rem -1rem 0 -1rem;
    }
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-right: 32px;
        text-decoration: none;
    }
    .nav-logo-text {
        color: #F7EFE3;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .nav-links {
        display: flex;
        gap: 4px;
    }
    .nav-link {
        color: #B89878;
        font-size: 13px;
        padding: 6px 16px;
        border-radius: 6px;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.2s;
    }
    .nav-link:hover { color: #F7EFE3; background: rgba(255,255,255,0.08); }
    .nav-link-active {
        color: #F7EFE3 !important;
        background: rgba(196,130,58,0.2);
    }
    
    /* Butoane */
    .stButton > button {
        border-radius: 8px !important;
        font-size: 13px !important;
        transition: all 0.2s !important;
    }
    .btn-primary > button {
        background: #2C1810 !important;
        color: #F7EFE3 !important;
        border: none !important;
    }
    .btn-amber > button {
        background: #C4823A !important;
        color: #F7EFE3 !important;
        border: none !important;
    }
    
    /* Carduri */
    .card {
        background: #FFFCF8;
        border: 0.5px solid #D9CDBF;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: #C4823A !important;
    }
    
    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: #FFFCF8;
        border: 1.5px dashed #C4A882;
        border-radius: 10px;
        padding: 8px;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #FFFCF8;
        border: 0.5px solid #D9CDBF;
        border-radius: 10px;
        padding: 12px 16px;
    }
    
    /* Titluri sectiuni */
    .sectiune-titlu {
        font-size: 13px;
        font-weight: 600;
        color: #5A3A2A;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 0.5px solid #D9CDBF;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: #EDE0CF;
        color: #5A3A2A;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 4px;
    }
    .badge-amber {
        background: rgba(196,130,58,0.15);
        color: #C4823A;
        border: 0.5px solid rgba(196,130,58,0.4);
    }
    
    /* Info box */
    .info-box {
        background: #EDE0CF;
        border-left: 3px solid #C4823A;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 10px 0;
        font-size: 12px;
        color: #5A3A2A;
        line-height: 1.6;
    }
    
    /* Epoca timeline */
    .epoca-card {
        background: #FFFCF8;
        border: 0.5px solid #D9CDBF;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .epoca-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #C4823A;
        margin: 0 auto 6px;
    }
    .epoca-an {
        font-size: 12px;
        font-weight: 600;
        color: #2C1810;
    }
    .epoca-desc {
        font-size: 10px;
        color: #9A7A5A;
        line-height: 1.4;
        margin-top: 3px;
    }
    
    /* Tip conservare */
    .tip-card {
        background: #FFFCF8;
        border: 0.5px solid #D9CDBF;
        border-radius: 8px;
        padding: 12px;
        font-size: 12px;
        color: #5A3A2A;
        line-height: 1.5;
    }
    .tip-icon {
        font-size: 18px;
        margin-bottom: 6px;
    }
    
    /* Hero */
    .hero-box {
        background: #2C1810;
        border-radius: 12px;
        padding: 32px 28px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .hero-titlu-licenta {
        color: #9A7A5A;
        font-size: 12px;
        font-style: italic;
        margin-bottom: 10px;
        line-height: 1.5;
    }
    .hero-titlu {
        color: #F7EFE3;
        font-size: 28px;
        font-weight: 600;
        line-height: 1.3;
        margin-bottom: 10px;
    }
    .hero-titlu span { color: #C4823A; }
    .hero-sub {
        color: #C4A882;
        font-size: 13px;
        line-height: 1.7;
        max-width: 540px;
        margin-bottom: 0;
    }
    
    /* Defect bars custom */
    .defect-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .defect-label { font-size: 12px; color: #2C1810; }
    .defect-pct { font-size: 12px; font-weight: 600; color: #C4823A; }
    .defect-bar-bg {
        height: 5px;
        background: #EDE0CF;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .defect-bar-fill {
        height: 5px;
        border-radius: 3px;
        background: #C4823A;
    }
    .defect-bar-fill-warn { background: #8B2500; }
    /* Butoane navigare — stil vintage */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: transparent !important;
        color: #2C1810 !important;
        border: 0.5px solid #D9CDBF !important;
        border-radius: 8px !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #EDE0CF !important;
        border-color: #C4823A !important;
    }
    /* Buton activ */
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background: #2C1810 !important;
        color: #F7EFE3 !important;
        border: none !important;
    }
            /* Fix text radio buttons */
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] p {
        color: #2C1810 !important;
    }

    /* Fix text general dar NU in butoane */
    .stMarkdown p,
    .stMarkdown span,
    .element-container p {
        color: #2C1810 !important;
    }

    /* Fix text upload zone */
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #2C1810 !important;
    }

    /* Butoanele primary raman cu text deschis */
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] {
        color: #F7EFE3 !important;
    }

    /* Butoanele secondary cu text inchis */
    .stButton > button[kind="secondary"] p,
    .stButton > button[kind="secondary"] {
        color: #2C1810 !important;
    }                
        /* Fix butoane download */
    .stDownloadButton > button {
        color: #F7EFE3 !important;
        background: #2C1810 !important;
        border: none !important;
    }

    .stDownloadButton > button p,
    .stDownloadButton > button span {
        color: #F7EFE3 !important;
    }        
        /* Fix caption imagini */
    .stImage caption,
    [data-testid="caption"] {
        color: #7A5A3A !important;
    }        
        /* Fix caption sub imagini */
    .stImage > div > div > small,
    .stImage + div small,
    figcaption,
    [data-testid="stImage"] + div,
    .stImage p {
        color: #7A5A3A !important;
    }

    /* Fix orice text mic/caption */
    small {
        color: #7A5A3A !important;
    }  
        /* Numele fisierului in upload zone */
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderFileName"],
    .uploadedFile span {
        color: #F7EFE3 !important;
    }

    /* Caption sub imagini */
    .stCaption, 
    .stCaption p,
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #7A5A3A !important;
        font-size: 12px !important;
    }

    /* Caption Original / Restaurat deasupra imaginilor */
    [data-testid="stImage"] figcaption,
    [data-testid="stImage"] + [data-testid="stCaptionContainer"] p {
        color: #7A5A3A !important;
    }              
</style>
""", unsafe_allow_html=True)

# ============================================================
# NAVIGARE
# ============================================================

PAGINI = {
    "🏠 Acasă": "home",
    "🔍 Clasificare": "clasificare",
    "✨ Restaurare": "restaurare",
    "🎨 Recolorare": "recolorare",
    "ℹ️ Despre": "despre"
}

if "pagina_curenta" not in st.session_state:
    st.session_state.pagina_curenta = "home"

# Nav bar orizontal
cols_nav = st.columns([2, 1, 1, 1, 1, 1])
with cols_nav[0]:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; padding:8px 0;">
        <div style="width:30px; height:30px; border-radius:50%; background:#C4823A; 
                    display:flex; align-items:center; justify-content:center;">
            <span style="color:#F7EFE3; font-size:14px;">📷</span>
        </div>
        <span style="color:#2C1810; font-size:16px; font-weight:600;">HeritageVision</span>
    </div>
    """, unsafe_allow_html=True)

for i, (eticheta, valoare) in enumerate(PAGINI.items()):
    with cols_nav[i + 1]:
        este_activ = st.session_state.pagina_curenta == valoare
        if st.button(
            eticheta,
            key=f"nav_{valoare}",
            use_container_width=True,
            type="primary" if este_activ else "secondary"
        ):
            st.session_state.pagina_curenta = valoare
            st.rerun()

st.markdown("<hr style='border:0.5px solid #D9CDBF; margin:8px 0 20px 0;'>", unsafe_allow_html=True)

# ============================================================
# IMPORT PAGINI
# ============================================================

pagina = st.session_state.pagina_curenta

if pagina == "home":
    from pagini.home import afiseaza_home
    afiseaza_home()
elif pagina == "clasificare":
    from pagini.clasificare import afiseaza_clasificare
    afiseaza_clasificare()
elif pagina == "restaurare":
    from pagini.restaurare4 import afiseaza_restaurare
    afiseaza_restaurare()
elif pagina == "recolorare":
    from pagini.recolorare import afiseaza_recolorare
    afiseaza_recolorare()
elif pagina == "despre":
    from pagini.despre import afiseaza_despre
    afiseaza_despre()