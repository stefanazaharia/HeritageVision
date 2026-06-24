import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF


class ResizeAndPadEdge:
    def __init__(self, target_size=256):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        ratio = self.target_size / max_side
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left
        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), padding_mode='edge')


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


class ClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64,  kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.attention = SpatialAttention()
        self.pool      = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout   = nn.Dropout(0.5)
        self.fc1       = nn.Linear(256 * 16 * 16, 512)
        self.fc2       = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.attention(x)
        x = x.view(-1, 256 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


@st.cache_resource
def incarca_model_clasificare():
    model = ClassifierCNN()
    model.load_state_dict(torch.load(
        r"D:\LICENTA\models\clasificator_versiuni\model_clasificator_versiunea2_0.pth",
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model


transformari_test = transforms.Compose([
    ResizeAndPadEdge(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ETICHETE = ["Decolorat", "Zgâriat", "Pete", "Rupt"]
PRAGURI  = {"Decolorat": 35, "Zgâriat": 20, "Pete": 40, "Rupt": 20}
ICONURI  = {"Decolorat": "🟡", "Zgâriat": "🔶", "Pete": "🟤", "Rupt": "🔴"}

# Etichete vizuale — "Rupt" apare ca "Rupt și Îndoit" in interfata
ETICHETE_AFISARE = {
    "Decolorat": "Decolorat",
    "Zgâriat":   "Zgâriat",
    "Pete":      "Pete",
    "Rupt":      "Rupt și Îndoit",
}

DESCRIERI = {
    "Decolorat": "Fotografia și-a pierdut tonurile originale din cauza expunerii la lumină sau umiditate.",
    "Zgâriat":   "Suprafața fotografiei prezintă zgârieturi fizice cauzate de manipulare incorectă.",
    "Pete":      "Pete de umiditate, chimice sau de altă natură afectează suprafața fotografiei.",
    "Rupt":      "Fotografia prezintă rupturi, îndoituri sau lipsuri fizice la margini sau în suprafață.",
}


def afiseaza_clasificare():
    
    st.markdown("""
    <div class="card" style="margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="font-size:28px;">🔍</div>
            <div>
                <div style="font-size:16px; font-weight:600; color:#2C1810;">Clasificare defecte fotografice</div>
                <div style="font-size:12px; color:#9A7A5A; margin-top:2px;">
                    Model CNN cu Spatial Attention — detectează automat tipul și severitatea degradării
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        model = incarca_model_clasificare()
    except FileNotFoundError:
        st.error("❌ Nu găsesc modelul de clasificare. Verifică calea din `pagini/clasificare.py`.")
        st.stop()
    
    col_stanga, col_dreapta = st.columns([1, 1], gap="large")
    
    with col_stanga:
        st.markdown('<div class="sectiune-titlu">📁 Încarcă fotografia</div>', 
                    unsafe_allow_html=True)
        
        fisier = st.file_uploader(
            "Trage fotografia aici sau apasă pentru a alege",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if fisier:
            imagine = Image.open(fisier).convert("RGB")
            # use_column_width in loc de use_container_width (Streamlit 1.38)
            st.image(imagine, caption="Fotografie încărcată", use_column_width=True)
            
            w, h = imagine.size
            size_kb = fisier.size / 1024
            
            st.markdown(f"""
            <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
                <span class="badge">📐 {w}×{h}px</span>
                <span class="badge">💾 {size_kb:.0f} KB</span>
                <span class="badge">🎨 RGB</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔍 Analizează defectele", use_container_width=True, type="primary"):
                with st.spinner("Modelul CNN analizează imaginea..."):
                    tensor = transformari_test(imagine).unsqueeze(0)
                    with torch.no_grad():
                        predictii = model(tensor)
                        probabilitati = torch.sigmoid(predictii).squeeze().tolist()
                
                st.session_state.rezultate_clasificare = {
                    eticheta: prob 
                    for eticheta, prob in zip(ETICHETE, probabilitati)
                }
                st.session_state.imagine_clasificata = imagine
                st.rerun()
    
    with col_dreapta:
        st.markdown('<div class="sectiune-titlu">📊 Rezultatele analizei</div>', 
                    unsafe_allow_html=True)
        
        if "rezultate_clasificare" not in st.session_state:
            st.markdown("""
            <div style="background:#FFFCF8; border:1.5px dashed #D9CDBF; border-radius:10px; 
                        padding:32px; text-align:center; color:#9A7A5A;">
                <div style="font-size:32px; margin-bottom:10px;">📷</div>
                <div style="font-size:13px; color:#7A5A3A;">
                    Încarcă o fotografie și apasă<br>"Analizează defectele"
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            rezultate = st.session_state.rezultate_clasificare
            
            defect_principal = max(rezultate, key=rezultate.get)
            prob_max = rezultate[defect_principal]
            
            st.markdown(f"""
            <div style="background:#EDE0CF; border-radius:10px; padding:14px; 
                        margin-bottom:16px; display:flex; align-items:center; gap:12px;">
                <div style="font-size:28px;">{ICONURI[defect_principal]}</div>
                <div>
                    <div style="font-size:11px; color:#7A5A3A;">Defect principal detectat</div>
                    <div style="font-size:16px; font-weight:600; color:#2C1810;">
                        {ETICHETE_AFISARE[defect_principal]}
                    </div>
                    <div style="font-size:11px; color:#C4823A;">{int(prob_max*100)}% probabilitate</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            for eticheta in ETICHETE:
                prob = rezultate[eticheta]
                procent = int(prob * 100)
                prag = PRAGURI[eticheta]
                depasit = procent >= prag
                culoare_bar = "#8B2500" if depasit else "#C4823A"
                
                st.markdown(f"""
                <div class="defect-row">
                    <span class="defect-label">{ICONURI[eticheta]} {ETICHETE_AFISARE[eticheta]}</span>
                    <span class="defect-pct">{procent}%</span>
                </div>
                <div class="defect-bar-bg">
                    <div style="width:{procent}%; height:5px; border-radius:3px; 
                                background:{culoare_bar};"></div>
                </div>
                """, unsafe_allow_html=True)
                
                if depasit:
                    st.caption(f"⚠️ Peste pragul de {prag}% — {DESCRIERI[eticheta]}")
            
            nr_defecte = sum(1 for e, p in rezultate.items() if int(p*100) >= PRAGURI[e])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if nr_defecte == 0:
                st.success("✅ Fotografia este în stare bună — nu necesită restaurare.")
            elif nr_defecte == 1:
                st.warning("⚠️ Un defect semnificativ detectat — restaurare recomandată.")
            else:
                st.error(f"🔴 {nr_defecte} defecte semnificative — restaurare necesară.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("✨ Trimite la restaurare", use_container_width=True):
                st.session_state.pagina_curenta = "restaurare"
                st.session_state.imagine_pentru_restaurare = st.session_state.imagine_clasificata
                st.rerun()