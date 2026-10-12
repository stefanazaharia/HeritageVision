"""
=============================================================
SCRIPT COMPLET: Re-etichetare Hough + Antrenare Model v2.0
=============================================================

STRUCTURA SCRIPTULUI:
  Secțiunea 0 — Imports și setări globale
  Secțiunea 1 — Detectori clasici (Hough pentru Rupt, Laplacian pentru Zgâriat)
  Secțiunea 2 — Re-etichetare dataset (rulează O SINGURĂ DATĂ)
  Secțiunea 3 — Arhitectura modelului v2 (CNN + BatchNorm + Spatial Attention)
  Secțiunea 4 — Focal Loss (penalizează mai mult clasele rare)
  Secțiunea 5 — Dataset și augmentări
  Secțiunea 6 — Antrenare cu progress tracking detaliat
  Secțiunea 7 — Main (rulează pașii în ordine)

RULARE PE COLAB:
  Poți rula tot scriptul dintr-o singură celulă, sau îl poți
  împărți în celule separate câte o secțiune.
=============================================================
"""

# ============================================================
# SECȚIUNEA 0 — IMPORTS ȘI SETĂRI GLOBALE
# ============================================================
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import random
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---- CĂI CĂTRE FIȘIERE ----
# CSV-ul complet cu TOATE pozele (degradate + curate)

CALE_CSV = '/content/drive/MyDrive/Licenta/etichetare_alg_clasificare_dog.csv'

# CSV-ul nou generat după re-etichetare (va fi creat automat)
#CALE_CSV_NOU       = '/content/drive/MyDrive/Licenta/etichetare_alg_clasificarev2_0.csv'

# Folderele cu poze
FOLDER_DEGRADATE   = '/content/drive/MyDrive/Licenta/Poze_Degradate'
FOLDER_CURATE      = '/content/drive/MyDrive/Licenta/Poze_Curate'

# Unde se salvează modelul antrenat
CALE_SALVARE_MODEL = '/content/drive/MyDrive/Licenta/model_clasificator_versiunea2_0.pth'

# ---- HIPERPARAMETRI ----
BATCH_SIZE   = 32
EPOCI        = 60
LR           = 0.00005    # Learning rate mic — pornim cu un model nou, vrem stabilitate
#ALPHA_HOUGH  = 0.65       # 65% detector clasic + 35% CLIP pentru Rupt și Zgâriat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Antrenamentul va rula pe: {device.type.upper()}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
# ============================================================
# SECȚIUNEA 1 — DETECTORI CLASICI
# ============================================================
# De ce avem nevoie de asta?
# CLIP (modelul folosit anterior pentru etichetare) nu face
# diferența bine între o zgârietură ușoară și o ruptură masivă.
# Detectorii clasici de mai jos măsoară geometric discontinuitățile
# din imagine — sunt obiectivi și deterministici.

def get_img_path(nume_fisier: str) -> str:
    """
    Caută poza întâi în folderul Degradate, apoi în Curate.
    Returnează calea completă sau None dacă nu există nicăieri.
    """
    path_deg = os.path.join(FOLDER_DEGRADATE, nume_fisier)
    if os.path.exists(path_deg):
        return path_deg

    path_cur = os.path.join(FOLDER_CURATE, nume_fisier)
    if os.path.exists(path_cur):
        return path_cur

    return None


def scor_rupt_hough(img_path: str) -> float:
    """
    Detectează rupturi și crăpături LINIARE cu Hough Line Transform.

    Cum funcționează:
      1. Egalizăm histograma — crăpăturile devin mai vizibile
      2. Canny detectează muchiile (zone de contrast abrupt)
      3. Hough caută linii drepte lungi în acele muchii
      4. Cu cât liniile sunt mai lungi și mai multe, scorul e mai mare

    Returnează: scor 0.0 - 100.0
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    h, w = img.shape

    # Egalizare histogramă — mărește contrastul local
    img_eq = cv2.equalizeHist(img)

    # Canny — detectăm muchiile imaginii
    edges = cv2.Canny(img_eq, 40, 120)

    # Hough Probabilistic — căutăm linii care traversează
    # cel puțin 28% din lățimea imaginii (filtrăm zgomotul scurt)
    linii = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=w * 0.28,
        maxLineGap=25
    )

    if linii is None:
        return 0.0

    # Calculăm lungimea totală a tuturor liniilor găsite
    lungime_totala = sum(
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        for x1, y1, x2, y2 in linii[:, 0]
    )

    # Normalizăm față de diagonala imaginii și limităm la 100
    diagonala = np.sqrt(h ** 2 + w ** 2)
    scor = min(lungime_totala / (diagonala * 1.5), 1.0)

    return round(scor * 100, 2)


def scor_zgariat_laplacian(img_path: str) -> float:
    """
    Detectează zgârieturi FINE cu operatorul Laplacian.

    Cum funcționează:
      1. Laplacian detectează variații locale abrupte de intensitate
      2. Zgârieturile = variații înguste și liniare (nu zone mari)
      3. Filtrăm cu morphology să păstrăm doar structurile liniare
      4. Scorul = procent din pixeli care sunt structuri liniare fine

    Returnează: scor 0.0 - 100.0
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    h, w = img.shape

    # Reducem zgomotul de fundal înainte de Laplacian
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Laplacian — detectăm variațiile de intensitate
    lap = cv2.Laplacian(img_blur, cv2.CV_64F)
    lap_abs = np.abs(lap)

    # Păstrăm doar vârfurile de intensitate (top 8%)
    prag = np.percentile(lap_abs, 92)
    maska = (lap_abs > prag).astype(np.uint8)

    # Filtrăm cu morphology — păstrăm doar structurile liniare
    # (orizontale și verticale), eliminăm petele și zgomotul
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    linii_h  = cv2.morphologyEx(maska, cv2.MORPH_OPEN, kernel_h)
    linii_v  = cv2.morphologyEx(maska, cv2.MORPH_OPEN, kernel_v)

    # Combinăm structurile orizontale și verticale
    structuri_liniare = cv2.bitwise_or(linii_h, linii_v)

    # Scor = procent pixeli liniari, amplificat și limitat la 100
    scor = min(structuri_liniare.sum() / (h * w) * 800, 1.0)

    return round(scor * 100, 2)

# ============================================================
# SECȚIUNEA 2 — RE-ETICHETARE DATASET
# ============================================================
# IMPORTANT: Rulează această funcție O SINGURĂ DATĂ.
# Rezultatul se salvează în CALE_CSV_NOU și îl refolosești
# pentru orice antrenare ulterioară.

def reeticheteaza_dataset() -> pd.DataFrame:
    """
    Citește CSV-ul complet (degradate + curate), rulează
    detectorii clasici pe pozele degradate și combină cu CLIP.

    Formula de combinare:
      Rupt_nou    = 0.65 * Hough    + 0.35 * CLIP_Rupt
      Zgariat_nou = 0.65 * Laplacian + 0.35 * CLIP_Zgariat
      Decolorat, Pete = NESCHIMBATE (CLIP e bun pe astea)

    Pozele CURATE rămân cu 0.0 la toate categoriile — sunt deja corecte.
    """
    print("\n📂 Citire CSV complet...")
    df = pd.read_csv(CALE_CSV_VECHI)

    # Curățăm valorile numerice (înlocuim virgula cu punct dacă e cazul)
    cols = ['Decolorat', 'Zgariat', 'Pete', 'Rupt']
    for col in cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.'),
            errors='coerce'
        ).fillna(0.0)

    print(f"   Total rânduri în CSV: {len(df)}")

    # Identificăm ce poze există fizic pe Drive
    poze_deg = set(os.listdir(FOLDER_DEGRADATE))
    poze_cur = set(os.listdir(FOLDER_CURATE))
    toate_pozele = poze_deg.union(poze_cur)

    # Eliminăm din tabel pozele care nu există pe disc
    df = df[df['Nume_Fisier'].isin(toate_pozele)].reset_index(drop=True)
    print(f"   Poze găsite fizic: {len(df)}")

    # Separăm pozele degradate de cele curate
    # (doar pe cele degradate rulăm detectorii)
    mask_degradate = df['Nume_Fisier'].isin(poze_deg)
    print(f"   Din care degradate: {mask_degradate.sum()}")
    print(f"   Din care curate:    {(~mask_degradate).sum()}")

    # Inițializăm coloanele noi cu valorile existente
    rupt_nou    = df['Rupt'].tolist()
    zgariat_nou = df['Zgariat'].tolist()

    print(f"\n🔍 Rulare detectori pe {mask_degradate.sum()} poze degradate...")
    print("   (Pozele curate sunt sărite — au deja 0.0 la toate categoriile)\n")

    # Iterăm doar prin pozele degradate
    indecsi_degradate = df[mask_degradate].index.tolist()

    for idx in tqdm(indecsi_degradate, desc="Re-etichetare"):
        row      = df.iloc[idx]
        img_path = get_img_path(row['Nume_Fisier'])

        if img_path is None:
            continue  # Poza lipsă — păstrăm valoarea CLIP

        # Rulăm detectorii clasici
        h_scor = scor_rupt_hough(img_path)
        l_scor = scor_zgariat_laplacian(img_path)

        # Combinăm cu etichetele CLIP
        rupt_nou[idx]    = round(ALPHA_HOUGH * h_scor    + (1 - ALPHA_HOUGH) * row['Rupt'],    2)
        zgariat_nou[idx] = round(ALPHA_HOUGH * l_scor    + (1 - ALPHA_HOUGH) * row['Zgariat'], 2)

    df['Rupt']    = rupt_nou
    df['Zgariat'] = zgariat_nou

    # Salvăm CSV-ul nou
    df.to_csv(CALE_CSV_NOU, index=False)

    # ---- Raport comparativ ----
    df_vechi = pd.read_csv(CALE_CSV_VECHI)
    for col in cols:
        df_vechi[col] = pd.to_numeric(
            df_vechi[col].astype(str).str.replace(',', '.'),
            errors='coerce'
        ).fillna(0.0)

    print("\n📊 Comparație medie înainte vs după re-etichetare:")
    print(f"   {'Categorie':<12} {'Înainte':>10} {'După':>10} {'Diferență':>12}")
    print(f"   {'-'*46}")
    for col in cols:
        inainte = df_vechi[col].mean()
        dupa    = df[col].mean()
        diff    = dupa - inainte
        semn    = '+' if diff >= 0 else ''
        print(f"   {col:<12} {inainte:>10.2f} {dupa:>10.2f} {semn+f'{diff:.2f}':>12}")

    print(f"\n✅ CSV nou salvat la: {CALE_CSV_NOU}")
    return df

# ============================================================
# SECȚIUNEA 3 — ARHITECTURA MODELULUI v2
# ============================================================

class SpatialAttention(nn.Module):
    """
    Mecanism de atenție spațială.

    De ce avem nevoie de asta?
    Un CNN normal tratează toți pixelii la fel. Atenția spațială
    învață să se concentreze pe zonele relevante — de exemplu,
    o linie de ruptură sau o zonă cu pete — și ignoră fundalul.

    Cum funcționează:
      - Calculăm media și maximul pe canalele feature map-ului
      - Le concatenăm și aplicăm o convoluție 7x7
      - Sigmoid produce o hartă de atenție (0=ignoră, 1=focalizează)
      - Înmulțim feature map-ul original cu harta de atenție
    """
    def __init__(self):
        super().__init__()
        # Convoluție 7x7 pentru a captura contexte spațiale largi
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Calculăm statistici pe dimensiunea canalelor
        avg_pool = torch.mean(x, dim=1, keepdim=True)   # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W]

        # Concatenăm și generăm harta de atenție
        attn = torch.cat([avg_pool, max_pool], dim=1)   # [B, 2, H, W]
        attn = torch.sigmoid(self.conv(attn))            # [B, 1, H, W]

        # Aplicăm atenția: modelul "vede" mai clar zonele importante
        return x * attn


class ClassifierCNNv2(nn.Module):
    """
    Arhitectura CNN îmbunătățită față de v1.

    Noutăți față de v1:
      ✅ BatchNorm2d după fiecare conv — antrenare mai stabilă,
         gradienți mai buni, convergență mai rapidă
      ✅ Spatial Attention după ultimul bloc — modelul se
         concentrează pe zonele cu defecte, nu pe toată poza
      ✅ Dropout redus la 0.5 — BatchNorm compensează overfitting-ul
         deci nu mai avem nevoie de 0.6

    Dimensiuni:
      Input:  [B, 3, 256, 256]
      Conv1 → Pool: [B, 32,  128, 128]
      Conv2 → Pool: [B, 64,   64,  64]
      Conv3 → Pool: [B, 128,  32,  32]
      Conv4 → Pool: [B, 256,  16,  16]
      Attention:    [B, 256,  16,  16]
      Flatten:      [B, 65536]
      FC1:          [B, 512]
      FC2 (output): [B, 4]    ← Decolorat, Zgâriat, Pete, Rupt
    """
    def __init__(self):
        super().__init__()

        # ---- Blocuri convoluționale cu BatchNorm ----
        self.conv1 = nn.Conv2d(3,   32,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,  64,  kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        # ---- Atenție spațială ----
        self.attention = SpatialAttention()

        # ---- Utilități ----
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # ---- Straturi fully connected ----
        # 256 canale × 16 × 16 pixeli = 65536 neuroni
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        # Trecem prin cele 4 blocuri conv → BN → ReLU → Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Aplicăm atenția spațială
        x = self.attention(x)

        # Aplatizăm în vector 1D
        x = x.view(-1, 256 * 16 * 16)

        # Luăm decizia finală
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Nota: Sigmoid-ul e aplicat intern de FocalBCELoss
        # pentru stabilitate numerică — nu îl aplicăm manual aici
        return x

# ============================================================
# SECȚIUNEA 4 — FOCAL LOSS
# ============================================================

class FocalBCELoss(nn.Module):
    """
    BCEWithLogitsLoss + factor focal pentru clase dezechilibrate.

    De ce avem nevoie de asta?
    Rupt are doar 4.9% poze cu degradare severă.
    Cu loss standard, modelul ignoră aceste cazuri rare și tot
    scade loss-ul prezicând valori mici mereu.
    Focal Loss penalizează MULT mai tare erorile pe cazurile rare.

    Parametri:
      pos_weight: tensor [4] — ponderi per clasă
                  [Decolorat=1.8, Zgâriat=4.5, Pete=2.2, Rupt=5.5]
      gamma:      2.0 = standard din paper-ul original Facebook AI
    """
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='none'   # vrem loss per element, nu media
        )

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        # pt = probabilitatea predicției corecte
        # Dacă modelul e sigur și corect → pt ≈ 1 → focal ≈ 0
        # Dacă modelul e sigur și greșit  → pt ≈ 0 → focal ≈ bce_loss
        pt    = torch.exp(-bce_loss)
        focal = ((1 - pt) ** self.gamma) * bce_loss

        return focal.mean()

# ============================================================
# SECȚIUNEA 5 — DATASET ȘI AUGMENTĂRI
# ============================================================

class SimuleazaCrapaturi:
    """
    Augmentare specifică pentru rupturi și îndoituri.

    Adaugă linii fine de contrast redus care simulează îndoituri
    ale fotografiei. Aceasta NU e degradare sintetică — e o tehnică
    standard de augmentare pentru a crește varietatea datelor de
    antrenare pe clasa subreprezentată (Rupt).

    Se aplică cu probabilitate p=0.35 — adică 35% din imagini
    vor primi această augmentare în mod aleatoriu la fiecare epocă.
    """
    def __init__(self, p: float = 0.35):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img   # Nu aplicăm augmentarea de data asta

        img_arr = np.array(img).copy()
        h, w    = img_arr.shape[:2]

        numar_linii = random.randint(1, 2)
        for _ in range(numar_linii):
            # Linie în zona centrală a pozei (evităm marginile)
            y       = random.randint(h // 5, 4 * h // 5)
            x_start = random.randint(0, w // 4)
            x_end   = random.randint(3 * w // 4, w)
            grosime = random.randint(1, 3)

            # Factor de întunecime aleatoriu — simulează îndoitura
            factor = random.uniform(0.45, 0.70)
            img_arr[y:y + grosime, x_start:x_end] = (
                img_arr[y:y + grosime, x_start:x_end] * factor
            ).astype(np.uint8)

        return Image.fromarray(img_arr)


class ResizeAndPadEdge:
    """
    Redimensionare cu păstrarea proporțiilor + padding cu pixeli de margine.

    De ce padding cu 'edge' și nu cu negru?
    Padding-ul cu negru ar introduce artefacte — rețeaua ar putea
    asocia colțurile negre cu un tip de degradare. Repetând ultimul
    pixel de margine (edge padding) evităm acest efect.
    """
    def __init__(self, target_size: int = 256):
        self.target_size = target_size

    def __call__(self, img):
        w, h  = img.size
        ratio = self.target_size / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img   = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        dw    = self.target_size - new_w
        dh    = self.target_size - new_h
        pad_l = dw // 2
        pad_r = dw - pad_l
        pad_t = dh // 2
        pad_b = dh - pad_t

        return TF.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode='edge')


# Pipeline de transformări pentru antrenare (cu augmentări)
transformari_train = transforms.Compose([
    ResizeAndPadEdge(256),
    SimuleazaCrapaturi(p=0.35),          # augmentare specifică pentru Rupt
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Pipeline pentru validare (fără augmentări — vrem evaluare corectă)
transformari_val = transforms.Compose([
    ResizeAndPadEdge(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class HeritageDataset(Dataset):
    """
    Dataset care caută pozele în ambele foldere (Degradate și Curate).
    """
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row      = self.dataframe.iloc[idx]
        img_path = get_img_path(row['Nume_Fisier'])

        if img_path is None:
            raise FileNotFoundError(f"Poza nu există: {row['Nume_Fisier']}")

        image  = Image.open(img_path).convert('RGB')
        labels = torch.tensor(
            row[['Decolorat', 'Zgariat', 'Pete', 'Rupt']].values.astype(float),
            dtype=torch.float32
        ) / 100.0   # Normalizăm procentele la 0-1

        if self.transform:
            image = self.transform(image)

        return image, labels
    
    # ============================================================
# SECȚIUNEA 6 — ANTRENARE CU PROGRESS TRACKING DETALIAT
# ============================================================

def antreneaza(df: pd.DataFrame):
    """
    Antrenează modelul v2 și afișează progresul detaliat
    la fiecare epocă + un sumar la final.
    """

    # ---- Pregătire date ----
    # Eliminăm pozele care nu există fizic pe disc
    poze_disponibile = (
        set(os.listdir(FOLDER_DEGRADATE)) |
        set(os.listdir(FOLDER_CURATE))
    )
    df = df[df['Nume_Fisier'].isin(poze_disponibile)].reset_index(drop=True)
    print(f"\n📊 Poze disponibile pentru antrenare: {len(df)}")

    # Împărțim 70% train, 30% validare
    df_train, df_val = train_test_split(df, test_size=0.3, random_state=42)
    print(f"   Train: {len(df_train)} | Validare: {len(df_val)}")

    ds_train = HeritageDataset(df_train, transformari_train)
    ds_val   = HeritageDataset(df_val,   transformari_val)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ---- Model ----
    model = ClassifierCNNv2().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model v2 inițializat")
    print(f"   Parametri antrenabili: {total_params:,}")

    if os.path.exists(CALE_SALVARE_MODEL):
      model.load_state_dict(torch.load(CALE_SALVARE_MODEL, map_location=device))
      print(f"✅ Model anterior încărcat din: {CALE_SALVARE_MODEL}")
    else:
      print("⚠️  Nu există model salvat — antrenare de la zero.")

    # ---- Loss cu ponderi calibrate pe distribuția reală ----
    # Ponderile sunt inverse proporționale cu frecvența cazurilor severe:
    #   Decolorat: 18.2% cazuri >50% → pondere 1.8 (distribuit bine)
    #   Zgâriat:    4.2% cazuri >50% → pondere 4.5 (rar)
    #   Pete:      14.4% cazuri >50% → pondere 2.2 (distribuit ok)
    #   Rupt:       4.9% cazuri >50% → pondere 5.5 (cel mai rar)
    pos_weight = torch.tensor([1.8, 5.0, 2.2, 6.0]).to(device)
    criterion  = FocalBCELoss(pos_weight=pos_weight, gamma=2.0)

    # ---- Optimizer ----
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # ---- Scheduler — reduce LR când loss-ul stagnează ----
    # Dacă în 5 epoci consecutive nu se îmbunătățește loss-ul de validare,
    # înjumătățim learning rate-ul automat (până la minim 1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='min',
      factor=0.5,
      patience=5,
      min_lr=1e-7
    )

    # ---- Bucla de antrenare ----
    cel_mai_bun_loss = float('inf')
    istoric_train    = []
    istoric_val      = []
    lr_curent        = LR

    print(f"\n{'='*65}")
    print(f"{'EPOCĂ':>6} | {'LOSS TRAIN':>10} | {'LOSS VAL':>10} | "
          f"{'LR':>10} | {'STATUS':>15}")
    print(f"{'='*65}")

    timp_start_total = time.time()

    for epoca in range(EPOCI):
        timp_start_epoca = time.time()

        # ---- FAZA TRAIN ----
        model.train()
        loss_train_total = 0.0

        for imagini, etichete in tqdm(
            train_loader,
            desc=f"  Epoca {epoca+1:3d}/{EPOCI}",
            leave=True   # Bara tqdm dispare după fiecare epocă
        ):
            imagini, etichete = imagini.to(device), etichete.to(device)

            optimizer.zero_grad()
            predictii = model(imagini)
            loss      = criterion(predictii, etichete)
            loss.backward()
            optimizer.step()

            loss_train_total += loss.item()

        media_train = loss_train_total / len(train_loader)

        # ---- FAZA VALIDARE ----
        model.eval()
        loss_val_total = 0.0

        with torch.no_grad():
            for imagini, etichete in val_loader:
                imagini, etichete = imagini.to(device), etichete.to(device)
                predictii = model(imagini)
                loss_val_total += criterion(predictii, etichete).item()

        media_val = loss_val_total / len(val_loader)

        # Actualizăm scheduler-ul cu loss-ul de validare
        lr_vechi = optimizer.param_groups[0]['lr']
        scheduler.step(media_val)
        lr_nou = optimizer.param_groups[0]['lr']

        # Salvăm în istoric pentru grafic final
        istoric_train.append(media_train)
        istoric_val.append(media_val)

        # ---- Determinăm statusul epocii ----
        timp_epoca = time.time() - timp_start_epoca

        if media_val < cel_mai_bun_loss:
            cel_mai_bun_loss = media_val
            torch.save(model.state_dict(), CALE_SALVARE_MODEL)
            status = "✅ SALVAT"
        elif lr_nou < lr_vechi:
            status = "⬇️  LR redus"
        else:
            status = ""

        # ---- Print progres ----
        print(
            f"{epoca+1:>6} | "
            f"{media_train:>10.4f} | "
            f"{media_val:>10.4f} | "
            f"{lr_nou:>10.2e} | "
            f"{status:>15}"
            f"  [{timp_epoca:.0f}s]"
        )

    # ---- Sumar final ----
    timp_total = time.time() - timp_start_total
    ore   = int(timp_total // 3600)
    min_  = int((timp_total % 3600) // 60)
    sec   = int(timp_total % 60)

    print(f"\n{'='*65}")
    print(f"✅ ANTRENARE FINALIZATĂ")
    print(f"   Timp total: {ore}h {min_}m {sec}s")
    print(f"   Cel mai bun Loss Validare: {cel_mai_bun_loss:.4f}")
    print(f"   Model salvat la: {CALE_SALVARE_MODEL}")
    print(f"{'='*65}")

    # Afișăm evoluția loss-ului din 10 în 10 epoci
    print(f"\n📈 Evoluție Loss (din 10 în 10 epoci):")
    print(f"   {'Epocă':>6} | {'Train':>8} | {'Val':>8}")
    print(f"   {'-'*28}")
    for i in range(0, EPOCI, 10):
        print(f"   {i+1:>6} | {istoric_train[i]:>8.4f} | {istoric_val[i]:>8.4f}")
    # Adăugăm și ultima epocă dacă nu e multiplu de 10
    if (EPOCI - 1) % 10 != 0:
        print(f"   {EPOCI:>6} | {istoric_train[-1]:>8.4f} | {istoric_val[-1]:>8.4f}")

# ============================================================
# SECȚIUNEA 7 — MAIN
# ============================================================

if __name__ == "__main__":

    # ---- PASUL 1: Re-etichetare ----
    # Rulează O SINGURĂ DATĂ — durează 30-60 min
    # Dacă ai deja etichetare_alg_clasificarev2_0.csv, comentează acest bloc
    # și citește direct CSV-ul existent.
    #print("=" * 65)
    #print("PASUL 1: Re-etichetare Rupt + Zgâriat cu detectori clasici")
    #print("=" * 65)
    #df_nou = reeticheteaza_dataset()

    # Dacă vrei să sari re-etichetarea și să folosești CSV-ul existent:
    print("📂 Citire CSV re-etichetat...")
    df_nou = pd.read_csv(CALE_CSV)
    for col in ['Decolorat', 'Zgariat', 'Pete', 'Rupt']:
        df_nou[col] = pd.to_numeric(
            df_nou[col].astype(str).str.replace(',', '.'),
            errors='coerce'
        ).fillna(0.0)
    print(f"✅ CSV încărcat: {len(df_nou)} rânduri")

    # ---- PASUL 2: Antrenare ----
    print("\n" + "=" * 65)
    print("PASUL 2: Antrenare Model v2.0")
    print("=" * 65)
    antreneaza(df_nou)
