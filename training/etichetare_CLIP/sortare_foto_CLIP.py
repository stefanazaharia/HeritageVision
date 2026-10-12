#CONECTARE LA DRIVE
from google.colab import drive
import os
import zipfile

# 1. Ne conectăm la Google Drive
drive.mount('/content/drive')

# 2. Definim căile
cale_arhiva = '/content/drive/MyDrive/Licenta/data_finale_model_restaurare.zip'
# Folderul temporar și foarte rapid din memoria Colab
folder_extragere = '/content/Poze_Raw'

# 3. Creăm folderul și dezarhivăm
os.makedirs(folder_extragere, exist_ok=True)

print("Incepem dezarhivarea... Te rog asteapta.")
with zipfile.ZipFile(cale_arhiva, 'r') as zip_ref:
    zip_ref.extractall(folder_extragere)

numar_poze = len(os.listdir(folder_extragere))
print(f"Gata! Au fost extrase {numar_poze} de imagini direct în memoria rapidă a Colab-ului.")

import os

# Îi spunem să intre UN NIVEL MAI JOS, direct în folderul extras
folder_real_cu_poze = '/content/Poze_Initiale/data_standardized'

numar_poze = len(os.listdir(folder_real_cu_poze))
print(f"Acum ne uităm în folderul corect! Am găsit: {numar_poze} imagini.")

import os

print("--- Ce vede Colab în Drive-ul tău principal ---")
print(os.listdir('/content/drive/MyDrive/'))

print("\n--- Ce vede Colab în folderul Licenta ---")
try:
    print(os.listdir('/content/drive/MyDrive/Licenta/'))
except FileNotFoundError:
    print("❌ EROARE: Folderul 'Licenta' nu există! Verifică cum l-ai scris cu litere mari/mici.")

    import os
import shutil
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# 1. Căile de intrare și ieșire
folder = '/content/Poze_Raw'

# Salvăm rezultatele permanent pe Drive-ul tău
folder_curate = '/content/drive/MyDrive/Licenta/Poze_RAW_Curate'
folder_degradate = '/content/drive/MyDrive/Licenta/Poze_RAW_Degradate'

os.makedirs(folder_curate, exist_ok=True)
os.makedirs(folder_degradate, exist_ok=True)

# 2. Încărcăm modelul CLIP (Folosim fix versiunile preinstalate de Google)
print("\nIncărcăm asistentul CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

texte_sortare = [
    "a clean, perfect, well-preserved photograph",
    "a damaged, scratched, stained, or faded photograph"
]

print(f"Începem sortarea. Folosim: {device.upper()}")

poze_procesate = 0
poze_bune = 0
poze_stricate = 0

# 3. Procesarea
for nume_fisier in tqdm(os.listdir(folder)):
    if not nume_fisier.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    cale_poza = os.path.join(folder, nume_fisier)

    try:
        imagine = Image.open(cale_poza).convert("RGB")
        inputs = processor(text=texte_sortare, images=imagine, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probabilitati = outputs.logits_per_image.softmax(dim=1)[0]

        scor_curat = probabilitati[0].item()
        scor_degradat = probabilitati[1].item()

        # 4. Mutăm pozele pe Drive în funcție de decizia CLIP
        if scor_degradat > scor_curat:
            shutil.move(cale_poza, os.path.join(folder_degradate, nume_fisier))
            poze_stricate += 1
        else:
            shutil.move(cale_poza, os.path.join(folder_curate, nume_fisier))
            poze_bune += 1

        poze_procesate += 1

    except Exception as e:
        print(f"Eroare la procesarea {nume_fisier}: {e}")

print("\n--- RAPORT FINAL SORTARE ---")
print(f"Mutate în /Poze_RAW_Curate: {poze_bune}")
print(f"Mutate în /Poze_RAW_Degradate: {poze_stricate}")
print("Verifică Google Drive-ul, folderele tale sunt acum pline și sortate!")

import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# 1. Încărcăm modelul CLIP de la OpenAI
print("Descărcăm modelul CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Definim etichetele de text (Dicționarul nostru)
# CLIP funcționează cel mai bine în engleză
etichete_text = [
    "a faded photograph",                  # [0] Decolorat
    "a photograph with scratches",         # [1] Zgariat
    "a photograph with stains",            # [2] Pete
    "a torn or folded photograph",         # [3] Rupt
    "a perfect, clean, intact photograph"  # [4] Curat
]

folder_degradate = '/content/drive/MyDrive/Licenta/Poze_Degradate'
rezultate = []

print(f"Începem analiza pe GPU: {device}")

# 3. Analizăm fiecare poză
for nume_fisier in tqdm(os.listdir(folder_degradate)):
    if not nume_fisier.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    cale_poza = os.path.join(folder_degradate, nume_fisier)

    try:
        imagine = Image.open(cale_poza).convert("RGB")
        inputs = processor(text=etichete_text, images=imagine, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # APLICĂM SOFTMAX AICI (Nu Sigmoid)
        logits_per_image = outputs.logits_per_image
        probabilitati = logits_per_image.softmax(dim=1)[0].cpu().numpy()

        # Salvăm rezultatele (ignorăm probabilitatea de pe indexul [4] care e "Curat",
        # pentru că ne interesează doar cât la sută din ea e degradare)
        rezultate.append({
            'Nume_Fisier': nume_fisier,
            'Decolorat': round(probabilitati[0] * 100, 2),
            'Zgariat': round(probabilitati[1] * 100, 2),
            'Pete': round(probabilitati[2] * 100, 2),
            'Rupt': round(probabilitati[3] * 100, 2)
        })

    except Exception as e:
        print(f"Eroare la procesarea {nume_fisier}: {e}")

# 4. Salvăm totul într-un Excel/CSV
df_rezultate = pd.DataFrame(rezultate)
cale_csv = '/content/drive/MyDrive/Licenta/etichete.csv'
df_rezultate.to_csv(cale_csv, index=False)

print(f"\nGata! Fișierul a fost salvat la: {cale_csv}")
print(df_rezultate.head()) # Afișăm primele 5 rânduri ca să vezi cum arată

import os
import pandas as pd

# 1. Încărcăm tabelul făcut adineauri de CLIP (cel cu pozele stricate)
cale_csv_initial = '/content/drive/MyDrive/Licenta/etichete.csv'
df_degradate = pd.read_csv(cale_csv_initial)

# 2. Definim unde stau pozele perfecte
folder_curate = '/content/drive/MyDrive/Licenta/Poze_Curate'
date_curate = []

# 3. Le dăm 0% la toate defectele
for nume_fisier in os.listdir(folder_curate):
    if nume_fisier.lower().endswith(('.png', '.jpg', '.jpeg')):
        date_curate.append({
            'Nume_Fisier': nume_fisier,
            'Decolorat': 0.0,
            'Zgariat': 0.0,
            'Pete': 0.0,
            'Rupt': 0.0
        })

# 4. Lipim cele două liste și salvăm tabelul final, suprem!
df_curate = pd.DataFrame(date_curate)
df_final = pd.concat([df_degradate, df_curate], ignore_index=True)

cale_csv_final = '/content/drive/MyDrive/Licenta/etichete_complete.csv'
df_final.to_csv(cale_csv_final, index=False)

print(f"Baza de date este 100% completă! Avem un total de {len(df_final)} fotografii pentru antrenare.")
print(f"Salvat în: {cale_csv_final}")
