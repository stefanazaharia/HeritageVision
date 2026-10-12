# ============================================================
# HERITAGEVISION - NOTEBOOK COLAB PENTRU ANTRENAMENT
# ============================================================
# Copiaza fiecare CELULA intr-o celula separata in Colab
# ============================================================


# ============================================================
# CELULA 1: Verificare GPU
# ============================================================

import torch

print(f"CUDA disponibil: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memorie GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  NU ai GPU activat!")
    print("Mergi la: Runtime -> Change runtime type -> T4 GPU")


# ============================================================
# CELULA 2: Conectare Google Drive
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# Verificare foldere
import os

CALE_LICENTA = '/content/drive/MyDrive/Licenta'
FOLDER_CURATE = f'{CALE_LICENTA}/Poze_RAW_Curate'
FOLDER_DEGRADATE = f'{CALE_LICENTA}/Poze_RAW_Degradate'

assert os.path.exists(FOLDER_CURATE), f"Nu gasesc {FOLDER_CURATE}"
assert os.path.exists(FOLDER_DEGRADATE), f"Nu gasesc {FOLDER_DEGRADATE}"

# Numaram pozele
extensii_imagini = ('.jpg', '.jpeg', '.png')
numar_curate = sum(
    1 for f in os.listdir(FOLDER_CURATE) if f.lower().endswith(extensii_imagini)
)
numar_degradate = sum(
    1 for f in os.listdir(FOLDER_DEGRADATE) if f.lower().endswith(extensii_imagini)
)

print(f"Poze curate:    {numar_curate}")
print(f"Poze degradate: {numar_degradate}")


# ============================================================
# CELULA 3: Setup folder de proiect si import scripturi
# ============================================================

import sys

# Creeaza folderele
FOLDER_PROIECT = f'{CALE_LICENTA}/HeritageVision'
FOLDER_COD = f'{FOLDER_PROIECT}/cod'

os.makedirs(FOLDER_PROIECT, exist_ok=True)
os.makedirs(FOLDER_COD, exist_ok=True)
os.makedirs(f'{FOLDER_PROIECT}/checkpoints', exist_ok=True)
os.makedirs(f'{FOLDER_PROIECT}/esantioane', exist_ok=True)

# Adauga folder-ul cu cod in path-ul Python
if FOLDER_COD not in sys.path:
    sys.path.insert(0, FOLDER_COD)

print(f"Folder cod:         {FOLDER_COD}")
print(f"Fisiere in folder:  {os.listdir(FOLDER_COD)}")
print()
print("IMPORTANT: Uploadeaza urmatoarele fisiere in:")
print(f"  {FOLDER_COD}")
print()
print("Fisiere necesare:")
print("  - incarcator_date.py")
print("  - arhitecturi_retele.py")
print("  - antrenor_cyclegan.py")
print("  - restaurator_imagini.py")


# ============================================================
# CELULA 4: Test rapid incarcator (30 secunde)
# ============================================================

from incarcator_date import testeaza_incarcator

dataset_test = testeaza_incarcator(
    folder_curate=FOLDER_CURATE,
    folder_degradate=FOLDER_DEGRADATE,
    dimensiune=128,
)


# ============================================================
# CELULA 5: Test rapid retele (30 secunde)
# ============================================================

from arhitecturi_retele import testeaza_retelele

generator, discriminator = testeaza_retelele(dimensiune_imagine=128, device='cuda')


# ============================================================
# CELULA 6: SMOKE TEST - antrenament scurt (~1-2 ore)
# ============================================================
# Ce verificam:
#   - Training-ul ruleaza fara erori
#   - Loss-urile se modifica (G loss 2-10, D loss 0.3-0.7)
#   - In folderul esantioane_test se salveaza poze cu progresul

from antrenor_cyclegan import porneste_antrenament, configuratie_test_rapid

configuratie = configuratie_test_rapid(CALE_LICENTA)

print("CONFIGURATIE TEST:")
for cheie, valoare in configuratie.items():
    print(f"  {cheie}: {valoare}")

porneste_antrenament(configuratie)


# ============================================================
# CELULA 7: Verifica rezultatele smoke test-ului
# ============================================================

from PIL import Image
import matplotlib.pyplot as plt

folder_esantioane_test = f'{FOLDER_PROIECT}/esantioane_test'
fisiere_esantioane = sorted(os.listdir(folder_esantioane_test))

print(f"Esantioane salvate: {fisiere_esantioane}")

if fisiere_esantioane:
    ultimul_esantion = fisiere_esantioane[-1]
    cale = f'{folder_esantioane_test}/{ultimul_esantion}'
    
    imagine_esantion = Image.open(cale)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(imagine_esantion)
    plt.title(
        f'Smoke test - {ultimul_esantion}\n'
        '(sus: poze degradate / jos: restaurate)'
    )
    plt.axis('off')
    plt.show()

# VERIFICARI:
# ✓ Imaginile de jos trebuie sa arate DIFERIT de cele de sus
# ✓ NU trebuie sa fie perfecte - e doar 10 epoci
# ✗ Daca sunt identice = e o problema
# ✗ Daca sunt pure noise = trebuie ajustata rata de invatare


# ============================================================
# CELULA 8: ANTRENAMENT COMPLET (zile intregi!)
# ============================================================
# Ruleaza asta doar dupa ce smoke test-ul merge OK
# Pe Colab gratuit: 5-8 zile cu deconectari si reluari

from antrenor_cyclegan import porneste_antrenament, configuratie_antrenament_complet

configuratie = configuratie_antrenament_complet(CALE_LICENTA)

# OPTIONAL: limiteaza pozele pentru antrenament mai rapid
# configuratie['limita_poze'] = 5000

print("CONFIGURATIE ANTRENAMENT COMPLET:")
for cheie, valoare in configuratie.items():
    print(f"  {cheie}: {valoare}")

porneste_antrenament(configuratie)


# ============================================================
# CELULA 9: RELUARE dupa deconectare Colab
# ============================================================
# Dupa ce Colab te deconecteaza, rulezi asta in loc de CELULA 8

import glob
from antrenor_cyclegan import porneste_antrenament, configuratie_antrenament_complet

configuratie = configuratie_antrenament_complet(CALE_LICENTA)

# Gaseste ultimul checkpoint disponibil
lista_checkpoints = sorted(
    glob.glob(f"{configuratie['folder_checkpoints']}/checkpoint_epoca_*.pth")
)

if lista_checkpoints:
    configuratie['reluare_din'] = lista_checkpoints[-1]
    print(f"Continui de la: {lista_checkpoints[-1]}")
else:
    print("Nu exista checkpoint, incep de la zero")

porneste_antrenament(configuratie)


# ============================================================
# CELULA 10: TESTARE INFERENTA - restaurare simpla
# ============================================================

from restaurator_imagini import RestauratorImagini
from PIL import Image
import matplotlib.pyplot as plt

# Incarcam modelul antrenat
cale_model = f'{FOLDER_PROIECT}/checkpoints/generator_restaurare_ultim.pth'
restaurator = RestauratorImagini(cale_model, dimensiune_imagine=256)

# Alege o poza degradata pentru test (schimba numele!)
cale_poza_test = f'{FOLDER_DEGRADATE}/numele_pozei_tale.jpg'

imagine_originala = Image.open(cale_poza_test)
imagine_restaurata = restaurator.restaureaza_complet(imagine_originala)

# Afisam comparatia
figura, axe = plt.subplots(1, 2, figsize=(14, 7))
axe[0].imshow(imagine_originala)
axe[0].set_title('Degradat')
axe[0].axis('off')
axe[1].imshow(imagine_restaurata)
axe[1].set_title('Restaurat')
axe[1].axis('off')
plt.show()


# ============================================================
# CELULA 11: TESTARE INFERENTA CU MASCA (control selectiv)
# ============================================================

from restaurator_imagini import RestauratorImagini
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

restaurator = RestauratorImagini(
    f'{FOLDER_PROIECT}/checkpoints/generator_restaurare_ultim.pth',
    dimensiune_imagine=256
)

imagine = Image.open(cale_poza_test)
latime, inaltime = imagine.size

# Construim o masca dreptunghiulara (simulam ce ar desena utilizatorul)
masca = Image.new('L', (latime, inaltime), 0)
desenator = ImageDraw.Draw(masca)
desenator.rectangle(
    (latime // 4, inaltime // 4, 3 * latime // 4, 3 * inaltime // 4),
    fill=255
)

# Restaurare selectiva
rezultat_selectiv = restaurator.restaureaza_cu_masca(
    imagine, masca, pixeli_tranzitie=15
)

# Afisam
figura, axe = plt.subplots(1, 3, figsize=(18, 6))

axe[0].imshow(imagine)
axe[0].set_title('Original')
axe[0].axis('off')

# Overlay masca pe original pentru vizualizare
suprapunere = np.array(imagine).copy()
masca_numpy = np.array(masca)
suprapunere[masca_numpy > 0] = (
    suprapunere[masca_numpy > 0] * 0.5 +
    np.array([255, 0, 0]) * 0.5
)
axe[1].imshow(suprapunere.astype(np.uint8))
axe[1].set_title('Masca (rosu = zona de reparat)')
axe[1].axis('off')

axe[2].imshow(rezultat_selectiv)
axe[2].set_title('Restaurat doar in masca')
axe[2].axis('off')

plt.show()
