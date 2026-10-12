"""
incarcator_date.py - Incarca imagini din Google Drive pentru CycleGAN
======================================================================

Citeste pozele din 2 foldere (curate si degradate) si le pregateste
ca tensori PyTorch pentru antrenament. Foloseste augmentari standard
(flip, crop, color jitter) - NU creeaza date sintetice noi.

IMPORTANT: imaginile sunt NEPERECHE. La fiecare iteratie, ia o poza
curata si o poza degradata aleator (nu sunt relationate intre ele).
Asta e specific CycleGAN-ului.
"""

import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SetDeImaginiNepereche(Dataset):
    """
    Dataset PyTorch care returneaza perechi aleatoare (curat, degradat)
    pentru antrenamentul CycleGAN.
    
    Parametri:
        folder_curate: calea catre /content/drive/MyDrive/Licenta/Poze_RAW_Curate
        folder_degradate: calea catre /content/drive/MyDrive/Licenta/Poze_RAW_Degradate
        dimensiune_imagine: 128 (test rapid) sau 256 (antrenament real)
        mod: 'antrenament' (cu augmentari) sau 'test' (fara)
        limita_poze: None = toate, sau un numar pentru smoke test
    """
    
    def __init__(self, folder_curate, folder_degradate, dimensiune_imagine=256,
                 mod='antrenament', limita_poze=None):
        
        self.folder_curate = Path(folder_curate)
        self.folder_degradate = Path(folder_degradate)
        self.dimensiune_imagine = dimensiune_imagine
        self.mod = mod
        
        # Colectam caile tuturor imaginilor (jpg, jpeg, png)
        extensii = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        
        self.cai_curate = []
        self.cai_degradate = []
        
        for extensie in extensii:
            self.cai_curate.extend(sorted(self.folder_curate.glob(extensie)))
            self.cai_degradate.extend(sorted(self.folder_degradate.glob(extensie)))
        
        # Limitam pentru test rapid (smoke test)
        if limita_poze is not None:
            self.cai_curate = self.cai_curate[:limita_poze]
            self.cai_degradate = self.cai_degradate[:limita_poze]
        
        # Validari
        if len(self.cai_curate) == 0:
            raise ValueError(f"Nu am gasit imagini in {folder_curate}")
        if len(self.cai_degradate) == 0:
            raise ValueError(f"Nu am gasit imagini in {folder_degradate}")
        
        print(f"[{mod}] Poze curate gasite:    {len(self.cai_curate)}")
        print(f"[{mod}] Poze degradate gasite: {len(self.cai_degradate)}")
        
        # Construim pipeline-ul de transformari
        self.transformari = self._construieste_transformari()
    
    def _construieste_transformari(self):
        """
        Pipeline de transformari:
        1. Resize putin mai mare decat target (pentru random crop)
        2. Random crop la dimensiunea finala
        3. Flip orizontal 50% sansa
        4. Color jitter usor (variatii mici de luminozitate/contrast)
        5. Conversie la tensor si normalizare la [-1, 1]
        """
        if self.mod == 'antrenament':
            return transforms.Compose([
                # Resize la 112% din target pentru a avea loc de random crop
                transforms.Resize(
                    int(self.dimensiune_imagine * 1.12),
                    transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomCrop(self.dimensiune_imagine),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.05
                ),
                transforms.ToTensor(),
                # Normalizare la [-1, 1] - cerinta pt generatorul cu Tanh
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ])
        else:
            # Modul test - fara augmentari random
            return transforms.Compose([
                transforms.Resize(
                    (self.dimensiune_imagine, self.dimensiune_imagine),
                    transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ])
    
    def __len__(self):
        # Iteram atat cat e cel mai mare folder
        return max(len(self.cai_curate), len(self.cai_degradate))
    
    def _incarca_imagine(self, cale):
        """Incarca o imagine ca RGB. Daca esueaza, returneaza una gri."""
        try:
            imagine = Image.open(cale).convert('RGB')
            return imagine
        except Exception as eroare:
            print(f"Eroare la incarcarea {cale}: {eroare}")
            return Image.new('RGB',
                           (self.dimensiune_imagine, self.dimensiune_imagine),
                           (128, 128, 128))
    
    def __getitem__(self, index):
        # Poza curata: secvential
        cale_curata = self.cai_curate[index % len(self.cai_curate)]
        
        # Poza degradata: aleatoare (asta face ca perechile sa fie NEPERECHE!)
        if self.mod == 'antrenament':
            index_degradat = random.randint(0, len(self.cai_degradate) - 1)
        else:
            index_degradat = index % len(self.cai_degradate)
        
        cale_degradata = self.cai_degradate[index_degradat]
        
        # Incarcam imaginile
        imagine_curata = self._incarca_imagine(cale_curata)
        imagine_degradata = self._incarca_imagine(cale_degradata)
        
        # Aplicam transformarile
        tensor_curat = self.transformari(imagine_curata)
        tensor_degradat = self.transformari(imagine_degradata)
        
        return {
            'curat': tensor_curat,          # Domeniul A - tinta
            'degradat': tensor_degradat,    # Domeniul B - input de reparat
            'cale_curat': str(cale_curata),
            'cale_degradat': str(cale_degradata),
        }


def testeaza_incarcator(folder_curate, folder_degradate, dimensiune=128):
    """Test rapid - verifica ca datasetul functioneaza corect."""
    print("=" * 60)
    print("TEST INCARCATOR DE DATE")
    print("=" * 60)
    
    dataset = SetDeImaginiNepereche(
        folder_curate=folder_curate,
        folder_degradate=folder_degradate,
        dimensiune_imagine=dimensiune,
        mod='antrenament',
        limita_poze=10,
    )
    
    print(f"\nLungime dataset: {len(dataset)}")
    
    # Luam un sample
    esantion = dataset[0]
    print(f"\nEsantion 0:")
    print(f"  Tensor curat - dimensiune: {esantion['curat'].shape}")
    print(f"  Tensor degradat - dimensiune: {esantion['degradat'].shape}")
    print(f"  Valori curat: min={esantion['curat'].min():.3f}, "
          f"max={esantion['curat'].max():.3f} (trebuie [-1, 1])")
    print(f"  Cale curat:    {esantion['cale_curat']}")
    print(f"  Cale degradat: {esantion['cale_degradat']}")
    
    # Verificam ca perechile sunt intr-adevar nepereche
    esantion2 = dataset[0]
    perechi_identice = esantion['cale_degradat'] == esantion2['cale_degradat']
    print(f"\nLa 2 apeluri cu index=0, calea degradata identica? {perechi_identice}")
    print("(Ar trebui False uneori - confirmare nepereche)")
    
    print("\n✓ Incarcator OK!")
    return dataset
