"""
restaurator_imagini.py - Foloseste modelul antrenat pentru restaurare
=======================================================================

Dupa ce antrenamentul e gata, acest fisier incarca doar generatorul de
restaurare (fara optimizatori, fara discriminatori) si il foloseste
pentru a restaura poze noi.

2 MODURI DE RESTAURARE:
    1. restaureaza_complet(imagine) - toata imaginea e transformata
    2. restaureaza_cu_masca(imagine, masca) - doar zonele din masca
       (restul ramane bit-perfect identic cu originalul)

Modul 2 e cheia pentru control selectiv in GUI:
    - Utilizatorul deseneaza/alege zona cu defectul specific
    - Modelul reparează doar acolo
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

import torch
import torchvision.transforms as transforms

from arhitecturi_retele import GeneratorUNet


class RestauratorImagini:
    """
    Incarca generatorul de restaurare (poze degradate -> poze curate)
    si ofera metode pentru a restaura imagini noi.
    
    Utilizare:
        restaurator = RestauratorImagini(
            '/content/drive/MyDrive/Licenta/HeritageVision/checkpoints/generator_restaurare_ultim.pth'
        )
        
        # Mod 1 - tot
        rezultat = restaurator.restaureaza_complet(imagine_pil)
        
        # Mod 2 - doar zona data de masca
        rezultat = restaurator.restaureaza_cu_masca(imagine_pil, masca_pil)
    """
    
    def __init__(self, cale_generator, dimensiune_imagine=256, device=None):
        
        self.dimensiune_imagine = dimensiune_imagine
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Incarca generatorul
        self.model = GeneratorUNet(dimensiune_imagine=dimensiune_imagine).to(self.device)
        stare_model = torch.load(cale_generator, map_location=self.device)
        self.model.load_state_dict(stare_model)
        self.model.eval()  # mod evaluare - dezactiveaza dropout
        
        # Transformari pentru input
        self.transforma_in_tensor = transforms.Compose([
            transforms.Resize(
                (dimensiune_imagine, dimensiune_imagine),
                transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),  # -> [-1, 1]
        ])
        
        print(f"✓ Restaurator incarcat pe {self.device}")
    
    @torch.no_grad()  # dezactivam calculul gradientilor - mai rapid
    def _proceseaza(self, imagine_pil):
        """Trece imaginea prin retea si returneaza tensorul output."""
        dimensiune_originala = imagine_pil.size  # (latime, inaltime)
        
        # Pregatim input-ul
        tensor_input = self.transforma_in_tensor(imagine_pil).unsqueeze(0).to(self.device)
        
        # Inferenta
        tensor_output = self.model(tensor_input)
        
        return tensor_output, dimensiune_originala
    
    def _tensor_la_pil(self, tensor, dimensiune_tinta):
        """Converteste tensor [-1, 1] la imagine PIL la dimensiunea originala."""
        tensor = (tensor.squeeze(0) + 1) / 2  # [-1,1] -> [0,1]
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte().cpu().numpy()
        tensor = tensor.transpose(1, 2, 0)  # CHW -> HWC
        
        imagine = Image.fromarray(tensor)
        if imagine.size != dimensiune_tinta:
            imagine = imagine.resize(dimensiune_tinta, Image.BICUBIC)
        return imagine
    
    def restaureaza_complet(self, imagine_pil):
        """
        Restaureaza toata imaginea.
        
        Parametri:
            imagine_pil: PIL Image (orice mod - se converteste la RGB)
        
        Returneaza:
            PIL Image RGB restaurata, la aceeasi dimensiune ca input-ul
        """
        if imagine_pil.mode != 'RGB':
            imagine_pil = imagine_pil.convert('RGB')
        
        tensor_out, dim_originala = self._proceseaza(imagine_pil)
        return self._tensor_la_pil(tensor_out, dim_originala)
    
    def restaureaza_cu_masca(self, imagine_pil, masca, pixeli_tranzitie=10):
        """
        Restaureaza DOAR zona indicata de masca.
        Restul ramane neatins.
        
        Parametri:
            imagine_pil: PIL Image RGB
            masca: PIL Image grayscale SAU numpy array
                - 0 (negru) = pastreaza originalul
                - 255 (alb) = restaureaza
                - Intermediare = amestec (blend)
            pixeli_tranzitie: pentru margini naturale (blur pe masca)
        
        Returneaza:
            PIL Image - combinatie intre original (unde masca=0)
                       si restaurat (unde masca=255)
        """
        if imagine_pil.mode != 'RGB':
            imagine_pil = imagine_pil.convert('RGB')
        
        # ---- Normalizam masca la PIL grayscale ----
        if isinstance(masca, np.ndarray):
            if masca.dtype != np.uint8:
                if masca.max() <= 1.0:
                    masca = (masca * 255).astype(np.uint8)
                else:
                    masca = masca.astype(np.uint8)
            masca = Image.fromarray(masca, mode='L')
        elif isinstance(masca, Image.Image):
            masca = masca.convert('L')
        else:
            raise ValueError("masca trebuie sa fie numpy array sau PIL Image")
        
        # Resize masca la dimensiunea imaginii
        if masca.size != imagine_pil.size:
            masca = masca.resize(imagine_pil.size, Image.BILINEAR)
        
        # Blur la margini pentru tranzitie naturala
        if pixeli_tranzitie > 0:
            masca = masca.filter(ImageFilter.GaussianBlur(radius=pixeli_tranzitie))
        
        # Genereaza versiunea restaurata completa
        imagine_restaurata = self.restaureaza_complet_hd(imagine_pil)
        
        # ---- Blend ----
        # formula: rezultat = original * (1-masca) + restaurat * masca
        arr_original = np.array(imagine_pil, dtype=np.float32)
        arr_restaurat = np.array(imagine_restaurata, dtype=np.float32)
        arr_masca = np.array(masca, dtype=np.float32) / 255.0
        
        # Adaugam dimensiune pentru broadcasting cu canalele RGB
        arr_masca = arr_masca[..., None]  # (H, W) -> (H, W, 1)
        
        rezultat = arr_original * (1 - arr_masca) + arr_restaurat * arr_masca
        rezultat = rezultat.clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(rezultat)
    
    def restaureaza_zone_dreptunghiulare(self, imagine_pil, zone, pixeli_tranzitie=10):
        """
        Convenienta: restaureaza zone dreptunghiulare.
        Util daca nu ai canvas de desenat in UI.
        
        Parametri:
            imagine_pil: PIL Image
            zone: lista de tuple (x1, y1, x2, y2) in coordonatele imaginii
            pixeli_tranzitie: pentru blur la margini
        """
        latime, inaltime = imagine_pil.size
        
        # Construim masca cu dreptunghiuri albe
        masca = Image.new('L', (latime, inaltime), 0)
        drawer = ImageDraw.Draw(masca)
        for zona in zone:
            drawer.rectangle(zona, fill=255)
        
        return self.restaureaza_cu_masca(imagine_pil, masca, pixeli_tranzitie)
    
    def restaureaza_complet_hd(self, imagine_pil, overlap=32):
        """
        Restaureaza imagini mari prin procesare pe patch-uri.
        Evita blur-ul cauzat de resize-ul la 256x256.
        
        Parametri:
            imagine_pil: PIL Image de orice dimensiune
            overlap: pixeli de suprapunere intre patch-uri (pentru tranzitii line)
        """
        if imagine_pil.mode != 'RGB':
            imagine_pil = imagine_pil.convert('RGB')
        
        latime, inaltime = imagine_pil.size
        dim = self.dimensiune_imagine  # 256
        pas = dim - overlap  # 224
        
        # Daca imaginea e mica, folosim metoda normala
        if latime <= dim and inaltime <= dim:
            return self.restaureaza_complet(imagine_pil)
        
        import numpy as np
        
        arr_original = np.array(imagine_pil, dtype=np.float32)
        arr_rezultat = np.zeros_like(arr_original)
        arr_greutati = np.zeros((inaltime, latime), dtype=np.float32)
        
        # Cream o masca de greutati pentru blending lin la margini
        masca = np.ones((dim, dim), dtype=np.float32)
        if overlap > 0:
            for i in range(overlap):
                factor = i / overlap
                masca[i, :] *= factor
                masca[-(i+1), :] *= factor
                masca[:, i] *= factor
                masca[:, -(i+1)] *= factor
        
        # Iteram prin patch-uri
        y_pozitii = list(range(0, inaltime - dim + 1, pas))
        if y_pozitii[-1] + dim < inaltime:
            y_pozitii.append(inaltime - dim)
        
        x_pozitii = list(range(0, latime - dim + 1, pas))
        if x_pozitii[-1] + dim < latime:
            x_pozitii.append(latime - dim)
        
        total_patch_uri = len(y_pozitii) * len(x_pozitii)
        print(f"Procesez {total_patch_uri} patch-uri...")
        
        for idx_y, y in enumerate(y_pozitii):
            for idx_x, x in enumerate(x_pozitii):
                # Extragem patch-ul
                patch = imagine_pil.crop((x, y, x + dim, y + dim))
                
                # Restauram patch-ul
                patch_restaurat = self.restaureaza_complet(patch)
                arr_patch = np.array(patch_restaurat, dtype=np.float32)
                
                # Adaugam cu greutati pentru blending
                for c in range(3):
                    arr_rezultat[y:y+dim, x:x+dim, c] += arr_patch[:,:,c] * masca
                arr_greutati[y:y+dim, x:x+dim] += masca
        
        # Normalizam dupa greutati
        for c in range(3):
            arr_rezultat[:,:,c] /= np.maximum(arr_greutati, 1e-8)
        
        rezultat = arr_rezultat.clip(0, 255).astype(np.uint8)
        return Image.fromarray(rezultat)
