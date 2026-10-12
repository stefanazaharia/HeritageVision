"""
arhitecturi_retele.py - Arhitectura Generator si Discriminator
================================================================

Defineste cele 2 tipuri de retele neuronale din CycleGAN:

1. GENERATOR (U-Net):
   - Ia o imagine de intrare (curata sau degradata)
   - Produce imaginea transformata in celalalt domeniu
   - Avem 2 generatoare in CycleGAN:
     * Unul transforma degradat -> curat (RESTAURARE - ce vrem)
     * Unul transforma curat -> degradat (necesar pentru cycle consistency)

2. DISCRIMINATOR (PatchGAN):
   - Ia o imagine si decide daca e reala sau generata
   - Output-ul e o matrice (nu un singur numar) - evalueaza patch-uri
   - Avem 2 discriminatoare:
     * Unul pentru domeniul curat
     * Unul pentru domeniul degradat

STRUCTURA U-NET:
    Encoder (comprima info) -> Bottleneck -> Decoder (reconstruieste)
    Cu skip connections (legaturi directe encoder <-> decoder) care 
    pastreaza detalii fine - critic pentru restaurare!
"""

import torch
import torch.nn as nn


# ============================================================
# COMPONENTE DE BAZA PENTRU GENERATOR
# ============================================================

class BlocDeComprimare(nn.Module):
    """
    Un bloc care REDUCE dimensiunea la jumatate (downsampling).
    Folosit in encoder-ul U-Net.
    
    Exemplu: 128x128 -> 64x64
    """
    
    def __init__(self, canale_intrare, canale_iesire, foloseste_normalizare=True):
        super().__init__()
        straturi = [
            nn.Conv2d(
                canale_intrare, canale_iesire,
                kernel_size=4, stride=2, padding=1, bias=False
            )
        ]
        if foloseste_normalizare:
            straturi.append(nn.InstanceNorm2d(canale_iesire))
        # LeakyReLU e standard pentru encoderul GAN-urilor
        straturi.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.blocul = nn.Sequential(*straturi)
    
    def forward(self, x):
        return self.blocul(x)


class BlocDeExpandare(nn.Module):
    """
    Un bloc care DUBLEAZA dimensiunea (upsampling).
    Folosit in decoder-ul U-Net.
    
    Exemplu: 64x64 -> 128x128
    """
    
    def __init__(self, canale_intrare, canale_iesire, foloseste_dropout=False):
        super().__init__()
        straturi = [
            nn.ConvTranspose2d(
                canale_intrare, canale_iesire,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(canale_iesire),
        ]
        if foloseste_dropout:
            # Dropout in primele 3 blocuri decoder = regularizare + variabilitate
            straturi.append(nn.Dropout(0.5))
        # ReLU standard pentru decoder
        straturi.append(nn.ReLU(inplace=True))
        
        self.blocul = nn.Sequential(*straturi)
    
    def forward(self, x):
        return self.blocul(x)


# ============================================================
# GENERATORUL U-NET
# ============================================================

class GeneratorUNet(nn.Module):
    """
    Generator U-Net pentru traducerea intre 2 domenii de imagini.
    
    FUNCTIONARE INTUITIVA:
    - Encoder: imaginea trece prin straturi tot mai mici (comprimare)
    - Bottleneck: reprezentarea cea mai comprimata (4x4)
    - Decoder: reconstruim imaginea din bottleneck
    - Skip connections: info din encoder e copiata direct in decoder
      (asta pastreaza detalii fine - chei, ochi, texturi)
    
    PROGRESIA DIMENSIUNILOR pentru imagine 256x256:
    256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 (bottleneck)
    Apoi invers, cu skip de la fiecare nivel encoder la acelasi nivel decoder.
    
    Pentru 128x128: un nivel mai putin (128 -> 64 -> ... -> 2).
    """
    
    def __init__(self, canale_intrare=3, canale_iesire=3, caracteristici_baza=64,
                 dimensiune_imagine=256):
        super().__init__()
        
        self.dimensiune_imagine = dimensiune_imagine
        # Numarul de comprimari depinde de dimensiune
        # 256 -> 6 comprimari pentru a ajunge la 4x4
        # 128 -> 5 comprimari pentru a ajunge la 4x4
        self.numar_comprimari = 6 if dimensiune_imagine == 256 else 5
        
        f = caracteristici_baza  # prescurtare, e 64 default
        
        # ==================== ENCODER ====================
        # Primul bloc nu are normalizare (standard)
        self.comprimare1 = BlocDeComprimare(canale_intrare, f, foloseste_normalizare=False)
        self.comprimare2 = BlocDeComprimare(f, f * 2)
        self.comprimare3 = BlocDeComprimare(f * 2, f * 4)
        self.comprimare4 = BlocDeComprimare(f * 4, f * 8)
        self.comprimare5 = BlocDeComprimare(f * 8, f * 8)
        
        if self.numar_comprimari == 6:
            self.comprimare6 = BlocDeComprimare(f * 8, f * 8)
        
        # ==================== BOTTLENECK ====================
        # Un ultim nivel de comprimare, fara normalizare
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # ==================== DECODER ====================
        # Logica canalelor pentru skip connections:
        # fiecare BlocDeExpandare primeste: output_anterior + skip_de_la_encoder
        # => canale_intrare = canale_expandare_anterioara + canale_comprimare_corespondenta
        #
        # Pentru 256x256 (6 comprimari):
        #   e1: bottleneck(512)                          -> 512
        #   e2: cat(e1=512, c6=512) = 1024              -> 512
        #   e3: cat(e2=512, c5=512) = 1024              -> 512
        #   e4: cat(e3=512, c4=512) = 1024              -> 256
        #   e5: cat(e4=256, c3=256) = 512               -> 128
        #   e6: cat(e5=128, c2=128) = 256               -> 64
        #   final: cat(e6=64, c1=64) = 128              -> 3
        #
        # Pentru 128x128 (5 comprimari):
        #   e1: bottleneck(512)                          -> 512
        #   e2: cat(e1=512, c5=512) = 1024              -> 512
        #   e3: cat(e2=512, c4=512) = 1024              -> 256
        #   e4: cat(e3=256, c3=256) = 512               -> 128
        #   e5: cat(e4=128, c2=128) = 256               -> 64
        #   final: cat(e5=64, c1=64) = 128              -> 3

        if self.numar_comprimari == 6:
            # Dropout in primele 3 blocuri decoder = regularizare
            self.expandare1 = BlocDeExpandare(f * 8,      f * 8, foloseste_dropout=True)
            self.expandare2 = BlocDeExpandare(f * 8 * 2,  f * 8, foloseste_dropout=True)
            self.expandare3 = BlocDeExpandare(f * 8 * 2,  f * 8, foloseste_dropout=True)
            self.expandare4 = BlocDeExpandare(f * 8 * 2,  f * 4)
            self.expandare5 = BlocDeExpandare(f * 4 * 2,  f * 2)
            self.expandare6 = BlocDeExpandare(f * 2 * 2,  f)
            self.strat_final = nn.Sequential(
                nn.ConvTranspose2d(f * 2, canale_iesire,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        else:
            # 128x128: un nivel mai putin
            self.expandare1 = BlocDeExpandare(f * 8,      f * 8, foloseste_dropout=True)
            self.expandare2 = BlocDeExpandare(f * 8 * 2,  f * 8, foloseste_dropout=True)
            self.expandare3 = BlocDeExpandare(f * 8 * 2,  f * 4, foloseste_dropout=True)
            self.expandare4 = BlocDeExpandare(f * 4 * 2,  f * 2)
            self.expandare5 = BlocDeExpandare(f * 2 * 2,  f)
            self.strat_final = nn.Sequential(
                nn.ConvTranspose2d(f * 2, canale_iesire,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
    
    def forward(self, x):
        # ---- ENCODER: comprimam imaginea, salvam activarile ----
        # Fiecare comprimare injumatateste spatiul si dubleaza canalele
        c1 = self.comprimare1(x)   # 64ch,  H/2
        c2 = self.comprimare2(c1)  # 128ch, H/4
        c3 = self.comprimare3(c2)  # 256ch, H/8
        c4 = self.comprimare4(c3)  # 512ch, H/16
        c5 = self.comprimare5(c4)  # 512ch, H/32
        
        if self.numar_comprimari == 6:
            # ---- 256x256 ----
            c6 = self.comprimare6(c5)  # 512ch, H/64
            b  = self.bottleneck(c6)   # 512ch, H/128

            # ---- DECODER: expandam cu skip connections ----
            # La fiecare pas: torch.cat(output_anterior, skip_encoder) pe canale
            e1 = self.expandare1(b)                        # 512ch
            e2 = self.expandare2(torch.cat([e1, c6], 1))  # cat(512+512=1024) -> 512ch
            e3 = self.expandare3(torch.cat([e2, c5], 1))  # cat(512+512=1024) -> 512ch
            e4 = self.expandare4(torch.cat([e3, c4], 1))  # cat(512+512=1024) -> 256ch
            e5 = self.expandare5(torch.cat([e4, c3], 1))  # cat(256+256=512)  -> 128ch
            e6 = self.expandare6(torch.cat([e5, c2], 1))  # cat(128+128=256)  -> 64ch
            iesire = self.strat_final(torch.cat([e6, c1], 1))  # cat(64+64=128) -> 3ch
        else:
            # ---- 128x128: un nivel mai putin ----
            b  = self.bottleneck(c5)   # 512ch, H/64

            e1 = self.expandare1(b)                        # 512ch
            e2 = self.expandare2(torch.cat([e1, c5], 1))  # cat(512+512=1024) -> 512ch
            e3 = self.expandare3(torch.cat([e2, c4], 1))  # cat(512+512=1024) -> 256ch
            e4 = self.expandare4(torch.cat([e3, c3], 1))  # cat(256+256=512)  -> 128ch
            e5 = self.expandare5(torch.cat([e4, c2], 1))  # cat(128+128=256)  -> 64ch
            iesire = self.strat_final(torch.cat([e5, c1], 1))  # cat(64+64=128) -> 3ch
        
        return iesire


# ============================================================
# DISCRIMINATORUL PATCHGAN
# ============================================================

class DiscriminatorPatch(nn.Module):
    """
    Discriminator de tip PatchGAN 70x70.
    
    Diferenta fata de un discriminator clasic:
    - NU produce un singur numar (real/fake)
    - Produce o HARTA de probabilitati, fiecare valoare evalueaza 
      un patch de 70x70 din imaginea originala
    - Avantaj: mai sensibil la texturi si detalii locale (important
      pentru fotografii istorice cu texturi specifice)
    """
    
    def __init__(self, canale_intrare=3, caracteristici_baza=64):
        super().__init__()
        
        f = caracteristici_baza
        
        self.model = nn.Sequential(
            # Bloc 1: 256 -> 128 (fara normalizare)
            nn.Conv2d(canale_intrare, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Bloc 2: 128 -> 64
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Bloc 3: 64 -> 32
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Bloc 4: 32 -> 31 (stride=1, nu reduce dimensiunea)
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Strat final: 31 -> 30 (harta de evaluare pe patch-uri)
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),
            # NU punem Sigmoid aici - folosim LSGAN loss care vrea output raw
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================
# INITIALIZARE GREUTATI (standard CycleGAN)
# ============================================================

def initializeaza_greutati(retea, castig=0.02):
    """
    Initializeaza greutatile cu Normal(0, 0.02).
    Asta e standardul pentru GAN-uri - modelele initiate cu default
    PyTorch nu converg bine.
    """
    def functie_init(modul):
        nume_clasa = modul.__class__.__name__
        if hasattr(modul, 'weight') and (
            nume_clasa.find('Conv') != -1 or nume_clasa.find('Linear') != -1
        ):
            nn.init.normal_(modul.weight.data, 0.0, castig)
            if hasattr(modul, 'bias') and modul.bias is not None:
                nn.init.constant_(modul.bias.data, 0.0)
        elif nume_clasa.find('InstanceNorm') != -1:
            if modul.weight is not None:
                nn.init.normal_(modul.weight.data, 1.0, castig)
                nn.init.constant_(modul.bias.data, 0.0)
    
    retea.apply(functie_init)
    return retea


# ============================================================
# TEST
# ============================================================

def testeaza_retelele(dimensiune_imagine=128, device='cpu'):
    """Verificare rapida ca retelele ruleaza corect."""
    print("=" * 60)
    print(f"TEST RETELE (dimensiune={dimensiune_imagine}, device={device})")
    print("=" * 60)
    
    # Input simulat: batch de 2 imagini
    x = torch.randn(2, 3, dimensiune_imagine, dimensiune_imagine).to(device)
    print(f"\nInput: {x.shape}")
    
    # Test Generator
    generator = GeneratorUNet(dimensiune_imagine=dimensiune_imagine).to(device)
    generator = initializeaza_greutati(generator)
    iesire_generator = generator(x)
    
    print(f"Output Generator: {iesire_generator.shape}")
    assert iesire_generator.shape == x.shape, "Dimensiuni gresite!"
    
    nr_param_gen = sum(p.numel() for p in generator.parameters())
    print(f"Parametri Generator: {nr_param_gen:,} ({nr_param_gen/1e6:.1f}M)")
    
    # Test Discriminator
    discriminator = DiscriminatorPatch().to(device)
    discriminator = initializeaza_greutati(discriminator)
    iesire_discriminator = discriminator(x)
    
    print(f"\nOutput Discriminator: {iesire_discriminator.shape} (harta patch-uri)")
    
    nr_param_disc = sum(p.numel() for p in discriminator.parameters())
    print(f"Parametri Discriminator: {nr_param_disc:,} ({nr_param_disc/1e6:.1f}M)")
    
    print("\n✓ Retelele functioneaza!")
    return generator, discriminator
