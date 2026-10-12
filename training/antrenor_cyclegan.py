"""
antrenor_cyclegan.py - Antrenamentul modelului CycleGAN
=========================================================

CycleGAN antreneaza simultan 4 retele cu 3 tipuri de loss-uri:

CELE 4 RETELE:
    G_Restaurare:   degradat -> curat (CE NE INTERESEAZA!)
    G_Degradare:    curat -> degradat (necesar pentru cycle consistency)
    D_Curat:        distinge poze curate reale vs generate
    D_Degradat:     distinge poze degradate reale vs generate

CELE 3 LOSS-URI:
    1. ADVERSARIAL LOSS: generatorul incearca sa pacaleasca discriminatorul
    2. CYCLE CONSISTENCY LOSS: G_Restaurare(G_Degradare(curat)) ≈ curat
       Asta e magia care permite antrenare FARA perechi!
    3. IDENTITY LOSS: daca dam o poza deja curata la G_Restaurare,
       ar trebui sa o lase aproape neschimbata

CONFIGURATIE PATHS:
    Totul se salveaza in /content/drive/MyDrive/Licenta/HeritageVision/
"""

import os
import time
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from incarcator_date import SetDeImaginiNepereche
from arhitecturi_retele import (
    GeneratorUNet,
    DiscriminatorPatch,
    initializeaza_greutati
)


# ============================================================
# BAZIN DE IMAGINI (pentru stabilizarea discriminatorului)
# ============================================================

class BazinDeImagini:
    """
    Pastreaza ultimele 50 de imagini generate pentru antrenarea
    discriminatorului. Truc din lucrarea Shrivastava et al. care
    stabilizeaza antrenamentul GAN.
    
    Cand D vede un mix de imagini generate noi + vechi, nu se poate
    specializa pe ultimele exemple si ramane mai robust.
    """
    
    def __init__(self, dimensiune_bazin=50):
        self.dimensiune_bazin = dimensiune_bazin
        self.numar_imagini = 0
        self.imagini = []
    
    def interogheaza(self, imagini_noi):
        import random
        
        if self.dimensiune_bazin == 0:
            return imagini_noi
        
        imagini_returnate = []
        for imagine in imagini_noi:
            imagine = imagine.unsqueeze(0)
            
            if self.numar_imagini < self.dimensiune_bazin:
                # Bazinul nu e plin - adaugam
                self.imagini.append(imagine)
                imagini_returnate.append(imagine)
                self.numar_imagini += 1
            else:
                # Bazin plin - 50% sansa sa returnam una veche
                if random.uniform(0, 1) > 0.5:
                    index_aleator = random.randint(0, self.dimensiune_bazin - 1)
                    imagine_veche = self.imagini[index_aleator].clone()
                    self.imagini[index_aleator] = imagine
                    imagini_returnate.append(imagine_veche)
                else:
                    imagini_returnate.append(imagine)
        
        return torch.cat(imagini_returnate, dim=0)


# ============================================================
# ANTRENOR CYCLEGAN
# ============================================================

class AntrenorCycleGAN:
    """
    Clasa principala care gestioneaza antrenamentul CycleGAN.
    Fiecare apel la pas_antrenament() face o iteratie pe un batch.
    """
    
    def __init__(self, configuratie):
        self.cfg = configuratie
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Dispozitiv: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        dim = configuratie['dimensiune_imagine']
        
        # ==================== RETELE ====================
        # Generator de restaurare: degradat -> curat (MODELUL PRINCIPAL!)
        self.generator_restaurare = GeneratorUNet(dimensiune_imagine=dim).to(self.device)
        # Generator de degradare: curat -> degradat (pentru cycle consistency)
        self.generator_degradare = GeneratorUNet(dimensiune_imagine=dim).to(self.device)
        # Discriminatoare
        self.discriminator_curat = DiscriminatorPatch().to(self.device)
        self.discriminator_degradat = DiscriminatorPatch().to(self.device)
        
        # Initializam greutatile
        for retea in [self.generator_restaurare, self.generator_degradare,
                      self.discriminator_curat, self.discriminator_degradat]:
            initializeaza_greutati(retea)
        
        # ==================== LOSS-URI ====================
        # LSGAN loss (MSE) - mai stabil decat BCE clasic pentru GAN-uri
        self.loss_adversarial = nn.MSELoss()
        # L1 loss pentru cycle consistency si identity (pastreaza detalii)
        self.loss_cycle = nn.L1Loss()
        self.loss_identity = nn.L1Loss()
        
        # ==================== OPTIMIZATORI ====================
        # Adam cu parametrii recomandati de lucrarea CycleGAN
        self.optimizator_generatoare = torch.optim.Adam(
            itertools.chain(
                self.generator_restaurare.parameters(),
                self.generator_degradare.parameters()
            ),
            lr=configuratie['rata_invatare'],
            betas=(0.5, 0.999)
        )
        # D invata de 4x mai incet decat G - previne dominarea discriminatorului
        self.optimizator_discriminatoare = torch.optim.Adam(
            itertools.chain(
                self.discriminator_curat.parameters(),
                self.discriminator_degradat.parameters()
            ),
            lr=configuratie['rata_invatare'] * 0.25,
            betas=(0.5, 0.999)
        )
        
        # ==================== BAZINE ====================
        self.bazin_curate_generate = BazinDeImagini(50)
        self.bazin_degradate_generate = BazinDeImagini(50)
        
        # ==================== STARE ====================
        self.epoca_start = 0
        self.istoric_loss = []
    
    def _seteaza_grad(self, retele, activ):
        """Activeaza/dezactiveaza gradientii pentru economie de memorie."""
        if not isinstance(retele, list):
            retele = [retele]
        for retea in retele:
            for parametru in retea.parameters():
                parametru.requires_grad = activ
    
    def pas_antrenament(self, batch):
        """
        Un pas de antrenament pe un batch.
        Returneaza dictionar cu valorile loss-urilor.
        """
        # Convenție: A=curat, B=degradat
        real_A = batch['curat'].to(self.device)     # poze curate reale
        real_B = batch['degradat'].to(self.device)  # poze degradate reale
        
        # ==========================================
        # PARTEA 1: ANTRENEAZA GENERATOARELE
        # ==========================================
        # Dezactivam gradientii discriminatoarelor (doar le evaluam)
        self._seteaza_grad(
            [self.discriminator_curat, self.discriminator_degradat],
            activ=False
        )
        self.optimizator_generatoare.zero_grad()
        
        # ---- LOSS IDENTITY ----
        # Daca dam curat la generatorul de restaurare, ar trebui sa-l pastreze curat
        # Asta previne modelul sa "inventeze" transformari cand nu e nevoie
        identitate_A = self.generator_restaurare(real_A)
        loss_idt_curat = self.loss_identity(identitate_A, real_A) * \
            self.cfg['lambda_cycle'] * self.cfg['lambda_identity']
        
        identitate_B = self.generator_degradare(real_B)
        loss_idt_degradat = self.loss_identity(identitate_B, real_B) * \
            self.cfg['lambda_cycle'] * self.cfg['lambda_identity']
        
        # ---- LOSS ADVERSARIAL ----
        # Generam: curat -> degradat fals
        fals_B = self.generator_degradare(real_A)
        predictie_fals_B = self.discriminator_degradat(fals_B)
        # Vrem ca D sa creada ca e real (target = 1)
        loss_gan_curat_spre_degradat = self.loss_adversarial(
            predictie_fals_B, torch.ones_like(predictie_fals_B)
        )
        
        # Generam: degradat -> curat fals (RESTAURARE!)
        fals_A = self.generator_restaurare(real_B)
        predictie_fals_A = self.discriminator_curat(fals_A)
        loss_gan_degradat_spre_curat = self.loss_adversarial(
            predictie_fals_A, torch.ones_like(predictie_fals_A)
        )
        
        # ---- LOSS CYCLE CONSISTENCY ----
        # Aici e magia CycleGAN! 
        # curat -> degradat_fals -> curat_reconstruit ≈ curat original
        reconstructie_A = self.generator_restaurare(fals_B)
        loss_cycle_curat = self.loss_cycle(reconstructie_A, real_A) * \
            self.cfg['lambda_cycle']
        
        # degradat -> curat_fals -> degradat_reconstruit ≈ degradat original
        reconstructie_B = self.generator_degradare(fals_A)
        loss_cycle_degradat = self.loss_cycle(reconstructie_B, real_B) * \
            self.cfg['lambda_cycle']
        
        # Loss total pentru generatoare
        loss_generatoare_total = (
            loss_gan_curat_spre_degradat + loss_gan_degradat_spre_curat +
            loss_cycle_curat + loss_cycle_degradat +
            loss_idt_curat + loss_idt_degradat
        )
        loss_generatoare_total.backward()
        self.optimizator_generatoare.step()
        
        # ==========================================
        # PARTEA 2: ANTRENEAZA DISCRIMINATOARELE
        # ==========================================
        self._seteaza_grad(
            [self.discriminator_curat, self.discriminator_degradat],
            activ=True
        )
        self.optimizator_discriminatoare.zero_grad()
        
        # ---- Discriminator curat: real (1) vs fals (0) ----
        predictie_real_A = self.discriminator_curat(real_A)
        loss_d_real_curat = self.loss_adversarial(
            predictie_real_A, torch.ones_like(predictie_real_A)
        )
        # Folosim bazinul pentru stabilitate
        fals_A_din_bazin = self.bazin_curate_generate.interogheaza(fals_A.detach())
        predictie_fals_A_d = self.discriminator_curat(fals_A_din_bazin)
        loss_d_fals_curat = self.loss_adversarial(
            predictie_fals_A_d, torch.zeros_like(predictie_fals_A_d)
        )
        loss_discriminator_curat = (loss_d_real_curat + loss_d_fals_curat) * 0.5
        
        # ---- Discriminator degradat: real (1) vs fals (0) ----
        predictie_real_B = self.discriminator_degradat(real_B)
        loss_d_real_degradat = self.loss_adversarial(
            predictie_real_B, torch.ones_like(predictie_real_B)
        )
        fals_B_din_bazin = self.bazin_degradate_generate.interogheaza(fals_B.detach())
        predictie_fals_B_d = self.discriminator_degradat(fals_B_din_bazin)
        loss_d_fals_degradat = self.loss_adversarial(
            predictie_fals_B_d, torch.zeros_like(predictie_fals_B_d)
        )
        loss_discriminator_degradat = (loss_d_real_degradat + loss_d_fals_degradat) * 0.5
        
        # Loss total discriminatoare
        loss_discriminatoare_total = loss_discriminator_curat + loss_discriminator_degradat
        loss_discriminatoare_total.backward()
        self.optimizator_discriminatoare.step()
        
        return {
            'generator_total': loss_generatoare_total.item(),
            'discriminator_total': loss_discriminatoare_total.item(),
            'adversarial': (loss_gan_curat_spre_degradat + loss_gan_degradat_spre_curat).item(),
            'cycle': (loss_cycle_curat + loss_cycle_degradat).item(),
            'identity': (loss_idt_curat + loss_idt_degradat).item(),
        }
    
    def salveaza_checkpoint(self, epoca, folder_checkpoints):
        """
        Salveaza TOT ce trebuie pentru a relua antrenamentul.
        Salveaza si un fisier separat doar cu generatorul de restaurare
        (folosit pentru inferenta in Streamlit).
        """
        cale_checkpoint = Path(folder_checkpoints) / f'checkpoint_epoca_{epoca:03d}.pth'
        
        torch.save({
            'epoca': epoca,
            'generator_restaurare': self.generator_restaurare.state_dict(),
            'generator_degradare': self.generator_degradare.state_dict(),
            'discriminator_curat': self.discriminator_curat.state_dict(),
            'discriminator_degradat': self.discriminator_degradat.state_dict(),
            'optimizator_generatoare': self.optimizator_generatoare.state_dict(),
            'optimizator_discriminatoare': self.optimizator_discriminatoare.state_dict(),
            'istoric_loss': self.istoric_loss,
            'configuratie': self.cfg,
        }, cale_checkpoint)
        
        # Fisier separat doar cu generatorul de restaurare (pentru Streamlit)
        cale_generator = Path(folder_checkpoints) / 'generator_restaurare_ultim.pth'
        torch.save(self.generator_restaurare.state_dict(), cale_generator)
        
        print(f"  ✓ Checkpoint salvat: {cale_checkpoint.name}")
        
        # Economie spatiu: stergem checkpoint-uri vechi, pastram doar
        # ultimele 3 + cele la fiecare 10 epoci
        self._curata_checkpoints_vechi(folder_checkpoints, epoca)
    
    def _curata_checkpoints_vechi(self, folder, epoca_curenta):
        """Sterge checkpoint-uri vechi pentru economie spatiu."""
        import glob
        toate = sorted(glob.glob(f'{folder}/checkpoint_epoca_*.pth'))
        
        # Pastram: ultimele 3 + cele la fiecare 10 epoci
        de_pastrat = set(toate[-5:])  # ultimele 3
        for cale in toate:
            # Extragem numarul epocii din nume
            try:
                nume = Path(cale).stem  # checkpoint_epoca_042
                nr_epoca = int(nume.split('_')[-1])
                if nr_epoca % 5 == 0:
                    de_pastrat.add(cale)
            except ValueError:
                continue
        
        for cale in toate:
            if cale not in de_pastrat:
                os.remove(cale)
    
    def incarca_checkpoint(self, cale_checkpoint):
        """Incarca un checkpoint pentru a continua antrenamentul."""
        checkpoint = torch.load(cale_checkpoint, map_location=self.device)
        
        self.generator_restaurare.load_state_dict(checkpoint['generator_restaurare'])
        self.generator_degradare.load_state_dict(checkpoint['generator_degradare'])
        self.discriminator_curat.load_state_dict(checkpoint['discriminator_curat'])
        self.discriminator_degradat.load_state_dict(checkpoint['discriminator_degradat'])
        self.optimizator_generatoare.load_state_dict(checkpoint['optimizator_generatoare'])
        self.optimizator_discriminatoare.load_state_dict(checkpoint['optimizator_discriminatoare'])
        
        self.epoca_start = checkpoint['epoca'] + 1
        self.istoric_loss = checkpoint.get('istoric_loss', [])
        
        print(f"✓ Checkpoint incarcat din epoca {checkpoint['epoca']}, "
              f"continui cu epoca {self.epoca_start}")
    
    def reseteaza_discriminatoare(self):
        """
        Reinitializeaza greutatile discriminatoarelor de la zero.
        Folosit cand D devine prea puternic si G stagneaza.
        Generatoarele raman neatinse - pastreaza tot ce au invatat.
        """
        initializeaza_greutati(self.discriminator_curat)
        initializeaza_greutati(self.discriminator_degradat)
        
        # Resetam si optimizatorul D - starile momentum sunt invalide dupa reset
        self.optimizator_discriminatoare = torch.optim.Adam(
            itertools.chain(
                self.discriminator_curat.parameters(),
                self.discriminator_degradat.parameters()
            ),
            lr=self.cfg['rata_invatare'] * 0.25,
            betas=(0.5, 0.999)
        )
        print("✓ Discriminatoarele au fost resetate si reinitializate")
    
    def salveaza_esantioane(self, batch_fix, epoca, folder_esantioane):
        """
        Salveaza imagini esantion pentru monitorizare vizuala.
        Randul de sus = input degradat, randul de jos = restaurare.
        """
        self.generator_restaurare.eval()
        
        with torch.no_grad():
            real_B = batch_fix['degradat'].to(self.device)[:4]  # primele 4
            restaurate = self.generator_restaurare(real_B)
            
            # Grid de comparatie: sus original, jos restaurat
            grid = torch.cat([real_B, restaurate], dim=0)
            grid = (grid + 1) / 2  # [-1, 1] -> [0, 1]
            grid = grid.clamp(0, 1)
            
            cale_esantion = Path(folder_esantioane) / f'epoca_{epoca:03d}.png'
            vutils.save_image(grid, cale_esantion, nrow=4)
            print(f"  ✓ Esantion salvat: {cale_esantion.name}")
        
        self.generator_restaurare.train()


# ============================================================
# FUNCTIA PRINCIPALA DE ANTRENAMENT
# ============================================================

def porneste_antrenament(configuratie):
    """Entry point pentru antrenare."""
    
    # Creeaza folderele necesare
    Path(configuratie['folder_checkpoints']).mkdir(parents=True, exist_ok=True)
    Path(configuratie['folder_esantioane']).mkdir(parents=True, exist_ok=True)
    
    # Dataset si DataLoader
    set_date = SetDeImaginiNepereche(
        folder_curate=configuratie['folder_curate'],
        folder_degradate=configuratie['folder_degradate'],
        dimensiune_imagine=configuratie['dimensiune_imagine'],
        mod='antrenament',
        limita_poze=configuratie.get('limita_poze'),
    )
    
    incarcator = DataLoader(
        set_date,
        batch_size=configuratie['dimensiune_batch'],
        shuffle=True,
        num_workers=configuratie.get('numar_workers', 2),
        pin_memory=True,
        drop_last=True,
    )
    
    # Antrenor
    antrenor = AntrenorCycleGAN(configuratie)
    
    # Reluare din checkpoint daca e specificat
    cale_reluare = configuratie.get('reluare_din')
    if cale_reluare and os.path.exists(cale_reluare):
        antrenor.incarca_checkpoint(cale_reluare)
    
    # Batch fix pentru vizualizare progres
    batch_fix = next(iter(incarcator))
    
    # Bucla principala
    total_epoci = configuratie['numar_epoci']
    print(f"\n{'='*60}")
    print(f"ANTRENAMENT: epoca {antrenor.epoca_start} -> {total_epoci}")
    print(f"{'='*60}\n")
    
    for epoca in range(antrenor.epoca_start, total_epoci):
        timp_start = time.time()
        loss_epoca = {
            'generator_total': 0,
            'discriminator_total': 0,
            'adversarial': 0,
            'cycle': 0,
            'identity': 0
        }
        numar_batch_uri = 0
        
        for i, batch in enumerate(incarcator):
            loss_batch = antrenor.pas_antrenament(batch)
            
            for cheie, valoare in loss_batch.items():
                loss_epoca[cheie] += valoare
            numar_batch_uri += 1
            
            if i % configuratie.get('frecventa_log', 50) == 0:
                print(f"  Epoca {epoca} | Batch {i}/{len(incarcator)} | "
                      f"G={loss_batch['generator_total']:.3f} "
                      f"D={loss_batch['discriminator_total']:.3f} "
                      f"cycle={loss_batch['cycle']:.3f}")
        
        # Medii pe epoca
        for cheie in loss_epoca:
            loss_epoca[cheie] /= numar_batch_uri
        antrenor.istoric_loss.append(loss_epoca)
        
        timp_epoca = time.time() - timp_start
        print(f"\n>>> EPOCA {epoca} COMPLETA | "
              f"G={loss_epoca['generator_total']:.3f} "
              f"D={loss_epoca['discriminator_total']:.3f} "
              f"cycle={loss_epoca['cycle']:.3f} | "
              f"timp: {timp_epoca/60:.1f} min")
        
        # Salvare periodica
        if (epoca + 1) % configuratie.get('frecventa_salvare', 1) == 0:
            antrenor.salveaza_checkpoint(epoca, configuratie['folder_checkpoints'])
            antrenor.salveaza_esantioane(
                batch_fix, epoca, configuratie['folder_esantioane']
            )
        
        print()
    
    print("ANTRENAMENT COMPLET!")


# ============================================================
# CONFIGURATII GATA DE FOLOSIT
# ============================================================

def configuratie_test_rapid(cale_licenta='/content/drive/MyDrive/Licenta'):
    """
    Configuratie pentru smoke test: 200 poze, 128x128, 10 epoci.
    Ruleaza ~1-2 ore pe Colab gratuit - sa vedem ca totul functioneaza.
    """
    return {
        # Paths
        'folder_curate': f'{cale_licenta}/Poze_RAW_Curate',
        'folder_degradate': f'{cale_licenta}/Poze_RAW_Degradate',
        'folder_checkpoints': f'{cale_licenta}/HeritageVision/checkpoints_test',
        'folder_esantioane': f'{cale_licenta}/HeritageVision/esantioane_test',
        
        # Date
        'dimensiune_imagine': 128,
        'limita_poze': 200,
        'dimensiune_batch': 4,
        'numar_workers': 2,
        
        # Antrenament
        'numar_epoci': 10,
        'rata_invatare': 2e-4,
        'lambda_cycle': 10.0,    # greutate cycle consistency
        'lambda_identity': 0.5,  # greutate identity loss
        
        # Logging
        'frecventa_log': 10,
        'frecventa_salvare': 2,
        
        'reluare_din': None,
    }


def configuratie_antrenament_complet(cale_licenta='/content/drive/MyDrive/Licenta'):
    """
    Configuratie pentru antrenamentul real: toate pozele, 256x256.
    Dureaza zile pe Colab gratuit.
    """
    return {
        'folder_curate': f'{cale_licenta}/Poze_RAW_Curate',
        'folder_degradate': f'{cale_licenta}/Poze_RAW_Degradate',
        'folder_checkpoints': f'{cale_licenta}/HeritageVision/checkpoints',
        'folder_esantioane': f'{cale_licenta}/HeritageVision/esantioane',
        
        'dimensiune_imagine': 256,
        'limita_poze': None,        # toate pozele
        'dimensiune_batch': 2,      # mic pentru T4 la 256x256
        'numar_workers': 2,
        
        'numar_epoci': 100,
        'rata_invatare': 2e-4,
        'lambda_cycle': 15.0,    # crescut de la 10 - forteaza G sa prioritizeze continutul
        'lambda_identity': 0.5,
        
        'frecventa_log': 50,
        'frecventa_salvare': 1,  # salvam la fiecare epoca - CRITIC
        
        'reluare_din': None,
    }
