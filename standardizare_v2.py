import os
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

INPUT_DIR = r"D:\LICENTA\poze stefan 2\SDASM Archives\William Fark Special Collection"
OUTPUT_DIR = r"D:\LICENTA\data_standardized"
TARGET_SIZE = 512
MIN_WIDTH = 256
MIN_HEIGHT = 256

# --- CLASA NOASTRĂ PERSONALIZATĂ PENTRU REDIMENSIONARE PERFECTĂ ---
class ResizeAndPadReflect:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        
        # 1. Aflăm care este latura cea mai mare și o scalăm la 512
        max_side = max(w, h)
        ratio = self.target_size / max_side
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Redimensionăm toată poza, păstrând proporțiile (nimic nu e tăiat, nimic nu e turtit)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 2. Calculăm cât spațiu gol a rămas
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        
        # Împărțim spațiul egal (stânga/dreapta sau sus/jos)
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left
        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top

        # padding_mode='reflect' salvează modelul de la a învăța margini negre false
        img_padded = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), padding_mode='edge')
        
        return img_padded

# Inițializăm transformarea noastră
preprocesare_perfecta = ResizeAndPadReflect(TARGET_SIZE)

def standardize_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_path = Path(INPUT_DIR)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'} 
    
    processed_count = 14621   #de aici sa continuie numaratoarea
    skipped_size = 0
    skipped_corrupt = 0

    print(f"Începem procesarea imaginilor din '{INPUT_DIR}'...\n")

    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            try:
                with Image.open(file_path) as img:
                    
                    if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
                        skipped_size += 1
                        continue
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # APLICĂM NOUA METODĂ: Toată poza + margini oglindite
                    final_img = preprocesare_perfecta(img)

                    # original_name = file_path.stem 
                    # new_filename = f"{original_name}.jpg"

                    # 4. SALVARE ÎN FORMAT JPG + REDENUMIRE (img_00001.jpg, img_00002.jpg etc.)
                    new_filename = f"img_{processed_count:05d}.jpg"
                    
                    output_path = Path(OUTPUT_DIR) / new_filename
                    final_img.save(output_path, "JPEG", quality=95)
                    processed_count += 1
                    
            except Exception as e:
                print(f"⚠️ Eroare la fișierul {file_path.name}: {e}")
                skipped_corrupt += 1

    print("\n" + "="*30)
    print("📋 REZUMAT STANDARDIZARE")
    print("="*30)
    print(f"✅ Procesate și salvate : {processed_count - 1}") 
    print(f"🗑️ Respinse (prea mici)  : {skipped_size}")
    print(f"❌ Respinse (corupte)    : {skipped_corrupt}")
    print("="*30)

if __name__ == "__main__":
    standardize_images()