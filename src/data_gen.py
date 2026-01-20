import numpy as np
import os
from skimage.draw import disk, ellipse
from skimage.io import imsave
from skimage.util import random_noise
import shutil

def create_brain_slice(condition):
    """Gera uma imagem 2D simulando um corte axial de PET cerebral."""
    img_size = 128
    img = np.zeros((img_size, img_size), dtype=np.float32)

    # Simula o crânio/formato do cérebro base
    rr, cc = ellipse(img_size/2, img_size/2, img_size/2.2, img_size/2.5)
    img[rr, cc] = 0.8 # Metabolismo base (cinza claro)

    # Adiciona "ventrículos" (áreas escuras no centro)
    rr_v, cc_v = ellipse(img_size/2, img_size/2, img_size/6, img_size/8)
    img[rr_v, cc_v] = 0.1

    if condition == 'Alzheimer':
        # Simula hipometabolismo (áreas escuras) nos lobos temporais/parietais
        # Lado esquerdo
        rr_l, cc_l = disk((img_size/1.8, img_size/3.5), img_size/10)
        img[rr_l, cc_l] *= 0.4 # Reduz brilho drasticamente
        
        # Lado direito
        rr_r, cc_r = disk((img_size/1.8, img_size - img_size/3.5), img_size/10)
        img[rr_r, cc_r] *= 0.4 # Reduz brilho drasticamente

    # Adiciona ruído para realismo
    img = random_noise(img, mode='gaussian', var=0.01)
    img = np.clip(img, 0, 1)
    # Converte para formato 8-bit (0-255) para salvar como PNG
    img_uint8 = (img * 255).astype(np.uint8)
    return img_uint8

def generate_dataset(n_samples_per_class=100):
    print("🧠 Generating synthetic brain PET slices...")
    raw_dir = 'data/raw'
    if os.path.exists(raw_dir): shutil.rmtree(raw_dir)
    os.makedirs(os.path.join(raw_dir, 'Normal'), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, 'Alzheimer'), exist_ok=True)

    for i in range(n_samples_per_class):
        # Gerar Normal
        img_norm = create_brain_slice('Normal')
        imsave(os.path.join(raw_dir, 'Normal', f'norm_{i}.png'), img_norm, check_contrast=False)
        
        # Gerar Alzheimer
        img_alz = create_brain_slice('Alzheimer')
        imsave(os.path.join(raw_dir, 'Alzheimer', f'alz_{i}.png'), img_alz, check_contrast=False)
    
    print(f"✅ Generated {n_samples_per_class*2} images in data/raw/")

if __name__ == "__main__":
    generate_dataset()
