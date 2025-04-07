import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importer nos modules de données
from src.data.data_loader import (
    get_gan_data_loader, 
    get_diffusion_data_loader, 
    get_transformer_data_loader
)
from src.data.preprocessing import (
    add_noise_for_diffusion, 
    normalize_images_for_gan
)

def test_gan_data_loader():
    print("Test du chargeur de données pour GAN...")
    loader = get_gan_data_loader('cifar10', batch_size=4)
    
    # Récupérer un batch et l'afficher
    batch = next(iter(loader))
    images = batch[0]  # Extrait les images (sans étiquettes)
    
    print(f"Forme du batch d'images: {images.shape}")
    print(f"Valeurs min/max: {images.min().item():.2f}/{images.max().item():.2f}")
    
    # Visualiser quelques images
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        img = (images[i] * 0.5 + 0.5).permute(1, 2, 0).numpy()  # Renormalise dans [0, 1]
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle("Images pour GAN")
    plt.savefig("gan_data_test.png")
    print("Visualisation sauvegardée dans gan_data_test.png")

def test_diffusion_data_loader():
    print("\nTest du chargeur de données pour modèle de diffusion...")
    loader = get_diffusion_data_loader('cifar10', batch_size=4)
    
    # Récupérer un batch
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch
    
    print(f"Forme du batch d'images: {images.shape}")
    
    # Simuler un schedule de diffusion
    num_timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Simuler différents niveaux de bruit
    timesteps = [0, 250, 500, 750, 999]
    
    plt.figure(figsize=(len(timesteps) * 5, 5))
    for i, t in enumerate(timesteps):
        # Appliquer le même niveau de bruit à toutes les images
        t_batch = torch.full((images.shape[0],), t, dtype=torch.long)
        noisy_images, _ = add_noise_for_diffusion(images, t_batch, alphas_cumprod)
        
        # Afficher la première image
        plt.subplot(1, len(timesteps), i+1)
        img = (noisy_images[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()  # Renormalise dans [0, 1]
        plt.imshow(img)
        plt.title(f"t={t}")
        plt.axis('off')
    
    plt.suptitle("Processus de diffusion à différents timesteps")
    plt.savefig("diffusion_data_test.png")
    print("Visualisation sauvegardée dans diffusion_data_test.png")

def test_transformer_data_loader():
    print("\nTest du chargeur de données pour modèle Transformer...")
    
    # Utiliser un petit dataset de texte
    train_loader, val_loader = get_transformer_data_loader('tiny_shakespeare', batch_size=2, block_size=64)
    
    # Extraire un batch
    batch = next(iter(train_loader))
    
    print(f"Forme des input_ids: {batch['input_ids'].shape}")
    print(f"Forme des labels: {batch['labels'].shape}")
    
    # Décoder un exemple pour vérification
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Afficher un exemple
    input_text = tokenizer.decode(batch['input_ids'][0])
    target_text = tokenizer.decode(batch['labels'][0])
    
    print("\nExemple d'entrée:")
    print(input_text[:100] + "...")  # Juste les 100 premiers caractères
    
    print("\nExemple de sortie (décalé d'un token):")
    print(target_text[:100] + "...")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Tester tous les chargeurs de données
    test_gan_data_loader()
    test_diffusion_data_loader()
    test_transformer_data_loader()