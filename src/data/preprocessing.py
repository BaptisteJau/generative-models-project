import numpy as np
import torch
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T

def preprocess_images(images, target_size=(64, 64), normalize=True, to_tensor=True):
    """
    Preprocess batch of images with various transformations
    
    Args:
        images: List of PIL images or numpy arrays
        target_size: Tuple of (height, width) to resize images
        normalize: Whether to normalize pixel values to [-1, 1]
        to_tensor: Whether to convert to PyTorch tensors
    
    Returns:
        Processed images
    """
    processed_images = []
    
    for image in images:
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Resize
        if target_size:
            image = image.resize(target_size)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize
        if normalize:
            img_array = (img_array - 127.5) / 127.5
        
        # Convert to tensor
        if to_tensor:
            img_tensor = torch.tensor(img_array, dtype=torch.float32)
            # Ensure channel-first format (C, H, W)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            processed_images.append(img_tensor)
        else:
            processed_images.append(img_array)
    
    # Stack if tensors
    if to_tensor:
        return torch.stack(processed_images)
    return processed_images

def tokenize_texts(texts, tokenizer_name='bert-base-uncased', max_length=128):
    """
    Tokenize a batch of texts using transformers tokenizer
    
    Args:
        texts: List of text strings
        tokenizer_name: Pretrained tokenizer name or path
        max_length: Maximum sequence length
    
    Returns:
        Dictionary containing input_ids, attention_mask, etc.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return encodings

def create_noise_for_diffusion(batch_size, image_size=64, channels=3, device="cpu"):
    """
    Create random noise for diffusion models
    
    Args:
        batch_size: Number of noise samples to generate
        image_size: Size of noise image (assumed square)
        channels: Number of channels (3 for RGB)
        device: Device to create tensor on (cpu/cuda)
    
    Returns:
        Tensor of noise with shape (batch_size, channels, image_size, image_size)
    """
    return torch.randn(batch_size, channels, image_size, image_size, device=device)

def create_noise_for_cnn(batch_size, latent_dim=100, device="cpu"):
    """
    Create random noise for CNN generator
    
    Args:
        batch_size: Number of noise samples
        latent_dim: Dimension of latent space
        device: Device to create tensor on
    
    Returns:
        Tensor of noise with shape (batch_size, latent_dim)
    """
    return torch.randn(batch_size, latent_dim, device=device)

def create_transformer_mask(size):
    """
    Create attention mask for transformer model to prevent attending to future tokens
    
    Args:
        size: Size of the square mask
    
    Returns:
        Mask tensor (size, size) where 1s allow attention and 0s prevent it
    """
    return torch.triu(torch.ones(size, size) == 1).transpose(0, 1)

def preprocess_data(data, data_type='image', **kwargs):
    """
    General preprocessing function that dispatches to specific preprocessing methods
    
    Args:
        data: Input data to preprocess
        data_type: Type of data ('image', 'text', 'noise_diffusion', 'noise_cnn')
        **kwargs: Additional arguments for specific preprocessing functions
    
    Returns:
        Preprocessed data
    """
    if data_type == 'image':
        return preprocess_images(data, **kwargs)
    elif data_type == 'text':
        return tokenize_texts(data, **kwargs)
    elif data_type == 'noise_diffusion':
        return create_noise_for_diffusion(**kwargs)
    elif data_type == 'noise_cnn':
        return create_noise_for_cnn(**kwargs)
    elif data_type == 'transformer_mask':
        return create_transformer_mask(**kwargs)
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Use 'image', 'text', 'noise_diffusion', 'noise_cnn' or 'transformer_mask'")

# Ajouter ces fonctions pour le traitement des données pour modèles de diffusion

def add_noise_for_diffusion(images, timesteps, noise_schedule):
    """
    Ajoute du bruit aux images selon le modèle de diffusion.
    
    Args:
        images: Tensor d'images [B, C, H, W] normalisées dans [-1, 1]
        timesteps: Indices de timestep pour chaque image [B]
        noise_schedule: Schedule de bruit (alphas_cumprod)
        
    Returns:
        tuple (images_bruitées, bruit)
    """
    batch_size = images.shape[0]
    
    # Générer du bruit
    noise = torch.randn_like(images)
    
    # Paramètres de bruitage
    sqrt_alphas_cumprod = torch.sqrt(noise_schedule)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - noise_schedule)
    
    # Sélectionner les paramètres pour les timesteps spécifiés
    sqrt_alphas = sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas = sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    
    # Ajouter le bruit selon la formule du processus de diffusion
    noisy_images = sqrt_alphas * images + sqrt_one_minus_alphas * noise
    
    return noisy_images, noise

def prepare_transformer_batch(tokenizer, texts, max_length=128):
    """
    Prépare un batch de textes pour l'entraînement d'un Transformer génératif.
    
    Args:
        tokenizer: Tokenizer à utiliser
        texts: Liste de textes
        max_length: Longueur maximale des séquences
        
    Returns:
        dict avec input_ids et attention_mask
    """
    # Tokenisation avec padding
    batch = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Pour la génération de texte, nous avons besoin de décaler les inputs pour avoir 
    # les targets (labels) automatiquement générés
    input_ids = batch['input_ids'][:, :-1]  # tous sauf le dernier token
    labels = batch['input_ids'][:, 1:]      # tous sauf le premier token
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": batch['attention_mask'][:, :-1]
    }

def normalize_images_for_gan(images):
    """
    Normalise les images pour l'entraînement GAN dans [-1, 1]
    
    Args:
        images: Images en format [0, 1] ou [0, 255]
        
    Returns:
        Images normalisées dans [-1, 1]
    """
    # Si les images sont déjà dans [0, 1]
    if images.max() <= 1.0:
        return images * 2 - 1
    else:
        # Si les images sont dans [0, 255]
        return (images / 127.5) - 1