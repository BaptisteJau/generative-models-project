import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision
from torchvision import transforms, datasets
from transformers import AutoTokenizer
import requests
import zipfile
import io
import yaml

def get_data_dir():
    """Return the path to the data directory, creating it if it doesn't exist"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Also create subdirectories for different types of data
    text_dir = os.path.join(data_dir, "text")
    os.makedirs(text_dir, exist_ok=True)
    
    return data_dir

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer=None, max_length=128):
        self.max_length = max_length
        
        # Load data
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
            self.texts = self.data['text'].tolist()
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.texts = f.readlines()
        else:
            raise ValueError("Unsupported file format. Use .csv or .txt")
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for key in encoding.keys():
            encoding[key] = encoding[key].squeeze(0)
            
        return encoding

def get_standard_dataset(name, root="./data", train=True, download=True, transform=None):
    """Get standard datasets like MNIST, CIFAR10, etc."""
    if name.lower() == 'mnist':
        return datasets.MNIST(root=root, train=train, download=download, transform=transform)
    elif name.lower() == 'cifar10':
        return datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    elif name.lower() == 'fashion_mnist':
        return datasets.FashionMNIST(root=root, train=train, download=download, transform=transform)
    elif name.lower() == 'celeba':
        return datasets.CelebA(root=root, split='train' if train else 'test', download=download, transform=transform)
    else:
        raise ValueError(f"Dataset {name} not supported. Choose from 'mnist', 'cifar10', 'fashion_mnist', 'celeba'.")

def get_text_dataset(name, root="./data"):
    """Get standard text datasets."""
    os.makedirs(root, exist_ok=True)
    
    if name.lower() == "wikitext":
        from datasets import load_dataset
        return load_dataset("wikitext", "wikitext-2-v1")
    
    elif name.lower() == "tiny_shakespeare":
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        file_path = os.path.join(root, "tiny_shakespeare.txt")
        
        if not os.path.exists(file_path):
            print(f"Downloading {name}...")
            response = requests.get(url)
            with open(file_path, 'wb') as f:
                f.write(response.content)
                
        return TextDataset(file_path)
    
    else:
        raise ValueError(f"Dataset {name} not supported. Choose from 'wikitext', 'tiny_shakespeare'.")

def get_image_loader(source, batch_size=32, transform=None, split_ratio=0.8):
    """Get image data loader with train/val split"""
    if isinstance(source, str):
        # Check if it's a standard dataset name or directory path
        if source.lower() in ['mnist', 'cifar10', 'fashion_mnist', 'celeba']:
            # Standard dataset
            if transform is None:
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            
            train_dataset = get_standard_dataset(source, train=True, download=True, transform=transform)
            val_dataset = get_standard_dataset(source, train=False, download=True, transform=transform)
        else:
            # Custom directory
            if not os.path.isdir(source):
                raise ValueError(f"Directory {source} does not exist")
                
            dataset = ImageDataset(source, transform)
            train_size = int(split_ratio * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        # Assume it's already a dataset and just split it
        train_size = int(split_ratio * len(source))
        val_size = len(source) - train_size
        train_dataset, val_dataset = random_split(source, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_text_loader(source, batch_size=32, tokenizer=None, max_length=128, split_ratio=0.8):
    """Get text data loader with train/val split"""
    if isinstance(source, str):
        # Check if it's a standard dataset name or file path
        if source.lower() in ['wikitext', 'tiny_shakespeare']:
            dataset = get_text_dataset(source)
        else:
            # Assume it's a file path
            dataset = TextDataset(source, tokenizer, max_length)
    else:
        # Assume it's already a dataset
        dataset = source
    
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def load_data(config):
    """Load data based on configuration"""
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_type = cfg.get('data_type', 'image')
    dataset_name = cfg.get('dataset_name', None)
    dataset_path = cfg.get('dataset_path', None)
    batch_size = cfg.get('batch_size', 32)
    
    if data_type == 'image':
        transform = None
        if 'augmentation' in cfg:
            transform_list = []
            aug = cfg['augmentation']
            
            # Add resizing
            transform_list.append(transforms.Resize((cfg.get('input_shape', {}).get('height', 64), 
                                                  cfg.get('input_shape', {}).get('width', 64))))
            
            # Add data augmentation if specified
            if aug.get('horizontal_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug.get('rotation_range', 0) > 0:
                transform_list.append(transforms.RandomRotation(aug['rotation_range']))
            
            # Add to tensor and normalize
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            
            transform = transforms.Compose(transform_list)
        
        return get_image_loader(dataset_name or dataset_path, batch_size, transform)
    
    elif data_type == 'text':
        tokenizer = None
        if cfg.get('tokenizer', None):
            tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer'])
            
        return get_text_loader(dataset_name or dataset_path, batch_size, tokenizer, 
                              max_length=cfg.get('max_sequence_length', 128))
    
    else:
        raise ValueError(f"Data type {data_type} not supported. Choose from 'image', 'text'.")

def preconvert_dataset_to_tensorflow(dataset, batch_size=32, normalize_range=(-1, 1)):
    """Convertit tout le dataset en tensors TensorFlow à l'avance"""
    from src.utils.framework_utils import FrameworkBridge
    import tensorflow as tf
    
    converted_data = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, _ = batch
        else:
            imgs = batch
        
        # Conversion en batch
        tf_imgs = FrameworkBridge.pytorch_to_tensorflow(imgs, normalize_range=normalize_range)
        converted_data.append(tf_imgs)
    
    # Combiner tous les batches en un seul dataset TensorFlow
    return tf.data.Dataset.from_tensor_slices(converted_data).batch(batch_size)

def get_gan_data_loader(source, batch_size=32, image_size=64, preconvert=False):
    """
    Chargeur de données spécialement conçu pour les GAN (CNN génératif).
    Renvoie seulement des images sans étiquettes.
    
    Args:
        source: Nom du dataset ou chemin vers les images
        batch_size: Taille du batch
        image_size: Taille des images (carrées)
        preconvert: Convertir à l'avance en tensors TensorFlow
    
    Returns:
        DataLoader pour entraînement GAN
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation dans [-1, 1]
    ])
    
    # Obtenir le dataset
    if source.lower() in ['mnist', 'cifar10', 'fashion_mnist', 'celeba']:
        dataset = get_standard_dataset(source, transform=transform)
        # Convertir les datasets avec étiquettes en datasets sans étiquettes
        images_only = []
        for img, _ in dataset:
            images_only.append(img)
        tensor_images = torch.stack(images_only)
        dataset = TensorDataset(tensor_images)
    else:
        # Utiliser un dataset personnalisé avec dossier d'images
        dataset = ImageOnlyDataset(source, transform=transform)
    
    # Ajouter l'option de préconversion
    if preconvert:
        return preconvert_dataset_to_tensorflow(dataset, batch_size)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class ImageOnlyDataset(Dataset):
    """Dataset qui charge uniquement des images sans étiquettes, adapté pour les GANs"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) 
                          if os.path.isfile(os.path.join(image_dir, f)) 
                          and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return (image,)  # Retourne un tuple avec une seule valeur pour compatibilité

def get_diffusion_data_loader(source, batch_size=32, image_size=64):
    """
    Chargeur de données pour les modèles de diffusion.
    
    Args:
        source: Nom du dataset ou chemin vers les images
        batch_size: Taille du batch
        image_size: Taille des images (carrées)
    
    Returns:
        DataLoader pour modèle de diffusion
    """
    # Même transformation que pour GAN mais nous n'avons pas besoin d'extraire
    # seulement les images car le modèle de diffusion s'occupe du bruitage
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Utiliser la fonction existante mais adapter le format
    train_loader, _ = get_image_loader(source, batch_size=batch_size, transform=transform)
    
    return train_loader

class TransformerTextDataset(Dataset):
    def __init__(self, text_data=None, text_path=None, tokenizer=None, block_size=128):
        """
        Dataset pour modèles Transformer de génération de texte
        
        Args:
            text_data: Texte brut ou chemin vers un fichier texte
            tokenizer: Tokenizer à utiliser (si None, utilise GPT2Tokenizer)
            block_size: Taille maximale des séquences
        """
        self.block_size = block_size
        
        # Pour compatibilité avec les appels existants
        if text_data is None and text_path is not None:
            text_data = text_path
        
        # Charger le texte si c'est un chemin de fichier
        if isinstance(text_data, str) and os.path.isfile(text_data):
            with open(text_data, 'r', encoding='utf-8') as f:
                text_data = f.read()
        
        # Initialiser le tokenizer
        if tokenizer is None:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # Ajout de token spécial si nécessaire pour GPT2
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Tokenize le texte
        self.encodings = self.tokenizer(text_data, return_tensors='pt', truncation=True)
        self.input_ids = self.encodings.input_ids[0]
        
        # Stocker les exemples sous forme de liste pour __len__
        self.examples = []
        for i in range(0, len(self.input_ids) - block_size + 1, 1):  # Step par 1 pour maximiser le nombre de séquences
            self.examples.append((
                self.input_ids[i:i+block_size],
                self.input_ids[i+1:i+block_size+1]
            ))
        
        # Stocker la taille du vocabulaire
        self.vocab_size = len(self.tokenizer)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        inputs, targets = self.examples[idx]
        return inputs, targets

def get_transformer_data_loader(source='tiny_shakespeare', batch_size=32, tokenizer=None, block_size=128):
    """
    Get data loaders for transformer models
    
    Args:
        source: Path to text file or dataset name
        batch_size: Batch size for training
        tokenizer: Optional pretrained tokenizer
        block_size: Context window size
    
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    # Handle standard datasets
    if source == 'tiny_shakespeare':
        source = os.path.join(get_data_dir(), 'text/tiny_shakespeare.txt')
        
        # Check if the file exists, download if it doesn't
        if not os.path.exists(source):
            print(f"Downloading tiny_shakespeare.txt to {source}")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            os.makedirs(os.path.dirname(source), exist_ok=True)
            response = requests.get(url)
            with open(source, 'wb') as f:
                f.write(response.content)
                
    elif not os.path.exists(source):
        raise ValueError(f"Source {source} not found")
    
    # Create dataset
    transformer_dataset = TransformerTextDataset(
        text_data=source,
        tokenizer=tokenizer,
        block_size=block_size
    )
    
    # Get vocab size from the dataset's tokenizer for model creation
    vocab_size = len(transformer_dataset.tokenizer.vocab) if hasattr(transformer_dataset.tokenizer, 'vocab') else transformer_dataset.tokenizer.vocab_size
    
    # Store the vocab size as an attribute of the dataset for later use
    transformer_dataset.vocab_size = vocab_size
    
    # Split into train and validation
    train_size = int(0.9 * len(transformer_dataset))
    val_size = len(transformer_dataset) - train_size
    
    train_dataset, val_dataset = random_split(transformer_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=transformer_collate_fn  # Ajouter cette ligne
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=transformer_collate_fn  # Ajouter cette ligne
    )
    
    return train_loader, val_loader

def transformer_collate_fn(batch):
    """Collate function pour Transformer qui aligne les séquences à la même longueur"""
    
    # Séparer les entrées et les cibles
    inputs, targets = zip(*batch)
    
    # Trouver la longueur maximale dans ce batch
    max_len = max(inp.size(0) for inp in inputs)
    
    # Fonction pour padder un tensor
    def pad_tensor(x, length):
        # Créer un nouveau tensor avec la longueur cible
        pad_size = length - x.size(0)
        if pad_size <= 0:
            return x
            
        # Le padding token est généralement 0 dans la plupart des tokenizers
        padding = torch.zeros(pad_size, dtype=x.dtype, device=x.device)
        padded = torch.cat([x, padding], dim=0)
        return padded
    
    # Padder tous les inputs et targets
    padded_inputs = torch.stack([pad_tensor(inp, max_len) for inp in inputs])
    padded_targets = torch.stack([pad_tensor(tgt, max_len) for tgt in targets])
    
    return padded_inputs, padded_targets