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

def get_gan_data_loader(source, batch_size=32, image_size=64):
    """
    Chargeur de données spécialement conçu pour les GAN (CNN génératif).
    Renvoie seulement des images sans étiquettes.
    
    Args:
        source: Nom du dataset ou chemin vers les images
        batch_size: Taille du batch
        image_size: Taille des images (carrées)
    
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
    
    # Créer le DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader

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
    """Dataset avec fonctionnalités avancées de tokenisation pour les modèles Transformer"""
    def __init__(self, file_path=None, text_data=None, tokenizer=None, block_size=128, stride=64):
        """
        Args:
            file_path: Chemin vers le fichier texte
            text_data: Liste de textes si déjà chargés
            tokenizer: Tokenizer à utiliser (sinon utilise GPT-2)
            block_size: Taille maximale des séquences
            stride: Chevauchement entre blocs consécutifs
        """
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        else:
            self.tokenizer = tokenizer
            
        self.block_size = block_size
        self.stride = stride
        
        # Charger les textes
        if file_path is not None:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.examples = self._tokenize_and_chunk(text)
            elif file_path.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(file_path)
                text_column = 'text' if 'text' in df.columns else df.columns[0]
                texts = df[text_column].tolist()
                self.examples = []
                for text in texts:
                    self.examples.extend(self._tokenize_and_chunk(text))
        elif text_data is not None:
            self.examples = []
            for text in text_data:
                self.examples.extend(self._tokenize_and_chunk(text))
    
    def _tokenize_and_chunk(self, text):
        """Tokenize et divise le texte en chunks avec chevauchement"""
        tokenized = self.tokenizer.encode(text)
        examples = []
        
        # Créer des chunks avec chevauchement
        for i in range(0, len(tokenized) - self.block_size + 1, self.stride):
            input_ids = tokenized[i:i + self.block_size]
            examples.append({
                "input_ids": input_ids[:-1],  # entrée: tous sauf le dernier token
                "labels": input_ids[1:]       # cible: tous sauf le premier token (décalage de 1)
            })
            
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

def get_transformer_data_loader(source, batch_size=16, tokenizer=None, block_size=128):
    """
    Chargeur de données spécialisé pour les modèles Transformer génératifs
    
    Args:
        source: Chemin vers le fichier texte ou nom du dataset
        batch_size: Taille du batch
        tokenizer: Tokenizer à utiliser
        block_size: Taille maximale des séquences
        
    Returns:
        DataLoader pour modèle Transformer génératif
    """
    # Pour des datasets standards
    if source.lower() in ['wikitext', 'tiny_shakespeare']:
        dataset = get_text_dataset(source)
        
        # Convertir en format pour Transformer
        transformer_dataset = TransformerTextDataset(
            text_data=dataset.texts if hasattr(dataset, 'texts') else dataset,
            tokenizer=tokenizer,
            block_size=block_size
        )
    else:
        # Fichier texte ou CSV
        transformer_dataset = TransformerTextDataset(
            file_path=source,
            tokenizer=tokenizer,
            block_size=block_size
        )
    
    # Diviser en train/val
    train_size = int(0.9 * len(transformer_dataset))
    val_size = len(transformer_dataset) - train_size
    train_dataset, val_dataset = random_split(transformer_dataset, [train_size, val_size])
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=transformer_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=transformer_collate_fn
    )
    
    return train_loader, val_loader

def transformer_collate_fn(batch):
    """Fonction de collate personnalisée pour les batches Transformer"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}