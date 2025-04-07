import os
import sys
import torch
import yaml
import argparse
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_loader import load_data, get_transformer_data_loader

def load_configuration(config_path):
    """Charge la configuration depuis un fichier YAML"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def build_model(model_type, config):
    """
    Construit un modèle selon le type et la configuration
    
    Args:
        model_type: Type de modèle ('cnn', 'transformer', 'diffusion')
        config: Configuration du modèle
    
    Returns:
        Instance du modèle
    """
    if model_type.lower() == 'cnn':
        # Import only when needed
        from src.models.cnn.deep_cnn import DeepCNN
        
        # Pour le modèle CNN génératif (GAN)
        input_shape = (
            config.get('input_shape', {}).get('height', 64),
            config.get('input_shape', {}).get('width', 64),
            config.get('input_shape', {}).get('channels', 3)
        )
        latent_dim = config.get('latent_dim', 100)
        return DeepCNN(input_shape=input_shape, latent_dim=latent_dim)
        
    elif model_type.lower() == 'transformer':
        # Import only when needed
        from src.models.transformer.transformer_model import TransformerModel
        
        # Pour le modèle Transformer
        vocab_size = config.get('vocab_size', 30000)
        d_model = config.get('embedding_dim', 512)
        nhead = config.get('num_heads', 8)
        num_encoder_layers = config.get('num_layers', 6)
        num_decoder_layers = config.get('num_layers', 6)
        dim_feedforward = d_model * 4  # Taille classique: 4x dimension du modèle
        dropout = config.get('dropout_rate', 0.1)
        
        return TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
    elif model_type.lower() == 'diffusion':
        # Import only when needed
        from src.models.diffusion.diffusion_model import DiffusionModel
        
        # Pour le modèle de diffusion
        diffusion_config = {
            'image_size': config.get('image_size', 64),
            'num_channels': config.get('num_channels', 3),
            'batch_size': config.get('batch_size', 32),
            'beta_start': config.get('beta_start', 1e-4),
            'beta_end': config.get('beta_end', 0.02),
            'num_timesteps': config.get('num_timesteps', 1000),
            'learning_rate': config.get('learning_rate', 1e-4),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        return DiffusionModel(diffusion_config)
        
    else:
        raise ValueError(f"Model type not supported: {model_type}")

def setup_optimizers(model, model_type, config):
    """Configure les optimiseurs pour le modèle"""
    lr = config.get('learning_rate', 0.001)
    
    if model_type.lower() == 'cnn':
        # Pour CNN (GAN)
        if not hasattr(model, 'optimizer'):
            # Vérifier si c'est un modèle Keras (qui utilise Sequential)
            if hasattr(model, 'generator') and 'keras' in str(type(model.generator)):
                # Utiliser les optimiseurs Keras pour les modèles Keras
                import tensorflow as tf
                g_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr,
                    beta_1=0.5, 
                    beta_2=0.999
                )
                d_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr,
                    beta_1=0.5, 
                    beta_2=0.999
                )
                # Attacher les optimiseurs au modèle
                model.g_optimizer = g_optimizer
                model.d_optimizer = d_optimizer
            else:
                # Pour les modèles PyTorch standard
                g_optimizer = torch.optim.Adam(
                    model.generator.parameters(),
                    lr=lr,
                    betas=(0.5, 0.999)
                )
                d_optimizer = torch.optim.Adam(
                    model.discriminator.parameters(),
                    lr=lr,
                    betas=(0.5, 0.999)
                )
                # Attacher les optimiseurs au modèle
                model.g_optimizer = g_optimizer
                model.d_optimizer = d_optimizer
        
    elif model_type.lower() == 'transformer':
        # Create optimizer and attach it to the model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.optimizer = optimizer
        
    elif model_type.lower() == 'diffusion':
        # L'optimiseur est généralement géré dans le DiffusionModel
        if not hasattr(model, 'optimizer'):
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=config.get('weight_decay', 1e-4)
            )
            model.optimizer = optimizer
            
    return model

def get_data_loaders(model_type, config):
    """Obtient les data loaders appropriés selon le type de modèle"""
    batch_size = config.get('batch_size', 32)
    dataset_name = config.get('dataset_name', None)
    dataset_path = config.get('dataset_path', None)
    
    # Set default dataset based on model type
    if model_type.lower() == 'transformer':
        # For transformer models, use a text dataset by default
        source = dataset_name or dataset_path or 'tiny_shakespeare'  # Default to tiny_shakespeare for text
    else:
        # For image-based models (CNN, diffusion), use an image dataset by default
        source = dataset_name or dataset_path or 'cifar10'  # Default to cifar10 for images
    
    if model_type.lower() == 'cnn':
        # Import only when needed
        from src.data.data_loader import get_gan_data_loader
        
        # Pour les GANs, nous avons besoin d'un chargeur spécial
        train_loader = get_gan_data_loader(
            source, 
            batch_size=batch_size, 
            image_size=config.get('input_shape', {}).get('height', 64)
        )
        return train_loader, None  # GANs n'ont généralement pas de validation
        
    elif model_type.lower() == 'diffusion':
        # Import only when needed
        from src.data.data_loader import get_diffusion_data_loader
        
        # Pour les modèles de diffusion
        train_loader = get_diffusion_data_loader(
            source, 
            batch_size=batch_size, 
            image_size=config.get('image_size', 64)
        )
        return train_loader, None  # Modèles de diffusion n'utilisent pas de validation classique
        
    elif model_type.lower() == 'transformer':
        # Import only when needed
        from src.data.data_loader import get_transformer_data_loader
        
        # Pour les modèles Transformer
        tokenizer = None  # On peut ajouter le chargement du tokenizer spécifique ici
        block_size = config.get('max_sequence_length', 128)
        
        return get_transformer_data_loader(
            source, 
            batch_size=batch_size, 
            tokenizer=tokenizer, 
            block_size=block_size
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(model_type, config_path):
    """
    Fonction principale pour entraîner un modèle génératif
    
    Args:
        model_type: Type de modèle ('cnn', 'transformer', 'diffusion')
        config_path: Chemin vers le fichier de configuration
    """
    # Charger la configuration
    config = load_configuration(config_path)
    
    # Créer le répertoire pour les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results', f"{model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    config['save_dir'] = output_dir
    
    # Configurer device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data loaders first to get vocab size for transformer
    train_loader, val_loader = get_data_loaders(model_type, config)
    
    # Capture vocab size from data loader if it's a transformer
    if (model_type.lower() == 'transformer'):
        # Extract vocab size from the dataset of the train loader
        vocab_size = train_loader.dataset.dataset.vocab_size if hasattr(train_loader.dataset, 'dataset') else 50257
        # Update config with the correct vocab size
        config['vocab_size'] = vocab_size
    
    # Build model with updated config
    model = build_model(model_type, config)
    
    # Configurer les optimiseurs
    model = setup_optimizers(model, model_type, config)
    
    # Import the trainer creator only when needed
    from src.training.trainer import create_trainer
    
    # Créer le trainer adapté
    trainer = create_trainer(model_type, model, train_loader, val_loader, config)
    
    # Entraîner le modèle
    history = trainer.train()
    
    # Sauvegarder le modèle final
    final_model_path = os.path.join(output_dir, f"{model_type}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    
    return model, history

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Train a generative model")
    parser.add_argument("--model", type=str, required=True, choices=['cnn', 'transformer', 'diffusion'],
                      help="Type of model to train")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    
    args = parser.parse_args()
    
    train_model(args.model, args.config)

if __name__ == "__main__":
    main()