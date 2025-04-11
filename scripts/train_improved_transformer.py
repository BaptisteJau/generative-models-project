import os
import sys
import torch
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import configure_logging
from src.training.train import build_model, setup_optimizers, get_data_loaders
from src.training.trainer import create_trainer
from src.models.transformer.transformer_model import TransformerModel

# Configuration du logger
logger = configure_logging(level=logging.INFO)

def load_config(config_path):
    """Chargement et validation de la configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Vérifications de base
    assert config.get('model_type', '').lower() == 'transformer', "Config must be for transformer model"
    assert config.get('vocab_size', 0) > 0, "Vocab size must be positive"
    
    logger.info(f"Configuration chargée depuis {config_path}")
    return config

def prepare_output_dir(config):
    """Prépare le répertoire de sortie avec horodatage"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"transformer_improved_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer les sous-répertoires
    models_dir = os.path.join(output_dir, "models")
    logs_dir = os.path.join(output_dir, "logs")
    samples_dir = os.path.join(output_dir, "samples")
    
    for directory in [models_dir, logs_dir, samples_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Sauvegarder la configuration utilisée
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    logger.info(f"Répertoire de sortie préparé: {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Entraîner un modèle Transformer amélioré")
    
    parser.add_argument("--config", type=str, default="configs/transformer_improved_config.yaml",
                      help="Chemin vers le fichier de configuration")
    parser.add_argument("--resume_from", type=str, default=None,
                      help="Chemin vers un checkpoint pour reprendre l'entraînement")
    parser.add_argument("--epochs", type=int, default=None,
                      help="Nombre d'époques (écrase la valeur dans la config)")
    parser.add_argument("--batch_size", type=int, default=None,
                      help="Taille du batch (écrase la valeur dans la config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                      help="Learning rate (écrase la valeur dans la config)")
    parser.add_argument("--warm_start_from", type=str, default=None,
                      help="Initialiser les poids à partir d'un modèle pré-existant")
    
    args = parser.parse_args()
    
    # Charger la configuration
    config = load_config(args.config)
    
    # Appliquer les arguments de ligne de commande qui écrasent la configuration
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
        logger.info(f"Écrasement du nombre d'époques: {args.epochs}")
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        logger.info(f"Écrasement de la taille du batch: {args.batch_size}")
    
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
        logger.info(f"Écrasement du learning rate: {args.learning_rate}")
    
    # Préparer le répertoire de sortie
    output_dir = prepare_output_dir(config)
    config['output_dir'] = output_dir
    
    try:
        # Obtenir les data loaders avec la configuration actuelle
        logger.info("Chargement des données...")
        train_loader, val_loader = get_data_loaders('transformer', config)
        
        # Mettre à jour la taille du vocabulaire si disponible dans le data loader
        if hasattr(train_loader.dataset, 'vocab_size'):
            config['vocab_size'] = train_loader.dataset.vocab_size
            logger.info(f"Taille du vocabulaire mise à jour: {config['vocab_size']}")
        
        # Construire le modèle
        logger.info("Construction du modèle...")
        if args.resume_from:
            # Charger le modèle à partir d'un checkpoint
            model = TransformerModel(
                vocab_size=config['vocab_size'],
                d_model=config.get('embedding_dim', 256),
                nhead=config.get('num_heads', 4),
                num_encoder_layers=config.get('num_layers', 4),
                num_decoder_layers=config.get('num_layers', 4),
                dim_feedforward=config.get('dim_feedforward', 1024),
                dropout=config.get('dropout_rate', 0.1)
            )
            model.load_state_dict(torch.load(args.resume_from, map_location='cpu', weights_only=True))
            logger.info(f"Modèle chargé depuis {args.resume_from}")
        else:
            # Construire un nouveau modèle
            model = build_model('transformer', config)
            
            # Initialisation à partir d'un modèle existant si demandé
            if args.warm_start_from:
                try:
                    # Chargement partiel des poids (compatibles)
                    pretrained_dict = torch.load(args.warm_start_from, map_location='cpu', weights_only=True)
                    model_dict = model.state_dict()
                    
                    # Filtrer les poids qui correspondent aux dimensions
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                      if k in model_dict and v.shape == model_dict[k].shape}
                    
                    # Mettre à jour le dictionnaire d'état
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    
                    logger.info(f"Initialisation partielle à partir de {args.warm_start_from}")
                    logger.info(f"Chargé {len(pretrained_dict)}/{len(model_dict)} couches compatible")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'initialisation à partir de {args.warm_start_from}: {e}")
        
        # Configuration des optimiseurs
        logger.info("Configuration des optimiseurs...")
        model = setup_optimizers(model, 'transformer', config)
        
        # Créer le trainer
        logger.info("Création du trainer...")
        trainer = create_trainer('transformer', model, train_loader, val_loader, config)
        
        # Lancer l'entraînement
        logger.info("Début de l'entraînement...")
        history = trainer.train()
        
        logger.info("Entraînement terminé!")
        logger.info(f"Modèle final sauvegardé dans: {output_dir}/models/transformer_final.pt")
        
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())