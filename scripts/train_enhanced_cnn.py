import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import configure_logging
from src.models.cnn.deep_cnn import DeepCNN
# Import correct avec la fonction de préconversion
from src.data.data_loader import get_gan_data_loader, preconvert_dataset_to_tensorflow

# Configuration du logger
logger = configure_logging()

def main():
    parser = argparse.ArgumentParser(description="Entraînement amélioré de modèle CNN/GAN")
    parser.add_argument("--config", type=str, help="Chemin vers le fichier de configuration")
    parser.add_argument("--batch_size", type=int, default=64, help="Taille du batch")
    parser.add_argument("--num_epochs", type=int, default=30, help="Nombre d'époques")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Taux d'apprentissage")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension de l'espace latent")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 pour l'optimiseur Adam")
    parser.add_argument("--subset_size", type=float, default=1.0, help="Fraction du dataset à utiliser (0-1)")
    parser.add_argument("--output_dir", type=str, default="results/enhanced_cnn", help="Répertoire de sortie")
    parser.add_argument("--save_interval", type=int, default=5, help="Intervalle de sauvegarde des checkpoints")
    parser.add_argument("--use_spectral_norm", action="store_true", help="Utiliser la normalisation spectrale")
    parser.add_argument("--use_wasserstein", action="store_true", help="Utiliser la perte Wasserstein")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba", "custom"], 
                      help="Dataset à utiliser")
    
    args = parser.parse_args()
    
    # Charger la configuration si spécifiée
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration chargée depuis {args.config}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return 1
    
    # Écraser avec les arguments spécifiés en ligne de commande
    for arg, value in vars(args).items():
        if arg != 'config' and value is not None:
            config[arg] = value
            
    # Créer le répertoire de sortie avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.get('output_dir', 'results/enhanced_cnn'), f"cnn_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    config['output_dir'] = output_dir
    
    # Sauvegarder la configuration utilisée
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Charger les données
    logger.info("Chargement des données...")
    train_loader = get_gan_data_loader(
        source=config.get('dataset', 'cifar10'),
        batch_size=config.get('batch_size', 64),
        image_size=config.get('image_size', 64),
        preconvert=True
    )
    
    # Déterminer la forme d'entrée depuis le dataset
    for batch in train_loader.take(1):
        logger.info(f"Format de batch: {batch.shape}")
        input_shape = (batch.shape[1], batch.shape[2], batch.shape[3])  # (H, W, C)
        logger.info(f"Format des images: {input_shape}")
        break
    
    # Créer et configurer le modèle amélioré
    logger.info("Construction du modèle CNN amélioré...")
    model = DeepCNN(
        input_shape=input_shape,
        latent_dim=config.get('latent_dim', 100),
        use_spectral_norm=config.get('use_spectral_norm', False),
        use_wasserstein=config.get('use_wasserstein', False)
    )
    
    # Entraîner le modèle
    logger.info("Début de l'entraînement...")
    history = model.train(
        train_loader=train_loader,
        epochs=config.get('num_epochs', 30),
        log_interval=10,
        plot_interval=200,
        save_interval=config.get('save_interval', 5)
    )
    
    # Sauvegarder le modèle final
    final_model_path = os.path.join(output_dir, 'models')
    os.makedirs(final_model_path, exist_ok=True)
    model.save_model(os.path.join(final_model_path, 'cnn_final'))
    logger.info(f"Modèle final sauvegardé dans {final_model_path}")
    
    # Générer des images finales
    logger.info("Génération d'images finales...")
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Générer 16 images avec différents niveaux de troncation
    for truncation in [0.7, 1.0, 1.3]:
        images = model.generate_images(16, truncation=truncation)
        model.save_images(images, os.path.join(sample_dir, f'final_samples_truncation_{truncation}.png'))
    
    logger.info(f"Entraînement terminé! Résultats dans {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())