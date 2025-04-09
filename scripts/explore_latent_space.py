import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import configure_logging
from src.models.cnn.deep_cnn import DeepCNN
from src.creative.latent_exploration import LatentExplorer
from src.creative.conditional_generation import AttributeController

# Configuration du logger
logger = configure_logging(level=logging.INFO)

def load_gan_model(model_path, input_shape=(32, 32, 3), latent_dim=None):
    """Charge un modèle GAN à partir des fichiers sauvegardés"""
    # Réutiliser la fonction load_gan_model de evaluate_gan.py
    from scripts.evaluate_gan import load_gan_model as load_model
    return load_model(model_path, input_shape, latent_dim)

def main():
    parser = argparse.ArgumentParser(description="Exploration de l'espace latent d'un GAN")
    
    parser.add_argument("--model_path", type=str, required=True,
                      help="Chemin vers le modèle GAN à explorer")
    parser.add_argument("--mode", type=str, default="interpolation",
                      choices=["interpolation", "dimension", "grid", "attributes", "pca"],
                      help="Mode d'exploration")
    parser.add_argument("--output_dir", type=str, default="results/latent_exploration",
                      help="Répertoire de sortie pour les résultats")
    parser.add_argument("--n_steps", type=int, default=10,
                      help="Nombre d'étapes pour les interpolations")
    parser.add_argument("--dimensions", type=str, default="0,1",
                      help="Dimensions à explorer (séparées par des virgules)")
    parser.add_argument("--animate", action="store_true",
                      help="Créer une animation pour les interpolations")
    parser.add_argument("--range", type=float, nargs=2, default=[-2, 2],
                      help="Plage de valeurs pour l'exploration")
    parser.add_argument("--input_shape", type=str, default="32,32,3",
                      help="Forme des images d'entrée (format: H,W,C)")
    parser.add_argument("--latent_dim", type=int, default=None,
                      help="Dimension latente du GAN (none pour auto-détection)")
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parser input_shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Charger le modèle
    model = load_gan_model(args.model_path, input_shape, args.latent_dim)
    
    # Créer l'explorateur d'espace latent
    explorer = LatentExplorer(model, output_dir=args.output_dir)
    
    if args.mode == "interpolation":
        # Interpolation entre deux points dans l'espace latent
        save_path = os.path.join(args.output_dir, "interpolation.gif" if args.animate else "interpolation.png")
        
        logger.info("Interpolation entre deux points aléatoires de l'espace latent...")
        images, vectors = explorer.latent_space_interpolation(
            n_steps=args.n_steps,
            method='spherical',
            animate=args.animate,
            save_path=save_path
        )
        
        logger.info(f"Interpolation sauvegardée: {save_path}")
        
    elif args.mode == "dimension":
        # Exploration d'une dimension spécifique
        dims = list(map(int, args.dimensions.split(',')))
        
        for dim in dims:
            save_path = os.path.join(args.output_dir, f"dimension_{dim}.png")
            
            logger.info(f"Exploration de la dimension {dim}...")
            images, vectors = explorer.explore_dimension(
                dimension=dim,
                n_steps=args.n_steps,
                other_dims_range=args.range,
                save_path=save_path
            )
        
    elif args.mode == "grid":
        # Création d'une grille en variant deux dimensions
        dims = list(map(int, args.dimensions.split(',')))
        if len(dims) < 2:
            dims = [0, 1]  # Dimensions par défaut
            
        save_path = os.path.join(args.output_dir, f"grid_{dims[0]}_{dims[1]}.png")
        
        logger.info(f"Création d'une grille en variant les dimensions {dims[0]} et {dims[1]}...")
        grid = explorer.create_latent_space_grid(
            dim1=dims[0],
            dim2=dims[1],
            n_steps=min(args.n_steps, 10),  # Limiter à 10 pour éviter des grilles trop grandes
            value_range=args.range,
            save_path=save_path
        )
        
    elif args.mode == "attributes":
        # Découverte et application d'attributs
        controller = AttributeController(model, explorer, os.path.join(args.output_dir, "attributes"))
        
        # Découvrir trois attributs via PCA
        logger.info("Découverte d'attributs via analyse en composantes principales...")
        controller.interactive_exploration(
            n_attributes=3, 
            save_dir=os.path.join(args.output_dir, "attributes")
        )
        
        # Générer quelques images avec différentes combinaisons d'attributs
        for i in range(5):
            attributes = {}
            for j in range(min(3, len(controller.attribute_directions))):
                attr_name = f"attribute_{j+1}"
                attributes[attr_name] = np.random.uniform(-1.5, 1.5)
                
            save_path = os.path.join(args.output_dir, "attributes", f"generated_{i+1}.png")
            controller.generate_with_attributes(attributes, save_path=save_path)
                
    elif args.mode == "pca":
        # Exploration des composantes principales
        save_dir = os.path.join(args.output_dir, "pca")
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info("Analyse des composantes principales de l'espace latent...")
        results = explorer.explore_principal_directions(
            n_components=5,
            n_steps=args.n_steps,
            save_dir=save_dir
        )

if __name__ == "__main__":
    main()