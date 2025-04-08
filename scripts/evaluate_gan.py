import os
import sys
import argparse
import logging

# Ajouter le répertoire parent au path pour l'import des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import configure_logging
from src.models.cnn.deep_cnn import DeepCNN
from src.data.data_loader import get_gan_data_loader
from src.evaluation.gan_evaluator import GANEvaluator

# Configuration du logger
logger = configure_logging(level=logging.INFO)

# Modifier la fonction load_gan_model pour une meilleure gestion des erreurs:

def load_gan_model(model_path, input_shape=(32, 32, 3), latent_dim=None):
    """Charge un modèle GAN à partir des fichiers sauvegardés
    
    Args:
        model_path: Chemin vers le répertoire contenant les fichiers du modèle
        input_shape: Forme des images générées
        latent_dim: Dimension de l'espace latent (None pour détection automatique)
        
    Returns:
        Instance DeepCNN chargée avec les poids
    """
    logger.info(f"Chargement du modèle GAN depuis {model_path}")
    
    # Recherche des fichiers de modèle
    if not os.path.isdir(model_path):
        model_dir = os.path.dirname(model_path)
        model_prefix = os.path.basename(model_path)
    else:
        # Chercher le dernier checkpoint dans le dossier
        model_dir = model_path
        checkpoints = [f for f in os.listdir(model_dir) if f.endswith("_generator.h5")]
        
        if not checkpoints:
            raise FileNotFoundError(f"Aucun checkpoint trouvé dans {model_dir}")
        
        # Trier par numéro d'epoch et prendre le plus récent
        latest = sorted(checkpoints, key=lambda x: int(x.split("_")[-3]) 
                        if "epoch" in x else 0)[-1]
        model_prefix = latest.replace("_generator.h5", "")
    
    # Afficher des détails sur le modèle trouvé
    full_path = os.path.join(model_dir, model_prefix)
    logger.info(f"Fichiers de modèle: {full_path}_generator.h5, {full_path}_discriminator.h5")
    
    try:
        # Analyse préalable pour déterminer la dimension latente (si non fournie)
        if latent_dim is None:
            # Essayer d'extraire la dimension latente directement du modèle
            try:
                import h5py
                with h5py.File(f"{full_path}_generator.h5", 'r') as f:
                    # Explorer la structure pour trouver l'information sur la dimension d'entrée
                    for key in f.keys():
                        if key.startswith('layer_') or key == 'model_weights':
                            for layer in f[key]:
                                if 'input' in layer.lower() or layer == 'dense':
                                    if 'kernel' in f[key][layer]:
                                        # La forme du noyau de la première dense peut indiquer la dimension latente
                                        shape = f[key][layer]['kernel'].shape
                                        latent_dim = shape[0] if len(shape) >= 2 else 100
                                        logger.info(f"Dimension latente extraite du fichier h5: {latent_dim}")
                                        break
            except Exception as e:
                logger.warning(f"Impossible d'extraire la dimension latente du fichier h5: {e}")
                latent_dim = 64  # Valeur la plus probable basée sur l'erreur précédente
        
        # Créer une instance du modèle avec la dimension latente détectée ou par défaut
        model = DeepCNN(input_shape=input_shape, latent_dim=latent_dim or 64)
        model.load_model(full_path)
        
        # Afficher les informations du modèle chargé
        logger.info(f"Modèle chargé avec succès: dimension latente = {model.latent_dim}")
        
        # Tester que la génération fonctionne
        test_imgs = model.generate_images(num_images=2)
        logger.info(f"Test de génération réussi: {test_imgs.shape}")
        
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Évaluation approfondie d'un modèle GAN")
    
    parser.add_argument("--model_path", type=str, required=True,
                      help="Chemin vers le modèle GAN à évaluer")
    parser.add_argument("--data_source", type=str, default="cifar10",
                      help="Source des données réelles (nom du dataset ou chemin)")
    parser.add_argument("--n_samples", type=int, default=1000,
                      help="Nombre d'échantillons pour l'évaluation")
    parser.add_argument("--input_shape", type=str, default="32,32,3",
                      help="Forme des images d'entrée (format: H,W,C)")
    parser.add_argument("--latent_dim", type=int, default=100,
                      help="Dimension de l'espace latent du GAN")
    parser.add_argument("--output_dir", type=str, default="results/evaluations",
                      help="Répertoire de sortie pour les résultats")
    
    args = parser.parse_args()
    
    # Parser input_shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Charger le modèle
    model = load_gan_model(args.model_path, input_shape, args.latent_dim)
    
    # Charger les données réelles
    data_loader = get_gan_data_loader(
        args.data_source, 
        batch_size=32, 
        image_size=input_shape[0]
    )
    
    # Créer l'évaluateur
    evaluator = GANEvaluator(
        model=model,
        data_loader=data_loader,
        output_dir=args.output_dir
    )
    
    # Exécuter l'évaluation complète
    results = evaluator.run_full_evaluation(n_samples=args.n_samples)
    
    # Afficher un résumé des résultats
    print("\n=== Résumé de l'Évaluation ===")
    print(f"Inception Score: {results['inception_score']:.3f} ± {results['inception_score_std']:.3f}")
    print(f"FID Score: {results['fid']:.3f}")
    print(f"Diversité: {results['diversity_score']:.5f}")
    print(f"Qualité moyenne: {results['quality_mean']:.3f} ± {results['quality_std']:.3f}")
    print(f"\nRapport complet disponible dans: {os.path.join(args.output_dir, 'evaluation_report.html')}")

if __name__ == "__main__":
    main()