import os
import sys
import torch
import argparse
import logging
import glob

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import configure_logging
from src.models.transformer.transformer_model import TransformerModel

# Configuration du logger
logger = configure_logging(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Générer du texte avec un modèle Transformer")
    
    parser.add_argument("--model_path", type=str, required=True,
                      help="Chemin vers le modèle Transformer entraîné")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                      help="Texte de départ pour la génération")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Température de sampling (0.7=équilibré, <1=conservateur, >1=créatif)")
    parser.add_argument("--top_k", type=int, default=50,
                      help="Limiter le choix aux k tokens les plus probables")
    parser.add_argument("--top_p", type=float, default=0.95,
                      help="Échantillonnage nucleus (filtrer les tokens peu probables)")
    parser.add_argument("--repetition_penalty", type=float, default=2.0, 
                      help="Pénalité pour les répétitions (min: 1.0, recommandé: 1.5-2.5)")
    parser.add_argument("--max_length", type=int, default=200,
                      help="Longueur maximale du texte généré")
    parser.add_argument("--output_file", type=str, default=None,
                      help="Fichier pour sauvegarder le texte généré")
    
    args = parser.parse_args()
    
    # Résoudre le chemin du modèle automatiquement si nécessaire
    if "*" in args.model_path or not os.path.exists(args.model_path):
        # Si le chemin contient un wildcard ou n'existe pas
        potential_paths = []
        
        # Cas 1: Le chemin contient un wildcard
        if "*" in args.model_path:
            potential_paths = glob.glob(args.model_path)
        
        # Cas 2: Le chemin est spécifique mais n'existe pas - essayer de trouver un modèle similaire
        else:
            base_dir = os.path.dirname(os.path.dirname(args.model_path))
            model_name = os.path.basename(args.model_path)
            pattern = os.path.join(base_dir, "*", "models", model_name)
            potential_paths = glob.glob(pattern)
        
        if potential_paths:
            # Utiliser le modèle le plus récent
            args.model_path = max(potential_paths, key=os.path.getctime)
            logger.info(f"Modèle résolu automatiquement: {args.model_path}")
        else:
            logger.error(f"Aucun modèle trouvé correspondant au chemin {args.model_path}")
            return 1
    
    # Ajuster les paramètres pour éviter les répétitions pathologiques
    if args.repetition_penalty < 1.5:
        logger.warning(f"Pénalité de répétition faible ({args.repetition_penalty}), augmentée à 1.5 pour éviter les boucles")
        args.repetition_penalty = 1.5
    
    # Charger le modèle
    try:
        logger.info(f"Chargement du modèle depuis {args.model_path}")
        model = TransformerModel(
            vocab_size=50257,  # Cette valeur est correcte (vocabulaire GPT-2)
            d_model=128,       # Réduire de 256 à 128
            nhead=2,           # Réduire de 4 à 2
            num_encoder_layers=2,  # Réduire de 4 à 2
            num_decoder_layers=2,  # Réduire de 4 à 2
            dim_feedforward=512,   # Cette valeur semble correcte
            dropout=0.1           # Cette valeur semble correcte
        )
        model.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=True))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Générer du texte
        logger.info(f"Génération de texte à partir du prompt: '{args.prompt}'")
        generated_text = model.generate(
            args.prompt, 
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
        # Afficher le résultat
        print("\nTexte généré:")
        print("-" * 40)
        print(generated_text)
        print("-" * 40)
        
        # Sauvegarder dans un fichier si demandé
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            logger.info(f"Texte sauvegardé dans {args.output_file}")
                
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())