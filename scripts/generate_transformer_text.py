import os
import sys
import torch
import argparse
import logging

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
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                      help="Pénalité pour les tokens répétés (>1 réduit les répétitions)")
    parser.add_argument("--max_length", type=int, default=200,
                      help="Longueur maximale du texte généré")
    parser.add_argument("--output_file", type=str, default=None,
                      help="Fichier pour sauvegarder le texte généré")
    
    args = parser.parse_args()
    
    # Charger le modèle
    try:
        logger.info(f"Chargement du modèle depuis {args.model_path}")
        model = TransformerModel(
            vocab_size=50257,  # GPT-2 vocabulary size
            d_model=256,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=512,
            dropout=0.1
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