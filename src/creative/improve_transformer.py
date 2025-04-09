import os
import sys
import torch
import argparse
import logging
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.transformer.transformer_model import TransformerModel

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("transformer_improvements.log")
        ]
    )
    return logging.getLogger(__name__)

def load_transformer(model_path, vocab_size=50257):
    """Charge un modèle transformer avec les paramètres appropriés"""
    logger = logging.getLogger(__name__)
    
    try:
        # Créer le modèle avec la même architecture
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=128,    # Utilisez les valeurs d'origine pour la compatibilité
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1
        )
        
        # Charger les poids
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        logger.info(f"Modèle chargé avec succès: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

def generate_comparison(model, prompts, output_dir="results/improved_transformer"):
    """Génère des comparaisons avec différents paramètres"""
    logger = logging.getLogger(__name__)
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Paramètres à tester
    configurations = [
        {"name": "default", "temp": 1.0, "top_k": 0, "top_p": 1.0, "rep_penalty": 1.0},
        {"name": "creative", "temp": 0.9, "top_k": 50, "top_p": 0.95, "rep_penalty": 1.1},
        {"name": "balanced", "temp": 0.7, "top_k": 40, "top_p": 0.92, "rep_penalty": 1.2},
        {"name": "focused", "temp": 0.5, "top_k": 30, "top_p": 0.9, "rep_penalty": 1.3},
        {"name": "deterministic", "temp": 0.3, "top_k": 20, "top_p": 0.85, "rep_penalty": 1.5}
    ]
    
    # Fichier de sortie
    output_file = os.path.join(output_dir, "generation_comparison.md")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Comparaison des méthodes de génération\n\n")
        
        for prompt in prompts:
            f.write(f"## Prompt: \"{prompt}\"\n\n")
            
            for config in configurations:
                f.write(f"### Configuration: {config['name']}\n")
                f.write(f"- Température: {config['temp']}\n")
                f.write(f"- Top-k: {config['top_k']}\n")
                f.write(f"- Top-p: {config['top_p']}\n")
                f.write(f"- Pénalité de répétition: {config['rep_penalty']}\n\n")
                
                try:
                    # Générer le texte avec ces paramètres
                    generated = model.generate(
                        prompt=prompt,
                        max_length=150,
                        temperature=config['temp'],
                        top_k=config['top_k'],
                        top_p=config['top_p'],
                        repetition_penalty=config['rep_penalty']
                    )
                    
                    f.write("```\n")
                    f.write(generated)
                    f.write("\n```\n\n")
                    
                    logger.info(f"Génération réussie pour '{prompt}' avec config '{config['name']}'")
                except Exception as e:
                    error_msg = f"Erreur lors de la génération: {str(e)}"
                    logger.error(error_msg)  # Log l'erreur avant d'écrire dans le fichier
                    f.write(f"**ERREUR**: {error_msg}\n\n")
            
            f.write("---\n\n")
    
    logger.info(f"Comparaison sauvegardée dans {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Améliorer les résultats de génération du modèle Transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle Transformer")
    parser.add_argument("--output_dir", type=str, default="results/improved_transformer", help="Répertoire de sortie")
    
    args = parser.parse_args()
    
    # Configuration du logger
    logger = setup_logging()
    
    # Charger le modèle
    model = load_transformer(args.model_path)
    
    # Prompts de test
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where technology",
        "Scientists recently discovered",
        "The most important lesson I learned"
    ]
    
    # Générer les comparaisons
    output_file = generate_comparison(model, prompts, args.output_dir)
    
    print(f"Comparaison terminée. Résultats sauvegardés dans: {output_file}")

if __name__ == "__main__":
    main()