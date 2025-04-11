import os
import sys
import torch
import argparse
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import numpy as np
from tabulate import tabulate

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import configure_logging
from src.models.transformer.transformer_model import TransformerModel

logger = configure_logging()

def load_model(model_path, config):
    """Charge le modèle avec les paramètres appropriés"""
    logger.info(f"Chargement du modèle depuis {model_path}")
    
    # Extraire les paramètres du modèle depuis le fichier config
    model = TransformerModel(
        vocab_size=config.get('vocab_size', 50257),
        d_model=config.get('embedding_dim', 256),
        nhead=config.get('num_heads', 4),
        num_encoder_layers=config.get('num_layers', 4),
        num_decoder_layers=config.get('num_layers', 4),
        dim_feedforward=config.get('dim_feedforward', 512),
        dropout=config.get('dropout_rate', 0.1)
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model

def load_config(config_path):
    """Charge la configuration du modèle"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_metrics(log_dir):
    """Charge et analyse les métriques d'entraînement à partir des logs"""
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    try:
        # Chercher le fichier de métriques d'entraînement
        metrics_file = os.path.join(log_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            # Analyser le fichier de log si pas de métriques directes
            log_file = os.path.join(log_dir, 'training.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Extraire les métriques des lignes de log
                for line in lines:
                    if "Epoch" in line and "Train loss" in line:
                        parts = line.split()
                        try:
                            epoch = int(parts[parts.index("Epoch") + 1].split("/")[0])
                            train_loss = float(parts[parts.index("loss:") + 1])
                            metrics['epochs'].append(epoch)
                            metrics['train_loss'].append(train_loss)
                        except (ValueError, IndexError):
                            continue
                            
                    if "Validation loss" in line:
                        try:
                            val_loss = float(line.split("loss:")[1].strip().split()[0])
                            metrics['val_loss'].append(val_loss)
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        logger.error(f"Erreur lors du chargement des métriques: {e}")
    
    return metrics

def generate_example_texts(model, prompts, configs):
    """Génère des exemples de texte avec différentes configurations"""
    results = []
    
    for prompt in prompts:
        prompt_results = []
        
        for config_name, params in configs.items():
            try:
                generated = model.generate(
                    prompt,
                    max_length=150,
                    temperature=params['temperature'],
                    top_k=params['top_k'],
                    top_p=params['top_p'],
                    repetition_penalty=params['repetition_penalty']
                )
                prompt_results.append((config_name, params, generated))
            except Exception as e:
                logger.error(f"Erreur pendant la génération avec config {config_name}: {e}")
                prompt_results.append((config_name, params, f"[ERROR] {str(e)}"))
                
        results.append((prompt, prompt_results))
    
    return results

def plot_training_metrics(metrics, output_dir):
    """Crée des graphiques des métriques d'entraînement"""
    if not metrics['train_loss']:
        logger.warning("Pas de métriques d'entraînement disponibles pour les graphiques")
        return None
        
    plt.figure(figsize=(10, 6))
    
    # Tracer les pertes d'entraînement
    plt.plot(metrics['epochs'], metrics['train_loss'], 'b-', label='Perte d\'entraînement')
    
    # Tracer les pertes de validation si disponibles
    if metrics['val_loss'] and len(metrics['val_loss']) == len(metrics['epochs']):
        plt.plot(metrics['epochs'], metrics['val_loss'], 'r-', label='Perte de validation')
    
    plt.title('Évolution des pertes pendant l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Enregistrer le graphique
    chart_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(chart_path, dpi=100)
    plt.close()
    
    return chart_path

def create_html_report(model_dir, model, config, metrics, generation_examples, output_file):
    """Crée un rapport HTML complet"""
    # Générer le graphique des métriques
    chart_path = plot_training_metrics(metrics, os.path.dirname(output_file))
    chart_rel_path = os.path.basename(chart_path) if chart_path else None
    
    # Calculer le nombre de paramètres du modèle
    num_params = sum(p.numel() for p in model.parameters())
    
    # Créer le contenu HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rapport du Modèle Transformer Amélioré</title>
<style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }}
    h1, h2, h3 {{ color: #333; }}
    .chart-container {{ margin: 30px 0; text-align: center; }}
    .model-section {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    .param-name {{ font-weight: bold; }}
    .generated-text {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007bff; margin: 10px 0; font-family: 'Courier New', monospace; white-space: pre-wrap; }}
    .config-item {{ background-color: #f0f0f0; padding: 5px; margin: 2px; border-radius: 3px; display: inline-block; }}
</style>
</head>
<body>
    <h1>Rapport du Modèle Transformer Amélioré</h1>
    <p>Date de génération: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Informations sur le Modèle</h2>
    <div class="model-section">
        <p><span class="param-name">Répertoire du modèle:</span> {model_dir}</p>
        <p><span class="param-name">Nombre de paramètres:</span> {num_params:,}</p>
        
        <h3>Configuration</h3>
        <table>
            <tr><th>Paramètre</th><th>Valeur</th></tr>
"""

    # Ajouter les paramètres de configuration
    for key, value in config.items():
        html_content += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
    
    html_content += """
        </table>
    </div>
    
"""

    # Ajouter la section des métriques d'entraînement si disponible
    if chart_rel_path:
        html_content += f"""
    <h2>Métriques d'Entraînement</h2>
    <div class="chart-container">
        <img src="{chart_rel_path}" alt="Évolution de la perte pendant l'entraînement" style="max-width:100%;">
    </div>
    
"""

    # Ajouter les exemples de génération de texte
    html_content += """
    <h2>Exemples de Génération de Texte</h2>
"""

    for prompt, results in generation_examples:
        html_content += f"""
    <div class="model-section">
        <h3>Prompt: "{prompt}"</h3>
"""
        
        for config_name, params, generated_text in results:
            param_text = ", ".join([f'<span class="config-item">{k}={v}</span>' for k, v in params.items()])
            html_content += f"""
        <h4>Configuration: {config_name}</h4>
        <p>{param_text}</p>
        <div class="generated-text">{generated_text}</div>
"""
            
        html_content += """
    </div>
"""
        
    html_content += """
    <h2>Conclusion</h2>
    <p>Ce rapport présente les résultats de l'entraînement et des exemples de génération de texte
    du modèle Transformer amélioré. Les différentes configurations de génération montrent l'impact
    des paramètres comme la température et la pénalité de répétition sur la qualité du texte généré.</p>
</body>
</html>
"""
    
    # Écrire le fichier HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Génère un rapport d'évaluation pour un modèle transformer")
    parser.add_argument("--model_dir", type=str, required=True, help="Répertoire contenant le modèle")
    args = parser.parse_args()
    
    # Gérer les wildcards avec le module glob
    import glob
    if '*' in args.model_dir:
        model_dirs = glob.glob(args.model_dir)
        if not model_dirs:
            logger.error(f"Aucun répertoire correspondant au pattern: {args.model_dir}")
            return 1
        # Utiliser le répertoire le plus récent
        model_dir = max(model_dirs, key=os.path.getctime)
        logger.info(f"Utilisation du répertoire le plus récent: {model_dir}")
    else:
        model_dir = args.model_dir
    
    # Vérifier que le répertoire existe
    if not os.path.exists(model_dir):
        logger.error(f"Le répertoire du modèle n'existe pas: {model_dir}")
        return 1
        
    # Déterminer le chemin du fichier de sortie
    output_file = args.output_file
    if not output_file:
        output_file = os.path.join(model_dir, 'model_report.html')
        
    try:
        # Charger la configuration
        config_path = os.path.join(model_dir, 'config.yaml')
        if not os.path.exists(config_path):
            logger.error(f"Fichier de configuration non trouvé: {config_path}")
            return 1
            
        config = load_config(config_path)
        
        # Charger le modèle
        model_path = os.path.join(model_dir, 'models', 'transformer_final.pt')
        if not os.path.exists(model_path):
            logger.error(f"Fichier de modèle non trouvé: {model_path}")
            return 1
            
        model = load_model(model_path, config)
        
        # Charger les métriques d'entraînement
        log_dir = os.path.join(model_dir, 'logs')
        metrics = load_training_metrics(log_dir)
        
        # Configurer les prompts et les configurations de génération
        prompts = [
            "Once upon a time",
            "The future of artificial intelligence",
            "In a world where technology",
            "Scientists recently discovered"
        ]
        
        generation_configs = {
            "conservative": {
                "temperature": 0.3,
                "top_k": 20,
                "top_p": 0.85,
                "repetition_penalty": 1.5
            },
            "balanced": {
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.92,
                "repetition_penalty": 1.2
            },
            "creative": {
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.1
            }
        }
        
        # Générer des exemples de texte
        logger.info("Génération d'exemples de texte...")
        generation_examples = generate_example_texts(model, prompts, generation_configs)
        
        # Créer le rapport HTML
        logger.info("Création du rapport HTML...")
        report_path = create_html_report(model_dir, model, config, metrics, generation_examples, output_file)
        
        logger.info(f"Rapport généré avec succès: {report_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())