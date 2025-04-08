# Nouveau fichier pour centraliser la configuration des loggers

import logging
import os

def configure_logging(level=logging.INFO):
    """Configure le système de logging global
    
    Args:
        level: Niveau de log par défaut
    """
    # Configurer un format cohérent
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configurer le logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Supprimer les handlers existants pour éviter les duplications
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Ajouter un handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Créer un répertoire pour les logs
    logs_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Ajouter un handler pour les fichiers
    file_handler = logging.FileHandler(
        os.path.join(logs_dir, 'generative_models.log'),
        mode='a'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger