import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Arrêt précoce pour éviter le surapprentissage.
    Arrête l'entraînement quand la métrique surveillée cesse de s'améliorer.
    """
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience: Nombre d'époques à attendre après la dernière amélioration
            min_delta: Amélioration minimale pour être considérée significative
            restore_best_weights: Si True, restaure les meilleurs poids quand l'arrêt est déclenché
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None
        
    def __call__(self, val_loss, model):
        """
        Vérifie si l'entraînement doit être arrêté
        
        Args:
            val_loss: Perte de validation courante
            model: Modèle dont on peut sauvegarder les poids
            
        Returns:
            True si l'entraînement doit être arrêté
        """
        score = -val_loss  # Score plus haut = meilleur
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping: Pas d\'amélioration depuis {self.counter} époques')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_state_dict is not None:
                    logger.info('EarlyStopping: Restauration des meilleurs poids')
                    model.load_state_dict(self.best_state_dict)
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
        return self.early_stop
            
    def save_checkpoint(self, model):
        """Sauvegarde l'état du modèle"""
        self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}