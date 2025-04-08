import logging
import numpy as np

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports conditionnels pour éviter les erreurs si un framework n'est pas disponible
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch n'est pas disponible")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow n'est pas disponible")

class FrameworkBridge:
    """Classe utilitaire pour gérer les conversions entre PyTorch et TensorFlow"""
    
    @staticmethod
    def is_pytorch_tensor(tensor):
        """Détecte si un objet est un tensor PyTorch de manière robuste"""
        if not TORCH_AVAILABLE:
            return False
        # Vérification plus robuste du type pytorch
        return TORCH_AVAILABLE and (
            isinstance(tensor, torch.Tensor) or
            hasattr(tensor, 'detach') or  # Attributs spécifiques à PyTorch
            str(type(tensor)).find('torch') >= 0
        )
    
    @staticmethod
    def is_tensorflow_tensor(tensor):
        """Détecte si un objet est un tensor TensorFlow de manière robuste"""
        if not TF_AVAILABLE:
            return False
        # Vérification plus robuste du type tensorflow
        return TF_AVAILABLE and (
            isinstance(tensor, tf.Tensor) or
            hasattr(tensor, 'numpy') or  # Attributs spécifiques à TensorFlow
            str(type(tensor)).find('tensorflow') >= 0
        )
    
    @staticmethod
    def to_numpy(tensor):
        """Convertit n'importe quel tensor en numpy array de manière sécurisée"""
        try:
            if FrameworkBridge.is_pytorch_tensor(tensor):
                logger.debug("Conversion PyTorch -> NumPy")
                return tensor.detach().cpu().numpy()
            elif FrameworkBridge.is_tensorflow_tensor(tensor):
                logger.debug("Conversion TensorFlow -> NumPy")
                return tensor.numpy()
            elif isinstance(tensor, np.ndarray):
                return tensor
            else:
                return np.array(tensor)
        except Exception as e:
            logger.error(f"Erreur lors de la conversion vers NumPy: {e}")
            # Fallback: tentative de conversion de manière plus directe
            return np.array(tensor)
    
    @staticmethod
    def pytorch_to_tensorflow(tensor, normalize_range=None):
        """Convertit un tensor PyTorch vers un tensor TensorFlow"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow n'est pas disponible pour la conversion")
        
        try:
            # Conversion en NumPy d'abord
            array = FrameworkBridge.to_numpy(tensor)
            
            # Réarrangement NCHW -> NHWC si nécessaire
            if len(array.shape) == 4 and array.shape[1] in [1, 3]:
                original_shape = array.shape
                array = np.transpose(array, (0, 2, 3, 1))
                logger.info(f"Format converti de {original_shape} -> {array.shape}")
            
            # Normalisation
            if normalize_range:
                min_val, max_val = array.min(), array.max()
                target_min, target_max = normalize_range
                
                if min_val != target_min or max_val != target_max:
                    range_orig = max_val - min_val
                    range_target = target_max - target_min
                    
                    # Éviter la division par zéro
                    if range_orig > 0:
                        array = (array - min_val) * (range_target / range_orig) + target_min
                        logger.info(f"Valeurs normalisées de [{min_val:.4f}, {max_val:.4f}] à {normalize_range}")
            
            # Conversion finale vers TensorFlow
            return tf.convert_to_tensor(array, dtype=tf.float32)
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion PyTorch -> TensorFlow: {str(e)}")
            raise ValueError(f"Échec de conversion: {str(e)}")
    
    @staticmethod
    def tensorflow_to_pytorch(tensor, to_format='NCHW', device='cpu'):
        """Convertit un tensor TensorFlow vers un tensor PyTorch"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible pour la conversion")
            
        try:
            # Conversion en NumPy d'abord
            array = FrameworkBridge.to_numpy(tensor)
            
            # Réorganisation NHWC -> NCHW si nécessaire
            if to_format == 'NCHW' and len(array.shape) == 4:
                original_shape = array.shape
                array = np.transpose(array, (0, 3, 1, 2))
                logger.info(f"Format converti de {original_shape} -> {array.shape}")
            
            # Détermination du device
            if device == 'auto' and TORCH_AVAILABLE:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Conversion vers PyTorch
            result = torch.tensor(array, dtype=torch.float32, device=device)
            logger.info(f"Tensor converti vers PyTorch sur {device}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion TensorFlow -> PyTorch: {str(e)}")
            raise ValueError(f"Échec de conversion: {str(e)}")
    
    @staticmethod
    def normalize_batch_shape(tensor, expected_format='NHWC'):
        """Normalise la forme d'un batch de données pour le rendre compatible avec le format attendu
        
        Args:
            tensor: Tensor d'entrée (TensorFlow ou numpy)
            expected_format: Format attendu ('NHWC' ou 'NCHW')
            
        Returns:
            Tensor avec la forme normalisée
        """
        if not isinstance(tensor, (tf.Tensor, np.ndarray)):
            return tensor
            
        # Obtenir la forme actuelle
        shape = tensor.shape
        
        # Cas des tenseurs avec dimensions supplémentaires
        if len(shape) > 4:
            logger.warning(f"Tensor a trop de dimensions: {shape}, tentative de correction")
            
            # Si première dimension est 1, la supprimer (common broadcasting issue)
            if shape[0] == 1 and len(shape) == 5:
                if isinstance(tensor, tf.Tensor):
                    tensor = tf.squeeze(tensor, axis=0)
                else:
                    tensor = np.squeeze(tensor, axis=0)
                
                logger.info(f"Dimension supplémentaire supprimée: {tensor.shape}")
        
        # Cas des tenseurs avec dimensions manquantes
        if len(tensor.shape) == 3:  # Manque dimension batch
            if isinstance(tensor, tf.Tensor):
                tensor = tf.expand_dims(tensor, axis=0)
            else:
                tensor = np.expand_dims(tensor, axis=0)
            logger.info(f"Dimension batch ajoutée: {tensor.shape}")
        
        # Vérifier et corriger l'ordre des canaux
        shape = tensor.shape
        if len(shape) == 4:
            # Détection automatique du format actuel
            is_nchw = shape[1] <= shape[2] and shape[1] <= shape[3] and shape[1] in [1, 3, 4]
            
            if is_nchw and expected_format == 'NHWC':
                # Convertir NCHW -> NHWC
                if isinstance(tensor, tf.Tensor):
                    tensor = tf.transpose(tensor, [0, 2, 3, 1])
                else:
                    tensor = np.transpose(tensor, [0, 2, 3, 1])
                logger.info(f"Format converti de NCHW à NHWC: {tensor.shape}")
        
        return tensor

    # Améliorer les méthodes de détection de types de framework

    @staticmethod
    def is_pytorch_model(model):
        """Vérifie si un modèle utilise PyTorch
        
        Args:
            model: Instance d'un modèle à vérifier
            
        Returns:
            bool: True si c'est un modèle PyTorch
        """
        # Vérifier la présence d'attributs caractéristiques de PyTorch
        if hasattr(model, 'state_dict') and callable(getattr(model, 'state_dict')):
            return True
            
        # Vérifier si le modèle a des paramètres torch.nn
        if hasattr(model, 'parameters') and callable(getattr(model, 'parameters')):
            try:
                next(model.parameters())
                return True
            except (StopIteration, AttributeError, TypeError):
                pass
        
        return False

    @staticmethod
    def is_tensorflow_model(model):
        """Vérifie si un modèle utilise TensorFlow/Keras
        
        Args:
            model: Instance d'un modèle à vérifier
            
        Returns:
            bool: True si c'est un modèle TensorFlow/Keras
        """
        # Vérifier la présence d'attributs caractéristiques de TensorFlow/Keras
        if hasattr(model, 'save') and callable(getattr(model, 'save')):
            return True
            
        # Vérifier s'il s'agit d'un modèle composite avec composants TensorFlow
        if hasattr(model, 'generator') and hasattr(model.generator, 'save'):
            return True
            
        # Vérifier s'il contient des layers Keras
        if hasattr(model, 'layers') or ('keras' in str(type(model)).lower()) or ('tensorflow' in str(type(model)).lower()):
            return True
            
        return False

# Fonction utilitaire pour vérifier la disponibilité des frameworks
def get_available_frameworks():
    """Renvoie un dictionnaire indiquant quels frameworks sont disponibles"""
    return {
        'pytorch': TORCH_AVAILABLE,
        'tensorflow': TF_AVAILABLE
    }