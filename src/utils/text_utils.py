import torch
import numpy as np
from typing import List, Union
import logging
import os

logger = logging.getLogger(__name__)

def tokenize_text(text: str) -> List[int]:
    """
    Tokenise un texte en liste d'identifiants de tokens.
    
    Args:
        text: Texte à tokeniser
        
    Returns:
        Liste d'identifiants de tokens
    """
    try:
        # Si transformers est disponible, utiliser GPT2Tokenizer
        from transformers import AutoTokenizer
        
        # Utiliser un tokenizer préentraîné
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Encoder le texte
        tokens = tokenizer.encode(text, return_tensors="pt")[0].tolist()
        return tokens
    except ImportError:
        logger.warning("La bibliothèque 'transformers' n'est pas installée. Utilisation d'un tokenizer basique.")
        
        # Tokenizer basique (caractère par caractère)
        # Pour une vraie application, utilisez un vrai tokenizer
        return [ord(c) % 10000 for c in text]

def detokenize_text(tokens: Union[List[int], np.ndarray]) -> str:
    """
    Détokenise une liste d'identifiants de tokens en texte.
    
    Args:
        tokens: Liste d'identifiants de tokens
        
    Returns:
        Texte détokenisé
    """
    try:
        # Si transformers est disponible, utiliser GPT2Tokenizer
        from transformers import AutoTokenizer
        
        # Utiliser un tokenizer préentraîné
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Convertir en liste si c'est un numpy array
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
            
        # Décoder les tokens
        text = tokenizer.decode(tokens)
        return text
    except ImportError:
        logger.warning("La bibliothèque 'transformers' n'est pas installée. Utilisation d'un détokenizer basique.")
        
        # Détokenizer basique (caractère par caractère)
        return ''.join([chr(t % 128 + 32) for t in tokens])