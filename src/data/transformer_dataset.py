import torch
from torch.utils.data import Dataset
import os
import logging

from src.utils.text_utils import SimpleTokenizer, load_text_file, clean_text

logger = logging.getLogger(__name__)

class TransformerTextDataset(Dataset):
    """
    Dataset pour l'entraînement d'un modèle Transformer sur du texte
    """
    
    def __init__(self, text_data=None, tokenizer=None, block_size=128, stride=64):
        """
        Args:
            text_data: Chemin vers un fichier texte ou texte brut
            tokenizer: Tokenizer à utiliser
            block_size: Taille des séquences
            stride: Pas entre les séquences consécutives
        """
        self.block_size = block_size
        self.stride = stride
        
        # Charger le texte si c'est un chemin de fichier
        if isinstance(text_data, str) and os.path.exists(text_data):
            logger.info(f"Chargement du texte depuis {text_data}")
            self.text = load_text_file(text_data)
        else:
            self.text = text_data
            
        # Nettoyage du texte
        if self.text:
            self.text = clean_text(self.text)
            
        # Créer ou utiliser le tokenizer fourni
        if tokenizer is None:
            logger.info("Création d'un nouveau tokenizer")
            self.tokenizer = SimpleTokenizer(self.text, vocab_size=5000, tokenize_type='char')
        else:
            self.tokenizer = tokenizer
            
        # Tokeniser tout le texte
        self.tokens = self.tokenizer.encode(self.text)
        self.vocab_size = len(self.tokenizer.vocab)
        logger.info(f"Taille du vocabulaire: {self.vocab_size}")
        
        # Créer des exemples
        self.examples = []
        # Gérer le cas où le texte est plus court que block_size
        if len(self.tokens) <= block_size:
            if len(self.tokens) > 1:  # Au moins 2 tokens pour avoir input et target
                input_tokens = self.tokens[:-1]
                target_tokens = self.tokens[1:]  # Décalage de 1
                # Padding si nécessaire
                if len(input_tokens) < block_size:
                    pad_length = block_size - len(input_tokens)
                    pad_id = self.tokenizer.vocab.get('<pad>', 0)
                    input_tokens = [pad_id] * pad_length + input_tokens
                    target_tokens = [pad_id] * pad_length + target_tokens
                self.examples.append((input_tokens, target_tokens))
        else:
            for i in range(0, len(self.tokens) - block_size, stride):
                input_tokens = self.tokens[i:i + block_size]
                target_tokens = self.tokens[i + 1:i + block_size + 1]  # Décalage de 1 pour la prédiction
                self.examples.append((input_tokens, target_tokens))
        
        logger.info(f"Dataset créé avec {len(self.examples)} exemples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_tokens, target_tokens = self.examples[idx]
        return torch.tensor(input_tokens), torch.tensor(target_tokens)