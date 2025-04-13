import re
import os
import torch
import numpy as np
from collections import Counter
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class SimpleTokenizer:
    """Tokenizer simple basé sur des caractères ou des mots"""
    
    def __init__(self, text=None, vocab_size=None, tokenize_type='char'):
        """
        Args:
            text: Corpus de texte pour construire le vocabulaire
            vocab_size: Taille maximale du vocabulaire
            tokenize_type: 'char' pour tokeniser par caractère, 'word' pour tokeniser par mot
        """
        self.tokenize_type = tokenize_type
        self.vocab_size = vocab_size
        
        # Tokens spéciaux
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        # Mappings
        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1, 
            self.bos_token: 2,
            self.eos_token: 3
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Construire le vocabulaire si du texte est fourni
        if text:
            self.build_vocab(text)
    
    def build_vocab(self, text):
        """Construit le vocabulaire à partir d'un corpus de texte"""
        # Tokenisation selon le type choisi
        if self.tokenize_type == 'char':
            tokens = list(text)
        else:
            tokens = text.split()
        
        # Compter les fréquences
        counter = Counter(tokens)
        
        # Ajouter les tokens les plus fréquents au vocabulaire
        vocab_size = self.vocab_size - 4 if self.vocab_size else len(counter)
        for token, _ in counter.most_common(vocab_size):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.reverse_vocab[len(self.vocab) - 1] = token
    
    def encode(self, text, add_special_tokens=False):
        """Convertit du texte en IDs de tokens"""
        if self.tokenize_type == 'char':
            tokens = list(text)
        else:
            tokens = text.split()
        
        ids = []
        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])
            
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.unk_token])
                
        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])
            
        return ids
    
    def decode(self, ids):
        """Convertit des IDs de tokens en texte"""
        tokens = [self.reverse_vocab.get(id, self.unk_token) for id in ids]
        
        # Filtrer les tokens spéciaux si nécessaire
        special_tokens = [self.pad_token, self.bos_token, self.eos_token]
        tokens = [token for token in tokens if token not in special_tokens]
        
        if self.tokenize_type == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def save(self, path):
        """Sauvegarde le tokenizer dans un fichier"""
        data = {
            'vocab': self.vocab,
            'reverse_vocab': self.reverse_vocab,
            'tokenize_type': self.tokenize_type,
            'vocab_size': self.vocab_size
        }
        torch.save(data, path)
    
    def load(self, path):
        """Charge le tokenizer depuis un fichier"""
        data = torch.load(path)
        self.vocab = data['vocab']
        self.reverse_vocab = data['reverse_vocab']
        self.tokenize_type = data['tokenize_type']
        self.vocab_size = data['vocab_size']
        return self

def tokenize_text(text, tokenizer=None):
    """
    Tokenise un texte avec le tokenizer spécifié ou par caractère par défaut
    
    Args:
        text: Texte à tokeniser
        tokenizer: Tokenizer à utiliser (SimpleTokenizer ou compatible)
        
    Returns:
        Liste d'IDs de tokens
    """
    if tokenizer is None:
        # Tokenisation par caractère simple si aucun tokenizer n'est fourni
        chars = list(text)
        # Créer un mapping simple caractère -> id
        char_to_id = {c: i+4 for i, c in enumerate(set(chars))}  # +4 pour les tokens spéciaux
        return [char_to_id.get(c, 1) for c in chars]  # 1 = <unk>
    
    return tokenizer.encode(text)

def detokenize_text(token_ids, tokenizer=None):
    """
    Convertit des IDs de tokens en texte
    
    Args:
        token_ids: Liste d'IDs de tokens
        tokenizer: Tokenizer à utiliser (SimpleTokenizer ou compatible)
        
    Returns:
        Texte reconstitué
    """
    if tokenizer is None:
        # Si pas de tokenizer, difficile de détokeniser correctement
        return "[Tokens: " + str(token_ids[:10]) + "...]"
    
    return tokenizer.decode(token_ids)

def load_text_file(file_path):
    """
    Charge un fichier texte
    
    Args:
        file_path: Chemin vers le fichier texte
        
    Returns:
        Contenu du fichier
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(text):
    """Nettoyage basique du texte"""
    # Suppression des multiples espaces et sauts de ligne
    text = re.sub(r'\s+', ' ', text)
    return text.strip()