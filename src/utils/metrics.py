import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Essayer de télécharger les ressources NLTK nécessaires (silencieusement)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class InceptionScore:
    """Calcul du score d'inception pour évaluer la qualité et la diversité des images générées"""
    
    def __init__(self, device=None, batch_size=32):
        """
        Args:
            device: Appareil sur lequel exécuter le modèle (None pour auto-détection)
            batch_size: Taille des batchs pour l'inférence
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Charger le modèle Inception pré-entraîné
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()  # Enlever la couche de classification
        self.model.eval()
        self.model.to(self.device)
        
    def __call__(self, images):
        """Calcule l'Inception Score
        
        Args:
            images: Lot d'images tensorielles [N, 3, 299, 299] avec valeurs dans [0, 1]
            
        Returns:
            tuple(score, std): Score moyen et écart-type
        """
        # S'assurer que les images sont au format adapté
        if images.shape[2] != 299 or images.shape[3] != 299:
            raise ValueError(f"Les images doivent être redimensionnées à 299x299, actuellement {images.shape[2]}x{images.shape[3]}")
        
        # Diviser en lots
        n_batches = (images.shape[0] + self.batch_size - 1) // self.batch_size
        
        probs_list = []
        
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, images.shape[0])
                batch = images[start:end].to(self.device)
                
                # Forward pass
                features = self.model(batch)
                probs = nn.functional.softmax(features, dim=1)
                probs_list.append(probs.cpu().numpy())
        
        # Concaténer les probabilités
        probs = np.concatenate(probs_list, axis=0)
        
        # Calculer le score
        py = np.mean(probs, axis=0)
        scores = []
        
        for i in range(probs.shape[0]):
            pyx = probs[i, :]
            scores.append(np.sum(pyx * (np.log(pyx) - np.log(py))))
        
        # Retourner moyenne et écart-type
        return np.exp(np.mean(scores)), np.exp(np.std(scores))


class FrechetInceptionDistance:
    """Calcul de la distance de Fréchet (FID) entre deux ensembles d'images"""
    
    def __init__(self, device=None, batch_size=32):
        """
        Args:
            device: Appareil sur lequel exécuter le modèle (None pour auto-détection)
            batch_size: Taille des batchs pour l'inférence
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Charger le modèle Inception pré-entraîné
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()  # Utiliser les caractéristiques avant la dernière couche
        self.model.eval()
        self.model.to(self.device)
        
    def extract_features(self, images):
        """Extrait les features de l'avant-dernière couche d'Inception
        
        Args:
            images: Lot d'images tensorielles [N, 3, H, W] avec valeurs dans [0, 1]
            
        Returns:
            Caractéristiques extraites [N, 2048]
        """
        # Redimensionner si nécessaire
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        features_list = []
        n_batches = (images.shape[0] + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, images.shape[0])
                batch = images[start:end].to(self.device)
                
                # Forward pass
                features = self.model(batch)
                features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
        
    def __call__(self, real_images, generated_images):
        """Calcule le score FID entre images réelles et générées
        
        Args:
            real_images: Images réelles, tenseur [N, 3, H, W] avec valeurs dans [0, 1]
            generated_images: Images générées, tenseur [N, 3, H, W] avec valeurs dans [0, 1]
            
        Returns:
            Score FID (plus bas = meilleur)
        """
        # Extraire les caractéristiques
        real_features = self.extract_features(real_images)
        gen_features = self.extract_features(generated_images)
        
        # Calculer statistiques
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        # Calculer FID
        diff = mu1 - mu2
        
        # Ajouter une petite constante aux diagonales pour la stabilité
        eps = 1e-6
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
        
        # Calcul de la racine carrée matricielle: sqrt(sigma1 * sigma2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Gérer la partie imaginaire numérique
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Formule FID
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)


def calculate_accuracy(y_true, y_pred):
    """Calcule la précision de la classification

    Args:
        y_true: Étiquettes réelles (tenseur ou array)
        y_pred: Prédictions (tenseur ou array)

    Returns:
        Précision (float)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        # Multi-classe: prendre l'indice max comme prédiction
        y_pred = np.argmax(y_pred, axis=1)
    
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    
    return correct_predictions / total_predictions


def calculate_perplexity(log_probs):
    """Calcule la perplexité à partir des log-probabilités

    Args:
        log_probs: Log-probabilités (tenseur ou array)

    Returns:
        Perplexité (float)
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.cpu().detach().numpy()
    
    # Perplexité = exp(-moyenne(log-probabilités))
    return float(np.exp(-np.mean(log_probs)))


def calculate_bleu_score(references, hypotheses, weights=None):
    """Calcule le score BLEU pour évaluer la génération de texte

    Args:
        references: Liste de listes de phrases de référence
        hypotheses: Liste de phrases générées
        weights: Pondération des n-grammes (défaut: équipondéré jusqu'à 4-grammes)

    Returns:
        Score BLEU (float)
    """
    if weights is None:
        weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-4 grams
    
    # Tokeniser si nécessaire
    tokenized_references = []
    for refs in references:
        if isinstance(refs, str):
            refs = [refs]  # Convertir une seule référence en liste
        tokenized_refs = []
        for ref in refs:
            if isinstance(ref, str):
                tokenized_refs.append(nltk.word_tokenize(ref.lower()))
            else:
                tokenized_refs.append(ref)  # Supposé déjà tokenisé
        tokenized_references.append(tokenized_refs)
    
    tokenized_hypotheses = []
    for hyp in hypotheses:
        if isinstance(hyp, str):
            tokenized_hypotheses.append(nltk.word_tokenize(hyp.lower()))
        else:
            tokenized_hypotheses.append(hyp)  # Supposé déjà tokenisé
    
    return corpus_bleu(tokenized_references, tokenized_hypotheses, weights=weights)


def calculate_rouge_score(references, hypotheses, rouge_type='rouge-l'):
    """Calcule le score ROUGE pour évaluer la génération de texte

    Args:
        references: Liste de phrases de référence
        hypotheses: Liste de phrases générées
        rouge_type: Type de métrique ROUGE ('rouge-1', 'rouge-2', 'rouge-l')

    Returns:
        Scores ROUGE (F1, Précision, Rappel)
    """
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    
    rouge_score = scores[rouge_type]
    return {
        'f1': rouge_score['f'],
        'precision': rouge_score['p'],
        'recall': rouge_score['r']
    }


def calculate_meteor_score(references, hypothesis):
    """Calcule le score METEOR pour évaluer la génération de texte

    Args:
        references: Liste de phrases de référence
        hypothesis: Phrase générée à évaluer

    Returns:
        Score METEOR (float)
    """
    if isinstance(references, str):
        references = [references]
    
    return meteor_score(references, hypothesis)


def evaluate_generated_text(references, hypotheses):
    """Évalue la qualité du texte généré avec plusieurs métriques

    Args:
        references: Liste de textes de référence
        hypotheses: Liste de textes générés

    Returns:
        Dictionnaire avec les différentes métriques
    """
    results = {}
    
    # BLEU scores
    results['bleu-1'] = calculate_bleu_score(
        [[r] for r in references], hypotheses, weights=(1.0, 0.0, 0.0, 0.0))
    results['bleu-2'] = calculate_bleu_score(
        [[r] for r in references], hypotheses, weights=(0.5, 0.5, 0.0, 0.0))
    results['bleu-4'] = calculate_bleu_score(
        [[r] for r in references], hypotheses)
    
    # ROUGE scores
    rouge_scores = calculate_rouge_score(references, hypotheses)
    results['rouge-l-f1'] = rouge_scores['f1']
    
    # METEOR score (moyenne sur tous les exemples)
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        meteor_scores.append(calculate_meteor_score([ref], hyp))
    results['meteor'] = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    
    return results


def evaluate_generated_images(real_images, generated_images):
    """Évalue la qualité des images générées avec plusieurs métriques

    Args:
        real_images: Images réelles, tenseur [N, 3, H, W] avec valeurs dans [0, 1]
        generated_images: Images générées, tenseur [N, 3, H, W] avec valeurs dans [0, 1]

    Returns:
        Dictionnaire avec les différentes métriques
    """
    results = {}
    
    # FID score
    fid_calculator = FrechetInceptionDistance()
    results['fid'] = fid_calculator(real_images, generated_images)
    
    # Inception score
    is_calculator = InceptionScore()
    is_mean, is_std = is_calculator(generated_images)
    results['inception_score'] = is_mean
    results['inception_score_std'] = is_std
    
    return results


def calculate_loss(loss_function, outputs, targets):
    return loss_function(outputs, targets)


def calculate_fid(real_images, generated_images):
    fid_calculator = FrechetInceptionDistance()
    return fid_calculator(real_images, generated_images)


def calculate_is(generated_images):
    is_calculator = InceptionScore()
    is_mean, is_std = is_calculator(generated_images)
    return is_mean, is_std


def log_metrics(epoch, metrics):
    print(f"Epoch: {epoch}, Metrics: {metrics}")