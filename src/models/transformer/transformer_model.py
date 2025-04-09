import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding et positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important pour la cohérence des dimensions
        )
        
        # Couche de sortie (projection vers le vocabulaire)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Pour initialiser les paramètres
        self.init_weights()
        
        # Stockage des dimensions pour référence
        self.d_model = d_model
        self.vocab_size = vocab_size
        
    def init_weights(self):
        # Initialisation standard des poids pour les transformers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, trg=None, src_mask=None, trg_mask=None, 
                src_padding_mask=None, trg_padding_mask=None, labels=None):
        """
        Forward pass du modèle Transformer
        
        Args:
            src: Tensor d'entrée source [batch_size, seq_len]
            trg: Tensor d'entrée cible [batch_size, seq_len] (peut être None pour l'inférence)
            src_mask: Masque pour la séquence source
            trg_mask: Masque pour la séquence cible
            src_padding_mask: Masque de padding pour la séquence source
            trg_padding_mask: Masque de padding pour la séquence cible
            labels: Étiquettes pour le calcul de la perte (peut être identique à trg)
            
        Returns:
            Si labels est fourni: tuple (logits de sortie, perte)
            Sinon: logits de sortie
        """
        # Si trg n'est pas fourni mais labels l'est, utiliser labels comme cible
        if trg is None and labels is not None:
            trg = labels
            
        # Appliquer l'embedding et l'encodage positionnel à la source
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        # Pour la cible également si elle est fournie
        if trg is not None:
            trg = self.embedding(trg) * math.sqrt(self.d_model)
            trg = self.positional_encoding(trg)
            
            # Créer un masque d'attention pour empêcher les positions de regarder les positions futures
            if trg_mask is None:
                trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        
        # Passer par le transformer
        if trg is not None:
            output = self.transformer(src, trg, src_mask, trg_mask,
                                     src_padding_mask, trg_padding_mask)
        else:
            # En mode inférence, nous n'avons pas de cible
            # Utiliser src comme entrée du décodeur 
            output = self.transformer(src, src, src_mask, None,
                                     src_padding_mask, None)
        
        # Passer par la couche de sortie
        output = self.fc_out(output)
        
        # Calculer la perte si les labels sont fournis
        loss = None
        if labels is not None:
            # Reshape pour correspondre à l'attente de CrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output.view(-1, self.vocab_size), labels.view(-1))
            return output, loss
            
        return output
        
    def generate_square_subsequent_mask(self, sz):
        """Génère un masque pour empêcher l'attention aux positions futures."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def generate_text(self, start_tokens, max_length=100, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2):
        """
        Génère du texte amélioré à partir de tokens de départ avec diverses techniques anti-répétition
        
        Args:
            start_tokens: Tensor contenant les tokens de départ [batch_size, seq_len]
            max_length: Longueur maximale de la séquence à générer
            temperature: Contrôle la randomisation (0.7 = équilibre entre créativité et cohérence)
            top_k: Limite les choix aux k tokens les plus probables (50 = valeur recommandée)
            top_p: Échantillonnage nucleus (0.95 = 95% de la masse de probabilité)
            repetition_penalty: Pénalise les tokens déjà utilisés (>1.0 réduit les répétitions)
            
        Returns:
            Tensor contenant les tokens générés [batch_size, max_length]
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            batch_size = start_tokens.shape[0]
            cur_tokens = start_tokens.clone()
            
            # Garder trace des tokens récemment générés pour pénaliser les répétitions
            recent_tokens_window = min(100, start_tokens.shape[1] + max_length)  # Évite les valeurs trop grandes
            
            for i in range(max_length - start_tokens.shape[1]):
                # Limiter le contexte si trop long pour économiser la mémoire
                if cur_tokens.shape[1] > 1024:
                    cur_tokens = cur_tokens[:, -1024:]
                    
                # Obtenir les prédictions pour la séquence actuelle
                logits = self(cur_tokens)
                
                # Extraire les logits pour le prochain token (dernier de la séquence)
                next_token_logits = logits[:, -1, :].clone()
                
                # Appliquer la pénalité de répétition
                if cur_tokens.shape[1] > 1 and repetition_penalty != 1.0:
                    # Identifier les tokens déjà utilisés récemment
                    last_tokens = cur_tokens[:, -min(recent_tokens_window, cur_tokens.shape[1]):]
                    
                    # Pour chaque batch
                    for batch_idx in range(batch_size):
                        # Créer un tensor unique de tokens utilisés (pour éviter les doublons)
                        used_tokens = torch.unique(last_tokens[batch_idx])
                        
                        # Filtrer les tokens invalides (comme le padding)
                        valid_tokens = used_tokens[used_tokens > 0]
                        
                        # Pénaliser les tokens déjà utilisés
                        if len(valid_tokens) > 0:  # Vérifier qu'il y a des tokens valides
                            # Créer un masque pour la pénalisation
                            if repetition_penalty > 1.0:
                                # Pénaliser les tokens répétés (diminuer leur probabilité)
                                next_token_logits[batch_idx, valid_tokens] /= repetition_penalty
                            else:
                                # Favoriser les tokens répétés (augmenter leur probabilité)
                                next_token_logits[batch_idx, valid_tokens] *= repetition_penalty
                
                # Appliquer la température pour contrôler la randomisation
                next_token_logits = next_token_logits / temperature
                
                # Masquer les tokens de faible probabilité (sampling top-k)
                if top_k > 0:
                    # Conserver uniquement les top_k tokens les plus probables
                    top_k = min(top_k, next_token_logits.size(-1))  # Cas où top_k > taille vocabulaire
                    
                    # Identifier les valeurs et indices des top-k tokens pour chaque exemple dans le batch
                    values, indices = torch.topk(next_token_logits, top_k, dim=-1)
                    
                    # Créer un masque pour filtrer uniquement les top-k tokens
                    mask = torch.zeros_like(next_token_logits).scatter_(1, indices, 1)
                    next_token_logits = torch.where(mask.bool(), next_token_logits, 
                                                  torch.full_like(next_token_logits, float('-inf')))
                
                # Échantillonnage nucleus (top-p)
                if 0.0 < top_p < 1.0:
                    # Calculer les probabilités (softmax)
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Trier les probabilités par ordre décroissant
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    
                    # Calculer la probabilité cumulative
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Créer un masque pour les tokens dont la probabilité cumulative < top_p
                    nucleus_mask = cumulative_probs < top_p
                    
                    # S'assurer qu'au moins un token est inclus
                    nucleus_mask[:, 0] = True
                    
                    # Pour chaque élément du batch, inclure tous les tokens jusqu'au dernier qui respecte top_p
                    limit_indices = torch.sum(nucleus_mask, dim=1, keepdim=True)
                    for batch_idx in range(batch_size):
                        nucleus_mask[batch_idx, :limit_indices[batch_idx]] = True
                    
                    # Obtenir les indices des tokens à conserver dans l'ordre original
                    tokens_to_keep = torch.gather(sorted_indices, 1, 
                                                 torch.where(nucleus_mask, 
                                                           torch.arange(sorted_indices.size(1), device=device).expand_as(nucleus_mask),
                                                           torch.zeros_like(nucleus_mask, dtype=torch.long)))
                    
                    # Reconstruire les logits filtrés
                    next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
                    
                    # Pour chaque batch, mettre à jour les logits des tokens conservés
                    for batch_idx in range(batch_size):
                        batch_tokens = tokens_to_keep[batch_idx, nucleus_mask[batch_idx]]
                        batch_probs = sorted_probs[batch_idx, nucleus_mask[batch_idx]]
                        next_token_logits_filtered[batch_idx, batch_tokens] = torch.log(batch_probs)
                    
                    next_token_logits = next_token_logits_filtered
                
                # Échantillonner le prochain token
                # On utilise les probabilités directement au lieu des logits pour éviter des problèmes numériques
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ajouter le nouveau token à la séquence
                cur_tokens = torch.cat([cur_tokens, next_token], dim=1)
            
            return cur_tokens

    def generate(self, prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2):
        """
        Génère du texte à partir d'un prompt en texte brut
        
        Args:
            prompt: Texte de départ (string)
            max_length: Longueur maximale de la séquence générée
            temperature: Contrôle la randomisation (0.7 = équilibre)
            top_k: Limite les choix aux k tokens les plus probables
            top_p: Échantillonnage nucleus (0.95 = 95% de la masse de probabilité) 
            repetition_penalty: Pénalise les tokens déjà utilisés

        Returns:
            Texte généré (string)
        """
        try:
            # Tokeniser le prompt
            from src.utils.text_utils import tokenize_text, detokenize_text
            tokens = tokenize_text(prompt)
            if not tokens:
                return "Erreur: impossible de tokeniser le prompt"
                
            # Convertir en tensor pour la génération
            start_tokens = torch.tensor([tokens], device=next(self.parameters()).device)
            
            # Générer la séquence
            generated = self.generate_text(
                start_tokens,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            # Convertir les tokens en texte
            return detokenize_text(generated[0].cpu().numpy())
        except Exception as e:
            return f"[ERROR] Génération échouée: {str(e)}"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)