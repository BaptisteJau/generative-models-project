import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 use_layer_norm=False, norm_first=False):
        super(TransformerModel, self).__init__()
        
        # Stocker explicitement la taille du vocabulaire
        self.vocab_size = vocab_size
        
        # Embedding et positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # LayerNorm supplémentaire avant et après les embeddings
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.emb_norm = nn.LayerNorm(d_model)
        
        # Transformer avec normalisation pré-attention si demandé
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first  # Pre-LN ou Post-LN architecture
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first
        )
        
        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_encoder_layers,
            norm=encoder_norm
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_decoder_layers,
            norm=decoder_norm
        )
        
        # Couche de sortie (projection vers le vocabulaire)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Pour initialiser les paramètres
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src):
        # src shape: [batch_size, seq_len]
        
        # Créer un masque pour empêcher l'attention aux positions futures
        src_mask = generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # Embedding + positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_emb = self.positional_encoding(src_emb)
        
        # LayerNorm supplémentaire sur les embeddings si activée
        if self.use_layer_norm:
            src_emb = self.emb_norm(src_emb)
        
        # Pas de masque de padding pour simplifier
        memory = self.transformer_encoder(src_emb)
        output = self.transformer_decoder(src_emb, memory, tgt_mask=src_mask)
        
        return self.fc_out(output)
        
    # Ajout d'une fonction auxiliaire pour générer le masque causal
    def generate(self, prompt, max_length=100, temperature=0.9, top_k=40, top_p=0.92, repetition_penalty=1.5):
        """
        Génération de texte améliorée avec techniques anti-répétition
        
        Args:
            prompt: Texte ou tokens de départ
            max_length: Longueur maximale à générer
            temperature: Contrôle la créativité (>1 plus créatif, <1 plus conservateur)
            top_k: Nombre de tokens les plus probables à considérer (0 pour désactiver)
            top_p: Fraction de la masse de probabilité à considérer (1.0 pour désactiver)
            repetition_penalty: Pénalité pour répéter des tokens (>1.0 pénalise les répétitions)
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Tokeniser le prompt
        if isinstance(prompt, str):
            if hasattr(self, 'tokenizer'):
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_ids = torch.tensor([tokens]).to(device)
            else:
                # Fallback simple: encoder par caractère
                input_ids = torch.tensor([[ord(c) % 256 for c in prompt]]).to(device)
        else:
            input_ids = prompt.to(device)
        
        # Historique pour la détection des répétitions
        generated_tokens = input_ids[0].tolist().copy()
        
        for _ in range(max_length):
            with torch.no_grad():
                # Forward pass pour obtenir les prédictions
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :].clone()
                
                # 1. Appliquer la température
                next_token_logits = next_token_logits / max(0.7, temperature)
                
                # 2. Pénaliser les répétitions
                if repetition_penalty > 1.0:
                    # Vérifier les 10 derniers tokens pour éviter les répétitions locales
                    recent_tokens = generated_tokens[-10:]
                    for token_id in set(recent_tokens):
                        # Compter les occurrences
                        count = recent_tokens.count(token_id)
                        if count > 1 and token_id < next_token_logits.size(1):
                            # Pénalité proportionnelle au nombre d'occurrences
                            penalty = repetition_penalty * (1 + 0.1 * (count - 1))
                            next_token_logits[0, token_id] /= penalty
                
                # 3. Top-K sampling
                if top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 4. Top-p (nucleus) sampling
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Supprimer les tokens avec une probabilité cumulative > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Toujours garder le premier token
                    sorted_indices_to_remove[..., 0] = 0
                    # Décalage pour appliquer sur les bons indices
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    
                    # Créer un masque pour les logits originaux
                    indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    indices_to_remove = indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove.unsqueeze(0)
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 5. Échantillonnage selon la distribution de probabilité
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 6. Vérifier si on génère un token de fin de séquence
                if hasattr(self, 'tokenizer') and next_token.item() == self.tokenizer.vocab.get('<eos>', -1):
                    break
                
                # 7. Ajouter le token à la séquence d'entrée
                input_ids = torch.cat((input_ids, next_token), dim=1)
                generated_tokens.append(next_token.item())
        
        # Convertir les tokens en texte
        if hasattr(self, 'tokenizer'):
            output_text = self.tokenizer.decode(generated_tokens)
        else:
            # Fallback
            output_text = "".join([chr(min(t, 127)) for t in generated_tokens])
        
        # Post-traitement pour nettoyer les répétitions
        output_text = clean_repetitions(output_text)
        
        return output_text

def generate_square_subsequent_mask(sz):
    """Génère un masque causal pour l'auto-attention"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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

def clean_repetitions(text):
    """Nettoie les répétitions pathologiques dans le texte généré"""
    import re
    
    # 1. Répétition de mots consécutifs
    pattern1 = r'\b(\w+)(\s+\1){2,}\b'
    while re.search(pattern1, text):
        text = re.sub(pattern1, r'\1', text)
    
    # 2. Répétition de groupes de mots (2-4 mots)
    for n in range(2, 5):
        # Capture un groupe de n mots qui se répètent au moins 2 fois
        pattern = r'((?:\w+\W+){' + str(n) + r'})(\1)+'
        while re.search(pattern, text):
            # Remplace par une seule occurrence
            text = re.sub(pattern, r'\1', text)
    
    # 3. Répétition excessive de caractères individuels (plus de 3 fois)
    pattern3 = r'([^\w\s])(\1{3,})'  # Caractères non alphanumériques
    text = re.sub(pattern3, r'\1\1', text)
    
    # 4. Répétition excessive de ponctuations
    pattern4 = r'([,.!?;:]){3,}'
    text = re.sub(pattern4, r'\1\1', text)
    
    return text