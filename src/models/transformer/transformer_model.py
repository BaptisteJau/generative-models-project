from torch import nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.src_mask = None
        self.trg_mask = None
        self.d_model = d_model  # Store d_model for scaling
        self.vocab_size = vocab_size

    def generate_square_subsequent_mask(self, sz):
        return (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Handle dictionary input
        if isinstance(input_ids, dict):
            labels = input_ids.get('labels')
            attention_mask = input_ids.get('attention_mask')
            input_ids = input_ids.get('input_ids')
        
        # If input_ids is batched, handle appropriately
        if input_ids.dim() == 2:
            # Switch from batch_first to seq_first for transformer
            input_ids = input_ids.transpose(0, 1)
            
        # Use input_ids as both src and trg by default
        src = input_ids
        trg = input_ids
        
        # Convert to embeddings
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        trg = self.embedding(trg) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Create masks
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self.generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            device = trg.device
            mask = self.generate_square_subsequent_mask(len(trg)).to(device)
            self.trg_mask = mask

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask)
        return self.fc_out(output)

    def generate(self, input_ids=None, attention_mask=None, max_length=100, temperature=1.0, do_sample=False, 
                 top_p=0.9, top_k=50, **kwargs):
        """
        Generate text from input tokens
        
        Args:
            input_ids: Starting token ids (can be a tensor or inside a dict)
            attention_mask: Attention mask (optional)
            max_length: Maximum sequence length to generate
            temperature: Temperature for sampling
            do_sample: Whether to sample or use greedy decoding
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token ids
        """
        # Extract input_ids from dict if passed as a dict
        if isinstance(input_ids, dict):
            input_ids = input_ids.get('input_ids')
        
        # If we just have a single input sequence
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        
        # Switch to evaluation mode
        self.eval()
        
        # Generate up to max_length tokens
        with torch.no_grad():
            batch_size = input_ids.size(0)
            current_seq = input_ids
            
            for _ in range(max_length):
                # Get model output for current sequence
                if current_seq.size(1) > 1024:
                    # Truncate to avoid exceeding max length
                    current_seq = current_seq[:, -1024:]
                
                current_seq_t = current_seq.transpose(0, 1)  # Switch to seq_first for transformer
                outputs = self(current_seq_t, current_seq_t)
                
                # Get logits for the next token
                next_token_logits = outputs[-1].transpose(0, 1)  # Get last position and put batch first
                
                if do_sample:
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding - simply take the most likely token
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to current sequence
                current_seq = torch.cat((current_seq, next_token), dim=1)
                
                # Check for EOS token - stop if all sequences have ended
                # Assuming EOS token is typically the last token in the vocabulary
                if (next_token == self.vocab_size - 1).all():
                    break
            
            return current_seq