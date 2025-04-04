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

    def generate_square_subsequent_mask(self, sz):
        return (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

    def forward(self, src, trg):
        src = self.embedding(src) * torch.sqrt(torch.tensor(src.size(-1), dtype=torch.float32))
        trg = self.embedding(trg) * torch.sqrt(torch.tensor(trg.size(-1), dtype=torch.float32))
        
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

    def generate(self, input_seq, max_length):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                output = self(input_seq, input_seq)
                next_token = output[-1, :].argmax(dim=-1).unsqueeze(0)
                input_seq = torch.cat((input_seq, next_token), dim=0)
        return input_seq