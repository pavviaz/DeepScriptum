import torch
from torch import nn
from torch import Tensor
from model.cnn_encoder import Encoder
from model.positional_embedding import PositionalEncoding
from model.token_embedding import TokenEmbedding


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 device: str = 'cpu'):
        super(Seq2SeqTransformer, self).__init__()
        self.cnn_encoder = Encoder(emb_size, device).to(device)
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True,
                                       norm_first=True).to(device)
        self.generator = nn.Linear(emb_size, tgt_vocab_size).to(device)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size).to(device)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout).to(device)
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor):
        image_features = self.cnn_encoder(src)
        
        trg_inp = trg[:, :-1]
        
        _, tgt_mask, _, tgt_padding_mask = self.create_mask(image_features, trg_inp, 0)
        
        src_emb = self.positional_encoding(image_features)
        
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg_inp))
        transformers_out = self.transformer(src_emb, tgt_emb, None, tgt_mask, None,
                                None, tgt_padding_mask, None)
        logits = self.generator(transformers_out)

        return logits
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, pad_idx):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == pad_idx)
        tgt_padding_mask = (tgt == pad_idx)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
