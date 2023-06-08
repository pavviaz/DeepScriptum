import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import repeat
from models.torch.positional_embedding import PositionalEncoding

from models.torch.token_embedding import TokenEmbedding


class PatchEmbedding(nn.Module):
        def __init__(self, img_H: int, img_W: int, patch_size: int = 16, emb_size: int = 512):
            self.patch_size = patch_size
            super().__init__()
            self.projection = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.LazyConv2d(emb_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
            self.linear = nn.LazyLinear(emb_size)
            self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
            self.positions = nn.Parameter(torch.randn((img_H // patch_size) * (img_W // patch_size) + 1, emb_size))

        def forward(self, x):
            b, c, h, w = x.shape
            
            # patches = []
            # for h_p in range(0, h, self.patch_size):
            #     for w_p in range(0, w, self.patch_size):
            #         patches.append(x[:, :, h_p: h_p + self.patch_size, w_p: w_p + self.patch_size])
            # patches = torch.stack(patches)
            # p, b, c, _, _ = patches.shape
            
            # x = patches.view(b, p, c * self.patch_size * self.patch_size)
            # x = self.linear(x)
            
            x = self.projection(x)
            cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
            # prepend the cls token to the input
            x = torch.cat([cls_tokens, x], dim=1)
            # add position embedding
            x += self.positions
            return x


class ViT(nn.Module):
        def __init__(self, img_H: int, img_W: int, nhead=8, patch_size: int = 16, emb_size: int = 512, depth=6):
            super().__init__()
            self.embedding = PatchEmbedding(img_H, img_W, patch_size, emb_size)
            self.encoder = torch.nn.TransformerEncoder(nn.TransformerEncoderLayer(emb_size,
                                                      nhead=nhead,
                                                      activation='gelu',
                                                      batch_first=True,
                                                      norm_first=True), depth)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            
            return x


class Seq2SeqTransformerVIT(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 img_H: int,
                 img_W:int,
                 tgt_vocab_size: int,
                 criterion,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 device: str = 'cpu'):
        super(Seq2SeqTransformerVIT, self).__init__()
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.vit_encoder = ViT(img_H=img_H, img_W=img_W, nhead=nhead, depth=num_encoder_layers, emb_size=emb_size)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=emb_size,
                                                      nhead=nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation='gelu',
                                                      batch_first=True), num_decoder_layers)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.criterion = criterion
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor):
        loss = 0
        trg_inp = trg[:, :-1]
        trg_true = trg[:, 1:]
        
        img_features = self.vit_encoder(src)
        
        _, tgt_mask, _, tgt_padding_mask = self.create_mask(img_features, trg_inp, 0)
        
        # src_emb = self.positional_encoding(self.src_tok_emb(src))
        
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg_inp))
        transformers_out = self.decoder(tgt_emb, img_features, tgt_mask, None, tgt_padding_mask, None)
        logits = self.generator(transformers_out)
        
        for pred, trgt in zip(logits, trg_true):
            loss += torch.sum(self.criterion(pred, trgt))
        
        return loss / torch.sum(~(trg_true == 0)), logits
        

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
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