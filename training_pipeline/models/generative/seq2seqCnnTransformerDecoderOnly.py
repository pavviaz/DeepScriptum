import torch
from torch import nn
from torch import Tensor
from .cnn_encoder import Encoder
from .positional_embedding import PositionalEncoding
from .token_embedding import TokenEmbedding


class Seq2SeqTransformerDecOnly(nn.Module):
    def __init__(self,
                 emb_size: int,
                 nhead: int,
                #  src_vocab_size: int,
                 tgt_vocab_size: int,
                 criterion,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_encoder_layers: int = None,
                 num_decoder_layers: int = None,
                 device: str = 'cpu'):
        super(Seq2SeqTransformerDecOnly, self).__init__()
        self.cnn_encoder = Encoder(emb_size, device).to(device)
        # self.transformer = nn.Transformer(d_model=emb_size,
        #                                nhead=nhead,
        #                                num_encoder_layers=num_encoder_layers,
        #                                num_decoder_layers=num_decoder_layers,
        #                                dim_feedforward=dim_feedforward,
        #                                dropout=dropout,
        #                                batch_first=True,
        #                                norm_first=True).to(device)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=emb_size,
                                       nhead=nhead,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True,
                                       norm_first=True), 
            num_decoder_layers).to(device)
        self.generator = nn.Linear(emb_size, tgt_vocab_size).to(device)
        # self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size).to(device)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout).to(device)
        self.criterion = criterion
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor):
        loss = 0
        image_features = self.cnn_encoder(src)
        
        trg_inp = trg[:, :-1]
        trg_true = trg[:, 1:]
        
        _, tgt_mask, _, tgt_padding_mask = self.create_mask(image_features, trg_inp, 0)
        
        src_emb = self.positional_encoding(image_features)
        
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg_inp))
        
        transformers_out = self.transformer_decoder(tgt_emb, src_emb, tgt_mask, None, tgt_padding_mask, None)
        logits = self.generator(transformers_out)
        
        # batch_size =  logits.shape[0]
        # for i in range(batch_size):
        #     loss += torch.mean(self.criterion(logits[:, i, :], trg_true[:, i]))
        
        # for pred, trgt in zip(logits, trg_true):
        #     for pred1, trgt1 in zip(pred, trgt):
        #         l = self.criterion(pred1, trgt1)
        #         loss += (l)
        
        # for pred, trgt in zip(logits, trg_true):
        #     loss += torch.mean(self.criterion(pred, trgt))
            
        for pred, trgt in zip(logits, trg_true):
            loss += torch.sum(self.criterion(pred, trgt))
        
        # loss = self.criterion(logits.reshape(-1, logits.shape[-1]), trg_true.reshape(-1))
        # return loss, loss / trg.shape[1]
        # w = torch.sum(~(trg_true == 0))
        return loss / torch.sum(~(trg_true == 0)), logits
    
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
