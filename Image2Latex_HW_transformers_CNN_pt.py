import collections
import inspect
import json
import logging
import math
import os
import random
import shutil
import time
from matplotlib import pyplot as plt
from munch import munchify
import keras
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from keras_preprocessing.text import tokenizer_from_json
from models.Transformer.resnet import ResBlock, ResNet34
from utils.images_preprocessing import make_fix_size, compile_handwritten_image
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat


def enable_gpu(enable:bool):
    if not enable:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def log_init(path, name, mode):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:    %(message)s")

    file_handler = logging.FileHandler(f"{path}/{name}.txt", mode=mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class Checkpointing():
    def __init__(self, path, max_to_keep=10, **kwargs) -> None:
        assert all([(nn.Module in type(el).mro() or torch.optim.Optimizer in type(el).mro()) for el in kwargs.values()])
        
        self.max_to_keep = max_to_keep
        self.modules = munchify({k: v for k, v in kwargs.items() if nn.Module in type(v).mro()})
        self.optims = munchify({k: v for k, v in kwargs.items() if torch.optim.Optimizer in type(v).mro()})
        
        if len(self.modules) == 0:
            print("Warning: no modules specified for saving/loading")
        if len(self.modules) == 0:
            print("Warning: no optimizers specified for saving/loading")
        
        self.path = path
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
    
    def __get_files_and_ckpt_idx(self, return_all_checkpoints=False):
        dir_files = sorted([s for s in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, s))], 
                           key=lambda s: os.path.getmtime(os.path.join(self.path, s)))
        try:
            checkpoing_idx = 0 if len(dir_files) == 0 else ((int(dir_files[-1].replace(".tar", "").split("_")[-1]) + 1) if not return_all_checkpoints else [int(idx.replace(".tar", "").split("_")[-1]) + 1 for idx in dir_files])
        except:
            raise IOError
        return dir_files, checkpoing_idx
        
    def save(self):
        dir_files, checkpoing_idx = self.__get_files_and_ckpt_idx()
        
        del_idxs = len(dir_files) - self.max_to_keep + 1
        
        if del_idxs > 0:
            [os.remove(os.path.join(self.path, dir_files[idx])) for idx in range(del_idxs)]
            
        try:
            torch.save({k: v.state_dict() for k, v in self.modules.items()} 
                       |
                       {k: v.state_dict() for k, v in self.optims.items()}
                       , os.path.join(self.path, f"checkpoint_{checkpoing_idx}.tar"))
        except Exception as e:
            raise e
    
    def load(self, idx=None, print_avail_ckpts=False, return_idx=False):
        _, checkpoing_idx = self.__get_files_and_ckpt_idx(return_all_checkpoints=True)
        
        if print_avail_ckpts:
            print("Following checkpoints are available:")
            [print(f"{idx+1}) {ckpt - 1}", sep=", ", end=" ") for idx, ckpt in enumerate(checkpoing_idx)]
        
        checkpoing_idx = checkpoing_idx[-1] if not idx else idx + 1
        
        try:
            checkpoint = torch.load(os.path.join(self.path, f"checkpoint_{checkpoing_idx - 1}.tar"))
            [v.load_state_dict(checkpoint[k]) for k, v in self.modules.items()]
            [v.load_state_dict(checkpoint[k]) for k, v in self.optims.items()]
        except Exception as e:
            raise e
        
        if return_idx:
            return checkpoing_idx


class Encoder(nn.Module):
        def __init__(self, emb_dim, device):
            super().__init__()
            
            self.device = device
            # self.conv1_block1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding='same')
            
            # self.conv1_block2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
            
            # self.conv1_block3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same')
            # self.conv2_block3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same')
            
            # self.conv1_block4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same')
            # self.conv2_block4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')
            # self.resnet = ResNet18(3, ResBlock)
            self.resnet = ResNet34(3, ResBlock)
            
            self.fc = nn.LazyLinear(out_features=emb_dim)

            
        def forward(self, x):
            x = self.resnet(x)
            # x = torch.max_pool2d(input=torch.relu(self.conv1_block1(x)), kernel_size=(2, 2))
            
            # x = torch.max_pool2d(input=torch.relu(self.conv1_block2(x)), kernel_size=(2, 2))
            
            # x = torch.relu(self.conv1_block3(x))
            # x = torch.max_pool2d(input=torch.relu(self.conv2_block3(x)), kernel_size=(2, 2))
            
            # x = torch.max_pool2d(input=torch.relu(self.conv1_block4(x)), kernel_size=(2, 2))
            # x = torch.relu(self.conv2_block4(x))
            
            # features_block4 = self.dropout_block4(torch.max_pool2d(input=torch.relu(self.conv1_block4(features_block3)), kernel_size=(2, 2)))
            
            x = x.permute(0, 2, 3, 1)
            
            x = self.add_timing_signal_nd(x, min_timescale=10.0)
                        
            x = torch.reshape(x, (x.shape[0], -1, x.shape[3]))
            
            x = torch.relu(self.fc(x))
            
            return x
        
        def add_timing_signal_nd(self, x:Tensor, min_timescale=5.0, max_timescale=1.0e4):
            """Adds a bunch of sinusoids of different frequencies to a Tensor.
            Each channel of the input Tensor is incremented by a sinusoid of a different
            frequency and phase in one of the positional dimensions.
            This allows attention to learn to use absolute and relative positions.
            Timing signals should be added to some precursors of both the query and the
            memory inputs to attention.
            The use of relative position is possible because sin(a+b) and cos(a+b) can be
            experessed in terms of b, sin(a) and cos(a).
            x is a Tensor with n "positional" dimensions, e.g. one dimension for a
            sequence or two dimensions for an image
            We use a geometric sequence of timescales starting with
            min_timescale and ending with max_timescale.  The number of different
            timescales is equal to channels // (n * 2). For each timescale, we
            generate the two sinusoidal signals sin(timestep/timescale) and
            cos(timestep/timescale).  All of these sinusoids are concatenated in
            the channels dimension.
            Args:
                x: a Tensor with shape [batch, d1 ... dn, channels]
                min_timescale: a float
                max_timescale: a float
            Returns:
                a Tensor the same shape as x.
            """
            num_dims = 2
            channels = x.shape[-1]
            num_timescales = channels // (num_dims * 2)
            log_timescale_increment = (
                    math.log(float(max_timescale) / float(min_timescale)) /
                    (torch.Tensor([num_timescales]).type(torch.float32) - 1))
            inv_timescales = min_timescale * torch.exp(
                    torch.Tensor(range(0, num_timescales)).type(torch.float32) * -log_timescale_increment)
            for dim in range(num_dims):
                length = x.shape[dim + 1]
                position = torch.Tensor(range(0, length)).type(torch.float32) 
                scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(
                        inv_timescales, 0)
                signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1)
                prepad = dim * 2 * num_timescales
                postpad = channels - (dim + 1) * 2 * num_timescales
                signal = F.pad(signal, (prepad, postpad, 0, 0))
                for _ in range(1 + dim):
                    signal = torch.unsqueeze(signal, 0)
                for _ in range(num_dims - 1 - dim):
                    signal = torch.unsqueeze(signal, -2)
                x = x + signal.to(self.device)
            return x
        

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PatchEmbedding(nn.Module):
        def __init__(self, img_H: int, img_W: int, patch_size: int = 16, emb_size: int = 512):
            self.patch_size = patch_size
            super().__init__()
            self.projection = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.LazyConv2d(emb_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
            self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
            self.positions = nn.Parameter(torch.randn((img_H // patch_size) * (img_W // patch_size) + 1, emb_size))

        def forward(self, x):
            b, _, _, _ = x.shape
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
        self.vit_encoder = ViT(img_H=img_H, img_W=img_W, nhead=nhead, depth=num_encoder_layers, emb_size=512)
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
        
        trg_inp = trg[:, :-1]
        trg_true = trg[:, 1:]
        
        img_features = self.vit_encoder(src)
        
        _, tgt_mask, _, tgt_padding_mask = self.create_mask(img_features, trg_inp, 0)
        
        # src_emb = self.positional_encoding(self.src_tok_emb(src))
        
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg_inp))
        transformers_out = self.decoder(tgt_emb, img_features, tgt_mask, None, tgt_padding_mask, None)
        logits = self.generator(transformers_out)
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), trg_true.reshape(-1))
        return loss
        

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


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                #  src_vocab_size: int,
                 tgt_vocab_size: int,
                 criterion,
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
        transformers_out = self.transformer(src_emb, tgt_emb, None, tgt_mask, None,
                                None, tgt_padding_mask, None)
        logits = self.generator(transformers_out)
        
        for pred, trgt in zip(logits, trg_true):
            loss += torch.sum(self.criterion(pred, trgt))
        
        # loss = self.criterion(logits.reshape(-1, logits.shape[-1]), trg_true.reshape(-1))
        return loss / torch.sum(~(trg_true == 0))
        

    # def encode(self, src: Tensor, src_mask: Tensor):
    #     return self.transformer.encoder(self.positional_encoding(
    #                         self.src_tok_emb(src)), src_mask)

    # def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
    #     return self.transformer.decoder(self.positional_encoding(
    #                       self.tgt_tok_emb(tgt)), memory,
    #                       tgt_mask)
    
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


class HandwrittenLatexDataset(Dataset):
    def __init__(self, img_names, caps, resize_H, resize_W, transform=None, random_resize=False):
        self.img_names = img_names
        self.caps = caps
        self.transform = transform
        self.random_resize = random_resize
        self.resize_H = resize_H
        self.resize_W = resize_W
        assert len(self.img_names) == len(self.caps)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        background = read_image(self.img_names[idx][0], ImageReadMode.RGB_ALPHA)
        formula = read_image(self.img_names[idx][1], ImageReadMode.RGB_ALPHA)
        # plt.imshow(formula.permute(1, 2, 0))
        caption = self.caps[idx]
        # image = make_fix_size(image.permute(1, 2, 0).numpy(), self.resize_W, self.resize_H, random_resize=self.random_resize)
        image = compile_handwritten_image(background.permute(1, 2, 0).numpy(), formula.permute(1, 2, 0).numpy())
        if self.transform:
            image = self.transform(image)
        return image, caption


class Training:
    def __init__(self, device, **kwargs):
        self.params = munchify(kwargs)
        for item, v in zip(self.params, self.params.values()):
            print(f"k: {item} --- v: {v}")
        
        self.device = device
        
        
        if not self.params.train_from_ckpt:
            os.makedirs(os.path.dirname(self.params.meta_path), exist_ok=True)
            with open(self.params.meta_path, "w+") as meta:
                json.dump({"params": [kwargs]}, meta, indent=4)

            self.logger = log_init(self.params.model_path, "model_log", "w")
            self.logger.info(
                f"Model '{self.params.model_path}' has been created with these params:")

            self.logger.info(
                f"---------------MODEL AND UTILS SUMMARY---------------")
            self.logger.info(
                f"ENCODER:\n{inspect.getsource(Encoder)}")
            # self.logger.info(
            #     f"DECODER:\n{inspect.getsource(Decoder)}")
            self.logger.info(
                f"SEQ2SEQ:\n{inspect.getsource(Seq2SeqTransformer)}")
            # self.logger.info(f"RESIZE_HEIGHT: {RESIZED_IMG_H}")
            # self.logger.info(f"RESIZE_WIDTH: {RESIZED_IMG_W}")
            # self.logger.info(f"DATA_SPLIT_INDEX: {DATA_SPLIT}")
            for k, v in kwargs.items():
                self.logger.info(f"{k}: {v}")
            self.logger.info(
                f"-----------------------------------------------------")
        
        else:
            self.logger = log_init(self.params.model_path, "model_log", "a+")
            self.logger.info("Params have been loaded successfully for training from checkpoint!")
            
            
    def auto_train(self):
        self.data_preprocess()
        self.train()
    
    def train_from_ckpt(self):
        self.data_preprocess_from_ckpt()
        self.train()

    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    def get_image_to_caption(self, annotations_file):
        # словарь, в который автоматически будет добавляться лист при попытке доступа к несущестувующему ключу
        image_path_to_caption = collections.defaultdict(list)
        for val in annotations_file['annotations']:
            caption = f"<start> {val['caption']} <end>"
            backgrnd_path = self.params.background_path + val["image_id"] + ".png"
            formulas_path = self.params.formulas_path + val["image_id"] + ".png"
            image_path_to_caption[(backgrnd_path, formulas_path)].append(
                caption)  # словарь типа 'путь_фото': ['описание1', 'описание2', ...]

        return image_path_to_caption

    def data_preprocess_from_ckpt(self):
        with open(self.params.caption_path, 'r') as f:
            annotations = json.load(f)

        image_path_to_caption = self.get_image_to_caption(annotations)
        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)

        train_image_paths = image_paths[:self.params.image_count]
        self.logger.info(
            f"Annotations and image paths have been successfully extracted from {self.params.caption_path}\ntotal images count - {len(train_image_paths)}")

        train_captions = []
        self.img_name_vector = []

        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            self.img_name_vector.extend([image_path] * len(caption_list))

        train_seqs = self.params.tokenizer.texts_to_sequences(
            train_captions)

        cap_vector = keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding='post')

        self.img_to_cap_vector = collections.defaultdict(list)
        for img, cap in zip(self.img_name_vector, cap_vector):
            self.img_to_cap_vector[img].append(cap)
    
    def data_preprocess(self):
        with open(self.params.caption_path, 'r') as f:
            annotations = json.load(f)

        image_path_to_caption = self.get_image_to_caption(annotations)
        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)

        train_image_paths = image_paths[:self.params.image_count]
        self.logger.info(
            f"Annotations and image paths have been successfully extracted from {self.params.caption_path}\ntotal images count - {len(train_image_paths)}")

        train_captions = []  # описания для тренировки
        self.img_name_vector = []  # пути к фото

        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            self.img_name_vector.extend([image_path] * len(caption_list))

        os.makedirs(os.path.dirname(self.params.tokenizer_path), exist_ok=True)
        with open(self.params.tokenizer_path, 'w+', encoding='utf-8') as f:
            # Choose the top top_k words from the vocabulary
            self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.params.vocab_size, split=' ', oov_token="<unk>",
                                                                   lower=False, filters='%')

            self.tokenizer.fit_on_texts(train_captions)
            self.tokenizer.word_index['<pad>'] = 0
            # создание токенайзера и сохранение его в json
            self.tokenizer.index_word[0] = '<pad>'

            tokenizer_json = self.tokenizer.to_json()
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        self.logger.info("\nTokenizer has been created with params:")
        for k, v in self.tokenizer.get_config().items():
            self.logger.info(f"{k} - {v}")

        # Create the tokenized vectors
        train_seqs = self.tokenizer.texts_to_sequences(
            train_captions)  # преобразование текста в последовательность чисел
        # print(train_seqs[:100])

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        cap_vector = keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding='post')  # приведение всех
        # последовательностей к одинаковой длине
        # print(cap_vector[:100])

        # Calculates the max_length, which is used to store the attention weights

        # os.makedirs(os.path.dirname(tokenizer_params_path), exist_ok=True)
        self.max_length = self.calc_max_length(train_seqs)
        with open(self.params.meta_path, "r+") as meta:
            params = json.load(meta)
            params["params"][0].update({"max_len": self.max_length})
            meta.seek(0)
            json.dump(params, meta, indent=4)

        self.img_to_cap_vector = collections.defaultdict(list)
        for img, cap in zip(self.img_name_vector, cap_vector):
            self.img_to_cap_vector[img].append(cap)

        self.logger.info(
            f"Img_to_cap vector is compiled. Max cap length - {self.max_length}")
        
    def train(self):
        img_keys = list(self.img_to_cap_vector.keys())
        random.shuffle(img_keys)
        # print(img_keys[:100])

        slice_index = int(len(img_keys) * 1)
        img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

        img_name_train = []
        cap_train = []
        for imgt in img_name_train_keys:  # тут теперь идут все фотки, а не только 80%
            capt_len = len(self.img_to_cap_vector[imgt])
            img_name_train.extend([imgt] * capt_len)
            cap_train.extend(self.img_to_cap_vector[imgt])

        img_name_val = []
        cap_val = []
        for imgv in img_name_val_keys:
            capv_len = len(self.img_to_cap_vector[imgv])
            img_name_val.extend([imgv] * capv_len)
            cap_val.extend(self.img_to_cap_vector[imgv])

        # print(img_name_train[:20])
        # print(cap_train[:20])
        self.logger.info(
            f"Dataset has been splitted:\nIMG_TRAIN - {len(img_name_train)}\nCAP_TRAIN - {len(cap_train)}\nIMG_VAL - {len(img_name_val)}\nCAP_VAL - {len(cap_val)}")

        num_steps = len(img_name_train) // self.params.batch_size
        
        training_data = HandwrittenLatexDataset(img_name_train,
                                     cap_train, 
                                     self.params.image_size[0], 
                                     self.params.image_size[1], 
                                     torchvision.transforms.ToTensor(),
                                     True)
        #  val_data = LatexDataset(img_name_val, cap_val, self.params.image_size[0], self.params.image_size[1], torchvision.transforms.ToTensor())
        
        train_dataset = DataLoader(training_data, 
                                   batch_size=self.params.batch_size, 
                                   shuffle=True, 
                                   drop_last=False, 
                                #    pin_memory=True, 
                                #    num_workers=12,
                                #    persistent_workers=True
                                   )
        # val_dataset = DataLoader(val_data, batch_size=self.params.batch_size, shuffle=True, drop_last=False)
        
        for im, cap in train_dataset:
            plt.imshow(torchvision.utils.make_grid(im).permute(1, 2, 0))
            print(im[0].shape)
            print(im[0])
            print(cap[0].shape)
            print(cap)
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            
        model = Seq2SeqTransformer(num_encoder_layers=self.params.num_encoder_layers, 
                                   num_decoder_layers=self.params.num_decoder_layers, 
                                   emb_size=self.params.embedding_dim,
                                   nhead=self.params.nhead,
                                   tgt_vocab_size=self.params.vocab_size,
                                   criterion=criterion,
                                   dim_feedforward=self.params.ff_dim,
                                   device=self.device)
        
        # model = Seq2SeqTransformerVIT(num_encoder_layers=self.params.num_encoder_layers, 
        #                               num_decoder_layers=self.params.num_decoder_layers, 
        #                               emb_size=self.params.embedding_dim,
        #                               nhead=self.params.nhead,
        #                               img_H=self.params.image_size[0],
        #                               img_W=self.params.image_size[1],
        #                               tgt_vocab_size=self.params.vocab_size,
        #                               criterion=criterion,
        #                               dim_feedforward=self.params.ff_dim,
        #                               device=self.device).to(self.device)
                
        optimizer = torch.optim.Adam(model.parameters())
        
        self.start_ckpt_id = 0
        ckpt = Checkpointing(self.params.checkpoint_path,
                             max_to_keep=10,
                             model=model,
                             optimizer=optimizer)
        if self.params.train_from_ckpt:
            self.start_ckpt_id = ckpt.load(return_idx=True)
        
        # self.logger.info(f"OPTIMIZER SUMMARY:\n{optimizer.get_config()}")
        
        loss_plot = []

        for epoch in range(self.start_ckpt_id, self.params.epochs):
            start = time.time()
            epoch_loss = 0

            for (batch, (img_tensor, target)) in tqdm(enumerate(train_dataset)):
                img_tensor, target = img_tensor.to(self.device), target.type(torch.LongTensor).to(self.device)
                
                optimizer.zero_grad()
                
                # print(f"img_tensor shape = {img_tensor.shape}")
                # print(f"target shape = {target.shape}")

                # print(img_tensor.numpy())
                # print(target.numpy())
                
                loss = model(src=img_tensor, trg=target)
                
                # output = output[1:]
                # target = target[:, 1:].to(device)
                
                # output = output.view(-1, output.shape[-1])
                # target = target.reshape(-1)
                
                # loss = criterion(output, target)
                # for idx, t in enumerate(loss):
                #     print(f"{idx}) {t.item()}", end=' ', sep='\n')
                
                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                
                # total_loss += t_loss
                # print(f"b_loss = {batch_loss}")
                if batch % 100 == 0:
                    # average_batch_loss = batch_loss.numpy() / \
                    #     int(target.shape[1])
                    self.logger.info(
                        f'Epoch {epoch + 1} Batch {batch} Loss = {loss.item()}')
            # storing the epoch end loss value to plot later
            # loss_plot.append(total_loss / num_steps)

            ckpt.save()
            self.logger.info(f"Epoch {epoch} checkpoint saved!")

            # self.logger.info(
            #     f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
            self.logger.info(
                f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

        self.logger.info(f"Training is done!")


class Image2Latex_load:
    loaded = False
    background_path = "handlim_backgrounds\\"
    formulas_path = "handlim_formulas\\"
    c_p = "handlim_dataset.json"
    VOCAB_SIZE = 300
    IMAGE_COUNT = 100
    IMG_H = 110
    IMG_W = 940
    BATCH_SIZE = 24
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    NHEAD = 8
    EMBEDDING_DIM = 256
    FF_DIM = 512
    EPOCHS = 100

    def __init__(self, model_name: str, working_path: str = "", ckpt_idx=None, device='cpu'):
        self.model_path = os.path.abspath(
            ".") + "\\trained_models_transformer_handwritten_pt\\" + working_path + model_name
        
        self.device = device
        self.tokenizer_path = self.model_path + "\\tokenizer.json"
        self.checkpoint_path = self.model_path + "\\checkpoints\\"
        self.meta_path = self.model_path + "\\meta.json"

        if os.path.exists(self.checkpoint_path) and len(os.listdir(self.checkpoint_path)) != 0 and os.path.exists(self.tokenizer_path) and os.path.exists(self.meta_path):
            print(f"Model has been found at {self.model_path}")
            self.loaded = True
        elif not os.path.exists(self.model_path):
            print(f"Model will be created at {self.model_path}")
        else:
            if input(f"Model is corrupted. Remove? (y/AB): ") == "y":
                shutil.rmtree(self.model_path)
                self.loaded = False
            else:
                self.loaded = True

        if self.loaded:
            try:
                with open(self.meta_path, "r") as meta:
                    self.params = munchify(json.load(meta)["params"])[0]
            except:
                raise IOError("config file reading error!")
            
            try:
                with open(self.params.tokenizer_path, "r") as f:
                    data = json.load(f)
                    self.tokenizer = tokenizer_from_json(data)
            except:
                raise IOError("Something went wrong with initializing tokenizer")
                    
            self.model = Seq2SeqTransformer(num_encoder_layers=self.params.num_encoder_layers, 
                                   num_decoder_layers=self.params.num_decoder_layers, 
                                   emb_size=self.params.embedding_dim,
                                   nhead=self.params.nhead,
                                   tgt_vocab_size=self.params.vocab_size,
                                   criterion=None,
                                   dim_feedforward=self.params.ff_dim,
                                   device=self.device)
            
            # self.model = Seq2SeqTransformerVIT(num_encoder_layers=self.params.num_encoder_layers, 
            #                        num_decoder_layers=self.params.num_decoder_layers, 
            #                        emb_size=self.params.embedding_dim,
            #                        nhead=self.params.nhead,
            #                        img_H=self.params.image_size[0],
            #                        img_W=self.params.image_size[1],
            #                        tgt_vocab_size=self.params.vocab_size,
            #                        criterion=None,
            #                        dim_feedforward=self.params.ff_dim,
            #                        device=self.device)

            ckpt = Checkpointing(self.params.checkpoint_path,
                                 model=self.model)
            ckpt.load(idx=ckpt_idx)
        
    def get_model_path(self):
        return self.model_path

    def input(self, inp, val):
        return val if inp == "" else inp

    def train(self):
        print(f"Caption file must be in {os.path.abspath('.')}")
        print(f"Dataset folder must be in {os.path.abspath('.')}\\datasets\\")
        if self.loaded:
            if input("You will loose your model. Proceed? (y/AB): ") == "y":
                shutil.rmtree(self.model_path)
                self.loaded = False
            else:
                return

        if not self.loaded:
            print("\nPlease enter some data for new model: ")
            try:
                self.background_path = self.input(
                    input(f"background_path - Default = {self.background_path}: "), self.background_path)
                self.formulas_path = self.input(
                    input(f"formulas_path - Default = {self.formulas_path}: "), self.formulas_path)
                self.c_p = self.input(
                    input(f"caption file - Default = {self.c_p}: "), self.c_p)
                self.VOCAB_SIZE = int(
                    self.input(input(f"vocab size - Default = {self.VOCAB_SIZE} (number of top used words in caps): "),
                               self.VOCAB_SIZE))
                self.IMAGE_COUNT = int(
                    self.input(input(f"image_count - Default = {self.IMAGE_COUNT} (number of photos to be used): "),
                               self.IMAGE_COUNT))
                self.IMG_H = int(
                    self.input(input(f"image_height - Default = {self.IMG_H} (image final height after preprocessing): "),
                               self.IMG_H))
                self.IMG_W = int(
                    self.input(input(f"image_width - Default = {self.IMG_W} (image final width after preprocessing): "),
                               self.IMG_W))
                self.BATCH_SIZE = int(self.input(
                    input(f"BATCH_SIZE - Default = {self.BATCH_SIZE}: "), self.BATCH_SIZE))
                self.NUM_ENCODER_LAYERS = int(
                    self.input(input(f"NUM_ENCODER_LAYERS - Default = {self.NUM_ENCODER_LAYERS}: "), self.NUM_ENCODER_LAYERS))
                self.NUM_DECODER_LAYERS = int(
                    self.input(input(f"NUM_DECODER_LAYERS - Default = {self.NUM_DECODER_LAYERS}: "), self.NUM_DECODER_LAYERS))
                self.NHEAD = int(
                    self.input(input(f"NHEAD - Default = {self.NHEAD}: "), self.NHEAD))
                self.EMBEDDING_DIM = int(
                    self.input(input(f"embedding_dim - Default = {self.EMBEDDING_DIM}: "), self.EMBEDDING_DIM))
                self.FF_DIM = int(self.input(
                    input(f"ff_dim - Default = {self.FF_DIM} (feed-forward unit dimension): "), self.FF_DIM))
                self.EPOCHS = int(self.input(
                    input(f"EPOCHS - Default = {self.EPOCHS}: "), self.EPOCHS))
            except:
                raise TypeError("Model params initialization failed")

            self.loaded = True
            self.background_path = os.path.abspath(".") + "\\datasets\\" + self.background_path
            self.formulas_path = os.path.abspath(".") + "\\datasets\\" + self.formulas_path
            self.caption_path = os.path.abspath(".") + "\\" + self.c_p

            train = Training(device=self.device,
                             model_path=self.model_path,
                             background_path=self.background_path,
                             formulas_path=self.formulas_path,
                             caption_path=self.caption_path,
                             tokenizer_path=self.tokenizer_path,
                             checkpoint_path=self.checkpoint_path,
                             meta_path=self.meta_path,
                             vocab_size=self.VOCAB_SIZE,
                             image_count=self.IMAGE_COUNT,
                             image_size=(self.IMG_H, self.IMG_W),
                             batch_size=self.BATCH_SIZE,
                             num_encoder_layers=self.NUM_ENCODER_LAYERS,
                             num_decoder_layers=self.NUM_DECODER_LAYERS,
                             nhead=self.NHEAD,
                             embedding_dim=self.EMBEDDING_DIM,
                             ff_dim=self.FF_DIM,
                             epochs=self.EPOCHS,
                             train_from_ckpt=False)

            train.auto_train()
            
    def train_from_ckpt(self):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        train = Training(device=self.device,
                         model_path=self.params.model_path,
                         background_path=self.params.background_path,
                         formulas_path=self.params.formulas_path,
                         caption_path=self.params.caption_path,
                         tokenizer_path=self.params.tokenizer_path,
                         checkpoint_path=self.params.checkpoint_path,
                         meta_path=self.params.meta_path,
                         vocab_size=self.params.vocab_size,
                         image_count=self.params.image_count,
                         image_size=(self.params.image_size[0], self.params.image_size[1]),
                         batch_size=self.params.batch_size,
                         num_encoder_layers=self.params.num_encoder_layers,
                         num_decoder_layers=self.params.num_decoder_layers,
                         nhead=self.params.nhead,
                         embedding_dim=self.params.embedding_dim,
                         ff_dim=self.params.ff_dim,
                         epochs=self.params.epochs,
                         train_from_ckpt=True,
                         max_len=self.params.max_len,
                         tokenizer=self.tokenizer)

        train.train_from_ckpt()
            
    def predict(self, decoder_type: str, image_path: str, beam_width=4, temperature=0.3, postprocess=True, beam_return_only_correct=True):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        prediction = self.greedy_decoder(image_path=image_path) if decoder_type=="greedy" else self.beam_decoder(image_path, beam_width=beam_width, temperature=temperature, return_only_correct=beam_return_only_correct)
        return prediction[:prediction.index(" <end>")].replace("<start> ", "") if postprocess else prediction
        
    def index(self, array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx[1]

    def find_n_best(self, array, n):
        probs = [np.partition(array[0], i)[i] for i in range(-1, -n - 1, -1)]
        ids = [self.index(array, p) for p in probs]
        return [[prob, id] for prob, id in zip(probs, ids)]
    
    def beam_decoder(self, image, beam_width=10, temperature=0.3, return_only_correct=True):
        self.model.eval()
        with torch.no_grad():
            img = self.load_image(image, torchvision.transforms.ToTensor()).unsqueeze(0)
                
            img_features = self.model.cnn_encoder(img)

            # Generate the caption using the Transformer decoder
            decoded_caption = "<start> "

            tokenized_caption = keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([decoded_caption]), padding="post", maxlen=self.params.max_len)[:, :-1]
            _, tgt_mask, _, tgt_padding_mask = self.model.create_mask(img_features, tokenized_caption, 0)
            
            tgt_emb = self.model.positional_encoding(self.model.tgt_tok_emb(torch.tensor(tokenized_caption)))
            src_emb = self.model.positional_encoding(img_features)
            
            transformers_out = self.model.transformer(src_emb, tgt_emb, None, tgt_mask, None,
                                    None, torch.tensor(tgt_padding_mask), None)
            logits = self.model.generator(transformers_out)
            
            predictions = (torch.softmax(logits[0, 0, :], -1) / temperature).unsqueeze(0).numpy()
            init = self.find_n_best(predictions, beam_width)
            results = [[obj[0], obj[1], decoded_caption + self.tokenizer.index_word[int(obj[1])] + " "] for obj in
                    init]  # 0 - prob ; 1 - id ; 2 - hidden

            for i in range(1, self.params.max_len - 1):
                tmp_res = []

                for r in results:
                    tmp_tokenized_caption = keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([r[2]]), padding="post", maxlen=self.params.max_len)[:, :-1]
                    
                    _, tmp_tgt_mask, _, tmp_tgt_padding_mask = self.model.create_mask(img_features, tmp_tokenized_caption, 0)
                    tmp_tgt_emb = self.model.positional_encoding(self.model.tgt_tok_emb(torch.tensor(tmp_tokenized_caption)))
                                        
                    tmp_preds = torch.softmax(self.model.generator(self.model.transformer(src_emb, tmp_tgt_emb, None, tmp_tgt_mask, None,
                                    None, torch.tensor(tmp_tgt_padding_mask), None)), -1)
                    for obj in self.find_n_best((tmp_preds[0, i, :] / temperature).unsqueeze(0).numpy(), beam_width):
                        tmp_res.append(
                            [obj[0] + r[0], obj[1], r[2] + self.tokenizer.index_word[int(obj[1])] + " "])

                results.clear()
                tmp_res.sort(reverse=True, key=lambda x: x[0])
                for el in range(beam_width):
                    results.append(tmp_res[el])

                if all(['<end>' in r[2] for r in results]):
                    break
        
        results = [res[2]+" <end>" if not "<end>" in res[2] else res[2] for res in results]
        if return_only_correct:
            for el in results:
                if el.count("{") == el.count("}"):
                    return el 
            return "<start> Correct caption was not generated <end>"
        
        [print(el[2]) for el in results]
        return [el[2] for el in results]
        
    def greedy_decoder(self, image_path):
        self.model.eval()
        with torch.no_grad():
            # Read the image from the disk
            img = self.load_image(image_path, torchvision.transforms.ToTensor()).unsqueeze(0)
            
            img_features = self.model.cnn_encoder(img)
            # img_features = self.model.vit_encoder(img)

            # Generate the caption using the Transformer decoder
            decoded_caption = "<start> "
            for i in range(self.params.max_len):
                # tokenized_caption = vectorization([decoded_caption])[:, :-1]
                tokenized_caption = keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([decoded_caption]), padding="post", maxlen=self.params.max_len)[:, :-1]
                _, tgt_mask, _, tgt_padding_mask = self.model.create_mask(img_features, tokenized_caption, 0)
                
                tgt_emb = self.model.positional_encoding(self.model.tgt_tok_emb(torch.tensor(tokenized_caption)))
                src_emb = self.model.positional_encoding(img_features)
                # transformers_out = self.model.decoder(tgt_emb, img_features, tgt_mask, None, torch.tensor(tgt_padding_mask), None)
                transformers_out = self.model.transformer(src_emb, tgt_emb, None, tgt_mask, None,
                                None, torch.tensor(tgt_padding_mask), None)
                logits = self.model.generator(transformers_out)
                sampled_token_index = np.argmax(logits[0, i, :])
                sampled_token = self.tokenizer.index_word[sampled_token_index.item()]
                decoded_caption += " " + sampled_token
                if sampled_token == "<end>":
                    break
            
            return decoded_caption

    def load_image(self, img_name, transform):
        # raise NotImplementedError
        image = np.array(read_image(img_name, ImageReadMode.RGB).permute(1, 2, 0))
        if transform:
            image = transform(image)
        return image

    def random_predict(self, decoder_type, number=5):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        # with open(self.params.caption_path, 'r+') as file:
        #     capt = json.load(file)["annotations"]

        with open("5_dataset_large_val.json", 'r+') as file:
            capt = json.load(file)["annotations"]

        parameters = {'axes.labelsize': 10,
                      'axes.titlesize': 10,
                      'figure.subplot.hspace': 0.999,
                      'figure.subplot.wspace': 0.999}
        plt.rcParams.update(parameters)
        # print(plt.rcParams.keys())
        
        images = random.choices(capt, k=number)
        plotting = []
        for im in tqdm(images):
            image_path = "C:\\users\\shace\\desktop\\lol2\\" + im["image_id"] + ".png"
            plotting.append([im["caption"], self.predict(decoder_type, image_path, temperature=0.9),
                             plt.imread(image_path), im["image_id"]])  # real ; pred ; plt image

        cols = number / 5
        _, axes = plt.subplots(nrows=5, ncols=int(cols + (0 if cols.is_integer() else 1)))

        for ax, plot in zip(axes.flat, plotting):
            ax.imshow(plot[2])
            # edit_dist = levenshteinDistance(
            #     plot[0], plot[1][0].replace('<end>', '')[:-2])
            # bleu = nltk.translate.bleu_score.sentence_bleu([list(filter(lambda a: a != " ", plot[0].split(" ")))], list(
            #     filter(lambda a: a != " ", plot[1][0][:plot[1][0].index("<end>")].split(" "))))
            ax.set(
                title=f"img_name = {plot[-1]}\nreal = {plot[0]}\npred = {plot[1]}\nbleu = {0}")
            ax.axis('off')
            print(plot[1])
        plt.show()
     
     
if __name__ == "__main__":
    device = enable_gpu(False)
    van = Image2Latex_load("torch_transformers_2", device=device)
    # van.calulate_bleu_metric("C:\\Users\\shace\\Documents\\GitHub\\im2latex\\datasets\\formula_images_png_5_large_resized\\",
    #                       "C:\\Users\\shace\\Documents\\GitHub\\im2latex\\5_dataset_large.json")
    van.train()
    # van.train_from_ckpt()
    # van.random_predict("greedy", 10)
    print(van.predict("greedy", "C:\\Users\\shace\\Desktop\\dwd.png", temperature=0.9))
    # print(van.predict("greedy", "C:\\Users\\shace\\Desktop\\lol2\\180718.png", temperature=0.9))