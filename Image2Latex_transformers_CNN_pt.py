import collections
import inspect
import json
import os
import random
import shutil
import time
from matplotlib import pyplot as plt
from munch import munchify
import keras
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split

from torchinfo import summary
import torchvision
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from keras_preprocessing.text import tokenizer_from_json

from models.torch.cnn_encoder import Encoder
from models.torch.seq2seqCnnTransformer import Seq2SeqTransformer
from models.torch.seq2seqCnnTransformerDecoderOnly import Seq2SeqTransformerDecOnly
from models.torch.seq2seqVIT import Seq2SeqTransformerVIT
from utils.dataset_json_update import split_data

from utils.device_def import enable_gpu
from utils.logger_init import log_init
from utils.images_preprocessing import make_fix_size
from utils.checkpointing import Checkpointing
from utils.metrics import levenshteinDistance
from rouge_score import rouge_scorer
from nltk.translate import bleu_score

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import Unigram, WordLevel, WordPiece
from tokenizers.trainers import UnigramTrainer, WordLevelTrainer, WordPieceTrainer


class LatexDataset(Dataset):
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
        image = read_image(self.img_names[idx], ImageReadMode.RGB)
        caption = self.caps[idx]
        image = make_fix_size(image.permute(1, 2, 0).numpy(), self.resize_W, self.resize_H, random_resize=self.random_resize)
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
    
    def calc_max_length_hf(self, tensor, pad_id):
        for el in tensor:
            if not pad_id in el:
                return len(el)

    @staticmethod
    def accuracy(y_pred, y_true):
        assert y_pred.shape == y_true.shape
        a = torch.sum(y_pred == y_true)
        return a / y_pred.shape[0]

    def get_image_to_caption(self, annotations_file):
        # словарь, в который автоматически будет добавляться лист при попытке доступа к несущестувующему ключу
        image_path_to_caption = collections.defaultdict(list)
        for val in annotations_file['annotations']:
            caption = f"<start> {val['caption']} <end>"
            image_path = self.params.dataset_path + val["image_id"] + ".png"
            if os.path.exists(image_path):
                image_path_to_caption[image_path].append(
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

        if self.params.tokenizer_type == "keras":
            train_seqs = self.params.tokenizer.texts_to_sequences(
                train_captions)

            cap_vector = keras.preprocessing.sequence.pad_sequences(
                train_seqs, padding='post')
        elif self.params.tokenizer_type == "hf":
            cap_vector = np.array([el.ids for el in self.params.tokenizer.encode_batch(train_captions)])
        else:
            raise ValueError

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
        
        if self.params.tokenizer_type == "keras":
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
            
            self.max_length = self.calc_max_length(train_seqs)
        elif self.params.tokenizer_type == "hf":
            trainer = WordPieceTrainer(vocab_size=self.params.vocab_size, special_tokens=["<pad>", "<unk>", "<start>", "<end>"], show_progress=True)
            
            self.tokenizer = Tokenizer(WordPiece())
            self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            self.tokenizer.enable_padding(pad_token="<pad>", pad_id=0)

            self.tokenizer.train_from_iterator(train_captions, trainer)

            self.tokenizer.save(path=self.params.tokenizer_path, pretty=True)
            
            self.logger.info("\nTokenizer has been created")
            # for k, v in self.tokenizer.get_config().items():
            #     self.logger.info(f"{k} - {v}")
            
            cap_vector = np.array([el.ids for el in self.tokenizer.encode_batch(train_captions)])
            
            self.max_length = self.calc_max_length_hf(cap_vector, 0)
        else:
            raise ValueError

        # Calculates the max_length, which is used to store the attention weights

        # os.makedirs(os.path.dirname(tokenizer_params_path), exist_ok=True)
        
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

        slice_index = int(len(img_keys) * 0.9)
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
        
        training_data = LatexDataset(img_name_train,
                                     cap_train, 
                                     self.params.image_size[0], 
                                     self.params.image_size[1], 
                                     torchvision.transforms.ToTensor(),
                                     False)
        val_data = LatexDataset(img_name_val,
                                cap_val, 
                                self.params.image_size[0], 
                                self.params.image_size[1], 
                                torchvision.transforms.ToTensor(),
                                False)
        
        train_dataset = DataLoader(training_data, 
                                   batch_size=self.params.batch_size, 
                                   shuffle=True, 
                                   drop_last=False, 
                                   pin_memory=True, 
                                   num_workers=6,
                                   persistent_workers=True
                                   )
        val_dataset = DataLoader(val_data, 
                                   batch_size=self.params.batch_size, 
                                   shuffle=True, 
                                   drop_last=False, 
                                   pin_memory=True, 
                                   num_workers=6,
                                   persistent_workers=True
                                   )
        
        # val_dataset = DataLoader(val_data, batch_size=self.params.batch_size, shuffle=True, drop_last=False)
        
        # for im, cap in train_dataset:
        #     plt.imshow(torchvision.utils.make_grid(im).permute(1, 2, 0))
        #     print(im[0].shape)
        #     print(im[0])
        #     print(cap[0].shape)
        #     print(cap)
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            
        # model = Seq2SeqTransformer(num_encoder_layers=self.params.num_encoder_layers, 
        #                            num_decoder_layers=self.params.num_decoder_layers, 
        #                            emb_size=self.params.embedding_dim,
        #                            nhead=self.params.nhead,
        #                            tgt_vocab_size=self.params.vocab_size,
        #                            criterion=criterion,
        #                            dim_feedforward=self.params.ff_dim,
        #                            device=self.device)
        
        # model = Seq2SeqTransformerDecOnly(num_decoder_layers=self.params.num_decoder_layers, 
        #                                   emb_size=self.params.embedding_dim,
        #                                   nhead=self.params.nhead,
        #                                   tgt_vocab_size=self.params.vocab_size,
        #                                   criterion=criterion,
        #                                   dim_feedforward=self.params.ff_dim,
        #                                   device=self.device)
        
        model = Seq2SeqTransformerVIT(num_encoder_layers=self.params.num_encoder_layers, 
                                      num_decoder_layers=self.params.num_decoder_layers, 
                                      emb_size=self.params.embedding_dim,
                                      nhead=self.params.nhead,
                                      img_H=self.params.image_size[0],
                                      img_W=self.params.image_size[1],
                                      tgt_vocab_size=self.params.vocab_size,
                                      criterion=criterion,
                                      dim_feedforward=self.params.ff_dim,
                                      device=self.device).to(self.device)
                
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
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
                
                loss, train_logits = model(src=img_tensor, trg=target)
                
                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                
                # total_loss += t_loss
                # print(f"b_loss = {batch_loss}")
                if batch % 100 == 0:
                    # average_batch_loss = batch_loss.numpy() / \
                    #     int(target.shape[1])
                    self.logger.info(
                        f'Epoch {epoch + 1} Batch {batch} Loss = {loss.item()} Accuracy = {self.accuracy(torch.argmax(train_logits, dim=-1), target[:, 1:])}')
            # storing the epoch end loss value to plot later
            # loss_plot.append(total_loss / num_steps)
            
            val_loss = []
            val_acc = []
            with torch.no_grad():
                for (batch, (img_tensor, target)) in tqdm(enumerate(val_dataset)):
                    img_tensor, target = img_tensor.to(self.device), target.type(torch.LongTensor).to(self.device)
                    
                    loss, val_logits = model(src=img_tensor, trg=target)
                    
                    val_loss.append(loss.item())
                    val_acc.append(self.accuracy(torch.argmax(val_logits, dim=-1), target[:, 1:]).item())
                
                self.logger.info(f'Epoch {epoch + 1} Val Loss = {np.mean(val_loss)} Accuracy = {np.mean(val_acc).item()}')

            ckpt.save()
            self.logger.info(f"Epoch {epoch} checkpoint saved!")

            # self.logger.info(
            #     f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
            self.logger.info(
                f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

        self.logger.info(f"Training is done!")


class Image2Latex_load:
    loaded = False
    dataset_path = "images_150_merged_regenerated_ng_tiny\\"
    c_p = "dataset_NG_merged.json"
    VOCAB_SIZE = 300
    IMAGE_COUNT = 100
    IMG_H = 110
    IMG_W = 940
    # IMG_H = 112
    # IMG_W = 944
    TOKENIZER_TYPE = "hf"
    BATCH_SIZE = 24
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    NHEAD = 8
    EMBEDDING_DIM = 256
    FF_DIM = 512
    EPOCHS = 100

    def __init__(self, model_name: str, working_path: str = "", ckpt_idx=None, device='cpu'):
        self.model_path = os.path.abspath(
            ".") + "\\trained_models_transformers_pt\\" + working_path + model_name
        
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
                if self.params.tokenizer_type == "keras":
                    with open(self.params.tokenizer_path, "r") as f:
                        data = json.load(f)
                        self.tokenizer = tokenizer_from_json(data)
                elif self.params.tokenizer_type == "hf":
                    self.tokenizer = Tokenizer.from_file(self.params.tokenizer_path)
                    self.tokenizer.enable_padding(pad_token="<pad>", pad_id=0, length=self.params.max_len)
                else:
                    raise ValueError
            except Exception as e:
                print(e)
                raise IOError("Something went wrong with initializing tokenizer")
                    
            self.model = Seq2SeqTransformer(num_encoder_layers=self.params.num_encoder_layers, 
                                   num_decoder_layers=self.params.num_decoder_layers, 
                                   emb_size=self.params.embedding_dim,
                                   nhead=self.params.nhead,
                                   tgt_vocab_size=self.params.vocab_size,
                                   criterion=None,
                                   dim_feedforward=self.params.ff_dim,
                                   device=self.device)
            
            # self.model = Seq2SeqTransformerDecOnly(num_decoder_layers=self.params.num_decoder_layers, 
            #                                        emb_size=self.params.embedding_dim,
            #                                        nhead=self.params.nhead,
            #                                        tgt_vocab_size=self.params.vocab_size,
            #                                        criterion=None,
            #                                        dim_feedforward=self.params.ff_dim,
            #                                        device=self.device)
            
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
            # summary(self.model)
        
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
                self.dataset_path = self.input(
                    input(f"dataset_path - Default = {self.dataset_path}: "), self.dataset_path)
                self.c_p = self.input(
                    input(f"caption file - Default = {self.c_p}: "), self.c_p)
                self.TOKENIZER_TYPE = self.input(input(f"tokenizer type ('keras' of 'hf') - Default = {self.TOKENIZER_TYPE}: "),
                               self.TOKENIZER_TYPE)
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
            self.dataset_path = os.path.abspath(".") + "\\datasets\\" + self.dataset_path
            self.caption_path = os.path.abspath(".") + "\\datasets\\" + self.c_p

            train = Training(device=self.device,
                             model_path=self.model_path,
                             dataset_path=self.dataset_path,
                             caption_path=self.caption_path,
                             tokenizer_path=self.tokenizer_path,
                             checkpoint_path=self.checkpoint_path,
                             meta_path=self.meta_path,
                             vocab_size=self.VOCAB_SIZE,
                             tokenizer_type=self.TOKENIZER_TYPE,
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
                         dataset_path=self.params.dataset_path,
                         caption_path=self.params.caption_path,
                         tokenizer_path=self.params.tokenizer_path,
                         checkpoint_path=self.params.checkpoint_path,
                         meta_path=self.params.meta_path,
                         vocab_size=self.params.vocab_size,
                         tokenizer_type=self.params.tokenizer_type,
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
        
        # tokenizer_func = self.tokenizer.id_to_token if self.params.tokenizer_type == "hf" else  
        with torch.no_grad():
            img = self.load_image(image, torchvision.transforms.ToTensor()).unsqueeze(0)
                
            img_features = self.model.cnn_encoder(img)
            # img_features = self.model.vit_encoder(img)

            # Generate the caption using the Transformer decoder
            decoded_caption = "<start> "

            if self.params.tokenizer_type == "keras":
                tokenized_caption = keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([decoded_caption]), padding="post", maxlen=self.params.max_len)[:, :-1]
            elif self.params.tokenizer_type == "hf":
                tokenized_caption = np.expand_dims(np.array(self.tokenizer.encode(decoded_caption).ids), 0)[:, :-1]
                    
            _, tgt_mask, _, tgt_padding_mask = self.model.create_mask(img_features, tokenized_caption, 0)
            
            tgt_emb = self.model.positional_encoding(self.model.tgt_tok_emb(torch.tensor(tokenized_caption)))
            src_emb = self.model.positional_encoding(img_features)
            
            transformers_out = self.model.transformer(src_emb, tgt_emb, None, tgt_mask, None,
                                    None, torch.tensor(tgt_padding_mask), None)
            # transformers_out = self.model.decoder(tgt_emb, img_features, tgt_mask, None, torch.tensor(tgt_padding_mask), None)
            logits = self.model.generator(transformers_out)
            
            predictions = (torch.softmax(logits[0, 0, :], -1) / temperature).unsqueeze(0).numpy()
            init = self.find_n_best(predictions, beam_width)
            # self.tokenizer.index_word[int(obj[1])]
            results = [[obj[0], obj[1], decoded_caption + self.tokenizer.id_to_token(int(obj[1])) + " "] for obj in
                    init]  # 0 - prob ; 1 - id ; 2 - hidden

            for i in range(1, self.params.max_len - 1):
                tmp_res = []

                for r in results:
                    if self.params.tokenizer_type == "keras":
                        tmp_tokenized_caption = keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([r[2]]), padding="post", maxlen=self.params.max_len)[:, :-1]
                    elif self.params.tokenizer_type == "hf":
                        tmp_tokenized_caption = np.expand_dims(np.array(self.tokenizer.encode(r[2]).ids), 0)[:, :-1]
                    
                    _, tmp_tgt_mask, _, tmp_tgt_padding_mask = self.model.create_mask(img_features, tmp_tokenized_caption, 0)
                    tmp_tgt_emb = self.model.positional_encoding(self.model.tgt_tok_emb(torch.tensor(tmp_tokenized_caption)))
                                        
                    tmp_preds = torch.softmax(self.model.generator(self.model.transformer(src_emb, tmp_tgt_emb, None, tmp_tgt_mask, None,
                                    None, torch.tensor(tmp_tgt_padding_mask), None)), -1)
                    # tmp_preds = torch.softmax(self.model.generator(self.model.decoder(tmp_tgt_emb, img_features, tmp_tgt_mask, None, torch.tensor(tmp_tgt_padding_mask), None)), -1)
                    for obj in self.find_n_best((tmp_preds[0, i, :] / temperature).unsqueeze(0).numpy(), beam_width):
                        new_token = self.tokenizer.index_word[int(obj[1])] if self.params.tokenizer_type == "keras" else self.tokenizer.id_to_token(int(obj[1]))
                        tmp_res.append(
                            [obj[0] + r[0], obj[1], r[2] + new_token + " "])

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
        
        [print(el) for el in results]
        return [el for el in results]
        
    def greedy_decoder(self, image_path):
        self.model.eval()
        with torch.no_grad():
            # Read the image from the disk
            img = self.load_image(image_path, torchvision.transforms.ToTensor()).unsqueeze(0).to(self.device)
            
            img_features = self.model.cnn_encoder(img)
            # img_features = self.model.vit_encoder(img)

            # Generate the caption using the Transformer decoder
            decoded_caption = "<start> "
            for i in range(self.params.max_len):
                if self.params.tokenizer_type == "keras":
                    tokenized_caption = keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([decoded_caption]), padding="post", maxlen=self.params.max_len)[:, :-1]
                elif self.params.tokenizer_type == "hf":
                    tokenized_caption = np.expand_dims(np.array(self.tokenizer.encode(decoded_caption).ids), 0)[:, :-1]
                    
                _, tgt_mask, _, tgt_padding_mask = self.model.create_mask(img_features, tokenized_caption, 0)
                
                tgt_emb = self.model.positional_encoding(self.model.tgt_tok_emb(torch.tensor(tokenized_caption).to(self.device)))
                src_emb = self.model.positional_encoding(img_features)
                
                # transformers_out = self.model.decoder(tgt_emb, img_features, tgt_mask, None, torch.tensor(tgt_padding_mask), None)
                
                # transformers_out = self.model.transformer_decoder(tgt_emb, src_emb, tgt_mask, None, torch.tensor(tgt_padding_mask), None)
                
                transformers_out = self.model.transformer(src_emb, tgt_emb, None, tgt_mask, None, None, torch.tensor(tgt_padding_mask).to(self.device), None)
                logits = self.model.generator(transformers_out)
                sampled_token_index = np.argmax(logits.cpu()[0, i, :])
                sampled_token = self.tokenizer.index_word[sampled_token_index.item()] if self.params.tokenizer_type == "keras" else self.tokenizer.id_to_token(sampled_token_index.item())
                decoded_caption += " " + sampled_token
                # print(decoded_caption)
                if sampled_token == "<end>":
                    break
                

            # decoded_caption = decoded_caption.replace("<start> ", "")
            # decoded_caption = decoded_caption.replace(" <end>", "").strip()
            # print("Predicted Caption: ", decoded_caption)
            return decoded_caption

    def load_image(self, img_name, transform):
        # raise NotImplementedError
        image = read_image(img_name, ImageReadMode.RGB)
        image = make_fix_size(image.permute(1, 2, 0).numpy(), self.params.image_size[1], self.params.image_size[0], False)
        if transform:
            image = transform(image)
        return image
    
    # def predict(self, decoder, image_path: str):
    #     if not self.loaded:
    #         raise ValueError("Model is not loaded!")
    #     pred = Prediction()
    #     res = pred.predict(decoder, image_path)
    #     if decoder == "categorical" or decoder == "beam2" or decoder == "beam3":
    #         print(' '.join(res))
    #     if decoder == "beam":
    #         [print(r[:r.index("<end>")] + "\n") for r in res]
    
    def testing(self, val_dataset, labels):
        if not self.loaded:
            raise ValueError("Model is not loaded!")

        bleus = []
        levens = []
        rouge1 = []
        rouge2 = []
        exact_match = 0
        
        with open(labels, "r") as file:
            dataset = json.load(file)["annotations"]
        
        dataset = dataset[:1000]
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        chencherry =  bleu_score.SmoothingFunction()

        for img in tqdm(dataset):
            if len(img["caption"].split()) < 5:
                continue
            
            try:
                pred = self.predict(decoder_type="greedy", image_path=os.path.join(val_dataset, f"{img['image_id']}.png")).replace("  ", " ").replace(" ##", "").strip()
                
                b = bleu_score.sentence_bleu(references=[img["caption"].split()], 
                                             hypothesis=pred.split(), 
                                            #  smoothing_function=chencherry.method7
                                             )
                # if b - 0.05 <= 0:
                #     print()
                l = levenshteinDistance(img["caption"], pred)
                
                r_scores = scorer.score(img["caption"], pred)
                r1 = r_scores["rouge1"][-1]
                r2 = r_scores["rouge2"][-1]
                ex = 1 if img["caption"] == pred else 0
            except:
                continue
            
            bleus.append(b)
            levens.append(l)
            rouge1.append(r1)
            rouge2.append(r2)
            exact_match += ex
            
        print(f"BLEU STATS:\nMIN = {np.min(bleus)} --- MAX = {np.max(bleus)} --- MEAN = {np.mean(bleus)} --- STD = {np.std(bleus)}\n")
        print(f"LEVD STATS:\nMIN = {np.min(levens)} --- MAX = {np.max(levens)} --- MEAN = {np.mean(levens)} --- STD = {np.std(levens)}\n")
        print(f"ROG1 STATS:\nMIN = {np.min(rouge1)} --- MAX = {np.max(rouge1)} --- MEAN = {np.mean(rouge1)} --- STD = {np.std(rouge1)}\n")
        print(f"ROG2 STATS:\nMIN = {np.min(rouge2)} --- MAX = {np.max(rouge2)} --- MEAN = {np.mean(rouge2)} --- STD = {np.std(rouge2)}\n")
        print(f"EXACT_MATCH = {exact_match / len(dataset)}\n")
        print(f"NORM_LEVD = {1 - np.mean(levens / np.linalg.norm(levens))}\n")

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
        
        # capt = list(filter(lambda x: len(x["caption"]) > 200, capt))
        
        # random.seed("ERGDRJHUGREGT654RGHMKGAH")
        images = random.choices(capt, k=number)
        plotting = []
        for im in tqdm(images):
            image_path = "C:\\users\\shace\\Documents\\GitHub\\im2latex\\datasets\\images_150_val\\" + im["image_id"] + ".png"
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
    device = enable_gpu(True)

    # van = Image2Latex_load("torch_transformers_11_combined_pos_emb_RESNET_XXL", device=device)
    # van.random_predict(decoder_type="greedy", number=5)
    
    # van = Image2Latex_load("torch_transformers_12_VIT_0.1_of_default_lr", device=device)  
    # van = Image2Latex_load("torch_transformers_16", device=device)
    van = Image2Latex_load("torch_transformers_16_v3", device=device)

    van.train()
    # van.train_from_ckpt()
    # van.testing(val_dataset="C:/Users/shace/Documents/GitHub/im2latex/datasets/images_150_test_rgb", labels="C:/Users/shace/Documents/GitHub/im2latex/test_dataset.json")
    # van.random_predict(decoder_type="greedy", number=5)
