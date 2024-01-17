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
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm
from keras_preprocessing.text import tokenizer_from_json
from utils.device_def import enable_gpu
from utils.images_preprocessing import make_fix_size
from utils.logger_init import log_init
from utils.metrics import levenshteinDistance, bleu_score


class Checkpointing:
    def __init__(self, path, max_to_keep=10, **kwargs) -> None:
        assert all(
            [
                (nn.Module in type(el).mro() or torch.optim.Optimizer in type(el).mro())
                for el in kwargs.values()
            ]
        )

        self.max_to_keep = max_to_keep
        self.modules = munchify(
            {k: v for k, v in kwargs.items() if nn.Module in type(v).mro()}
        )
        self.optims = munchify(
            {k: v for k, v in kwargs.items() if torch.optim.Optimizer in type(v).mro()}
        )

        if len(self.modules) == 0:
            print("Warning: no modules specified for saving/loading")
        if len(self.modules) == 0:
            print("Warning: no optimizers specified for saving/loading")

        self.path = path

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __get_files_and_ckpt_idx(self):
        dir_files = sorted(
            [
                s
                for s in os.listdir(self.path)
                if os.path.isfile(os.path.join(self.path, s))
            ],
            key=lambda s: os.path.getmtime(os.path.join(self.path, s)),
        )
        try:
            checkpoing_idx = (
                0
                if len(dir_files) == 0
                else int(dir_files[-1].replace(".tar", "").split("_")[-1]) + 1
            )
        except:
            raise IOError
        return dir_files, checkpoing_idx

    def save(self):
        dir_files, checkpoing_idx = self.__get_files_and_ckpt_idx()

        del_idxs = len(dir_files) - self.max_to_keep + 1

        if del_idxs > 0:
            [
                os.remove(os.path.join(self.path, dir_files[idx]))
                for idx in range(del_idxs)
            ]

        try:
            torch.save(
                {k: v.state_dict() for k, v in self.modules.items()}
                | {k: v.state_dict() for k, v in self.optims.items()},
                os.path.join(self.path, f"checkpoint_{checkpoing_idx}.tar"),
            )
        except:
            raise IOError

    def load(self):
        _, checkpoing_idx = self.__get_files_and_ckpt_idx()

        try:
            checkpoint = torch.load(
                os.path.join(self.path, f"checkpoint_{checkpoing_idx - 1}.tar")
            )
            [v.load_state_dict(checkpoint[k]) for k, v in self.modules.items()]
            [v.load_state_dict(checkpoint[k]) for k, v in self.optims.items()]
        except:
            raise IOError


class BahdanauAttention(nn.Module):
    def __init__(self, units, device):
        super().__init__()
        self.W1 = nn.LazyLinear(units)
        self.W2 = nn.LazyLinear(units)
        self.V = nn.LazyLinear(1)
        self.device = device

    def forward(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, H*W, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = torch.unsqueeze(hidden, 1).to(self.device)

        # attention_hidden_layer shape == (batch_size, H*W, units)
        attention_hidden_layer = (
            torch.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        ).to(self.device)

        # score shape == (batch_size, H*W, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, H*W, 1)
        attention_weights = torch.softmax(score, dim=1).to(self.device)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


class Encoder(nn.Module):
    def __init__(self, emb_dim, device):
        super().__init__()

        self.device = device
        self.conv1_block1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3), padding="same"
        )

        self.conv1_block2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
        )

        self.conv1_block3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same"
        )
        self.conv2_block3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same"
        )

        self.conv1_block4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same"
        )
        self.conv2_block4 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"
        )

        self.fc = nn.LazyLinear(out_features=emb_dim)

    def forward(self, x):
        x = torch.max_pool2d(input=torch.relu(self.conv1_block1(x)), kernel_size=(2, 2))

        x = torch.max_pool2d(input=torch.relu(self.conv1_block2(x)), kernel_size=(2, 2))

        x = torch.relu(self.conv1_block3(x))
        x = torch.max_pool2d(input=torch.relu(self.conv2_block3(x)), kernel_size=(2, 2))

        x = torch.max_pool2d(input=torch.relu(self.conv1_block4(x)), kernel_size=(2, 2))
        x = torch.relu(self.conv2_block4(x))

        # features_block4 = self.dropout_block4(torch.max_pool2d(input=torch.relu(self.conv1_block4(features_block3)), kernel_size=(2, 2)))

        x = x.permute(0, 2, 3, 1)

        # x = self.add_timing_signal_nd(x, min_timescale=10.0)

        x = torch.reshape(x, (x.shape[0], -1, x.shape[3]))

        x = torch.relu(self.fc(x))

        return x

    def add_timing_signal_nd(self, x: Tensor, min_timescale=5.0, max_timescale=1.0e4):
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
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (torch.Tensor([num_timescales]).type(torch.float32) - 1)
        inv_timescales = min_timescale * torch.exp(
            torch.Tensor(torch.range(start=0, end=num_timescales - 1)).type(
                torch.float32
            )
            * -log_timescale_increment
        )
        for dim in range(num_dims):
            length = x.shape[dim + 1]
            position = torch.Tensor(torch.range(start=0, end=length - 1)).type(
                torch.float32
            )
            scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(
                inv_timescales, 0
            )
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


class Decoder(nn.Module):
    def __init__(self, embedding_dim, units, vocab_size, device):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTMCell(embedding_dim * 2, units)

        self.fc1 = nn.LazyLinear(out_features=units)

        self.fc2 = nn.LazyLinear(out_features=vocab_size)

        self.attention = BahdanauAttention(units, device=device).to(device)

        self.units = units
        self.vocab_size = vocab_size

    def forward(self, x, features, prev_hidden_state, prev_cell_state):
        context_vector, attention_weights = self.attention(features, prev_hidden_state)

        x = self.embedding(x)

        x = torch.cat([context_vector, torch.squeeze(x, dim=1)], dim=-1)

        hidden_state, cell_state = self.lstm(x, (prev_hidden_state, prev_cell_state))

        x = self.fc1(hidden_state)

        x = self.fc2(x)

        return x, hidden_state, cell_state, attention_weights

    def initial_lstm_state(self, batch_size):
        return (
            torch.zeros((batch_size, self.units), dtype=torch.float32)
            .unsqueeze(0)
            .repeat(2, 1, 1)
        )


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self, src: Tensor, trg: Tensor, criterion, teacher_forcing_ratio: float = 0.5
    ) -> Tensor:
        loss = 0

        # outputs = torch.zeros(trg.shape[1], trg.shape[0], self.decoder.vocab_size).to(self.device)

        hidden_state, cell_state = self.decoder.initial_lstm_state(trg.shape[0]).to(
            self.device
        )

        img_features = self.encoder(src)

        # first input to the decoder is the <start> token
        output = trg[:, 0].unsqueeze(-1)

        for t in range(1, trg.shape[1]):
            output, hidden_state, cell_state, _ = self.decoder(
                output, img_features, hidden_state, cell_state
            )

            loss_ = criterion(output, trg[:, t])
            loss += torch.mean(loss_)

            # outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]

            output = trg[:, t].unsqueeze(-1) if teacher_force else top1

        return loss, loss / trg.shape[1]


class LatexDataset(Dataset):
    def __init__(
        self, img_names, caps, resize_H, resize_W, transform=None, target_transform=None
    ):
        self.img_names = img_names
        self.caps = caps
        self.transform = transform
        self.target_transform = target_transform
        self.resize_H = resize_H
        self.resize_W = resize_W
        assert len(self.img_names) == len(self.caps)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = read_image(self.img_names[idx], ImageReadMode.RGB)
        caption = self.caps[idx]
        image = make_fix_size(
            image.permute(1, 2, 0).numpy(), self.resize_W, self.resize_H, False
        )
        if self.transform:
            image = self.transform(image)
        return image, caption


class Training:
    def __init__(self, device, **kwargs):
        self.params = munchify(kwargs)
        for item, v in zip(self.params, self.params.values()):
            print(f"k: {item} --- v: {v}")

        # os.makedirs(os.path.dirname(self.params.checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.params.meta_path), exist_ok=True)
        with open(self.params.meta_path, "w+") as meta:
            json.dump({"params": [kwargs]}, meta, indent=4)

        self.device = device
        self.logger = log_init(self.params.model_path, "model_log")
        self.logger.info(
            f"Model '{self.params.model_path}' has been created with these params:"
        )

        self.logger.info(f"---------------MODEL AND UTILS SUMMARY---------------")
        self.logger.info(f"ENCODER:\n{inspect.getsource(Encoder)}")
        self.logger.info(f"DECODER:\n{inspect.getsource(Decoder)}")
        self.logger.info(f"SEQ2SEQ:\n{inspect.getsource(Seq2Seq)}")
        # self.logger.info(f"RESIZE_HEIGHT: {RESIZED_IMG_H}")
        # self.logger.info(f"RESIZE_WIDTH: {RESIZED_IMG_W}")
        # self.logger.info(f"DATA_SPLIT_INDEX: {DATA_SPLIT}")
        for k, v in kwargs.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info(f"-----------------------------------------------------")

    def auto_train(self):
        self.data_preprocess()
        self.train()

    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    def get_image_to_caption(self, annotations_file):
        # словарь, в который автоматически будет добавляться лист при попытке доступа к несущестувующему ключу
        image_path_to_caption = collections.defaultdict(list)
        for val in annotations_file["annotations"]:
            caption = f"<start> {val['caption']} <end>"
            image_path = self.params.dataset_path + val["image_id"] + ".png"
            image_path_to_caption[image_path].append(
                caption
            )  # словарь типа 'путь_фото': ['описание1', 'описание2', ...]

        return image_path_to_caption

    def data_preprocess(self):
        with open(self.params.caption_path, "r") as f:
            annotations = json.load(f)

        # image_path_to_caption = caption_mapping
        image_path_to_caption = self.get_image_to_caption(annotations)
        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)

        train_image_paths = image_paths[: self.params.image_count]
        self.logger.info(
            f"Annotations and image paths have been successfully extracted from {self.params.caption_path}\ntotal images count - {len(train_image_paths)}"
        )

        train_captions = []  # описания для тренировки
        self.img_name_vector = []  # пути к фото

        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            self.img_name_vector.extend([image_path] * len(caption_list))

        os.makedirs(os.path.dirname(self.params.tokenizer_path), exist_ok=True)
        with open(self.params.tokenizer_path, "w+", encoding="utf-8") as f:
            # Choose the top top_k words from the vocabulary
            self.tokenizer = keras.preprocessing.text.Tokenizer(
                num_words=self.params.vocab_size,
                split=" ",
                oov_token="<unk>",
                lower=False,
                filters="%",
            )

            self.tokenizer.fit_on_texts(train_captions)
            self.tokenizer.word_index["<pad>"] = 0
            # создание токенайзера и сохранение его в json
            self.tokenizer.index_word[0] = "<pad>"

            tokenizer_json = self.tokenizer.to_json()
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        self.logger.info("\nTokenizer has been created with params:")
        for k, v in self.tokenizer.get_config().items():
            self.logger.info(f"{k} - {v}")

        # Create the tokenized vectors
        train_seqs = self.tokenizer.texts_to_sequences(
            train_captions
        )  # преобразование текста в последовательность чисел
        # print(train_seqs[:100])

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        cap_vector = keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding="post"
        )  # приведение всех
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
            f"Img_to_cap vector is compiled. Max cap length - {self.max_length}"
        )

    def train(self):
        img_keys = list(self.img_to_cap_vector.keys())
        random.shuffle(img_keys)
        # print(img_keys[:100])

        slice_index = int(len(img_keys) * 1)
        img_name_train_keys, img_name_val_keys = (
            img_keys[:slice_index],
            img_keys[slice_index:],
        )

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
            f"Dataset has been splitted:\nIMG_TRAIN - {len(img_name_train)}\nCAP_TRAIN - {len(cap_train)}\nIMG_VAL - {len(img_name_val)}\nCAP_VAL - {len(cap_val)}"
        )

        num_steps = len(img_name_train) // self.params.batch_size

        training_data = LatexDataset(
            img_name_train,
            cap_train,
            self.params.image_size[0],
            self.params.image_size[1],
            torchvision.transforms.ToTensor(),
        )
        #  val_data = LatexDataset(img_name_val, cap_val, self.params.image_size[0], self.params.image_size[1], torchvision.transforms.ToTensor())

        train_dataset = DataLoader(
            training_data,
            batch_size=self.params.batch_size,
            shuffle=True,
            drop_last=False,
            #    pin_memory=True,
            #    num_workers=6,
            #    persistent_workers=True
        )
        # val_dataset = DataLoader(val_data, batch_size=self.params.batch_size, shuffle=True, drop_last=False)

        # for im, cap in train_dataset:
        #     plt.imshow(torchvision.utils.make_grid(im).permute(1, 2, 0))
        #     print(im[0].shape)
        #     print(im[0])
        #     print(cap[0].shape)
        #     print(cap)

        encoder = Encoder(self.params.embedding_dim, device=self.device).to(self.device)
        # cnn_model = get_cnn_model((110, 940))
        decoder = Decoder(
            embedding_dim=self.params.embedding_dim,
            units=self.params.ff_dim,
            vocab_size=self.params.vocab_size,
            device=self.device,
        ).to(self.device)
        model = Seq2Seq(encoder, decoder, self.device)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")

        ckpt = Checkpointing(
            self.params.checkpoint_path,
            max_to_keep=10,
            encoder=encoder,
            decoder=decoder,
            model=model,
            optimizer=optimizer,
        )

        # ckpt.save()
        # ckpt.load()

        # ckpt = tf.train.Checkpoint(encoder=encoder,
        #                            decoder=decoder,
        #                            cnn=cnn_model)
        # ckpt_manager = tf.train.CheckpointManager(
        #     ckpt, self.params.checkpoint_path, max_to_keep=10)

        # self.logger.info(f"OPTIMIZER SUMMARY:\n{optimizer.get_config()}")

        loss_plot = []

        for epoch in range(self.params.epochs):
            start = time.time()
            epoch_loss = 0

            for batch, (img_tensor, target) in tqdm(enumerate(train_dataset)):
                img_tensor, target = img_tensor.to(self.device), target.type(
                    torch.LongTensor
                ).to(self.device)

                optimizer.zero_grad()

                # print(f"img_tensor shape = {img_tensor.shape}")
                # print(f"target shape = {target.shape}")

                # print(img_tensor.numpy())
                # print(target.numpy())

                loss, batch_loss = model(
                    src=img_tensor,
                    trg=target,
                    criterion=criterion,
                    teacher_forcing_ratio=1.0,
                )

                # output = output[1:]
                # target = target[:, 1:].to(device)

                # output = output.view(-1, output.shape[-1])
                # target = target.reshape(-1)

                # loss = criterion(output, target)
                # for idx, t in enumerate(loss):
                #     print(f"{idx}) {t.item()}", end=' ', sep='\n')

                loss.backward()

                optimizer.step()

                epoch_loss += batch_loss.item()

                # total_loss += t_loss
                # print(f"b_loss = {batch_loss}")
                if batch % 100 == 0:
                    # average_batch_loss = batch_loss.numpy() / \
                    #     int(target.shape[1])
                    self.logger.info(
                        f"Epoch {epoch + 1} Batch {batch} Loss = {batch_loss}"
                    )
            # storing the epoch end loss value to plot later
            # loss_plot.append(total_loss / num_steps)

            ckpt.save()
            self.logger.info(f"Epoch {epoch} checkpoint saved!")

            # self.logger.info(
            #     f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
            self.logger.info(f"Time taken for 1 epoch {time.time() - start:.2f} sec\n")

        self.logger.info(f"Training is done!")


class Image2Latex_load:
    loaded = False
    d_p = "images_150/"
    c_p = "5_dataset_large.json"
    VOCAB_SIZE = 300
    IMAGE_COUNT = 100
    IMG_H = 110
    IMG_W = 940
    BATCH_SIZE = 24
    BUFFER_SIZE = 100
    EMBEDDING_DIM = 200
    FF_DIM = 200
    EPOCHS = 100

    def __init__(self, model_name: str, working_path: str = "", device="cpu"):
        self.model_path = (
            os.path.abspath(".") + "/trained_models_rnn_pt/" + working_path + model_name
        )

        self.device = device
        self.tokenizer_path = self.model_path + "/tokenizer.json"
        self.checkpoint_path = self.model_path + "/checkpoints/"
        self.meta_path = self.model_path + "/meta.json"

        if (
            os.path.exists(self.checkpoint_path)
            and len(os.listdir(self.checkpoint_path)) != 0
            and os.path.exists(self.tokenizer_path)
            and os.path.exists(self.meta_path)
        ):
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

            self.cnn_model = Encoder(self.params.embedding_dim, device=self.device).to(
                device
            )
            self.decoder = Decoder(
                embedding_dim=self.params.embedding_dim,
                units=self.params.ff_dim,
                vocab_size=self.params.vocab_size,
                device=self.device,
            ).to(device)
            self.model = Seq2Seq(self.cnn_model, self.decoder, device)

            ckpt = Checkpointing(
                self.params.checkpoint_path,
                encoder=self.cnn_model,
                decoder=self.decoder,
                model=self.model,
            )
            ckpt.load()
            # summary(self.decoder, input_size=[(1, 1), (1, 348, 200), (1, 200), (1, 200)], dtypes=[torch.int64, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], device=self.device)

    def get_model_path(self):
        return self.model_path

    def input(self, inp, val):
        return val if inp == "" else inp

    def train(self):
        print(f"Caption file must be in {os.path.abspath('.')}")
        print(f"Dataset folder must be in {os.path.abspath('.')}/datasets/")
        if self.loaded:
            if input("You will loose your model. Proceed? (y/AB): ") == "y":
                shutil.rmtree(self.model_path)
                self.loaded = False
            else:
                return

        if not self.loaded:
            print("\nPlease enter some data for new model: ")
            try:
                self.d_p = self.input(
                    input(f"dataset_path - Default = {self.d_p}: "), self.d_p
                )
                self.c_p = self.input(
                    input(f"caption file - Default = {self.c_p}: "), self.c_p
                )
                self.VOCAB_SIZE = int(
                    self.input(
                        input(
                            f"vocab size - Default = {self.VOCAB_SIZE} (number of top used words in caps): "
                        ),
                        self.VOCAB_SIZE,
                    )
                )
                self.IMAGE_COUNT = int(
                    self.input(
                        input(
                            f"image_count - Default = {self.IMAGE_COUNT} (number of photos to be used): "
                        ),
                        self.IMAGE_COUNT,
                    )
                )
                self.IMG_H = int(
                    self.input(
                        input(
                            f"image_height - Default = {self.IMG_H} (image final height after preprocessing): "
                        ),
                        self.IMG_H,
                    )
                )
                self.IMG_W = int(
                    self.input(
                        input(
                            f"image_width - Default = {self.IMG_W} (image final width after preprocessing): "
                        ),
                        self.IMG_W,
                    )
                )
                self.BATCH_SIZE = int(
                    self.input(
                        input(f"BATCH_SIZE - Default = {self.BATCH_SIZE}: "),
                        self.BATCH_SIZE,
                    )
                )
                self.BUFFER_SIZE = int(
                    self.input(
                        input(f"BUFFER_SIZE - Default = {self.BUFFER_SIZE}: "),
                        self.BUFFER_SIZE,
                    )
                )
                self.EMBEDDING_DIM = int(
                    self.input(
                        input(f"embedding_dim - Default = {self.EMBEDDING_DIM}: "),
                        self.EMBEDDING_DIM,
                    )
                )
                self.FF_DIM = int(
                    self.input(
                        input(
                            f"ff_dim - Default = {self.FF_DIM} (feed-forward unit dimension): "
                        ),
                        self.FF_DIM,
                    )
                )
                self.EPOCHS = int(
                    self.input(
                        input(f"EPOCHS - Default = {self.EPOCHS}: "), self.EPOCHS
                    )
                )
            except:
                raise TypeError("Model params initialization failed")

            self.loaded = True
            self.dataset_path = os.path.abspath(".") + "/datasets/" + self.d_p
            self.caption_path = os.path.abspath(".") + "/" + self.c_p

            train = Training(
                device=self.device,
                model_path=self.model_path,
                dataset_path=self.dataset_path,
                caption_path=self.caption_path,
                tokenizer_path=self.tokenizer_path,
                checkpoint_path=self.checkpoint_path,
                meta_path=self.meta_path,
                vocab_size=self.VOCAB_SIZE,
                image_count=self.IMAGE_COUNT,
                image_size=(self.IMG_H, self.IMG_W),
                batch_size=self.BATCH_SIZE,
                buffer_size=self.BUFFER_SIZE,
                embedding_dim=self.EMBEDDING_DIM,
                ff_dim=self.FF_DIM,
                epochs=self.EPOCHS,
            )

            train.auto_train()

    def predict(self, image_path: str):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        return self.greedy_decoder(image_path=image_path)
        # return self.beam_decoder(image_path, beam_width=10, temperature=0.5)

    def index(self, array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx[1]

    def find_n_best(self, array, n):
        probs = [np.partition(array[0], i)[i] for i in range(-1, -n - 1, -1)]
        ids = [self.index(array, p) for p in probs]
        return [[prob, id] for prob, id in zip(probs, ids)]

    def beam_decoder(self, image, beam_width=10, temperature=0.3):
        raise NotImplementedError
        sample_img = self.load_image(image)

        # Pass the image to the CNN
        img = tf.expand_dims(sample_img, 0)
        img = self.cnn_model(img)

        # Pass the image features to the Transformer encoder
        encoded_img = self.encoder(img, training=False, mask=None)

        # Generate the caption using the Transformer decoder
        decoded_caption = "<start> "

        tokenized_caption = tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenizer.texts_to_sequences([decoded_caption]),
            padding="post",
            maxlen=self.params.max_len,
        )[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = self.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask, b_s_t=None
        )
        predictions = tf.expand_dims((predictions[0, 0, :] / temperature), 0).numpy()
        init = self.find_n_best(predictions, beam_width)
        results = [
            [
                obj[0],
                obj[1],
                decoded_caption + self.tokenizer.index_word[int(obj[1])] + " ",
            ]
            for obj in init
        ]  # 0 - prob ; 1 - id ; 2 - hidden

        for i in range(1, self.params.max_len):
            tmp_res = []

            for r in results:
                tmp_tokenized_caption = tf.keras.preprocessing.sequence.pad_sequences(
                    self.tokenizer.texts_to_sequences([r[2]]),
                    padding="post",
                    maxlen=self.params.max_len,
                )[:, :-1]
                tmp_mask = tf.math.not_equal(tmp_tokenized_caption, 0)
                tmp_preds = self.decoder(
                    tmp_tokenized_caption,
                    encoded_img,
                    training=False,
                    mask=tmp_mask,
                    b_s_t=None,
                )

                for obj in self.find_n_best(
                    tf.expand_dims((tmp_preds[0, i, :] / temperature), 0).numpy(),
                    beam_width,
                ):
                    tmp_res.append(
                        [
                            obj[0] + r[0],
                            obj[1],
                            r[2] + self.tokenizer.index_word[int(obj[1])] + " ",
                        ]
                    )  # multiplied scores, curr id, hidden, prev id

            results.clear()
            tmp_res.sort(reverse=True, key=lambda x: x[0])
            # attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            for el in range(beam_width):
                results.append(tmp_res[el])

            # for res, att in zip(results, attention_plots):
            #     att[i] = tf.reshape(res[5], (-1,)).numpy()

            # if any(self.tokenizer.index_word[int(results[i][1])] == '<end>' for i in range(len(results))):
            #     break

            if all(["<end>" in r[2] for r in results]):
                break

            # print()

        # for el in results:
        #     tf.print(el[3] + "\n")
        # tf.print(results[0][3])
        # return [results[0][3]], None
        return [el[2] for el in results]

    def greedy_decoder(self, image_path):
        with torch.no_grad():
            img = self.load_image(
                image_path, torchvision.transforms.ToTensor()
            ).unsqueeze(0)

            hidden_state, cell_state = self.decoder.initial_lstm_state(1)

            # Pass the image to the CNN
            features = self.cnn_model(img)

            # Generate the caption using the Transformer decoder
            dec_input = torch.unsqueeze(
                torch.tensor([self.tokenizer.word_index["<start>"]]), 0
            )
            result = []
            for i in range(self.params.max_len):
                output, hidden_state, cell_state, _ = self.decoder(
                    dec_input, features, hidden_state, cell_state
                )

                sampled_token_index = np.argmax(output[0, :])
                sampled_token = self.tokenizer.index_word[sampled_token_index.item()]
                if sampled_token == "<end>":
                    break
                result.append(sampled_token)
                dec_input = torch.unsqueeze(torch.tensor([sampled_token_index]), 0)

            # decoded_caption = decoded_caption.replace("<start> ", "")
            # decoded_caption = decoded_caption.replace(" <end>", "").strip()
            return " ".join(result)

    def load_image(self, img_name, transform):
        # raise NotImplementedError
        image = read_image(img_name, ImageReadMode.RGB)
        image = make_fix_size(
            image.permute(1, 2, 0).numpy(),
            self.params.image_size[1],
            self.params.image_size[0],
            False,
        )
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

    def random_predict(self, number=5):
        if not self.loaded:
            raise ValueError("Model is not loaded!")

        with open("5_dataset_large_val.json", "r+") as file:
            capt = json.load(file)["annotations"]

        parameters = {
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "figure.subplot.hspace": 0.999,
            "figure.subplot.wspace": 0.999,
        }
        plt.rcParams.update(parameters)
        # print(plt.rcParams.keys())

        images = random.choices(capt, k=number)
        plotting = []
        for im in tqdm(images):
            try:
                image_path = (
                    "images_150_val/"
                    + im["image_id"]
                    + ".png"
                )
                plotting.append(
                    [
                        im["caption"],
                        self.predict(
                            decoder_type="greedy",
                            image_path=image_path,
                            temperature=0.5,
                        ),
                    ]
                )  # real ; pred ; plt image
            except:
                continue

        l_dist = []
        b = []
        for plot in tqdm(plotting):
            try:
                edit_dist = levenshteinDistance(plot[0], plot[1])
                bleu = bleu_score(plot[0], plot[1])

                l_dist.append(edit_dist)
                b.append(bleu)
            except:
                continue
            # print(plot[1])
        # plt.show()
        print(f"MEAN LEV. DIST = {np.mean(l_dist)}")
        print(f"MEAN BLEU = {np.mean(bleu)}")


device = enable_gpu(False)
van = Image2Latex_load("model_latex_pt_5", device=device)

# print(van.predict("25356.png"))
