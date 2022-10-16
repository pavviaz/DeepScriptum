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
import numpy as np
import torch
from torch import Tensor, isnan
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.logger_init import log_init


def enable_gpu(enable:bool):
    if not enable:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
            [print(f"{idx + 1}) {ckpt - 1}", sep=", ", end=" ") for idx, ckpt in enumerate(checkpoing_idx)]
        
        checkpoing_idx = checkpoing_idx[-1] if not idx else idx + 1
        
        try:
            checkpoint = torch.load(os.path.join(self.path, f"checkpoint_{checkpoing_idx - 1}.tar"))
            [v.load_state_dict(checkpoint[k], strict=False) for k, v in self.modules.items()]
            [v.load_state_dict(checkpoint[k]) for k, v in self.optims.items()]
        except Exception as e:
            raise e
        
        if return_idx:
            return checkpoing_idx


class Scale_Dataset(Dataset):
    def __init__(self, dataset_path, speech_folder_name="speech", no_speech_folder_name="no_speech", transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.speech_folder_name = speech_folder_name
        self.no_speech_folder_name = no_speech_folder_name
        self.data = self.__load_features()

    def __load_features(self):
        speech_path = os.path.join(self.dataset_path, self.speech_folder_name)
        no_speech_path = os.path.join(self.dataset_path, self.no_speech_folder_name)
        if not os.path.exists(speech_path) or not os.path.exists(no_speech_path):
            raise FileExistsError("Something wrong with dataset directory")
        
        data = []
        data.extend([ (np.load( os.path.join( speech_path, el) ), 1) for el in tqdm(os.listdir(speech_path))])  # loading speech
        data.extend([ (np.load( os.path.join( no_speech_path, el ) ), 0) for el in tqdm(os.listdir(no_speech_path))])  # loading no speech
        # data = []
        # data.extend([ ( os.path.join( speech_path, el) , 1) for el in tqdm(os.listdir(speech_path))])  # loading speech
        # data.extend([ ( os.path.join( no_speech_path, el  ), 0) for el in tqdm(os.listdir(no_speech_path))])  # loading no speech
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        # return np.load(features), label
        return features, label
    
    
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
            # self.logger.info(
            #     f"ENCODER:\n{inspect.getsource(Encoder)}")
            # self.logger.info(
            #     f"DECODER:\n{inspect.getsource(Decoder)}")
            # self.logger.info(
            #     f"SEQ2SEQ:\n{inspect.getsource(Seq2SeqTransformer)}")
            # self.logger.info(f"RESIZE_HEIGHT: {RESIZED_IMG_H}")
            # self.logger.info(f"RESIZE_WIDTH: {RESIZED_IMG_W}")
            # self.logger.info(f"DATA_SPLIT_INDEX: {DATA_SPLIT}")
            for k, v in kwargs.items():
                self.logger.info(f"{k}: {v}")
            self.logger.info(
                f"-----------------------------------------------------")
        
        else:
            self.logger = log_init(self.params.model_path, "model_log", True)
            self.logger.info("Params have been loaded successfully for training from checkpoint!")
            
    def auto_train(self):
        self.train()
    
    def train_from_ckpt(self):
        self.train()

    # def data_preprocess_from_ckpt(self):
    #     with open(self.params.caption_path, 'r') as f:
    #         annotations = json.load(f)

    #     pass
    
    # def data_preprocess(self):
    #     with open(self.params.caption_path, 'r') as f:
    #         annotations = json.load(f)

    #     pass
        
    def train(self):
        # random.shuffle(img_keys)

        # self.logger.info(
        #     f"Dataset has been splitted:\nIMG_TRAIN - {len(img_name_train)}\nCAP_TRAIN - {len(cap_train)}\nIMG_VAL - {len(img_name_val)}\nCAP_VAL - {len(cap_val)}")

        # num_steps = len(img_name_train) // self.params.batch_size

        # training_data = VAD_Dataset(self.params.label_path,
        #                             self.params.audios_path,
        #                             self.params.features_type)
        
        training_data = Scale_Dataset("C:\\Users\\shace\\Desktop\\aga\\plp\\stride_32_ms")
        
        train_dataset = DataLoader(training_data, 
                                   batch_size=self.params.batch_size, 
                                   shuffle=True, 
                                   drop_last=False, 
                                #    pin_memory=True, 
                                #    num_workers=6,
                                #    persistent_workers=True
                                   )
        
        #  val_data = LatexDataset(img_name_val, cap_val, self.params.image_size[0], self.params.image_size[1], torchvision.transforms.ToTensor())
        

        # val_dataset = DataLoader(val_data, batch_size=self.params.batch_size, shuffle=True, drop_last=False)
        
        # for im, cap in train_dataset:
        #     plt.imshow(torchvision.utils.make_grid(im).permute(1, 2, 0))
        #     print(im[0].shape)
        #     print(im[0])
        #     print(cap[0].shape)
        #     print(cap)

        criterion = nn.CrossEntropyLoss()
            
        model = torch_model().to(self.device)
                
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        self.start_ckpt_id = 0
        ckpt = Checkpointing(self.params.checkpoint_path,
                             max_to_keep=10,
                             model=model,
                             optimizer=optimizer)
        if self.params.train_from_ckpt:
            self.start_ckpt_id = ckpt.load(return_idx=True)
                
        loss_plot = []

        for epoch in range(self.start_ckpt_id, self.params.epochs):
            start = time.time()
            epoch_loss = 0

            for (batch, (img_tensor, target)) in tqdm(enumerate(train_dataset)):
                img_tensor, target = img_tensor.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                prediction = model(img_tensor)

                loss = criterion(prediction, target)
                
                if isnan(loss):
                    
                    print(img_tensor)
                    print(prediction)
                
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


class VAD_load:
    loaded = False
    audios_path = "audios\\"
    # label_path = "ava_final_label_mini.json"
    label_path = "ava_final_label_medium.json"
    # label_path = "ava_final_label.json"
    features_type = "log_mel"
    BATCH_SIZE = 24
    EPOCHS = 100

    def __init__(self, model_name: str, working_path: str = "", ckpt_idx=None, device='cpu'):
        self.model_path = os.path.abspath(
            ".") + "\\trained_models\\" + working_path + model_name
        
        self.device = device
        self.checkpoint_path = self.model_path + "\\checkpoints\\"
        self.meta_path = self.model_path + "\\meta.json"

        if os.path.exists(self.checkpoint_path) and len(os.listdir(self.checkpoint_path)) != 0: # and os.path.exists(self.tokenizer_path) and os.path.exists(self.meta_path):
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
                    
            self.model = torch_model()

            ckpt = Checkpointing(self.params.checkpoint_path,
                                 model=self.model)
            ckpt.load(idx=ckpt_idx)

    def input(self, inp, val):
        return val if inp == "" else inp

    def train(self):
        print(f"Caption file must be in {os.path.abspath('.')}")
        print(f"Dataset folder must be in {os.path.abspath('.')}\\dataset\\")
        if self.loaded:
            if input("You will loose your model. Proceed? (y/AB): ") == "y":
                shutil.rmtree(self.model_path)
                self.loaded = False
            else:
                return

        if not self.loaded:
            print("\nPlease enter some data for new model: ")
            try:
                self.audios_path = self.input(
                    input(f"dataset_path - Default = {self.audios_path}: "), self.audios_path)
                self.label_path = self.input(
                    input(f"caption file - Default = {self.label_path}: "), self.label_path)
                self.features_type = self.input(
                    input(f"features type - Default = {self.features_type}: "), self.features_type)
                self.BATCH_SIZE = int(self.input(
                    input(f"BATCH_SIZE - Default = {self.BATCH_SIZE}: "), self.BATCH_SIZE))
                self.EPOCHS = int(self.input(
                    input(f"EPOCHS - Default = {self.EPOCHS}: "), self.EPOCHS))
            except:
                raise TypeError("Model params initialization failed")

            self.loaded = True
            self.label = os.path.abspath(".") + "\\dataset\\" + self.label_path
            self.audios = os.path.abspath(".") + f"\\dataset\\{self.audios_path}"

            train = Training(device=self.device,
                             model_path=self.model_path,
                             label_path=self.label,
                             meta_path=self.meta_path,
                             audios_path=self.audios,
                             features_type=self.features_type,
                             checkpoint_path=self.checkpoint_path,
                             batch_size=self.BATCH_SIZE,
                             epochs=self.EPOCHS,
                             train_from_ckpt=False)

            train.auto_train()
            
    def train_from_ckpt(self):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        train = Training(device=self.device,
                         model_path=self.params.model_path,
                         label_path=self.params.label_path,
                         audios_path=self.params.audios_path,
                         checkpoint_path=self.params.checkpoint_path,
                         features_type=self.params.features_type,
                         meta_path=self.params.meta_path,
                         batch_size=self.params.batch_size,
                         epochs=self.params.epochs,
                         train_from_ckpt=True)

        train.train_from_ckpt()
            
    def predict(self):
        pass
     
     
if __name__ == "__main__": 
    torch.set_printoptions(profile="full")
    device = enable_gpu(True)
    van = VAD_load("vad_model_8_plp_adamW", device=device)
    van.train()
    van.train_from_ckpt()