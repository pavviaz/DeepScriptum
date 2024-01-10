import inspect
import os
import random
import time
from collections import defaultdict
from io import StringIO

import matplotlib

os.environ["MATPLOTLIB_BACKEND"] = matplotlib.get_backend()

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.artifacts import download_artifacts
from munch import munchify
from torch.utils.data import DataLoader
import yaml

from compute_metrics import MetricsTracker
# from data_utils.extract_features_from_dataset import FeatureExtractor
# from data_utils.extract_features_from_dataset import CONFIG_DUMP_NAME
# from model_constructor import ModelFeatured
from matplotlib import pyplot as plt
# from meta_dicts import FEATURES, LOSS_FUNCS, MODELS, OPTIMS
# from dataloader import LoadDataset
from tqdm import tqdm
from utils import disable_plot_show, enable_plot_show, flatten_dict, log_init


class ModelManager:
    def __init__(
        self,
        config_path=None,
        model_name=None,
        default_mlflow_host="http://127.0.0.1:5000",
    ):
        self.log_str = StringIO()
        self.logger = log_init(self.log_str)

        if not model_name and not config_path:
            error_msg = "Either 'model_name' or 'config_path' must be specified"
            self.__invoke_exception(error_msg, ValueError)

        mlflow_host = os.getenv("MLFLOW_HOST")
        mlflow.set_tracking_uri(mlflow_host if mlflow_host else default_mlflow_host)
        self.client = MlflowClient()

        if model_name:
            self.model, run_id, inp_shape = self.__load_model(model_name)

            self.logger.info(f"Model '{model_name}' have been found")

            cfg = download_artifacts(
                run_id=run_id, artifact_path="configs/train_config.yaml"
            )
            data_cfg = download_artifacts(
                run_id=run_id, artifact_path="configs/data_config.yaml"
            )

            self.cfg = munchify(self.__config_reader(cfg))
            self.data_cfg = munchify(self.__config_reader(data_cfg))
            self.input_shape = inp_shape
            return

        config = self.__config_reader(config_path)
        self.cfg = munchify(config)

        self.logger.info("training params:")
        for k, v in self.cfg.items():
            self.logger.info(f"k: {k} --- v: {v.toDict() if v else v}")

        mandatory_params = [self.cfg.training, self.cfg.models_params]
        [self.__check_config(d) for d in mandatory_params]

        if "seed" in self.cfg.general and self.cfg.general.seed:
            self.__seed_everything()

        if self.cfg.general.continue_from and self.cfg.general.finetune_from:
            error_msg = "Either one of 'continue_from' and 'finetune_from' \
                         params, or none of them must be specified"
            self.__invoke_exception(error_msg, ValueError)
        elif self.cfg.general.continue_from:
            pass
        elif self.cfg.general.finetune_from:
            pass
        else:
            if not os.path.exists(self.cfg.training.dataset_path):
                error_msg = f"Dataset on '{self.cfg.training.dataset_path}' \
                              does not exist"
                self.__invoke_exception(error_msg, OSError)

            data_cfg = self.__config_reader(
                os.path.join(self.cfg.training.dataset_path, CONFIG_DUMP_NAME)
            )
            self.data_cfg = munchify(data_cfg)

            if not self.cfg.training.model_to_use in self.cfg.models_params:
                error_msg = f"No params specified for \
                             '{self.cfg.training.model_to_use}' in config file"
                self.__invoke_exception(error_msg, ValueError)

            m = MODELS.get(self.cfg.training.model_to_use, None)
            if not m:
                error_msg = f"Not exisiting model type \
                              '{self.cfg.training.model_to_use}'"
                self.__invoke_exception(error_msg, ValueError)

            if FEATURES[self.data_cfg.general.feature_type]["type"] != m["feature"]:
                error_msg = f"Passing not appropriate feature type \
                              '{self.data_cfg.general.feature_type}' \
                              for model '{self.cfg.training.model_to_use}'"
                self.__invoke_exception(error_msg, ValueError)
            self.m = m["cls"]

            self.device = self.__enable_gpu(self.cfg.general.device)

            self.out_classes = len(self.data_cfg.general.training_tokens)
            self.model = self.m(
                out_classes=self.out_classes,
                **self.cfg.models_params[self.cfg.training.model_to_use],
            )

        # if self.cfg.prepare_dataset_conf:
        #     self.cfg.training.dataset_dir = -1
        #     pass

    @staticmethod
    def __enable_gpu(device):
        if "cuda" in device:
            return device if torch.cuda.is_available() else "cpu"
        return "cpu"

    def __config_reader(self, config_path):
        if not os.path.exists(config_path):
            error_msg = f"Config file on {config_path} path does not exist"
            self.__invoke_exception(error_msg, OSError)

        with open(config_path) as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        return config

    def __invoke_exception(self, msg, exc):
        self.logger.error(msg)
        mlflow.end_run()

        raise exc(msg)

    def __check_config(self, _dict):
        for k, v in _dict.items():
            if isinstance(v, dict):
                self.__check_config(v)
            if v is None:
                error_msg = f"Value for {k} must be specified"
                self.__invoke_exception(error_msg, ValueError)

    def __seed_everything(self):
        random.seed(self.cfg.general.seed)
        np.random.seed(self.cfg.general.seed)
        torch.manual_seed(self.cfg.general.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __load_model(self, name):
        try:
            filter_string = f"name='{name}'"
            model = self.client.search_registered_models(
                filter_string=filter_string, max_results=1
            )

            metainf = model[0].latest_versions[-1]
            model_uri = f"models:/{name}/{metainf.version}"

            inp_shape = eval(
                mlflow.models.get_model_info(model_uri).signature.to_dict()["inputs"]
            )[0]["tensor-spec"]["shape"]
            inp_shape[0] = 1  # batch size

            return mlflow.pytorch.load_model(model_uri), metainf.run_id, inp_shape
        except:
            error_msg = f"No specified model '{name}' found"
            self.__invoke_exception(error_msg, OSError)

    def __log_essential_files(self):
        mlflow.log_params(flatten_dict(self.cfg, parent_key="training"))
        mlflow.log_params(flatten_dict(self.data_cfg, parent_key="data"))

        mlflow.log_dict(self.cfg, "configs/train_config.yaml")
        mlflow.log_dict(self.data_cfg, "configs/data_config.yaml")

        mlflow.log_artifact(os.path.realpath(__file__), "training_scripts")
        mlflow.log_text(
            inspect.getsource(MetricsTracker), "training_scripts/compute_metrics.py"
        )
        mlflow.log_text(
            "\n".join([str(FEATURES), str(MODELS), str(LOSS_FUNCS), str(OPTIMS)]),
            "training_scripts/meta_dicts.py",
        )
        mlflow.log_text(inspect.getsource(self.m), "training_scripts/model.py")
        mlflow.log_text(
            inspect.getsource(FeatureExtractor), "training_scripts/extractor.py"
        )

    def __log_all_metrics(self, stage, ep, step, all_steps, preserve=False):
        cur_step_metrics = (
            self.metrics_tracker.compute_current()
            if not preserve
            else self.metrics_tracker.compute_all()
        )
        cur_step_metrics.pop("confusion_matrix")

        if preserve:
            [
                self.preservers[stage][k].append(float(v))
                for k, v in cur_step_metrics.items()
            ]

        cur_step_metrics = {
            f"{stage}_{k}": float(v) for k, v in cur_step_metrics.items()
        }
        mlflow.log_metrics(metrics=cur_step_metrics, step=(step + ep * all_steps))
        mlflow.log_figure(
            figure=self.metrics_tracker.plot_cm(),
            artifact_file=f"{stage}_conf_mat/epoch_{ep + 1}_step_{step}.png",
        )

        msg = " --- ".join([f"{k} = {v}" for k, v in cur_step_metrics.items()])

        self.logger.info(
            f"{stage} epoch {ep + 1}, Step {step} {'(final)' if step + 1 == all_steps else ''}: {msg}"
        )

    def __train_step(self, img_tensor, target):
        img_tensor, target = img_tensor.type(torch.float32).to(
            self.device
        ), target.type(torch.int64).to(self.device)

        self.optimizer.zero_grad()
        prediction = self.model(img_tensor)

        loss = self.criterion(prediction, target)

        loss.backward()
        self.optimizer.step()

        self.metrics_tracker.update(
            y_pred=torch.argmax(prediction, dim=-1), y_true=target, loss_val=loss.item()
        )

    def __val_step(self, img_tensor, target):
        img_tensor, target = img_tensor.type(torch.float32).to(
            self.device
        ), target.type(torch.int64).to(self.device)

        prediction = self.model(img_tensor)
        loss = self.criterion(prediction, target)

        self.metrics_tracker.update(
            y_pred=torch.argmax(prediction, dim=-1), y_true=target, loss_val=loss.item()
        )

    def train(self):
        train_data_dir = os.path.join(
            self.cfg.training.dataset_path, self.data_cfg.general.train_folder
        )
        val_data_dir = os.path.join(
            self.cfg.training.dataset_path, self.data_cfg.general.val_folder
        )
        test_data_dir = os.path.join(
            self.cfg.training.dataset_path, self.data_cfg.general.test_folder
        )

        if any(
            not os.path.exists(d_dir)
            for d_dir in [train_data_dir, val_data_dir, test_data_dir]
        ):
            error_msg = "Train, validation or test dataset folders are not existing"
            self.__invoke_exception(error_msg, OSError)

        training_data = LoadDataset(train_data_dir)
        train_dataset = DataLoader(
            training_data,
            batch_size=None,
            shuffle=True,
            drop_last=False,
            **self.cfg.general.cuda_params,
        )
        train_steps = len(train_dataset)

        val_data = LoadDataset(val_data_dir)
        val_dataset = DataLoader(
            val_data,
            batch_size=None,
            shuffle=True,
            drop_last=False,
            **self.cfg.general.cuda_params,
        )
        val_steps = len(val_dataset)

        test_data = LoadDataset(test_data_dir)
        test_dataset = DataLoader(
            test_data,
            batch_size=None,
        )

        if not self.cfg.training.loss_func.type in LOSS_FUNCS:
            error_msg = f"Not exisiting loss type \
                          '{self.cfg.training.loss_func.type}'"
            self.__invoke_exception(error_msg, ValueError)

        if not self.cfg.training.optimizer.type in OPTIMS:
            error_msg = f"Not exisiting optimizer type \
                          '{self.cfg.training.optimizer.type}'"
            self.__invoke_exception(error_msg, ValueError)

        self.criterion = LOSS_FUNCS[self.cfg.training.loss_func.type](
            **self.cfg.training.loss_func.params
        )
        self.optimizer = OPTIMS[self.cfg.training.optimizer.type](
            self.model.parameters(), **self.cfg.training.optimizer.params
        )

        self.start_ckpt_id = 0
        self.metrics_tracker = MetricsTracker(
            num_classes=self.out_classes, labels=self.data_cfg.general.training_tokens
        )
        self.preservers = {"train": defaultdict(list), "val": defaultdict(list)}

        with mlflow.start_run(description=self.cfg.general.description) as run:
            self.__log_essential_files()

            self.model.to(self.device)

            best_f1 = -1
            signature = None
            for epoch in range(self.start_ckpt_id, self.cfg.training.epochs):
                start = time.time()

                self.model.train()
                for batch, (img_tensor, target) in tqdm(enumerate(train_dataset)):
                    self.__train_step(img_tensor, target)
                    if batch % 100 == 0:
                        self.__log_all_metrics(
                            stage="train", ep=epoch, step=batch, all_steps=train_steps
                        )

                self.__log_all_metrics(
                    stage="train",
                    ep=epoch,
                    step=batch,
                    all_steps=train_steps,
                    preserve=True,
                )
                self.metrics_tracker.reset()

                self.model.eval()
                with torch.no_grad():
                    for batch, (img_tensor, target) in tqdm(enumerate(val_dataset)):
                        self.__val_step(img_tensor, target)
                        if batch % 100 == 0:
                            self.__log_all_metrics(
                                stage="val", ep=epoch, step=batch, all_steps=val_steps
                            )

                self.__log_all_metrics(
                    stage="val",
                    ep=epoch,
                    step=batch,
                    all_steps=val_steps,
                    preserve=True,
                )
                self.metrics_tracker.reset()

                if not signature:
                    self.input_shape = img_tensor.numpy().shape
                    signature = infer_signature(
                        model_input=img_tensor.numpy(),
                        model_output=self.model(img_tensor.to(self.device))
                        .detach()
                        .cpu()
                        .numpy(),
                    )

                if self.cfg.general.save_only_best:
                    val_macro_f1 = self.preservers["val"]["macro_f1"][-1]
                    if val_macro_f1 > best_f1:
                        best_f1 = val_macro_f1
                        mlflow.pytorch.log_model(
                            pytorch_model=self.model,
                            artifact_path="basic_model",
                            signature=signature,
                            registered_model_name=self.cfg.general.model_name,
                        )
                        # add quantization
                        self.logger.info(
                            f"Epoch {epoch} checkpoint saved, f1 = {best_f1}"
                        )
                else:
                    mlflow.pytorch.log_model(
                        pytorch_model=self.model,
                        artifact_path="basic_model",
                        signature=signature,
                        registered_model_name=self.cfg.general.model_name,
                    )
                    self.logger.info(f"Epoch {epoch} checkpoint saved")

                self.logger.info(
                    f"Time taken for 1 epoch {time.time() - start:.2f} sec\n"
                )

            disable_plot_show()
            for k_train in self.preservers["train"]:
                fig, ax = plt.subplots(figsize=(6, 6))

                ax.set_xlabel("Epochs", fontsize=6)
                ax.set_ylabel(k_train, fontsize=6)
                ax.set_title(f"Train & val {k_train}", fontsize=6)

                train = self.preservers["train"][k_train]
                val = self.preservers["val"][k_train]

                ax.plot(range(len(train)), train, label="train")
                ax.plot(range(len(val)), val, label="val")
                plt.legend(prop={"size": 6})

                mlflow.log_figure(
                    figure=fig,
                    artifact_file=f"train_val_metrics/{k_train}.png",
                )
            enable_plot_show()

            self.logger.info(f"Training is done!")

            self.logger.info(f"Testing...")
            self.model, _, _ = self.__load_model(self.cfg.general.model_name)

            self.model.eval()
            with torch.no_grad():
                for batch, (img_tensor, target) in tqdm(enumerate(test_dataset)):
                    img_tensor, target = img_tensor.type(torch.float32).to(
                        self.device
                    ), target.type(torch.int64).to(self.device)

                    prediction = self.model(img_tensor)

                    self.metrics_tracker.update(
                        torch.argmax(prediction, dim=-1), target, 0
                    )

            self.__log_all_metrics(stage="test", ep=0, step=batch, all_steps=0)

            mlflow.log_text(
                text=self.log_str.getvalue(), artifact_file="training_log.txt"
            )

    def export_jit_model(self):
        f_model = self.get_featured_model()
        f_model.to("cpu")
        f_model.eval()

        dummy_inp = torch.rand(*self.input_shape)

        traced_model = torch.jit.trace(self.model, dummy_inp)

        os.makedirs(self.cfg.general.export_path, exist_ok=True)
        save_path = os.path.join(
            os.path.normpath(self.cfg.general.export_path),
            f"{self.cfg.general.model_name}.pt",
        )
        traced_model.save(save_path)

    def get_featured_model(self):
        return ModelFeatured(self.model, self.data_cfg)

    def get_model(self):
        return self.model

    def get_training_cfg(self):
        return self.cfg

    def get_data_cfg(self):
        return self.data_cfg

    def get_model_inp_shape(self):
        return self.input_shape


if __name__ == "__main__":
    manager = ModelManager("training_configs/training_config.yaml")
