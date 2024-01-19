import os
import pickle
import codecs
import yaml

import mlflow
import torch
from mlflow import MlflowClient
from ultralytics import YOLO
from mlflow.artifacts import download_artifacts
from munch import munchify

from meta_dicts import TYPE_TO_TRAINER


class ModelManager:
    def __init__(
        self,
        config_path=None,
        run_name=None,
        run_id=None,
        mlflow_host="http://127.0.0.1:5000",
    ):
        if not run_name and not run_id and not config_path:
            error_msg = (
                "Either 'model_name' or 'run_id' or 'config_path' must be specified"
            )
            self.__invoke_exception(error_msg, ValueError)

        os.environ["MLFLOW_TRACKING_URI"] = mlflow_host
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "Default"
        self.client = MlflowClient()

        if run_name or run_id:
            self.model, run_id = self.__load_model(run_name, run_id)
            return

        config = self.__config_reader(config_path)
        self.cfg = munchify(config)

        if len(self.cfg) > 2:
            error_msg = f"Only 'general' key with one of following \
            keys must be specified in the config: {TYPE_TO_TRAINER.keys()}"
            self.__invoke_exception(error_msg, OSError)

        self.trainer = TYPE_TO_TRAINER[[el for el in self.cfg if el != "general"][0]]

        if not os.path.exists(self.cfg.general.dataset_cfg_path):
            error_msg = f"Dataset cfg on '{self.cfg.general.dataset_cfg_path}' \
                          does not exist"
            self.__invoke_exception(error_msg, OSError)

        self.cfg.general.device = self.__enable_gpu(self.cfg.general.device)

    @staticmethod
    def __enable_gpu(device):
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.set_device(0)
            return device
        return "cpu"

    def __config_reader(self, config_path):
        if not os.path.exists(config_path):
            error_msg = f"Config file on {config_path} path does not exist"
            self.__invoke_exception(error_msg, OSError)

        with open(config_path) as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        return config

    def __invoke_exception(self, msg, exc):
        mlflow.end_run()

        raise exc(msg)

    def __load_model(self, name, run_id):
        try:
            filter_string = (
                f"attributes.run_id='{run_id}'"
                if run_id
                else f"attributes.run_name='{name}'"
            )
            run = self.client.search_runs(
                experiment_ids=0, filter_string=filter_string, max_results=1
            )[0]

            run_id = run.info.run_id
            model_type = run.data.tags["model_type"]

            weights = download_artifacts(run_id=run_id, artifact_path="last.pt")
            with open(weights) as w:
                buf = w.read()
            model = pickle.loads(codecs.decode(buf.encode(), "base64"))

            match model_type:
                case "yolo":
                    m = YOLO()
                    m.model = model
                    m.training = False
                    m.model.training = False
                case "detectron":
                    pass

            return m, run_id
        except:
            error_msg = f"Run named {name} or having id {run_id} is not found"
            self.__invoke_exception(error_msg, OSError)

    def train(self):
        self.trainer(self.cfg)

    def get_model(self):
        return self.model


if __name__ == "__main__":
    manager = ModelManager(run_id="d4160fdc084540ab8db281e50cf328c2", mlflow_host="http://89.105.198.171:5000")

    # manager.train()
