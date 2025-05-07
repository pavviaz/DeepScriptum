import inspect
import os

import torch
import yaml
from munch import munchify
from clearml import Task, TaskTypes, InputModel

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from meta_dicts import LOSS_FUNCS, MODELS, OPTIMS, DATAMODULES, PREPROCESSORS
from utils import flatten_dict, ClearMLModelCheckpoint
from logger import TrainingLogger


class ModelManager:
    def __init__(self, config_path: str):
        """
        Initialize the ModelManager object.
        TODO
        """
        if not os.path.exists(config_path):
            raise OSError(f"Config file on {config_path} path does not exist")

        with open(config_path, encoding="utf-8") as c:
            config = yaml.load(c, Loader=yaml.FullLoader)
        self.cfg = munchify(config)

        self.task = Task.init(
            project_name=self.cfg.general.project_name,
            task_name=self.cfg.general.task_name,
            task_type=TaskTypes.training,
            auto_connect_frameworks={"pytorch": False},
        )
        self.task.set_comment(self.cfg.general.description)
        self.log_obj = TrainingLogger(self.task.get_logger())

        if "seed" in self.cfg.general and self.cfg.general.seed:
            Task.set_random_seed(self.cfg.general.seed)

        self.log_obj.info("training params:")
        self.log_obj.info(f"k: task_id --- v: {self.task.id}")
        for k, v in self.cfg.items():
            self.log_obj.info(f"k: {k} --- v: {v.toDict() if v else v}")

        mandatory_params = [self.cfg.training, self.cfg.models_params]
        [self.__check_config(d) for d in mandatory_params]

        if not self.cfg.training.datasets.train_datasets.paths:
            error_msg = f"Dataset on '{self.cfg.training.dataset_path}' \
                          does not exist"
            self.log_obj.invoke_exception(error_msg, OSError, self.task)

        if all(
            not v in self.cfg.training
            for v in ["model_to_use", "model_id", "model_name", "model_metadata"]
        ):
            error_msg = f"At least one of model params must be specified: \
                          'model_to_use', 'model_id', 'model_name', 'model_metadata'"
            self.log_obj.invoke_exception(error_msg, ValueError, self.task)

        # FROM ZERO
        if "model_to_use" in self.cfg.training:
            self.m = MODELS.get(self.cfg.training.model_to_use)
            if not self.m:
                error_msg = f"Not exisiting model type \
                            '{self.cfg.training.model_to_use}'"
                self.log_obj.invoke_exception(error_msg, ValueError, self.task)

            if not self.cfg.training.datamodule in self.m["dms"]:
                error_msg = f"Passing not appropriate datamodule \
                            '{self.cfg.training.datamodule}' \
                            for model '{self.cfg.training.model_to_use}'"
                self.log_obj.invoke_exception(error_msg, ValueError, self.task)

            self.model = self.m["cls"](
                self.cfg.models_params[self.cfg.training.model_to_use],
                self.cfg.training.optimizer,
                self.log_obj,
                self.task,
                self.cfg.general.ckpt_monitor_metric,
            )

            self.cfg.training.metadata["init_task_id"] = self.task.id
            self.cfg.training.metadata["init_model_id"] = None

        # FROM PRETRAIN
        else:
            if "model_id" in self.cfg.training and self.cfg.training.model_id:
                m = InputModel(
                    project=self.cfg.general.project_name,
                    model_id=self.cfg.training.model_id,
                )
            else:
                m = InputModel.query_models(
                    project_name=self.cfg.general.project_name,
                    model_name=self.cfg.training.model_name,
                    max_results=1,
                    metadata=self.cfg.training.model_metadata,
                )[0]

            init_task = Task.get_task(task_id=m.task)
            init_config = munchify(
                init_task.artifacts["configs/train_config.yaml"].get()
            )

            self.m = MODELS[m.name]
            if not self.cfg.training.datamodule in self.m["dms"]:
                error_msg = f"Passing not appropriate datamodule \
                            '{self.cfg.training.datamodule}' \
                            for model '{self.cfg.training.model_to_use}'"
                self.log_obj.invoke_exception(error_msg, ValueError, self.task)


            self.model = self.m["cls"].load_from_checkpoint(
                m.get_local_copy(),
                model_params=init_config.models_params[m.name],
                loss_params=self.cfg.training.loss_func,
                optimizer_params=self.cfg.training.optimizer,
                log_obj=self.log_obj,
                task=self.task,
                ckpt_monitor_metric=self.cfg.general.ckpt_monitor_metric,
            )

            # hardcode?
            self.cfg.training.model_to_use = m.name
            self.cfg.training.metadata["init_task_id"] = init_task.id
            self.cfg.training.metadata["init_model_id"] = m.id

    def __check_config(self, _dict):
        """
        Recursively checks if all values
        in a nested dictionary are specified.

        Args:
        - _dict (dict): A nested
        dictionary to check for missing values.

        Returns:
        - None: The method does not return any value.
        It either completes successfully or
        raises a `ValueError` if a value is missing.
        """
        for k, v in _dict.items():
            if isinstance(v, dict):
                self.__check_config(v)
            if v is None:
                error_msg = f"Value for {k} must be specified"
                self.log_obj.invoke_exception(error_msg, ValueError)

    def __log_essential_files(self):
        """
        Log essential files and information to MLflow.

        This method logs the flattened configuration
        parameters, the configuration dictionaries,
        and various source code files related to the training process.

        :return: None
        """
        self.task.set_parameters_as_dict(flatten_dict(self.cfg, parent_key="cfg"))
        self.task.upload_artifact(
            name="configs/train_config.yaml", artifact_object=self.cfg
        )

        self.task.upload_artifact(
            name="training_scripts/trainer.py",
            artifact_object=os.path.realpath(__file__),
        )

        self.task.upload_artifact(
            name="training_scripts/meta_dicts.py",
            artifact_object="\n".join(
                [
                    str(PREPROCESSORS),
                    str(MODELS),
                    str(LOSS_FUNCS),
                    str(OPTIMS),
                    str(DATAMODULES),
                ]
            ),
        )
        self.task.upload_artifact(
            name="training_scripts/model.py",
            artifact_object=inspect.getsource(self.m["cls"]),
        )
        self.task.upload_artifact(
            name="training_scripts/datamodule.py",
            artifact_object=inspect.getsource(self.dm),
        )

    def train(self):
        """
        Trains the model using the specified dataset and configuration parameters.

        Args:
            None

        Returns:
        None
        """

        self.dm = DATAMODULES.get(self.cfg.training.datamodule)
        if not self.dm:
            error_msg = f"Not exisiting datamodule '{self.cfg.training.datamodule}'"
            self.log_obj.invoke_exception(error_msg, ValueError, self.task)

        self.datamodule = self.dm(
            self.cfg.training.datasets,
            self.cfg.datamodules_params[self.cfg.training.datamodule],
            self.log_obj,
            self.task,
        )

        self.__log_essential_files()

        checkpoint_callback = ClearMLModelCheckpoint(
            task=self.task,
            model_name=self.cfg.training.model_to_use,
            upload_ckpt=self.cfg.general.upload_ckpt,
            filename=f"{{epoch}}-{{{self.model.metric_name}:.2f}}",
            monitor=self.model.metric_name,
            metadata=self.cfg.training.metadata,
            mode="max",
            verbose=True,
            save_top_k=1,
        )

        earlystop_callback = EarlyStopping(
            monitor=self.model.metric_name,
            min_delta=0.005,
            patience=self.cfg.training.early_stopping_patience,
            verbose=True,
            mode="max",
        )

        trainer = Trainer(
            callbacks=[
                earlystop_callback,
                checkpoint_callback,
            ],
            default_root_dir="ckpts",
            fast_dev_run=False,
            max_epochs=self.cfg.training.max_epochs,
            strategy='ddp_find_unused_parameters_true',
            precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
            devices=self.cfg.training.devices,
        )
        trainer.fit(model=self.model, datamodule=self.datamodule)
        best_model_path = checkpoint_callback.best_model_path

        trainer.test(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=best_model_path,
        )

    def get_model(self):
        return self.model

    def get_training_cfg(self):
        return self.cfg


if __name__ == "__main__":
    manager = ModelManager("training_configs/config_train.yaml")
