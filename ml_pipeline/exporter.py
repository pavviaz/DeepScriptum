import os

import torch
import yaml
from tqdm import tqdm
from munch import munchify
from clearml import Task, InputModel
from sklearn.metrics import classification_report


class Exporter:
    def __init__(self, config_path: str, device="cuda:0"):
        from meta_dicts import MODELS, DATAMODULES
        
        """
        Initialize the Exporter object.
        TODO
        """
        if not os.path.exists(config_path):
            raise OSError(f"Config file on {config_path} path does not exist")
        
        self.device = device

        with open(config_path, encoding="utf-8") as c:
            config = yaml.load(c, Loader=yaml.FullLoader)
        self.cfg = munchify(config)

        if all(
            not v in self.cfg.general
            for v in ["model_id", "model_name", "model_metadata"]
        ):
            error_msg = f"At least one of model params must be specified: \
                          'model_id', 'model_name', 'model_metadata'"
            raise ValueError(error_msg)

        # FROM PRETRAIN
        if "model_id" in self.cfg.general and self.cfg.general.model_id:
            m = InputModel(
                project=self.cfg.general.project_name,
                model_id=self.cfg.general.model_id,
            )
        else:
            m = InputModel.query_models(
                project_name=self.cfg.general.project_name,
                model_name=self.cfg.general.model_name,
                max_results=1,
                metadata=self.cfg.general.model_metadata,
            )[0]

        init_task = Task.get_task(task_id=m.task)
        self.init_config = munchify(
            init_task.artifacts["configs/train_config.yaml"].get()
        )

        self.m = MODELS[m.name]
        self.local_path = m.get_local_copy()
        self.model = (
            self.m["cls"]
            .load_from_checkpoint(
                self.local_path,
                model_params=self.init_config.models_params[m.name],
                loss_params=self.init_config.training.loss_func,
                optimizer_params=self.init_config.training.optimizer,
                log_obj=None,
                task=None,
                ckpt_monitor_metric=self.init_config.general.ckpt_monitor_metric,
                classes_map=self.init_config.training.datasets.classes_map,
            )
        )

        if device:
            self.model.to(device)

        self.model.eval()

    def test_on_data(self):
        from meta_dicts import MODELS, DATAMODULES

        dm = DATAMODULES.get(self.init_config.training.datamodule)
        datamodule = dm(
            self.cfg.general.data,
            self.init_config.datamodules_params[self.init_config.training.datamodule],
            None,
            None,
        )
        datamodule.setup(stage="test")

        for idx, dataset in enumerate(datamodule.test_dataloader()):
            preds = []
            y_true = []
            with torch.no_grad():
                for batch in tqdm(dataset):
                    y_hat = self.model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                    )
                    y_hat = torch.argmax(y_hat, dim=-1).cpu()

                    preds.extend([el.item() for el in y_hat])
                    y_true.extend([el.item() for el in batch["labels"]])

                print(f"===== {datamodule.test_names[idx]} =====")
                print(classification_report(y_true, preds))
