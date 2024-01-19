import pickle
import codecs

import mlflow
from ultralytics import YOLO, settings


def model_type_callback(_):
    mlflow.set_tag("model_type", "yolo")


def logging_callback(trainer):
    mlflow.log_metrics(trainer.lr)
    mlflow.log_metrics(
        {k: v for k, v in zip(trainer.loss_names, list(trainer.tloss.cpu().numpy()))}
    )
    model_dump = codecs.encode(pickle.dumps(trainer.model), "base64").decode()
    mlflow.log_text(model_dump, "last.pt")


def train_yolo(cfg):
    settings.update(
        {
            "mlflow": True,
            "clearml": False,
            "clearml": False,
            "comet": False,
            "dvc": False,
            "hub": False,
            "neptune": False,
            "raytune": False,
            "tensorboard": False,
            "wandb": False,
        }
    )

    model = YOLO()
    model.to(cfg.general.device)

    model.add_callback("on_train_start", model_type_callback)
    model.add_callback("on_fit_epoch_end", logging_callback)

    model.train(
        data=cfg.general.dataset_cfg_path,
        seed=cfg.general.seed,
        project=cfg.general.export_path,
        name=cfg.general.model_name,
        device=cfg.general.device,
        save=False,
        plots=False,
        **cfg.yolo
    )
