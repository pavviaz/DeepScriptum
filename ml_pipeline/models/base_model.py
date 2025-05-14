from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Any, Dict

import torch
from lightning import LightningModule
from transformers import get_scheduler

from ml_pipeline.compute_metrics import Seq2SeqMetricsTracker
from ml_pipeline.logger import TrainingLogger


class BaseModel(LightningModule, ABC):
    def __init__(
        self,
        model_params: Dict,
        optimizer_params: Dict,
        log_obj: TrainingLogger,
        task: Any,
        ckpt_monitor_metric: str,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["log_obj", "task_clearml"])

        self.model_params = model_params
        self.optimizer_params_config = optimizer_params
        self.log_obj = log_obj
        self.task_clearml_obj = task
        self.ckpt_monitor_metric_name = ckpt_monitor_metric

        self.metric_name = f"val_{self.ckpt_monitor_metric_name.upper()}"

        self._model_init()

    def setup(self, stage: str):
        self.metrics_trackers = defaultdict(
            lambda: Seq2SeqMetricsTracker(
                tokenizer=self.trainer.datamodule.tokenizer,
                device=self.device,
                log_obj=self.log_obj,
            )
        )
        self.log_obj.info(f"Model setup complete. Monitoring: {self.metric_name}")

    @abstractmethod
    def _model_init(self):
        raise NotImplementedError

    def _log_epoch_metrics(self, stage_name: str, epoch: int, step: int):
        metrics_data = self.metrics_trackers[stage_name].compute_all()

        gathered_metrics_tensors = self.all_gather(metrics_data)

        final_metrics_to_log = {}
        for key, value in gathered_metrics_tensors.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                final_metrics_to_log[key] = torch.mean(value).item()
            elif (
                isinstance(value, (list, tuple))
                and len(value) > 0
                and isinstance(value[0], (float, int))
            ):
                final_metrics_to_log[key] = sum(value) / len(value)
            else:
                final_metrics_to_log[key] = metrics_data.get(key, 0.0)

        if self.trainer.is_global_zero:
            self.log_obj.info(
                f"--- {stage_name.upper()} EPOCH {epoch} (Global Step {step}) RESULTS ---"
            )
            for name, value in final_metrics_to_log.items():
                self.log_obj.info(f"{name}: {value:.4f}")
                self.log_obj.log_scalar(
                    title=f"{stage_name}_epoch/{name}",
                    series=name.upper(),
                    value=value,
                    iteration=step,
                )

            if stage_name != "train":
                samples = self.metrics_trackers[stage_name].get_prediction_samples()
                if samples:
                    report_str = f"--- {stage_name.upper()} EPOCH {epoch} - Prediction Samples ---\n"
                    for i, sample_pair in enumerate(samples):
                        report_str += (
                            f"Sample {i+1}:\n"
                            f"  REF: {sample_pair['reference'][:250]}...\n"
                            f"  GEN: {sample_pair['generated'][:250]}...\n\n"
                        )

                    logger_method_name = getattr(
                        self.log_obj,
                        "log_text_to_clearml",
                        getattr(self.log_obj, "report_text", None),
                    )
                    if logger_method_name:
                        try:
                            logger_method_name(
                                title=f"{stage_name.upper()}_Samples_Epoch_{epoch}",
                                text_string=report_str,
                                iteration=step,
                            )
                        except TypeError:
                            logger_method_name(
                                report_str,
                                title=f"{stage_name.upper()}_Samples_Epoch_{epoch}",
                                iteration=step,
                                print_console=False,
                            )
                    else:
                        self.log_obj.info(report_str)

            self.log_obj.info(f"--- END {stage_name.upper()} EPOCH {epoch} REPORT ---")

        if stage_name == "val":
            metric_val_for_ckpt = final_metrics_to_log.get(
                self.ckpt_monitor_metric_name.upper()
            )
            if metric_val_for_ckpt is not None:
                self.log(self.metric_name, metric_val_for_ckpt, sync_dist=True)
            else:
                self.log_obj.info(
                    f"Warning: Checkpoint metric '{self.ckpt_monitor_metric_name.upper()}' not found in val results. Available: {list(final_metrics_to_log.keys())}"
                )

        self.metrics_trackers[stage_name].reset()

    @abstractmethod
    def _training_step_logic(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Return loss
        raise NotImplementedError

    @abstractmethod
    def _evaluation_step_logic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # Returns {'loss': torch.Tensor, 'generated_ids': torch.Tensor, 'reference_ids': torch.Tensor}
        raise NotImplementedError

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self._training_step_logic(batch)

        self.log(
            "train/step_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.lr_schedulers():
            lr = self.lr_schedulers().get_last_lr()[0]
            self.log(
                "lr",
                lr,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

        self.metrics_trackers["train"].loss_values.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train", self.current_epoch + 1, self.global_step)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        eval_output = self._evaluation_step_logic(batch)
        loss = eval_output["loss"]
        generated_ids = eval_output["generated_ids"]
        reference_ids = eval_output["reference_ids"]

        self.log(
            "val/step_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.metrics_trackers["val"].update(generated_ids, reference_ids, loss.item())

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self._log_epoch_metrics("val", self.current_epoch + 1, self.global_step)

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        stage_name = "test"
        if (
            hasattr(self.trainer.datamodule, "test_names")
            and self.trainer.datamodule.test_names
        ):
            if dataloader_idx < len(self.trainer.datamodule.test_names):
                stage_name = self.trainer.datamodule.test_names[dataloader_idx]
            else:
                self.log_obj.info(
                    f"Warning: dataloader_idx {dataloader_idx} out of bounds for test_names. Using default 'test'."
                )

        eval_output = self._evaluation_step_logic(batch)
        loss = eval_output.get("loss")
        generated_ids = eval_output["generated_ids"]
        reference_ids = eval_output["reference_ids"]

        if loss is not None:
            self.log(
                f"{stage_name}/step_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        self.metrics_trackers[stage_name].update(
            generated_ids, reference_ids, loss.item() if loss is not None else None
        )

    def on_test_epoch_end(self):
        test_dataloader_names = ["test"]
        if (
            hasattr(self.trainer.datamodule, "test_names")
            and self.trainer.datamodule.test_names
        ):
            test_dataloader_names = self.trainer.datamodule.test_names

        for name in test_dataloader_names:
            if (
                name in self.metrics_trackers
                and self.metrics_trackers[name].loss_values
                or self.metrics_trackers[name].prediction_samples
            ):
                self._log_epoch_metrics(name, self.current_epoch + 1, self.global_step)
            else:
                self.log_obj.info(
                    f"No metrics tracker found or used for test stage '{name}'. Skipping logging."
                )

    def configure_optimizers(self) -> Any:
        from ml_pipeline.meta_dicts import OPTIMS

        opt_creator = OPTIMS.get(self.optimizer_params_config.type)
        if not opt_creator:
            error_msg = (
                f"Non-existent optimizer type '{self.optimizer_params_config.type}'"
            )
            self.log_obj.invoke_exception(error_msg, ValueError, self.task_clearml_obj)

        opt_actual_params = {
            k: v
            for k, v in self.optimizer_params_config.get("params", {}).items()
            if v is not None
        }

        optimizer = opt_creator(self.parameters(), **opt_actual_params)

        if (
            "scheduler_params" in self.optimizer_params_config
            and self.optimizer_params_config.scheduler_params
        ):
            sch_params = self.optimizer_params_config.scheduler_params

            num_warmup_steps = 0
            if "num_warmup_steps" in sch_params and sch_params.num_warmup_steps:
                num_warmup_steps = sch_params.num_warmup_steps

            if "num_training_steps" in sch_params and sch_params.num_training_steps:
                num_training_steps = sch_params.num_training_steps
            else:
                num_training_steps = (
                    len(self.trainer.datamodule.train_loader) * self.trainer.max_epochs
                )

            scheduler = get_scheduler(
                sch_params.type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=sch_params.get("kwargs", {}),
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return optimizer
