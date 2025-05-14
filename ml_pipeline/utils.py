import collections

from lightning.pytorch.callbacks import ModelCheckpoint
from clearml import Task, OutputModel


class ClearMLModelCheckpoint(ModelCheckpoint):
    def __init__(
        self, task: Task, model_name: str, upload_ckpt: bool, metadata, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.task = task
        self.output_model = None
        self.model_name = model_name
        self.upload_models = upload_ckpt
        self.metadata = metadata
        self.best_model_path_used = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        if self.upload_models and trainer.is_global_zero:
            if (
                self.best_model_path
                and self.best_model_path != self.best_model_path_used
            ):
                self.best_model_path_used = self.best_model_path

                if self.output_model is None:
                    self.output_model = OutputModel(
                        task=self.task,
                        name=self.model_name,
                        framework="PyTorch"
                    )
                    [
                        self.output_model.set_metadata(key=k, value=v)
                        for k, v in self.metadata.items()
                    ]

                # upload to storage?
                self.output_model.update_weights(
                    weights_filename=self.best_model_path_used
                )

        return self.best_model_path


def flatten_dict(dictionary, parent_key=False, separator="."):
    """
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten_dict(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten_dict({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))

    return dict(items)
