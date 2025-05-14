import os
import base64
import io
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
import orjson

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# class VisionTextSeq2SeqDataset(Dataset):
#     def __init__(
#         self,
#         data,
#         image_processor: Any,
#         tokenizer: Any,
#         max_target_length: int,
#     ):
#         self.data = data
#         self.image_processor = image_processor
#         self.tokenizer = tokenizer
#         self.max_target_length = max_target_length

#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         pil_img, markdown_text = self.data[idx]

#         image_inputs = self.image_processor(images=pil_img, return_tensors="pt")
#         pixel_values = image_inputs.pixel_values.squeeze(0)

#         tokenized_markdown = self.tokenizer(
#             text=markdown_text,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_target_length,
#             return_tensors="pt",
#             add_special_tokens=True,
#         )
#         labels = tokenized_markdown.input_ids.squeeze(0)

#         # labels = tokenized_markdown.input_ids
#         # attention_mask = tokenized_markdown.attention_mask

#         # final_labels = []
#         # final_mask = []
#         # for seq_ids, mask_bits in zip(labels, attention_mask):
#         #     label_version = list(seq_ids)
#         #     mask_version = list(mask_bits)

#         #     # hardcode for gpt
#         #     label_version.append(self.tokenizer.eos_token_id)
#         #     mask_version.append(1)

#         #     padding_to_add = self.max_target_length - len(label_version)
#         #     if padding_to_add > 0:
#         #         label_version.extend([-100] * padding_to_add)
#         #         mask_version.extend([0] * padding_to_add)
#         #     final_labels.append(label_version[:self.max_target_length])
#         #     final_mask.append(mask_version[:self.max_target_length])

#         # labels = torch.tensor(final_labels).squeeze(0)
#         # labels_attention_mask = torch.tensor(final_mask).squeeze(0)

#         # return {
#         #     "pixel_values": pixel_values,
#         #     "labels": labels,
#         #     "labels_attention_mask": labels_attention_mask,
#         # }

#         return {"pixel_values": pixel_values, "labels": labels}
    
class VisionTextSeq2SeqDataset(Dataset):
    def __init__(
        self,
        data,
        image_processor: Any,
        tokenizer: Any,
        max_target_length: int,
    ):
        self.data = data
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def decode_image_from_base64(base64_string: str):
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datafile = self.data[idx]

        with open(datafile) as f:
            json_data = f.read()
            data = orjson.loads(json_data)

        pil_img = self.decode_image_from_base64(data["page_screenshot"])
        markdown_text = data["processed_markdown"]

        image_inputs = self.image_processor(images=pil_img, return_tensors="pt")
        pixel_values = image_inputs.pixel_values.squeeze(0)

        tokenized_markdown = self.tokenizer(
            text=markdown_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        labels = tokenized_markdown.input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "labels": labels}


class ArxivOnePageDataModule(LightningDataModule):
    def __init__(
        self, datasets_cfg: Dict, datamodule_cfg: Dict, log_obj: Any, task: Any
    ):
        super().__init__()
        self.datasets_cfg = datasets_cfg
        self.log_obj = log_obj
        self.datamodule_cfg = datamodule_cfg
        self.task = task

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.datamodule_cfg.encoder_model
        )
        if hasattr(self.image_processor, "size"):
            self.image_processor.size = {
                "height": self.datamodule_cfg.target_image_size[0],
                "width": self.datamodule_cfg.target_image_size[1],
            }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.datamodule_cfg.decoder_model
        )
        if "gpt2" in self.datamodule_cfg.decoder_model:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_target_length = self.datamodule_cfg.max_length

    def __load_data(self, data_cfg: Dict) -> List[Tuple[Image.Image, str]]:
        from ml_pipeline.meta_dicts import PREPROCESSORS

        all_processed_data: List[Tuple[Image.Image, str]] = []
        for path_key in data_cfg.paths:
            path_config = data_cfg.paths[path_key]
            if "preprocessor" in path_config:
                preproc_params_cfg = path_config.preprocessor
            else:
                preproc_params_cfg = data_cfg.default_preprocessor

            preprocessor_func = PREPROCESSORS.get(preproc_params_cfg.name)

            if not preprocessor_func:
                error_msg = f"Non-existent preprocessor '{preproc_params_cfg.name}'"
                self.log_obj.invoke_exception(error_msg, ValueError, self.task)

            processed_list = preprocessor_func(
                path_config.path,
                self.log_obj,
                self.task,
                **(preproc_params_cfg.get("params", {})),
            )
            if processed_list:
                all_processed_data.extend(processed_list)
            else:
                self.log_obj.info(f"Preprocessor for path {path_key} returned no data.")

        if not all_processed_data:
            error_msg = "Dataset is empty after preprocessing all paths."
            self.log_obj.invoke_exception(error_msg, ValueError, self.task)

        return all_processed_data

    def __create_dataset(
        self, data_list: List[Tuple[Image.Image, str]]
    ) -> VisionTextSeq2SeqDataset:
        ds = VisionTextSeq2SeqDataset(
            data_list,
            self.image_processor,
            self.tokenizer,
            self.max_target_length,
        )
        if not len(ds):
            error_msg = f"Dataset object is empty after initialization."
            self.log_obj.invoke_exception(error_msg, ValueError, self.task)
        return ds

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            train_val_raw_data = self.__load_data(self.datasets_cfg.train_datasets)
            train_val_dataset = self.__create_dataset(train_val_raw_data)

            train_size = int(
                len(train_val_dataset) * self.datamodule_cfg.train_val_split
            )
            val_size = len(train_val_dataset) - train_size

            if train_size == 0 or val_size == 0:
                error_msg = f"Train ({train_size}) or Val ({val_size}) size is 0 after split. Total: {len(train_val_dataset)}"
                self.log_obj.invoke_exception(error_msg, ValueError, self.task)

            self.train_data, self.val_data = random_split(
                train_val_dataset, [train_size, val_size]
            )

            self.train_loader = DataLoader(
                dataset=self.train_data,
                **self.datasets_cfg.train_datasets.loader_params,
            )
            self.val_loader = DataLoader(
                dataset=self.val_data,
                **self.datasets_cfg.train_datasets.loader_params,
            )

        if stage == "test":
            raw_path_data: Dict[str, List[Tuple[Image.Image, str]]] = {}
            from ml_pipeline.meta_dicts import PREPROCESSORS

            test_paths_cfg = self.datasets_cfg.test_datasets.paths

            for path_key in test_paths_cfg:
                path_config = test_paths_cfg[path_key]
                if "preprocessor" in path_config:
                    preproc_params_cfg = path_config.preprocessor
                else:
                    preproc_params_cfg = (
                        self.datasets_cfg.test_datasets.default_preprocessor
                    )

                preprocessor_func = PREPROCESSORS.get(preproc_params_cfg.name)
                if not preprocessor_func:
                    error_msg = f"Non-existent preprocessor '{preproc_params_cfg.name}' for test path {path_key}"
                    self.log_obj.invoke_exception(error_msg, ValueError, self.task)

                _data = preprocessor_func(
                    path_config.path,
                    self.log_obj,
                    self.task,
                    **(preproc_params_cfg.get("params", {})),
                )
                if not _data:
                    self.log_obj.info(
                        f"Test dataset for path {path_key} is empty after preprocess."
                    )
                raw_path_data[path_key] = _data or []

            combined_raw_data: Dict[str, List[Tuple[Image.Image, str]]] = defaultdict(
                list
            )
            if (
                "combine_tests" in self.datasets_cfg.test_datasets
                and self.datasets_cfg.test_datasets.combine_tests
            ):
                for (
                    comb_name,
                    paths_to_combine,
                ) in self.datasets_cfg.test_datasets.combine_tests.items():
                    for path_key_to_add in paths_to_combine:
                        _list_data = raw_path_data.get(path_key_to_add)
                        if not _list_data:
                            error_msg = f"Path key '{path_key_to_add}' specified in combine_tests for '{comb_name}' not found or was empty."
                            self.log_obj.invoke_exception(
                                error_msg, ValueError, self.task
                            )
                        combined_raw_data[comb_name].extend(_list_data)
            else:
                # If no combine_tests, each path is its own test set
                combined_raw_data = raw_path_data

            self.test_datasets_map: Dict[str, Dataset] = {}
            for name, data_list in combined_raw_data.items():
                if not data_list:
                    self.log_obj.info(
                        f"Test dataset named '{name}' is empty after combining/loading. Skipping DataLoader creation for it."
                    )
                    continue
                dataset_obj = self.__create_dataset(data_list)
                self.test_datasets_map[name] = dataset_obj

            if not self.test_datasets_map:
                self.log_obj.info(
                    "Warning: No test datasets were successfully loaded or created."
                )
                self.test_names = []
                self.test_loaders = []
                return

            self.test_names = list(self.test_datasets_map.keys())
            self.test_loaders = [
                DataLoader(
                    dataset=ds_obj, **self.datasets_cfg.test_datasets.loader_params
                )
                for ds_obj in self.test_datasets_map.values()
            ]

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    def test_dataloader(self) -> List[DataLoader]:  # Returns a list of DataLoaders
        return self.test_loaders


if __name__ == "__main__":
    import yaml
    from munch import munchify

    with open("ml_pipeline/training_configs/dialogues/exp0001.yaml") as f:
        config = yaml.safe_load(f)

    class MockLogger:
        def info(self, text):
            print(text)

        def invoke_exception(self, text, *args):
            print(text)

    dataset_conf = munchify(config["training"]["datasets"])
    loader_conf = munchify(config["datamodules_params"]["arxiv_onepage_ved"])

    o = ArxivOnePageDataModule(dataset_conf, loader_conf, MockLogger(), None)
    o.setup()
