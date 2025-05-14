from typing import Any, Tuple

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
)

from ml_pipeline.logger import TrainingLogger
from .base_model import BaseModel


class VEDDuoModel(BaseModel):
    def __init__(
        self,
        model_params,
        optimizer_params,
        log_obj: TrainingLogger,
        task,
        ckpt_monitor_metric,
    ):
        super().__init__(
            model_params=model_params,
            optimizer_params=optimizer_params,
            log_obj=log_obj,
            task=task,
            ckpt_monitor_metric=ckpt_monitor_metric,
        )

    @staticmethod
    def resize_position_embeddings(model, new_max_length):
        if not isinstance(new_max_length, int) or new_max_length <= 0:
            raise ValueError("new_max_length must be a positive integer")

        model_type = model.config.model_type.lower()
        if "mbart" in model_type:
            position_embeddings = model.model.decoder.embed_positions
            config_key = "max_position_embeddings"
            offset = 2  # mBART offset
        elif "gpt" in model_type:
            position_embeddings = model.transformer.wpe
            config_key = (
                "n_positions"
                if hasattr(model.config, "n_positions")
                else "max_position_embeddings"
            )
            offset = 0
        elif "roberta" in model_type:
            position_embeddings = model.roberta.embeddings.position_embeddings
            config_key = "max_position_embeddings"
            offset = 2  # RoBERTa uses 0 for <s>, 1 for <pad>, so 514 covers 512 tokens
        else:
            raise ValueError("Unsupported model type.")

        embedding_class = type(position_embeddings)
        hidden_size = position_embeddings.weight.shape[1]
        new_embedding_size = new_max_length + offset

        # Create new instance of the same embedding class
        new_position_embeddings = embedding_class(new_embedding_size, hidden_size)
        old_embedding_size = position_embeddings.weight.shape[0]

        if new_embedding_size == old_embedding_size:
            print(
                f"Current embedding size ({old_embedding_size}) matches new size. No resizing needed."
            )
            return

        # Resize embeddings
        old_embeddings = position_embeddings.weight
        if new_embedding_size < old_embedding_size:
            new_embeddings = old_embeddings[:new_embedding_size, :]
        else:
            embeddings = old_embeddings.T.unsqueeze(0)
            new_embeddings = nn.functional.interpolate(
                embeddings, size=new_embedding_size, mode="linear", align_corners=False
            )
            new_embeddings = new_embeddings.squeeze(0).T

        with torch.no_grad():
            new_position_embeddings.weight.copy_(new_embeddings)

        # Update model
        if "mbart" in model_type:
            model.model.decoder.embed_positions = new_position_embeddings
        elif "gpt" in model_type:
            model.transformer.wpe = new_position_embeddings
        elif "roberta" in model_type:
            model.roberta.embeddings.position_embeddings = new_position_embeddings

        setattr(model.config, config_key, new_max_length)
        print(
            f"Position embeddings resized from {old_embedding_size} to {new_embedding_size} "
            f"for {model_type} model (max_length={new_max_length})."
        )

    def _model_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_params.decoder_model)
        if "gpt2" in self.model_params.decoder_model:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            self.model_params.encoder_model, self.model_params.decoder_model
        )
        self.resize_position_embeddings(
            self.model.decoder, new_max_length=self.model_params.max_length
        )

        model_type = self.model.decoder.config.model_type.lower()
        if "mbart" in model_type:
            self.model.config.decoder_start_token_id = self.tokenizer.eos_token_id
        elif "gpt" in model_type:
            self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.model.decoder.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.decoder.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.decoder.config.pad_token_id = self.tokenizer.pad_token_id
            # self.model.decoder.resize_token_embeddings(len(self.tokenizer))

            self.model.config.max_length = self.model_params.max_length

            self.model.config.early_stopping = True
            self.model.config.no_repeat_ngram_size = 3
            self.model.config.length_penalty = 2.0
            # self.model.config.num_beams = 4
        elif "roberta" in model_type:
            self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
            self.model.config.eos_token_id = self.tokenizer.sep_token_id
            self.model.config.max_length = self.model_params.max_length

            self.model.config.early_stopping = True
            self.model.config.no_repeat_ngram_size = 3
            self.model.config.length_penalty = 2.0
            self.model.config.num_beams = 4
        else:
            raise ValueError

        self.log_obj.info(
            f"Set model.config.decoder_start_token_id to: {self.model.config.decoder_start_token_id}"
        )

        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.log_obj.info(
                f"Set model.config.pad_token_id to: {self.model.config.pad_token_id}"
            )

        self.model.config.vocab_size = self.tokenizer.vocab_size

        self.model.train()

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
        # attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if labels is not None:
            outputs = self.model(
                pixel_values=pixel_values,
                labels=labels,
                # decoder_attention_mask=attention_mask,
            )
            return outputs.logits, outputs.loss
        else:
            outputs = self.model(pixel_values=pixel_values)
            return outputs.logits, None

    def _training_step_logic(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        # attention_mask = batch["labels_attention_mask"]

        _, loss = self.forward(pixel_values, labels)
        return loss

    def _evaluation_step_logic(self, batch: dict) -> dict:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        # attention_mask = batch["labels_attention_mask"]

        _, loss = self.forward(pixel_values, labels)

        # Generate sequences
        generated_ids = self.model.generate(
            pixel_values,
            # max_length=self.model_params.max_length,
            max_length=256,
            eos_token_id=self.model.config.eos_token_id,
        )

        return {
            "loss": loss,
            "generated_ids": generated_ids,
            "reference_ids": labels[:, :256],
        }
