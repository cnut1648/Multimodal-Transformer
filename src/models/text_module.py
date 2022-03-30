from typing import Any, List, Optional, Dict
import math

import torch, hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .base_model import BaseModule


class TextModule(BaseModule):
    def __init__(
        self, task: str, num_classes: int,
        model: DictConfig, optim: DictConfig, dataset: str,
        init: Optional[str] = None, ordinal_regression: Optional[str] = None,
        **kwargs
    ):
        # disable tokenizer fork
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        super().__init__(task, num_classes, model, optim, dataset, init, ordinal_regression)
    
    def preprocess_batch(self, batch: dict):
        """
        override plus tokenize
        """
        batch, labels = super().preprocess_batch(batch)
        text: List[str] = batch.pop("text")
        tokens = self.model.tokenize(text).to(self.device)
        return {**batch, **tokens}, labels

    def configure_optimizers(self):
        wd = self.hparams.optim.pop("weight_decay")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": wd},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = hydra.utils.instantiate(
            self.hparams.optim, params=optimizer_grouped_parameters,
            _convert_="partial"
        )
        scheduler = get_linear_schedule_with_warmup(
            # TODO warmup
            optimizer, num_warmup_steps=math.ceil(self.total_steps * 0.2), num_training_steps=self.total_steps
        )
        return [optimizer], [scheduler]