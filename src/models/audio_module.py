from typing import Any, List, Optional, Dict

import torch, hydra, math
from omegaconf import DictConfig
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .base_model import BaseModule


class AudioModule(BaseModule):
    def __init__(
        self, task: str, num_classes: int,
        model: DictConfig, optim: DictConfig, dataset: str,
        # optional scheduler
        scheduler: DictConfig = None, init: Optional[str] = None,
        ordinal_regression: Optional[str] = None,
        **kwargs
    ):
        super().__init__(task, num_classes, model, optim, dataset, init, ordinal_regression, scheduler=scheduler)

    def configure_optimizers(self):
        wd = self.hparams.optim.pop("weight_decay")
        # set BN weight_decay = 0
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": wd},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        opt = hydra.utils.instantiate(
            self.hparams.optim, params=optimizer_grouped_parameters,
            _convert_="all"
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        scheduler = get_linear_schedule_with_warmup(
            # TODO warmup
            opt, num_warmup_steps=math.ceil(self.total_steps * 0.2), num_training_steps=self.total_steps
        )
        return [opt], [scheduler]