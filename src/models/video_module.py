from typing import Any, List, Optional, Dict

import torch, hydra
from omegaconf import DictConfig

from .base_model import BaseModule


class VideoModule(BaseModule):
    def __init__(
        self, task: str, num_classes: int,
        model: DictConfig, optim: DictConfig, dataset: str,
        # optional scheduler
        scheduler: DictConfig=None, init: Optional[str] = None,
        ordinal_regression: Optional[str] = None,
        **kwargs
    ):
        super().__init__(task, num_classes, model, optim, dataset, init, ordinal_regression, scheduler=scheduler)

    def configure_optimizers(self):
        # set BN weight_decay = 0
        bn_params = []
        non_bn_params = []
        for name, p in self.named_parameters():
            if "bn" in name:
                bn_params.append(p)
            else:
                non_bn_params.append(p)

        optim_params = [
            {"params": bn_params, "weight_decay": 0.0},
            {"params": non_bn_params, "weight_decay": self.hparams.optim.weight_decay},
        ]
        opt = hydra.utils.instantiate(
            self.hparams.optim, params=optim_params,
            _convert_="all"
        )

        ret_dict = {'optimizer': opt}
        if self.hparams.scheduler:
            scheduler = hydra.utils.instantiate(
                self.hparams.scheduler, optimizer=opt,
                _convert_="all"
            )
            ret_dict["lr_scheduler"] = scheduler
            ret_dict["monitor"] = "valid/epoch/loss"
        
        return ret_dict
