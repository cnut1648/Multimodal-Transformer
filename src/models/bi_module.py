from typing import Any, List, Optional, Dict, Tuple
import math
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer

import torch, hydra
from torch import nn
from omegaconf import DictConfig
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from src.models import ModuleMetricMixin
from functools import partial

from src.utils.modeling import weights_init, get_module_by_name, freeze, unfreeze

# layer name that register hook (i.e. before norm), norm layer name
Before_And_Norm = Tuple[str, str]
# if 0th is True, exchange 1->2, otherwise disable
# if 1th is True, exchange 2->1, otherwise disable
IS_EXCHANGE_ENABLE = Tuple[bool, bool]
# block level, and Before_And_Norm for model 1 and 2
BlockStatId = Tuple[int, Before_And_Norm, Before_And_Norm, IS_EXCHANGE_ENABLE]

class BiModule(ModuleMetricMixin, LightningModule):
    """
    use two modalities
    assume video consist of .backbonoe and .out
    assume text has .tokenize() and .model as LM

    train:
        training_step -> compute_step (preprocess + forward with stat + update stat) -> backward and optimizer
    infernece:
        compute_step -> forward again
    """
    def __init__(
        self, task: str, num_classes: int, dataset: str, modality: str, 
        exchange_levels: List[BlockStatId], link_mode: str, 
        aligner_optim: DictConfig, model1: DictConfig, model2: DictConfig, 
        accumulate_grad_batches: int = 1, unfreeze_epoch: int = 10,
        init: Optional[str] = None, ordinal_regression: Optional[str] = None,
        **kwargs
    ):
        # disable tokenizer fork
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        super().__init__(task=task, num_classes=num_classes, dataset=dataset, ordinal_regression=ordinal_regression)
        self.modality = list(modality)
        assert len(self.modality) == 2
        assert link_mode in ["sync", "next"]

        self.optcfg = {
            "model1": [model1.pop("optim"), model1.pop("scheduler")],
            "model2": [model2.pop("optim"), model2.pop("scheduler")],
        }
        self.weight1, self.weight2 = model1.pop("weight"), model2.pop("weight")

        x = ModalityModel.load_ckpt()

        self.model1: ModalityModel = hydra.utils.instantiate(
            model1.model
        )
        self.model2: ModalityModel = hydra.utils.instantiate(
            model2.model, _recursive_=False
        )
        if init:
            self.apply(weights_init(init))
        
        if task == "clf":
            if dataset == "cmu_mosei" and num_classes == 1:
                self.criterion = torch.nn.L1Loss()
            else:
                self.criterion = torch.nn.CrossEntropyLoss() 
        else:
            self.criterion = torch.nn.MSELoss()

        self.aligner1 = nn.ModuleDict({
            f"{idx}th_aligner": nn.Linear(self.model1.hidden_size, self.model2.hidden_size)
            for idx in range(len(exchange_levels))
        })
        self.aligner2 = nn.ModuleDict({
            f"{idx}th_aligner": nn.Linear(self.model2.hidden_size, self.model1.hidden_size)
            for idx in range(len(exchange_levels))
        })

        # append valid and test metric for two submodules
        for split in ["train", "valid", "test"]:
            metric = self.metrics[f"{split}_metrics"]
            for i in [1, 2]:
                self.metrics[f"{split}_metrics{i}"] = metric.clone(
                    prefix=f"{split}/{i}/")
                if split != "test":
                    self.mean_losses[f"{split}_losses{i}"] = self.mean_losses[f"{split}_losses"].clone()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
    
    @property
    def automatic_optimization(self):
        return False

    def preprocess_batch(self, batch: dict):
        if self.hparams.task == "clf":
            labels = batch.pop("labels")
        else:
            valence, arousal = batch.pop("valence"), batch.pop("arousal")
            # (bsz, 2)
            labels = torch.stack([valence, arousal], 1)
        if "t" in self.modality:
            i = self.modality.index("t")
            text_model = self.model1 if i == 0 else self.model2
            text = batch.pop("text")
            tokens = text_model.tokenize(text).to(self.device)
            batch = {**batch, **tokens}
        return batch, labels
    
    def postprocess_batch(self, logits, labels):
        """
        compute loss and preds
        """
        if self.hparams.task == "clf":
            if self.hparams.dataset == "cmu_mosei" and self.hparams.num_classes == 1:
                assert logits.size(1) == 1
                # labels [0, 6], thus clip [0, 6], (bsz, 1) to (bsz, )
                preds = logits.clip(0., 6.).round().squeeze(-1).long()
                loss = self.criterion(logits, labels)
            else:
                preds = torch.argmax(logits, dim=1)
                loss = self.criterion(logits, labels)
        else:
            loss = self.criterion(logits, labels)
            preds = logits
        
        return loss, preds
    
    def forward(self, batch: dict):
        logits1 = self.model1(**batch)
        logits2 = self.model2(**batch)
        return logits1, logits2, 0.5 * logits1 + 0.5 * logits2

    def forward_with_stat(self, batch: dict, statistics_storages=None):
        """
        same as forward but compute stat for `exchange_levels`
        return 
        statistics_storages: [model1's output (in model2 space), model2's output (in model1 space)]
        """
        if statistics_storages is not None:
            is_second_time = True
        else:
            is_second_time = False
            statistics_storage1, statistics_storage2 = [], []
            statistics_storages = (statistics_storage1, statistics_storage2)

        def ComputeStatHook(module, input, output, aligner):
            """
            pytorch hook, module(input) = output
            called in the order from 0th layer to last
            use Linear to compute statistics
            """
            if aligner is self.aligner1:
                storage = statistics_storage1
            else:
                storage = statistics_storage2
            ith = len(storage)
            # (B, seq_len, hidden)
            hidden = input[0]
            # (B, seq_len, hidden)
            representation = aligner[f'{ith}th_aligner'](hidden)
            μ = representation.mean([0, 1])
            σ = representation.var([0, 1]).sqrt()
            storage.append([μ, σ])
        
        def ApplyStatHook(module, input, output, stats):
            """
            stats: mean & std from opposite model
            """
            μ, σ = stats
            input = input[0]
            normalized_input = (input - input.mean(-1, keepdim=True)) / input.std(-1, keepdim=True).add(module.eps)
            return normalized_input * σ + μ

        hook_handles = []
        if is_second_time:
            for idx, _, layers_to_apply, is_exchange_enables in self.hparams.exchange_levels:
                # normlayer: [model1 norm, model2 norm]
                # statistics_storages: [model1's output (in model2 space), model2's output (in model1 space)]
                for layername, mean_std_list, opposite_model, is_exchange_enable in zip(
                    layers_to_apply[::-1], statistics_storages, 
                    [self.model2, self.model1], is_exchange_enables):
                    if is_exchange_enable:
                        mean, std = mean_std_list[idx]
                        m1, m2 = self.modality
                        if opposite_model is self.model2:
                            direction = f"{m1}->{m2}"
                        else:
                            direction = f"{m2}->{m1}"
                        self.log(f"stat/{direction}/ℒ[{idx}]mean", mean.mean(), sync_dist=True, prog_bar=False, on_epoch=False, on_step=True)
                        self.log(f"stat/{direction}/ℒ[{idx}]std", std.mean(), sync_dist=True, prog_bar=False, on_epoch=False, on_step=True)
                        block = opposite_model.blocks[idx]
                        block = get_module_by_name(block, layername)
                        hook_handles.append(block.register_forward_hook(
                            partial(ApplyStatHook, stats=mean_std_list[idx])
                        ))
        else:
            for idx, layernorms, _, is_exchange_enables in self.hparams.exchange_levels:
                for layername, aligner_for_hook, model, is_exchange_enable in zip(
                    layernorms, [self.aligner1, self.aligner2], [self.model1, self.model2],
                    is_exchange_enables
                ):
                    if is_exchange_enable:
                        block = model.blocks[idx]
                        # before norm
                        block = get_module_by_name(block, layername)
                        hook_handles.append(block.register_forward_hook(
                            partial(ComputeStatHook, aligner=aligner_for_hook)
                        ))
        logits = self(batch)

        # remove hooks to avoid unnecessary mess
        for hook in hook_handles:
            hook.remove()

        return logits, statistics_storages    

    def compute_step(self, batch: dict, split: str):
        batch, labels = self.preprocess_batch(batch)
        # import numpy as np
        # batch["audios"] = [
        #     np.random.random(size=(230000, )).astype(batch["audios"][i].dtype)
        #     for i in range(4)
        # ]
        # logits := (logits1, logits2, combine logits)
        logits, statistics_storages = self.forward_with_stat(batch)
        if split == "train":
            # forward twice, 1st time (already did) get stat and exchange; 2nd time forward with new stat from this batch
            # (N, K)
            logits, _ = self.forward_with_stat(batch, statistics_storages)
        ret = {"targets": labels}
        losses = []
        for model_logit, suffix in zip(logits, ["1", "2", ""]):
            # metricname = f"{split}_metrics{suffix}"
            loss, preds = self.postprocess_batch(model_logit, labels)
            if suffix == "":
                # return combined
                loss = sum([l * w for l, w in zip(losses, [self.weight1, self.weight2])])
            # log train metrics
            # self.metrics[metricname](preds, labels)
            self.log(f"{split}/step/loss{suffix}", loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            ret.update({
                f"loss{suffix}": loss,
                f"preds{suffix}": preds,
            })
            losses.append(loss)
        return ret
    
    def compute_step_end(self, outputs, split: str):
        """
        log, since DDP logging must be put in *_step_end method
        """
        (loss, preds,
         loss1, preds1,
         loss2, preds2, labels) = (outputs["loss"], outputs["preds"], 
                                  outputs["loss1"], outputs["preds1"], 
                                  outputs["loss2"], outputs["preds2"], outputs["targets"])
        # log metrics
        if split != "test":
            self.mean_losses[f"{split}_losses"](loss)
            self.mean_losses[f"{split}_losses1"](loss1)
            self.mean_losses[f"{split}_losses2"](loss2)
        self.metrics[f"{split}_metrics"](preds, labels)
        self.metrics[f"{split}_metrics1"](preds1, labels)
        self.metrics[f"{split}_metrics2"](preds2, labels)

    def agg_epoch(self, outputs: List[Any], split: str):
        if split != "test":
            for suffix in ["1", "2", ""]:
                loss = self.mean_losses[f"{split}_losses{suffix}"].compute()
                self.log(f"{split}/epoch/loss{suffix}", loss, on_epoch=True, prog_bar=True)
                self.mean_losses[f"{split}_losses{suffix}"].reset()
        for suffix in ["1", "2", ""]:
            metrics = self.metrics[f"{split}_metrics{suffix}"].compute()
            # make classwise e.g. valid/1/accuracy_neu separate namespace
            for k in list(metrics):
                if "_" in k:
                    value = metrics.pop(k)
                    splitname, k = k.rsplit("/", 1)
                    metrics[f"{splitname}/classwise/{k}"] = value
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.metrics[f"{split}_metrics{suffix}"].reset()

    def training_step(self, batch: Any, batch_idx: int):
        if self.trainer.current_epoch == self.hparams.unfreeze_epoch:
            unfreeze(self.model1)
            unfreeze(self.model2)
        ret = self.compute_step(batch, "train")
        loss = ret["loss"]
        self.manual_backward(loss)

        if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
            # opt1, opt2
            opts: List[LightningOptimizer] = self.optimizers()
            for opt in opts:
                opt.step()
                opt.zero_grad()
        
        return ret
    
    def training_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "train")

    def training_epoch_end(self, outputs: List[Any]):
        self.agg_epoch(outputs, "train")
        schs = self.lr_schedulers()
        for sch in schs:
            sch.step()

    def validation_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "valid")

    def validation_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "valid")

    def validation_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "valid")

    def test_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "test")

    def test_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "test")

    def test_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "test")

    def setup(self, stage: Optional[str] = None) -> None:
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        effective_batch_size = (self.trainer.datamodule.hparams.batch_size *
                                max(1, self.trainer.num_gpus) * self.trainer.accumulate_grad_batches)
        self.total_steps = int(
            (len(train_loader.dataset) // effective_batch_size) * float(self.trainer.max_epochs))

    def configure_optimizers(self):
        # initially freeze
        freeze(self.model1)
        freeze(self.model2)
        unfreeze(self.model1.blocks[1])
        unfreeze(self.model1.blocks[-1])
        unfreeze(self.model2.blocks[1])
        unfreeze(self.model2.blocks[-1])

        optimizers, schedulers = [], []
        for i, (modality, model) in enumerate(zip(self.modality, [self.model1, self.model2]), 1):
            optim, sch = self.optcfg[f"model{i}"]
            scheduler = None
            optimizer = None
            if modality == "t":
                # text
                wd = optim.pop("weight_decay")
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": wd},
                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0},
                ]
                optimizer = hydra.utils.instantiate(
                    optim, params=optimizer_grouped_parameters,
                    _convert_="partial"
                )
                scheduler = get_linear_schedule_with_warmup(
                    # TODO warmup
                    optimizer, num_warmup_steps=math.ceil(self.total_steps * 0.2), num_training_steps=self.total_steps
                )
            elif modality == "a":
                # text
                wd = optim.pop("weight_decay")
                no_decay = ["bias", "layer_norm.weight"]
                optimizer_grouped_parameters = [
                    {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": wd},
                    {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0},
                ]
                optimizer = hydra.utils.instantiate(
                    optim, params=optimizer_grouped_parameters,
                    _convert_="all"
                )
                scheduler = get_linear_schedule_with_warmup(
                    # TODO warmup
                    optimizer, num_warmup_steps=math.ceil(self.total_steps * 0.2), num_training_steps=self.total_steps
                )
            elif modality == "v":
                # video
                # set BN weight_decay = 0
                bn_params = []
                non_bn_params = []
                for name, p in model.named_parameters():
                    if "bn" in name:
                        bn_params.append(p)
                    else:
                        non_bn_params.append(p)

                optim_params = [
                    {"params": bn_params, "weight_decay": 0.0},
                    {"params": non_bn_params, "weight_decay": optim.weight_decay},
                ]
                optimizer = hydra.utils.instantiate(
                    optim, params=optim_params,
                    _convert_="all"
                )
                if sch is not None:
                    scheduler = hydra.utils.instantiate(
                        sch, optimizer=optimizer,
                        _convert_="all"
                    )
            optimizers.append(optimizer)
            if scheduler is not None:
                schedulers.append(scheduler)
        # add aligner otpim
        wd = self.hparams.aligner_optim.pop("weight_decay")
        params = [
            {"params": self.aligner1.parameters(), "weight_decay": wd},
            {"params": self.aligner2.parameters(), "weight_decay": wd},
        ]
        optimizers.append( hydra.utils.instantiate(
            self.hparams.aligner_optim, params=params,
            _convert_="partial"
        ))
        return optimizers, schedulers