from typing import Any, List, Optional, Dict, Union

import torch, hydra
from pytorch_lightning import LightningModule

from omegaconf import DictConfig

from src.models import ModuleMetricMixin

from .components.TextModel import TextModel
from .components.VideoModel import VideoModel
from ..utils.modeling import weights_init
from coral_pytorch.losses import coral_loss, corn_loss
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label, corn_label_from_logits

class BaseModule(ModuleMetricMixin, LightningModule):
    def __init__(
        self, task: str, num_classes: int,
        model: DictConfig, optim: DictConfig, dataset: str,
        init: Optional[str] = None, ordinal_regression: Optional[str] = None,
        **kwargs
    ):
        super().__init__(task=task, num_classes=num_classes, dataset=dataset, ordinal_regression=ordinal_regression)

        self.model: Union[TextModel, VideoModel] = hydra.utils.instantiate(
            model, _recursive_=False
        )
        if init:
            self.apply(weights_init(init))

        if task == "clf":
            if ordinal_regression == "coral":
                self.criterion = coral_loss
                self.model.replace_linear(CoralLayer, num_classes)
            elif ordinal_regression == "corn":
                self.criterion = corn_loss
                self.model.replace_linear(torch.nn.Linear, num_classes-1)
            elif dataset in ["cmu_mosei", "cmu_mosi"] and num_classes == 1:
                # self.criterion = torch.nn.L1Loss()
                self.criterion = torch.nn.MSELoss()
            else:
                # vanila clf
                self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def forward(self, batch: dict):
        return self.model(**batch)
    
    def preprocess_batch(self, batch: dict):
        if self.hparams.task == "clf":
            labels = batch.pop("labels")
        else:
            valence, arousal = batch.pop("valence"), batch.pop("arousal")
            # (bsz, 2)
            labels = torch.stack([valence, arousal], 1)
        return batch, labels
    
    def postprocess_logits(self, logits, labels):
        """
        compute loss and preds and labels
        since using metrics, ensure labels are positive
        """
        if self.hparams.task == "reg":
            loss = self.criterion(logits, labels)
            preds = logits
        elif self.hparams.ordinal_regression == "coral":
            levels = levels_from_labelbatch(labels, num_classes=self.hparams.num_classes).to(self.device)
            loss = self.criterion(logits, levels)
            preds = proba_to_label(logits.sigmoid())
        elif self.hparams.ordinal_regression == "corn":
            loss = self.criterion(logits, labels, num_classes=self.hparams.num_classes)
            preds = corn_label_from_logits(logits)
        elif self.hparams.dataset in ["cmu_mosei", "cmu_mosi"] and self.hparams.num_classes == 1:
            assert logits.size(1) == 1
            # loss = self.criterion(logits, labels)
            loss = self.criterion(logits, labels.float())
            # labels [0, 6], thus clip [0, 6], (bsz, 1) to (bsz, )
            preds = logits.clip(-3., 3.).round().squeeze(-1).long() + 3
            labels = labels.clip(-3., 3.).round().long() + 3
        else:
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def compute_step(self, batch: Any, split: str):
        """
        compute one step and one backward
        """
        batch, labels = self.preprocess_batch(batch)
        logits = self.forward(batch)
        loss, preds, labels = self.postprocess_logits(logits, labels)
        self.log(f"{split}/step/loss", loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        ret = {"loss": loss, "preds": preds, "targets": labels}
        if self.hparams.dataset in ["cmu_mosei", "cmu_mosi"]:
            assert "labels2" in batch
            ret.update({"labels2": batch['labels2']})
        return ret
       
    def compute_step_end(self, outputs, split: str):
        """
        log, since DDP logging must be put in *_step_end method
        """
        losses, preds, labels = outputs["loss"], outputs["preds"], outputs["targets"]
        # log metrics
        if split != "test":
            self.mean_losses[f"{split}_losses"](losses)
        self.metrics[f"{split}_metrics"](preds, labels)
        if self.hparams.dataset in ["cmu_mosei", "cmu_mosi"]:
            metricname = f"{split}_{self.hparams.dataset.replace('cmu_','')}"
            self.metrics[metricname](preds, labels, outputs["labels2"])

    def agg_epoch(self, outputs: List[Any], split: str):
        # t = torch.cat([o["targets"] for o in outputs]).cpu().numpy()
        # p = torch.cat([o["preds"] for o in outputs]).cpu().numpy()
        if split != "test":
            loss = self.mean_losses[f"{split}_losses"].compute()
            self.mean_losses[f"{split}_losses"].reset()
            self.log(f"{split}/epoch/loss", loss, on_epoch=True, prog_bar=True)
        
        metrics = self.metrics[f"{split}_metrics"].compute()
        # make classwise e.g. valid/accuracy_neu separate namespace
        for k in list(metrics):
            if "_" in k:
                value = metrics.pop(k)
                splitname, k = k.split("/")
                metrics[f"{splitname}/classwise/{k}"] = value
        self.metrics[f"{split}_metrics"].reset()
        if self.hparams.dataset in ["cmu_mosei", "cmu_mosi"]:
            metricname = f"{split}_{self.hparams.dataset.replace('cmu_','')}"
            # eg {'corr': nan, 'acc2': 0.46153846153846156, 'f12': 0.291497975708502}
            mosei_metric = self.metrics[metricname].compute()
            for name, value in mosei_metric.items():
                metrics[f"{split}/{name}"] = value
            self.metrics[metricname].reset()

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def setup(self, stage: Optional[str] = None) -> None:
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        effective_batch_size = (self.trainer.datamodule.hparams.batch_size *
                                max(1, self.trainer.num_gpus) * self.trainer.accumulate_grad_batches)
        self.total_steps = int(
            (len(train_loader.dataset) // effective_batch_size) * float(self.trainer.max_epochs))

    def training_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "train")
    
    def training_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "train")

    def training_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "train")
    
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