import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only

from src import utils
from src.models.audio_module import AudioModule

log = utils.get_logger(__name__)


@rank_zero_only
def get_pl_logger(cfg: DictConfig) -> List[LightningLoggerBase]:
    loggers: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger = hydra.utils.instantiate(lg_conf)
                loggers.append(logger)
                while True:
                    try:
                        # sometimes fail for unknown reason
                        print(logger.experiment)
                        break
                    except BaseException:
                        pass

                # will not be in debug mode as in debug mode logger is deleted
                if "wandb" in lg_conf["_target_"] and "model_checkpoint" in cfg.callbacks:
                    # will upload this run to cloud in the end of the run
                    log.info(f"wandb url in {logger.experiment.url}")
                    project = logger.experiment.project
                    # get id from x-y-id
                    id = logger.experiment.name.rsplit('-', 1)[1]
                    cfg.callbacks.model_checkpoint.dirpath = os.path.join(
                        cfg.callbacks.model_checkpoint.dirpath, project, id, "ckpt"
                    )

    return loggers

def get_pl_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback {cb_name} <{cb_conf._target_}>")
                if "ModelCheckpoint" in cb_conf._target_:
                    task = cfg.datamodule.task
                    if task == "clf":
                        cb_conf["monitor"] = "valid/WF1"
                        cb_conf["filename"] = r"epoch{epoch:02d}-F1{valid/WF1:.2f}-acc{valid/WA:.2f}"
                    elif task == "reg":
                        cb_conf["monitor"] = "valid/valence_CCC"
                        cb_conf["filename"] = r"epoch{epoch:02d}-valence{valid/valence_CCC:.2f}-arousal{valid/arousal_CCC:.2f}"
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        # non recursive so that optim and scheulder can be passed as DictConfig
        config.model, _recursive_=False
    )

    # Init lightning loggers
    logger: List[LightningLoggerBase] = get_pl_logger(config)

    # Init lightning callbacks
    callbacks: List[Callback] = get_pl_callbacks(config)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if config.get("infer_mode"):
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path
        if ckpt_path != "auto":
            assert os.path.exists(ckpt_path)
        else:
            # get best ckpt from wandb API
            pass
        log.info("Starting testing!")
        ########################################################################
        # save huggingface ckpt
        ########################################################################
        # m = model.load_from_checkpoint(ckpt_path, strict=False)
        # # m = m.model.model
        # out = "/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-text-clf/fold1-23"
        # m.model.model.save_pretrained(out)
        # m.model.tokenizer.save_pretrained(out)
        # return None

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        return None

    # Train the model
    log.info("Starting training!")
    ckpt_path = None
    if config.get("resume_from_ckpt"):
        ckpt_path = config.get("resume_from_ckpt")
        assert os.path.exists(ckpt_path)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        if config.get("test_without_fit"):
            ckpt_path = None
        else:
            ckpt_path = "best"
            log.info(
                f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
            if trainer.logger is not None:
                trainer.logger.log_hyperparams({"best_model_path": trainer.checkpoint_callback.best_model_path})
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Return metric score for hyperparameter optimization
    return score
