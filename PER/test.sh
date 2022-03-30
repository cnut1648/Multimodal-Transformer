#! /bin/bash

/home/ICT2000/jxu/miniconda3/envs/conv/bin/python run2.py -m \
    exp=msp_improv modal=roberta_ER \
    datamodule.fold=1 datamodule.batch_size=128,2,3,1,23 \
    model.optim.lr=1e-5 \
#     hydra/launcher=submitit_local hydra.launcher.gpus_per_node=5