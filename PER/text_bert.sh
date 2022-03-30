#! /bin/bash

########################################
# MSP reg
########################################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=msp_improv modal=roberta_ER \
#     datamodule.batch_size=128 datamodule.fold=1,2,3,4,5,6 \
#     model.optim.lr=1e-5 \
#     model.model.arch_name='bert-base-uncased' model.model.freeze_strategy="freeze"

# roberta
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=msp_improv modal=roberta_ER \
#     datamodule.batch_size=128 datamodule.fold=1,2,3,4,5,6 \
#     model.optim.lr=5e-3 \
#     model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze"

# bert
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=msp_improv modal=roberta_ER \
#     datamodule.batch_size=128 datamodule.fold=1,2,3,4,5,6 \
#     model.optim.lr=0.0008723037378105589 \
#     model.model.arch_name='bert-base-uncased' model.model.freeze_strategy="no-freeze"
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=msp_improv modal=roberta_ER \
#     datamodule.fold=$1 datamodule.batch_size=128 \
#     model.optim.lr=1e-5 \
#     hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1


# CMU MOSEI
# roberta
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei modal=roberta_ER \
#     datamodule.num_classes=7 \
#     datamodule.batch_size=32 \
#     model.optim.lr=5e-3 \
#     model.ordinal_regression=coral \
#     model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze"