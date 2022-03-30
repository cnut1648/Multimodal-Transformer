#! /bin/bash

# debug run
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py \
#     exp=msp_improv_reg modal=roberta_ER \
#     datamodule.batch_size=64 datamodule.fold=1 \
#     model.optim.lr=5e-3 \
#     model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze"
########################################
# MSP reg, bsz=128 (~47114mb)
########################################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=msp_improv_reg modal=roberta_ER \
#     datamodule.batch_size=128 datamodule.fold='range(1,13)' \
#     model.optim.lr='4e-5,1e-3,1e-4,5e-6' \
#     model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze"

########################################
# MSP clf, bsz=128 (~47114mb)
########################################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=msp_improv_clf modal=roberta_ER \
#     datamodule.batch_size=128 datamodule.fold='range(1,13)' \
#     model.optim.lr='4e-5,1e-3,1e-4,5e-6' \
#     model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze"


########################################
# CMU MOSEI, bsz=32 ~42830mb
########################################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=roberta_ER \
#     datamodule.batch_size=32 \
#     model.optim.lr=1e-5 \
#     model.ordinal_regression=coral,corn,null \
#     model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze"

/home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
    exp=cmu_mosei_clf7,cmu_mosei_l1 \
    modal=roberta_ER \
    ++datamodule.batch_size=16 \
    +trainer.accumulate_grad_batches=4 \
    model.optim.lr=0.000018534291964296485 \
    model.optim.weight_decay=0.0052960752776401955 \
    model.ordinal_regression=null \
    model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze" \


########################################
# CMU MOSEI hsearch, bsz=32 ~42830mb
########################################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=roberta_ER \
#     datamodule.batch_size=16 \
#     model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze" \
#     hparams_search=WA