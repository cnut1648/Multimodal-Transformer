#! /bin/bash

# Roberta
/home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py \
    exp=cmu_mosei_clf7 modal=roberta_ER \
    datamodule.batch_size=32 \
    model.model.arch_name='roberta-large' model.model.freeze_strategy="no-freeze" \
    mode=infer ckpt_path='/home/ICT2000/jxu/PER/logs/save/CMU-MOSEI-text-clf/15/ckpt/epoch18-F10.50-acc0.51.ckpt'
