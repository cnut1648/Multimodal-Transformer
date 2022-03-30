#! /bin/bash

# use aff-wild2 pretrained

########################################
# MSP reg, bsz=32 (~26000mb)
########################################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=msp_improv_reg modal=metatimesformer_ER \
#     datamodule.fold='range(1,13)' datamodule.batch_size=32 \
#     model.optim.lr=1e-5 \
#     model.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/Aff-Wild2.pyth'

########################################
# MSP clf, bsz=32 (~26000mb)
########################################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=msp_improv_clf modal=metatimesformer_ER \
#     datamodule.fold='range(1,13)' datamodule.batch_size=32 \
#     model.optim.lr=1e-5 \
#     model.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/Aff-Wild2.pyth'


########################################
# CMU MOSEI clf 7, bsz=32 (~26000mb)
########################################
/home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
    exp=cmu_mosei_clf7 modal=metatimesformer_ER \
    datamodule.batch_size=32 \
    +trainer.accumulate_grad_batches=1,2,3 \
    model.optim.lr=1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-6,5e-6 \
    model.ordinal_regression=null \
    model.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/Aff-Wild2.pyth'
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_l1,cmu_mosei_clf7 modal=metatimesformer_ER \
#     ++datamodule.batch_size=32 \
#     +trainer.accumulate_grad_batches=2 \
#     model.optim.lr=0.00009206306424314276 \
#     model.optim.weight_decay=0.008589238107591518 \
#     model.ordinal_regression=null \
#     model.model.pretrained_model='/shares/perception/yufeng/project/personalized_emotion_recognition/myTimesFormer/checkpoints/Aff-Wild2.pyth'

########################################
# CMU MOSEI clf 7 hsearch, bsz=32 (~26000mb)
########################################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=metatimesformer_ER \
#     datamodule.batch_size=32 \
#     hparams_search=WA \
#     model.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/Aff-Wild2.pyth'