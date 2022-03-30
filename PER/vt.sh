#! /bin/bash

# aff-wild2 pretrained
####################
# MOSEI bsz=8 (~32717 mb)
####################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=single_metatimesformer_ER.yaml \
#     datamodule.batch_size=16 \
#     model.accumulate_grad_batches=4 \
#     model.text_model.optim.lr='choice(1E-4,1E-5,1E-3)' \
#     model.text_model.optim.weight_decay=0.001781 \
#     model.text_model.model.freeze_strategy='no-freeze' \
#     model.video_model.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
#     model.video_model.optim.lr='choice(1E-4,1E-3,1E-5)' \
#     model.video_model.optim.weight_decay=0.007237

# grid
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=single_metatimesformer_ER_noexchange.yaml \
#     datamodule.batch_size=8 \
#     model.link_mode=sync \
#     model.accumulate_grad_batches=4 \
#     model.model1.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
#     model.model1.optim.lr=1e-5 \
#     model.model1.optim.weight_decay=0.007237 \
#     model.model1.weight=1 \
#     model.model2.optim.lr=3e-5 \
#     model.model2.optim.weight_decay=0.001781 \
#     model.model2.model.freeze_strategy='no-freeze' \
#     model.model2.weight=10,30,50,100 \
#     model.aligner_optim.lr=1e-5
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=single_metatimesformer_ER.yaml \
#     datamodule.batch_size=8 \
#     model.link_mode=sync \
#     model.accumulate_grad_batches=4 \
#     model.model1.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
#     model.model1.optim.lr=1e-5 \
#     model.model1.optim.weight_decay=0.007237 \
#     model.model1.weight=1 \
#     model.model2.optim.lr=3e-5 \
#     model.model2.optim.weight_decay=0.001781 \
#     model.model2.model.freeze_strategy='no-freeze' \
#     model.model2.weight=10,30,50,100 \
#     model.aligner_optim.lr=1e-5

# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=single_metatimesformer_ER.yaml \
#     datamodule.batch_size=8 \
#     model.link_mode=sync \
#     model.accumulate_grad_batches=4 \
#     model.model1.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
#     model.model1.optim.lr=4e-5 \
#     model.model1.optim.weight_decay=0.007237 \
#     model.model1.weight=1 \
#     model.model2.optim.lr=3e-6 \
#     model.model2.optim.weight_decay=0.001781 \
#     model.model2.model.freeze_strategy='no-freeze' \
#     model.model2.weight=10,30,50,100 \
#     model.aligner_optim.lr=1e-5

# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=single_metatimesformer_ER.yaml \
#     datamodule.batch_size=8 \
#     model.link_mode=sync \
#     model.accumulate_grad_batches=4 \
#     model.model1.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
#     model.model1.optim.lr=4e-5 \
#     model.model1.optim.weight_decay=0.007237 \
#     model.model1.weight=1 \
#     model.model2.optim.lr=7e-6 \
#     model.model2.optim.weight_decay=0.001781 \
#     model.model2.model.freeze_strategy='no-freeze' \
#     model.model2.weight=10,30,50,100 \
#     model.aligner_optim.lr=1e-5

/home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
    exp=cmu_mosei_clf7 modal=single_metatimesformer_ER.yaml \
    datamodule.batch_size=8 \
    model.link_mode=sync \
    model.accumulate_grad_batches=4 \
    model.model1.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
    model.model1.optim.lr=6e-5 \
    model.model1.optim.weight_decay=0.007237 \
    model.model1.weight=1 \
    model.model2.optim.lr=1e-6 \
    model.model2.optim.weight_decay=0.001781 \
    model.model2.model.freeze_strategy='no-freeze' \
    model.model2.weight=10,30,50,100 \
    model.aligner_optim.lr=1e-5
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=single_metatimesformer_ER.yaml \
#     datamodule.batch_size=8 \
#     model.link_mode=sync \
#     model.accumulate_grad_batches=2,4,8 \
#     model.model1.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
#     model.model1.optim.lr='choice(5E-4,5E-3,5E-5)' \
#     model.model1.optim.weight_decay=0.007237 \
#     model.model1.weight='${ratio: 0.3222, 0.5056}' \
#     model.model2.optim.lr='choice(5E-4,5E-5,5E-3)' \
#     model.model2.optim.weight_decay=0.001781 \
#     model.model2.model.freeze_strategy='no-freeze' \
#     model.model2.weight='${ratio: 50.56, 32.22}' \
#     model.aligner_optim.lr='choice(5E-4,5E-5,5E-3)' 

# 1
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=single_metatimesformer_ER.yaml \
#     datamodule.batch_size=8 \
#     model.accumulate_grad_batches=4 \
#     model.link_mode=sync \
#     model.model1.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
#     model.model1.optim.lr=0.0000010496586466748968 \
#     model.model1.optim.weight_decay=0.008506483774969657 \
#     model.model1.weight=1 \
#     model.model2.optim.lr=0.0000010088701793574817 \
#     model.model2.optim.weight_decay=0.0038415503235051103 \
#     model.model2.model.freeze_strategy='no-freeze' \
#     model.model2.weight=10 \
#     model.aligner_optim.lr=0.00004199053071449519

####################
# MOSEI hsearch bsz=8 (~32717 mb)
####################
# /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
#     exp=cmu_mosei_clf7 modal=single_metatimesformer_ER.yaml \
#     datamodule.batch_size=8 \
#     model.model1.weight='${ratio: 0.3222, 0.5056}' \
#     model.model1.model.pretrained_model='/home/ICT2000/yin/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth' \
#     model.model2.model.freeze_strategy='no-freeze' \
#     model.model2.weight='${ratio: 50.56, 32.22}' \
#     hparams_search=biWA