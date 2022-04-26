#! /bin/bash
cuda=$1
acc=$2
lr=$3
modality=$4
if [[ $cuda == *","* ]]; then
    # multi GPU, use DDP
    trainer="ddp"
else
    trainer="default"
fi;

pretrain="/shares/perception/yufeng/project/personalized_emotion_recognition/myTimesFormer/checkpoints/K600.pyth"
# pretrain="/home/ICT2000/jxu/Multimodal-Transformer/TimeSformer/ckpt/TimeSformer_divST_8x32_224_K600.pyth"
bsz=8;
exp=drive_and_act_clf;

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
echo "cuda=$cuda $exp $modality+[bsz=$bsz, acc=$acc, lr=$lr]";
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
CUDA_VISIBLE_DEVICES=$cuda /home/ICT2000/jxu/miniconda3/envs/benchmark/bin/python run.py -m \
    modal=v_timesformer \
    exp=$exp model.model.pretrained_model=$pretrain \
    datamodule.batch_size=$bsz \
    model.optim.lr=$lr \
    model.optim.weight_decay=0.01 \
    trainer=$trainer \
    trainer.accumulate_grad_batches=$acc \
    model.modality=$modality

    # mode=test_without_fit \