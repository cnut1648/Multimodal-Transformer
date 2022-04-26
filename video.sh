#! /bin/bash
cuda=$1
ds=$2
acc=$3
lr=$4
folds=$5
if [[ $cuda == *","* ]]; then
    # multi GPU, use DDP
    trainer="ddp"
else
    trainer="default"
fi;

pretrain="/shares/perception/yufeng/project/personalized_emotion_recognition/myTimesFormer/checkpoints/Aff-Wild2.pyth"
if [[ $ds == "iemocap" ]]; then
    echo "iemocap";
    bsz=32;
    exp=iemocap_clf;
    folds=${5:-1,2,3,4,5}
elif [[ $ds == "msp" ]]; then
    echo "msp";
    bsz=4;
    exp=msp_improv_clf;
    folds=${5:-1,2,3,4,5,6,7,8,9,10,11,12}
elif [[ $ds == "mosei" ]]; then
    echo "mosei";
    bsz=32;
    exp=cmu_mosei_clf7;
    folds=1
elif [[ $ds == "mosi" ]]; then
    echo "mosi";
    bsz=32;
    exp=cmu_mosi_clf7;
    folds=1
elif [[ $ds == "nturgb" ]]; then
    echo "ntu rgb";
    bsz=8;
    exp=ntu_rgb60_clf;
    folds=${5:xview,xsub};
    pretrain="/home/ICT2000/jxu/Multimodal-Transformer/TimeSformer/ckpt/TimeSformer_divST_8x32_224_K600.pyth"
else
    echo "unknown ds";
    exit 0;
fi;

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
echo "cuda=$cuda $exp FOLD$folds+[bsz=$bsz, acc=$acc, lr=$lr]";
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
CUDA_VISIBLE_DEVICES=$cuda /home/ICT2000/jxu/miniconda3/envs/benchmark/bin/python run.py -m \
    modal=v_timesformer \
    exp=$exp model.model.pretrained_model=$pretrain \
    datamodule.batch_size=$bsz \
    model.optim.lr=$lr \
    model.optim.weight_decay=0.01 \
    trainer=$trainer \
    trainer.accumulate_grad_batches=$acc \
    datamodule.fold=$folds;

    # mode=test_without_fit \