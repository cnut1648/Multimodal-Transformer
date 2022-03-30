#! /bin/bash
cuda=$1
ds=$2
acc=$3
lr1=$4
lr2=$5
lr_align=$6
folds=$7
if [[ $cuda == *","* ]]; then
    # multi GPU, use DDP
    trainer="ddp"
else
    trainer="default"
fi;

if [[ $ds == "iemocap" ]]; then
    echo "iemocap";
    bsz=4;
    exp=iemocap_clf;
    folds=${7:-1,2,3,4,5};
    # folds="1,2,3,4,5"
elif [[ $ds == "mosei" ]]; then
    echo "mosei";
    bsz=2;
    exp=cmu_mosei_l1;
    folds=1
else
    echo "unknown ds";
    exit 0;
fi;

# loop comma separate string
for fold in ${folds//,/ }; do
    if [[ $fold == 1 ]]; then
        ckpt1='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-audio-clf/fold1-43';
        ckpt2='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-text-clf/fold1-23';
    elif [[ $fold == 2 ]]; then
        ckpt1='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-audio-clf/fold2-53';
        ckpt2='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-text-clf/fold2-26';
    elif [[ $fold == 3 ]]; then
        ckpt1='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-audio-clf/fold3-54';
        ckpt2='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-text-clf/fold3-30';
    elif [[ $fold == 4 ]]; then
        ckpt1='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-audio-clf/fold4-48';
        ckpt2='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-text-clf/fold4-36';
    else
        ckpt1='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-audio-clf/fold5-50';
        ckpt2='/home/ICT2000/jxu/PER/selected_ckpt/IEMOCAP-text-clf/fold5-35';
    fi;
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    echo "cuda=$cuda $exp [bsz=$bsz, acc=$acc, lr=$lr1|$lr2|$lr_align]";
    echo "FOLD $fold, use ckpt1=$ckpt1 & ckpt2=$ckpt2"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    CUDA_VISIBLE_DEVICES=$cuda /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
        exp=$exp modal=at_hubert_roberta_skip \
        callbacks.early_stopping.patience=100 \
        model.unfreeze_epoch=5 \
        datamodule.batch_size=$bsz \
        model.accumulate_grad_batches=$acc \
        model.link_mode=sync \
        model.model1.optim.lr=$lr1 \
        model.model1.optim.weight_decay=0.01 \
        model.model1.model.pretrain_path=$ckpt1 \
        model.model1.weight=1 \
        model.model2.optim.lr=$lr2 \
        model.model2.optim.weight_decay=0.01 \
        model.model2.model.arch_name=$ckpt2 \
        model.model2.weight=1 \
        model.aligner_optim.lr=$lr_align \
        trainer=$trainer \
        datamodule.fold=$fold;
done;