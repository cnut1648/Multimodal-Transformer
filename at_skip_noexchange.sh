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
        ckpt1='M2022-IEMOCAP-audio-clf:3';
        ckpt2='M2022-IEMOCAP-text-clf:4';
    elif [[ $fold == 2 ]]; then
        ckpt1='M2022-IEMOCAP-audio-clf:4';
        ckpt2='M2022-IEMOCAP-text-clf:5';
    elif [[ $fold == 3 ]]; then
        ckpt1='M2022-IEMOCAP-audio-clf:5';
        ckpt2='M2022-IEMOCAP-text-clf:6';
    elif [[ $fold == 4 ]]; then
        ckpt1='M2022-IEMOCAP-audio-clf:6';
        ckpt2='M2022-IEMOCAP-text-clf:7';
    else
        ckpt1='M2022-IEMOCAP-audio-clf:7';
        ckpt2='M2022-IEMOCAP-text-clf:8';
    fi;
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    echo "cuda=$cuda $exp [bsz=$bsz, acc=$acc, lr=$lr1|$lr2|$lr_align]";
    echo "FOLD $fold, use ckpt1=$ckpt1 & ckpt2=$ckpt2"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    CUDA_VISIBLE_DEVICES=$cuda /home/ICT2000/jxu/miniconda3/envs/conv/bin/python run.py -m \
        model._target_='src.models.bi_module_linear_stat.BiModule' \
        exp=$exp modal=at_hubert_roberta_noexchange \
        callbacks.early_stopping.patience=20 \
        model.unfreeze_epoch=5 \
        datamodule.batch_size=$bsz \
        model.accumulate_grad_batches=$acc \
        model.link_mode=sync \
        model.model1.optim.lr=$lr1 \
        model.model1.optim.weight_decay=0.01 \
        model.model1.wandb_path=$ckpt1 \
        model.model1.weight=1 \
        model.model2.optim.lr=$lr2 \
        model.model2.optim.weight_decay=0.01 \
        model.model2.wandb_path=$ckpt2 \
        model.model2.weight=1 \
        model.aligner_optim.lr=$lr_align \
        trainer=$trainer \
        datamodule.fold=$fold;
done;