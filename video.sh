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

if [[ $ds == "iemocap" ]]; then
    echo "iemocap";
    bsz=32;
    exp=iemocap_clf;
    folds=${5:-1,2,3,4,5}
    # folds="1,2,3,4,5"
    # folds=1
    # folds=2
elif [[ $ds == "mosei" ]]; then
    echo "mosei";
    bsz=32;
    exp=cmu_mosei_l1;
    folds=1
else
    echo "unknown ds";
    exit 0;
fi;

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
echo "cuda=$cuda $exp FOLD$folds+[bsz=$bsz, acc=$acc, lr=$lr]";
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
CUDA_VISIBLE_DEVICES=$cuda /home/ICT2000/jxu/miniconda3/envs/benchmark/bin/python run.py -m \
    exp=$exp modal=v_timesformer \
    datamodule.batch_size=$bsz \
    model.optim.lr=$lr \
    model.optim.weight_decay=0.01 \
    trainer=$trainer \
    trainer.accumulate_grad_batches=$acc \
    datamodule.fold=$folds;

    # mode=test_without_fit \