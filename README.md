# PER

control param in `configs/`

mostly you only need to specify `exp=?` and `modal=?`. exp controls which dataset (and in which mode eg L1 loss or CE loss by num of clases model outputs)

`modal` decide the model + modality. It will load `model` config group. The most important config are
```
model.optim.lr
datamodule.batch_size
trainer.accumlate_grad_batches
```


1. To run video modality ER model with timesformer, run

```
python run.py exp=msp_improv modal=timesformer_ER
```

2. To run text modality ER model with roberta, run

```
python run.py exp=msp_improv modal=roberta_ER
```

3. To run multimodal ER model with roberta (freeze) and meta's timesformer, run

```
python run.py exp=msp_improv modal=multi_metatimesformer_ER
    model.text_model.model.freeze_strategy=freeze \
```

4. 
Tune the param by
```
python run.py -m model.optim.lr='range(1e-5, 9e-5, 1e-5)' model.optim.weight_decay='choice(1e-2, 1e-4)'
```

OR use optuna. This one tune for valid acc, see `configs/hparams_search/acc.yaml` for the params that are tuned
```
python run.py -m \
    exp=msp_improv modal=roberta_ER \
    model.model.arch_name='bert-base-uncased' \
    hparams_search=acc
```


## Reproduce nan

Run 
```
python run.py exp=cmu_mosei_l1 \
    datamodule.num_workers=1 modal=a_conformer \
    datamodule.batch_size=1
```