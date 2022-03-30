#! /bin/bash

/home/ICT2000/jxu/miniconda3/envs/benchmark/bin/python run.py -m \
    restore_from_run='CMU-MOSEI-text-clf:15' \
    ++model.ordinal_regression=coral,corn