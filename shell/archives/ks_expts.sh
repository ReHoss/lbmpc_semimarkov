#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/gymnasium_env/lib
export TF_FORCE_GPU_ALLOW_GROWTH='true'
export HYDRA_FULL_ERROR=1

python run.py -m name=hyperfit_ks fit_hypers=true env=ks num_iters=1 num_init_data=1500 test_set_size=1500 num_eval_trials=5 save_figures=true
