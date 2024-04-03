#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/gymnasium_env/lib
export TF_FORCE_GPU_ALLOW_GROWTH='true'
export HYDRA_FULL_ERROR=1

# python policy/on_policy.py
python policy/replay_policy.py