#!/bin/bash
#SBATCH --job-name=barl_pend
#SBATCH --output=slurm-m_pendulum_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=2gb
#SBATCH --array=100

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/gymnasium_env/lib
export TF_FORCE_GPU_ALLOW_GROWTH='true'
export HYDRA_FULL_ERROR=1

python run.py -m name=roll_pendulum_$SLURM_ARRAY_TASK_ID  alg=barl      num_iters=150  alg.simple_rollout_sampling=true   n_rand_acqopt=$SLURM_ARRAY_TASK_ID  eval_frequency=10  env=pendulum  save_figures=true  seed="range(4)" hydra/launcher=joblib
python run.py -m name=m_barl_pendulum_$SLURM_ARRAY_TASK_ID  alg=barl      num_iters=150  alg.simple_rollout_sampling=false  n_rand_acqopt=$SLURM_ARRAY_TASK_ID  eval_frequency=10  env=pendulum  save_figures=true  seed="range(4)" hydra/launcher=joblib
python run.py -m name=opt_pendulum_$SLURM_ARRAY_TASK_ID   alg=opt_barl  num_iters=150  alg.simple_rollout_sampling=false  n_rand_acqopt=$SLURM_ARRAY_TASK_ID  eval_frequency=10  env=pendulum  save_figures=true  seed="range(4)" hydra/launcher=joblib
