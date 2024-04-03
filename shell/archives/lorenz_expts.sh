#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/gymnasium_env/lib
export TF_FORCE_GPU_ALLOW_GROWTH='true'
export HYDRA_FULL_ERROR=1

# python run.py -m name=hyperfit_lorenz_fixed fit_hypers=true env=lorenz num_iters=1 num_init_data=1500 test_set_size=1500 num_eval_trials=5 save_figures=true

# python run.py -m name=opt_fixed_short_lorenz_1000      alg=opt_barl alg.simple_rollout_sampling=false num_iters=150 n_rand_acqopt=1000 eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&
# python run.py -m name=barl_fixed_short_lorenz_1000     alg=barl     alg.simple_rollout_sampling=false num_iters=150 n_rand_acqopt=1000 eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&
# python run.py -m name=rollout_fixed_short_lorenz_1000  alg=barl     alg.simple_rollout_sampling=true  num_iters=400 n_rand_acqopt=1000 eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&

# python run.py -m name=opt_noisy_short_lorenz_100     alg=opt_barl alg.simple_rollout_sampling=false num_iters=150 n_rand_acqopt=100  eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&
# python run.py -m name=barl_noisy_short_lorenz_100    alg=barl alg.simple_rollout_sampling=false     num_iters=150 n_rand_acqopt=100  eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&
# python run.py -m name=rollout_noisy_short_lorenz_100 alg=barl alg.simple_rollout_sampling=true      num_iters=400 n_rand_acqopt=100  eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&

# python run.py -m name=opt_noisy_short_lorenz_50      alg=opt_barl alg.simple_rollout_sampling=false num_iters=150 n_rand_acqopt=50   eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&
# python run.py -m name=barl_noisy_short_lorenz_50     alg=barl alg.simple_rollout_sampling=false     num_iters=150 n_rand_acqopt=50   eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&
# python run.py -m name=rollout_noisy_short_lorenz_50  alg=barl alg.simple_rollout_sampling=true      num_iters=400 n_rand_acqopt=50   eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&

# python run.py -m name=opt_fixed_short_lorenz_10      alg=opt_barl alg.simple_rollout_sampling=false  num_iters=150 n_rand_acqopt=10   eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&
# python run.py -m name=barl_fixed_short_lorenz_10     alg=barl     alg.simple_rollout_sampling=false  num_iters=150 n_rand_acqopt=10   eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&
# python run.py -m name=rollout_fixed_short_lorenz_10  alg=barl     alg.simple_rollout_sampling=true   num_iters=400 n_rand_acqopt=10   eval_frequency=10 env=lorenz save_figures=true seed="range(4)" hydra/launcher=joblib #&


# wait