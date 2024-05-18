export TF_FORCE_GPU_ALLOW_GROWTH='true'
export HYDRA_FULL_ERROR=1
#--multirun
# python run.py -m name=barl_pooling alg=barl env=pendulum num_iters=200 n_rand_acqopt=1000 eval_frequency=10 seed="range(5)" hydra/launcher=joblib

# python run.py -m name=compare_orig_1000 alg=barl alg.compare_mode=true env=pendulum num_iters=150 n_rand_acqopt=1000 eval_frequency=2 seed="range(4)" hydra/launcher=joblib
# python run.py -m name=compare_orig_100 alg=barl alg.compare_mode=true env=pendulum num_iters=150 n_rand_acqopt=100 eval_frequency=2 seed="range(4)" hydra/launcher=joblib
# python run.py -m name=compare_orig_50 alg=barl alg.compare_mode=true env=pendulum num_iters=150 n_rand_acqopt=50 eval_frequency=2 seed="range(4)" hydra/launcher=joblib
# python run.py -m name=compare_opt_1000 alg=opt_barl alg.compare_mode=true env=pendulum num_iters=150 n_rand_acqopt=1000 eval_frequency=2 seed="range(4)" hydra/launcher=joblib
# python run.py -m name=compare_opt_100 alg=opt_barl alg.compare_mode=true env=pendulum num_iters=150 n_rand_acqopt=100 eval_frequency=2 seed="range(4)" hydra/launcher=joblib
# python run.py -m name=compare_opt_50 alg=opt_barl alg.compare_mode=true env=pendulum num_iters=150 n_rand_acqopt=50 eval_frequency=2 seed="range(4)" hydra/launcher=joblib
# python run.py -m name=compare_orig_10 alg=barl alg.compare_mode=true env=pendulum num_iters=150 n_rand_acqopt=10 eval_frequency=2 seed="range(4)" hydra/launcher=joblib
# python run.py -m name=compare_opt_10 alg=opt_barl alg.compare_mode=true env=pendulum num_iters=150 n_rand_acqopt=10 eval_frequency=2 seed="range(4)" hydra/launcher=joblib

python run.py -m name=timed_rollout alg=barl num_iters=150 alg.simple_rollout_sampling=true n_rand_acqopt=1000 eval_frequency=2 env=pendulum seed="range(4)" hydra/launcher=joblib