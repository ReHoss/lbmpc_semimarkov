import logging
from gymnasium import register
from barl.envs.pendulum import PendulumEnv, pendulum_reward
from barl.envs.pilco_cartpole import (
    CartPoleSwingUpEnv,
    pilco_cartpole_reward,
    tf_pilco_cartpole_reward,
)

from barl.envs.lava_path import LavaPathEnv, lava_path_reward, ShortLavaPathEnv
from barl.envs.weird_gain import WeirdGainEnv, weird_gain_reward, WeirderGainEnv

# register each environment we wanna use
register(
    id="bacpendulum-v0",
    entry_point=PendulumEnv,
)
register(
    id="pilcocartpole-v0",
    entry_point=CartPoleSwingUpEnv,
)
register(
    id="lavapath-v0",
    entry_point=LavaPathEnv,
)
register(
    id="shortlavapath-v0",
    entry_point=ShortLavaPathEnv,
)
register(
    id="weirdgain-v0",
    entry_point=WeirdGainEnv,
)
register(
    id="weirdergain-v0",
    entry_point=WeirderGainEnv,
)
reward_functions = {
    "bacpendulum-v0": pendulum_reward,
    "pilcocartpole-v0": pilco_cartpole_reward,
    "lavapath-v0": lava_path_reward,
    "shortlavapath-v0": lava_path_reward,
    "weirdgain-v0": weird_gain_reward,
    "weirdergain-v0": weird_gain_reward,
}
tf_reward_functions = {
    "bacpendulum-v0": pendulum_reward,
    "pilcocartpole-v0": tf_pilco_cartpole_reward,
}
# mujoco stuff
try:
    from barl.envs.reacher import BACReacherEnv, reacher_reward, tf_reacher_reward

    register(
        id="bacreacher-v0",
        entry_point=BACReacherEnv,
    )
    reward_functions["bacreacher-v0"] = reacher_reward
except Exception as e:
    logging.warning("mujoco not found, skipping those envs")
    logging.warning(e)
# try:
#     from barl.envs.beta_tracking_env import BetaTrackingGymEnv, beta_tracking_rew
#     from barl.envs.tracking_env import TrackingGymEnv, tracking_rew
#     from barl.envs.betan_env import BetanRotationEnv, betan_rotation_reward

#     register(
#         id="betatracking-v0",
#         entry_point=BetaTrackingGymEnv,
#     )
#     reward_functions["betatracking-v0"] = beta_tracking_rew
# except:
#     logging.info("old fusion dependencies not found, skipping")
# try:
#     from fusion_control.envs.gym_env import BetaTrackingGymEnv as NewBetaTrackingGymEnv
#     from fusion_control.envs.gym_env import (
#         BetaRotationTrackingGymEnv,
#         BetaRotationTrackingMultiGymEnv,
#     )
#     from fusion_control.envs.rewards import TrackingReward

#     register(
#         id="newbetatracking-v0",
#         entry_point=NewBetaTrackingGymEnv,
#     )
#     _beta_env = NewBetaTrackingGymEnv()
#     reward_functions["newbetatracking-v0"] = _beta_env.get_reward
#     register(
#         id="newbetarotation-v0",
#         entry_point=BetaRotationTrackingGymEnv,
#     )
#     _beta_rotation_env = BetaRotationTrackingGymEnv()
#     reward_functions["newbetarotation-v0"] = _beta_rotation_env.get_reward
#     register(
#         id="multibetarotation-v0",
#         entry_point=BetaRotationTrackingMultiGymEnv,
#     )
#     _multi_beta_rotation_env = BetaRotationTrackingMultiGymEnv()
#     reward_functions["multibetarotation-v0"] = _multi_beta_rotation_env.get_reward
# except:
#     logging.warning("new fusion dependencies not found, skipping")


# Remy environments
try:
    from barl.envs.lorenz import LorenzEnv, lorenz_reward
    from barl.envs.lorenz_new import LorenzEnvNew, lorenz_reward_new
    from barl.envs.kuramoto_sivashinsky import KuramotoSivashinsky, KS_reward
    from barl.envs.pendulum_semimarkov import PendulumEnvSemiMarkov, pendulum_semimarkov_reward
    from barl.envs.pendulum_semimarkov_new import PendulumEnvSemiMarkovNew, pendulum_semimarkov_reward_new
    register(
        id="lorenz-v0",
        entry_point=LorenzEnv,
    )
    reward_functions["lorenz-v0"] = lorenz_reward
    tf_reward_functions["lorenz-v0"] = lorenz_reward

    register(
        id="lorenz-new-v0",
        entry_point=LorenzEnvNew,
    )
    reward_functions["lorenz-new-v0"] = lorenz_reward_new
    tf_reward_functions["lorenz-new-v0"] = lorenz_reward_new

    register(
        id='ks-v0',
        entry_point=KuramotoSivashinsky,
    )
    reward_functions['ks-v0'] = KS_reward
    tf_reward_functions['ks-v0'] = KS_reward

    register(
        id='bacpendulum-semimarkov-v0',
        entry_point=PendulumEnvSemiMarkov,
    )
    reward_functions['bacpendulum-semimarkov-v0'] = pendulum_semimarkov_reward
    tf_reward_functions['bacpendulum-semimarkov-v0'] = pendulum_semimarkov_reward

    register(
        id='bacpendulum-semimarkov-new-v0',
        entry_point=PendulumEnvSemiMarkovNew,
    )
    reward_functions['bacpendulum-semimarkov-new-v0'] = pendulum_semimarkov_reward_new
    tf_reward_functions['bacpendulum-semimarkov-new-v0'] = pendulum_semimarkov_reward_new


except Exception as err:
    print(err)
    raise "Custom environments not found"
