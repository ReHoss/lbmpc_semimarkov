from collections import defaultdict
from barl.viz.plot import (
    plot_pendulum,
    plot_cartpole,
    plot_pilco_cartpole,
    plot_acrobot,
    # noop,
    plot_lava_path,
    plot_weird_gain,
    make_plot_obs,
    plot_generic,
    scatter,
    plot,
    noop,
    plot_lorenz,  # CHANGES @REMY: Added plot_lorenz
    plot_lorenz_new,  # CHANGES @REMY: Added plot_lorenz (new)
    plot_pendulum_semimarkov,  # CHANGES @REMY: Added plot_pendulum_semimarkov
    plot_pendulum_semimarkov_new,  # CHANGES @REMY: Added plot_pendulum_semimarkov_new
    plot_pendulum_trigo,
)



_plotters = {
    "bacpendulum-v0": plot_pendulum,
    "bacpendulum-test-v0": plot_pendulum,
    "bacpendulum-tight-v0": plot_pendulum,
    "bacpendulum-medium-v0": plot_pendulum,
    "petscartpole-v0": plot_cartpole,
    "pilcocartpole-v0": plot_pilco_cartpole,
    "bacrobot-v0": plot_acrobot,
    "bacswimmer-v0": plot_generic,
    "bacreacher-v0": plot_generic,
    "bacreacher-tight-v0": plot_generic,
    "lavapath-v0": plot_lava_path,
    "shortlavapath-v0": plot_lava_path,
    "betatracking-v0": plot_generic,
    "betatracking-fixed-v0": plot_generic,
    "plasmatracking-v0": plot_generic,
    "bachalfcheetah-v0": noop,
    "weirdgain-v0": plot_weird_gain,
    "lorenz-v0": plot_lorenz,  # CHANGES @REMY: Added plot_lorenz
    "lorenz-new-v0": plot_lorenz_new,  # CHANGES @REMY: Added plot_lorenz
    "bacpendulum-semimarkov-v0": plot_pendulum_semimarkov,  # CHANGES @REMY: Added plot_pendulum_semimarkov
    "bacpendulum-semimarkov-new-v0": plot_pendulum_semimarkov_new,  # CHANGES @REMY: Added plot_pendulum_semimarkov_new
    "bacpendulum-trigo-v0": plot_pendulum_trigo,
}
plotters = defaultdict(lambda: plot_generic)
plotters.update(_plotters)

# Changes @REMY: Add global variable to store the environment name

TUPLE_ENVIRONMENTS_NAME = ("bacpendulum-trigo-v0", "ks-v0")
