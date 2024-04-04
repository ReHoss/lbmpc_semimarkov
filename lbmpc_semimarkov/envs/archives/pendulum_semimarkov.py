import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from os import path


# noinspection DuplicatedCode
class PendulumEnvSemiMarkov(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self,
                 seed,
                 dtype,
                 dict_pde_config,
                 dict_init_condition_config,
                 render_mode=None):

        # Assert dict_pde_config contains the right keys
        assert dict_pde_config.keys() == {"ep_length", "dt", "max_delay", "min_delay", "fixed_dt"}

        # Set gym env seed
        self.seed(seed)

        # Set semi-markov parameters
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.pendulum_length = 1.0
        self.viewer = None
        self.horizon = dict_pde_config["ep_length"]
        self.periodic_dimensions = []
        self.render_mode = render_mode

        # Set semi-markov parameters  # INFO @REMY: added by Remy from the original
        self.min_delay = dict_pde_config["min_delay"]
        self.max_delay = dict_pde_config["max_delay"]
        assert self.min_delay < self.max_delay, "Wrong value: self.min_delay >= self.max_delay"

        # Set the delay mode
        self.fixed_dt = dict_pde_config["fixed_dt"]
        assert type(self.fixed_dt) is bool, "Wrong type: self.fixed_dt should be a boolean"

        self.total_time_upper_bound = self.max_delay * self.horizon
        # self.name_delay_distribution = "uniform"
        self.time = None
        self.current_iteration_index = 0  # INFO @REMY: this should be bounded by horizon

        # Set initial condition parameters
        self.init_cond_type = dict_init_condition_config["init_cond_type"]
        self.init_cond_scale = dict_init_condition_config["init_cond_scale"]

        assert self.init_cond_type in {"uniform", "equilibrium"}
        if self.init_cond_type == "uniform":
            assert self.init_cond_scale == 0.0, "init_cond_scale should be 0.0 for uniform init cond"
        elif self.init_cond_type == "equilibrium":
            assert self.init_cond_scale >= 0.0, "init_cond_scale should be positive for equilibrium init cond"

        # Set rendering attributes to None
        self.pole_transform = None
        self.imgtrans = None
        self.img = None

        # Initialize attributes
        self.last_u = None
        self.t0 = None
        self.state = None

        # Set gym env spaces  # INFO @REMY: CFO means "change from original"

        self.max_angle = self.max_speed * self.horizon * self.dt
        observation_high = np.array([self.max_angle, self.max_speed], dtype=dtype)  # CFO
        observation_low = np.array([-self.max_angle, -self.max_speed], dtype=dtype)  # INFO @REMY: CFO

        action_high = np.array([self.max_torque, self.max_delay], dtype=dtype)  # INFO @REMY: CFO
        action_low = np.array([-self.max_torque, self.min_delay], dtype=dtype)  # INFO @REMY: CFO
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=dtype)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=dtype)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        """
        Here, P( x_t+1, \tau_t+1 | x_t, \tau_t, u_t ) is modelled.
        """
        # INFO @STELLA: this is where the real environment is queried, we will need this call count. ARLEADY EXISTS
        th, thdot = self.state  # th := theta

        assert self.current_iteration_index <= self.horizon, \
            f"current_iteration_index={self.current_iteration_index} > horizon={self.horizon}"

        force = u[0]
        if self.fixed_dt:
            new_random_dt = self.dt
        else:
            new_random_dt = u[1]

        assert new_random_dt <= self.max_delay, f"new_random_dt={new_random_dt} > max_delay={self.max_delay}"
        assert new_random_dt >= self.min_delay, f"new_random_dt={new_random_dt} < min_delay={self.min_delay}"
        # TODO: treat the case when adding new_random_dt to self.time exceeds self.total_time_upper_bound
        g = self.g
        m = self.m
        pendulum_length = self.pendulum_length

        force = np.clip(force, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering

        # INFO @REMY: I think the change of variable for \theta_here = \theta - \pi is done here.
        # INFO @REMY: Hence vertical orientation is considered;
        # INFO @REMY: we observe the dynamics of y = theta + pi and ydot = xdot in fact; sin(x + pi) = -sin(x)
        newthdot = (thdot + (-3 * g / (2 * pendulum_length) * np.sin(th + np.pi) + 3.0 / (
                m * pendulum_length ** 2) * force) * new_random_dt)
        unnorm_newth = th + newthdot * new_random_dt
        newth = unnorm_newth  # INFO @REMY: warning here HUGE difference ! this cause a discontinuity
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        # Update time
        self.time += new_random_dt
        self.current_iteration_index += 1

        costs = (angle_normalize(newth) - np.pi) ** 2 + 0.1 * newthdot ** 2 + 0.001 * (force ** 2)
        delta_s = np.array([unnorm_newth - th, newthdot - thdot])

        self.state = np.array([newth, newthdot]).astype(np.float64)
        done = False
        return self._get_obs(), -costs, done, False, {"delta_obs": delta_s}

    # noinspection PyUnresolvedReferences
    def reset(self, obs=None, seed=None, options=None):  # INFO @REMY: modified by Stella to be Gymnasium compliant
        super().reset(seed=seed)  # CHANGE @REMY: added seed
        if obs is None:
            if self.init_cond_type == "uniform":
                self.state = self.np_random.uniform(low=[0, - 1], high=[2 * np.pi, 1])
            elif self.init_cond_type == "equilibrium":
                array_noise = self.np_random.normal(loc=0.0, scale=self.init_cond_scale, size=2)
                self.state = np.array([0.0, 0.0]) + array_noise
        else:
            self.state = obs
        # TODO: we might want to reset the time t0 to 0,0 at any time ! this won't affect the dynamics
        self.current_iteration_index = 0  # INFO @REMY: this should be bounded by self.horizon
        self.time = 0.0  # INFO @REMY: added by Remy from the original
        self.last_u = None
        self.state = np.float64(self.state)
        return self.state, {}

    def _get_obs(self):
        return self.state

    def render(self):
        if self.viewer is None:
            # noinspection PyUnresolvedReferences
            from gymnasium.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(
        x):  # INFO @REMY: x is supposed to live in [-\pi, \pi] units while originally (books) it is in [0, 2\pi]
    return x % (2 * np.pi)  # INFO @REMY: dynamics is centered


# noinspection PyUnusedLocal
def pendulum_semimarkov_reward(x, next_obs, current_step):
    del current_step
    th = next_obs[..., 0]
    thdot = next_obs[..., 1]
    u = x[..., 2]  # INFO @REMY: here we shift by one index as the original
    costs = (angle_normalize(th) - np.pi) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    return -costs
