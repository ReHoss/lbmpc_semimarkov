import gymnasium


class EnvBARL(gymnasium.Env):
    def __init__(
        self,
        action_max_value: float,
        action_space: gymnasium.spaces.Box,
        dt: float,
        horizon: int,
        state_space: gymnasium.spaces.Box,
        total_time_upper_bound: int,
    ):
        self.action_max_value: float = action_max_value
        self.dt: float = dt
        self.horizon: int = horizon  # TODO: Centralise with run.py version
        self.total_time_upper_bound: float = total_time_upper_bound

        self.action_space: gymnasium.spaces.Box = action_space
        self.state_space: gymnasium.spaces.Box = state_space
        super().__init__()
