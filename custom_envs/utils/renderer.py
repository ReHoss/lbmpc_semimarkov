"""
Renderer class to store and render trajectories.
TODO:
- There is an option to record all the trajectories in a .csv file, but it as to be activated in some way.
Like using a dict rendering instead of a path in order to flag the trajectory writing.


"""

import datetime
import pathlib

import numpy as np
import pandas as pd


class Renderer:
    def __init__(self,
                 nx: int,
                 na: int,
                 t_max: int,
                 render_mode: str,
                 path_rendering: str,
                 path_output_data: str):
        self.nx = nx
        self.na = na
        self.t_max = t_max
        # Trajectory matrix storage not yet initialised.
        self.matrix_current_trajectory = None
        self.matrix_current_control = None
        self.matrix_current_action = None
        self.list_datetime = None
        # Initialisation
        self.reset_trajectory()

        self.trajectory_rendered = 0
        self.render_mode = render_mode
        self.path_rendering = path_rendering
        self.path_output_data = path_output_data
        # Create directory to store the potentially rendered data
        pathlib.Path(self.path_rendering).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{self.path_output_data}/trajectories").mkdir(parents=True, exist_ok=True)

        # Initialise empty trajectory database, order matters for writing .csv
        self.list_multi_index = [(i, j) for i in ["state", "control"]
                                 for j in range(self.nx)] + [('action', i) for i in range(self.na)]
        # (pd.DataFrame(columns=pd.MultiIndex.from_tuples(self.list_multi_index),
        #               index=pd.MultiIndex.from_tuples([], names=["trajectory_id", "time", "step"]))
        #  .to_csv(path_or_buf=f"{self.path_output_data}/trajectories/trajectory.csv"))

    def reset_trajectory(self):
        self.matrix_current_trajectory = np.zeros((self.t_max + 1, self.nx))
        self.matrix_current_control = np.zeros((self.t_max + 1, self.nx))
        self.matrix_current_action = np.zeros((self.t_max + 1, self.na))
        self.list_datetime = []

    def render(self, t: float, array_state: np.array, array_action: np.array, array_control: np.array):
        self.matrix_current_trajectory[t] = array_state
        self.matrix_current_action[t] = array_action
        self.matrix_current_control[t] = array_control
        self.list_datetime.append(datetime.datetime.now())

        if t == self.t_max:
            # Process data

            df_trajectory = (pd.concat([pd.DataFrame(self.matrix_current_trajectory),
                                        pd.DataFrame(self.matrix_current_control),
                                        pd.DataFrame(self.matrix_current_action)], axis="columns",
                                       keys=['state', 'control', 'action'])
                             .assign(trajectory_id=self.trajectory_rendered,
                                     time=self.list_datetime,
                                     step=list(range(self.t_max + 1)))
                             .set_index(["trajectory_id", "time", "step"]))

            # Write data for post-processing
# df_trajectory.to_csv(path_or_buf=f"{self.path_output_data}/trajectories/trajectory.csv", mode="a", header=False)
            # Write data for rendering
            df_trajectory.to_csv(path_or_buf=f"{self.path_rendering}/trajectory.csv")
            # Reinitialise data:
            self.reset_trajectory()
            self.trajectory_rendered += 1
