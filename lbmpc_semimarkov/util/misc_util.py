"""
Miscellaneous utilities.
"""
import pathlib
from argparse import Namespace
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
import logging
import pickle
from collections import defaultdict
from scipy.stats import norm

# CHANGES @REMY: Start - Tensorboard logging
import io
import matplotlib.pyplot as plt
import random

import gpflow
import omegaconf

import lbmpc_semimarkov
# CHANGES @REMY: End - Tensorboard logging

def configure_seed_dtypes(dict_config: omegaconf.DictConfig) -> None:
    # Set plot settings
    logging.getLogger("matplotlib.font_manager").disabled = True

    # Set random seed
    seed = dict_config.seed
    random.seed(seed)  # CHANGES @REMY: add the ramdom module method to set the seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.run_functions_eagerly(dict_config.tf_eager)
    tf_dtype = lbmpc_semimarkov.util.misc_util.get_tf_dtype(dict_config.tf_precision)
    str_dtype = str(tf_dtype).split("'")[1]
    tf.keras.backend.set_floatx(str_dtype)
    gpflow.config.set_default_float(tf_dtype.as_numpy_dtype())

    # Check fixed_start_obs and num_samples_mc compatability
    assert (not dict_config.fixed_start_obs) or dict_config.num_samples_mc == 1, (
        f"Need to have a fixed start obs"
        f" ({dict_config.fixed_start_obs}) or only 1 mc sample"
        f" ({dict_config.num_samples_mc})"
    )  # NOQA

def dict_to_namespace(params):
    """
    If params is a dict, convert it to a Namespace, and return it.

    Parameters ----------
    params : Namespace_or_dict
        Namespace or dict.

    Returns
    -------
    params : Namespace
        Namespace of params
    """
    # If params is a dict, convert to Namespace
    if isinstance(params, dict):
        params = Namespace(**params)

    return params


class suppress_stdout_stderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    Source: https://stackoverflow.com/q/11130156
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class Dumper:
    def __init__(self, experiment_name, filename="info", path_expdir=None):
        cwd = Path.cwd()
        # this should be the root of the repo
        self.expdir = cwd
        logging.info(f"Dumper dumping to {cwd}")
        self.info = defaultdict(list)
        if path_expdir is not None:  # CHANGES @REMY: specify path of the folder where to dump the experiment data; indeed so far it dump wrongly in cwd ?!
            self.expdir = pathlib.Path(path_expdir)
        self.info_path = self.expdir / f"{filename}.pkl"

        # CHANGES @REMY: Start - Tensorboard logging
        # Define the path where to store the tensorboard data
        self.path_tensorboard_data = f"{self.expdir}/tensorboard_data"
        Path(self.path_tensorboard_data).mkdir()
        self.list_name_images = ["ground_truth_1d", "sampled_1d", "postmean_1d"]
        # self.dict_tf_file_writer = {name: tf.summary.create_file_writer(self.path_tensorboard_data + f"/plots/{name}")
        #                             for name in self.list_name_images}
        self.tf_file_writer = tf.summary.create_file_writer(self.path_tensorboard_data + "/plots")

    def tf_plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def get_base_path(self):
        """
        Get project base path
        """
        current_path = os.path.dirname(os.path.abspath(__file__))
        return rf"{current_path}/../../"
    
    # CHANGES @STELLA: retreive ols experiment data
    def retrieve(self, experiment_name:str) -> int:
        """
        Retrieve data from previous experiment and copy to new experiment path
        This method runs on all seeds, do not put `seed_n` on experiment_name!
        :param str experiment_name: path of the previous experiment relative to project base
        :return int: iteration index to resume from
        """
        seed_n = str(self.expdir).split(r'/')[-1]
        expdir = rf"{self.get_base_path()}{experiment_name}/{seed_n}/"
        info_path = f"{expdir}info.pkl"
        try:
            with open(info_path, "rb") as f:
                self.info = pickle.load(f)
            self.save()
            logging.info(f"Resuming past experiment {expdir}")
            logging.info(f"Dumper dumping to {self.expdir}")
        except Exception as err:
            logging.error(err)
            logging.error(f"Could not retrieve past experiment on {info_path}")
        finally:
            return max(0, len(self.info['x']) - 1)

    def add(self, name, val, verbose=True, log_mean_std=False):
        if verbose:
            try:
                val = float(val)
                logging.info(f"{name}: {val:.3f}")
            except TypeError:
                logging.info(f"{name}: {val}")
            if log_mean_std:
                valarray = np.array(val)
                logging.info(
                    f"{name}: mean={valarray.mean():.3f} std={valarray.std():.3f}"
                )
        self.info[name].append(val)

    def extend(self, name, vals, verbose=False):
        if verbose:
            disp_vals = [f"{val:.3f}" for val in vals]
            logging.info(f"{name}: {disp_vals}")
        self.info[name].extend(vals)

    def save(self):
        logging.info(f"Saving the dump to {self.info_path}")
        with self.info_path.open("wb+") as f:
            pickle.dump(self.info, f)
        logging.info("Dump saved")

def batch_function(f, **kwargs):
    # naively batch a function by calling it on each element separately and making a list of those
    def batched_f(x_list, **kwargs):
        y_list = []
        for x in x_list:
            y_list.append(f(x, **kwargs))
        return y_list

    return batched_f


def make_postmean_fn(model, use_tf=False):
    def postmean_fn(x):
        mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)  # INFO @REMY: model is of class BatchMultiGpfsGp
        mu_list = np.array(mu_list)
        mu_tup_for_x = list(zip(*mu_list))
        return mu_tup_for_x

    if not use_tf:
        return postmean_fn

    def tf_postmean_fn(x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
        mu_tup_for_x = list(mu_list.numpy())
        return mu_tup_for_x

    return tf_postmean_fn


def mse(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    return np.mean(np.sum(np.square(y_hat - y), axis=1))


def model_likelihood(model, x, y):
    """
    assume x is list of n d_x-dim ndarrays
    and y is list of n d_y-dim ndarrays
    """
    # mu should be list of d_y n-dim ndarrays
    # cov should be list of d_y n-dim ndarrays
    n = len(x)
    mu, cov = model.get_post_mu_cov(x)
    y = np.array(y).flatten()
    mu = np.array(mu).T.flatten()
    cov = np.array(cov).T.flatten()
    white_y = (y - mu) / np.sqrt(cov)
    logpdfs = norm.logpdf(white_y)
    logpdfs = logpdfs.reshape((n, -1))
    avg_likelihood = logpdfs.sum(axis=1).mean()
    return avg_likelihood


def get_tf_dtype(precision):
    if precision == 32:
        return tf.float32
    elif precision == 64:
        return tf.float64
    else:
        raise ValueError(f"TF Precision {precision} not supported")


def flatten(policy_list):
    out = []
    for item in policy_list:
        if type(item) is list:
            out += item
        else:
            out.append(item)
    return out
