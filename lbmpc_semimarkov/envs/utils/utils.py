import numpy as np


def complex_to_real_flattened(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.real, x.imag])


def real_flattened_to_complex(x: np.ndarray) -> np.ndarray:
    x_complex = x[:x.shape[0] // 2] + 1j * x[x.shape[0] // 2:]
    return x_complex
