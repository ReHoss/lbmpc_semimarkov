import numpy as np
from scipy import linalg


# noinspection PyPep8Naming
def riccati(A: np.array, B: np.array, Q: np.array, R: np.array):
    X = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.conj().T @ X)
    E, _ = np.linalg.eig(A - B @ K)

    return X, K, E
