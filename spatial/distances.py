from scipy.spatial import distance_matrix as dm
from unittest import TestCase
import numpy as np

def distance_matrix(A, B):
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), "Matrices should be numpy arrays"
    assert A.shape[1] == B.shape[1], "Matrices's row vectors should have same dimension."

    AA = (A * A).sum(axis=1).reshape(-1, 1)
    BB = (B * B).sum(axis=1)
    AB = A @ B.T

    return np.sqrt(AA + BB - 2 * AB)

if __name__ == "__main__":
    X = np.array(
        [[1, 2, 3],
        [1, 0, 1],
        [-1, 0, 1]]
    )

    target = np.array(
        [[1, 2, 3],
        [1, 0, 1]]
    )

    assert np.all(dm(X, target) == distance_matrix(X, target)), "Test failed."
