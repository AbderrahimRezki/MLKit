from base_model import Model
from scipy.spatial import distance_matrix
from unittest import TestCase
import numpy as np

def distance_matrix(A, B):
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), "Matrices should be numpy arrays"
    assert A.shape[1] == B.shape[1], "Matrices's row vectors should have same dimension."



if __name__ == "__main__":
    X = np.array(
        [[1, 2, 3],
        [1, 0, 1],
        [-1, 0, 1]]
    )

    knn = KNearestNeighbors()
    knn.fit(X, np.zeros(X.shape[0]))
    knn.predict([[1, 2, 3]])
