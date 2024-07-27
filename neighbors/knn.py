import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from base import Predictor
from spatial.distances import distance_matrix

from scipy.spatial import distance_matrix as dm

def mode(x):
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts, axis=0)]

class KNeighborsClassifier(Predictor):
    def __init__(self, k = 1):
        assert k > 0 and isinstance(k, int), "Param k must be a positive integer."
        self.k = k

    def fit(self, X, y):
        assert isinstance(X, np.ndarray), "X must be a numpy array."
        assert isinstance(y, np.ndarray), "y must be a numpy array."
        assert X.shape[0] == y.shape[0] , "Number of instances and number of labels should match."
        assert X.shape[0] >= self.k     , f"Expected k <= n_samples_fit, but k = {self.k}, n_samples_fit = {X.shape[0]}"

        self.X = X
        self.y = y

    def predict(self, X):
        predictions = []

        for i, x in enumerate(X):
            new_X = np.concatenate([X[:i], X[i+1:]], axis=0)
            new_y = np.concatenate([y[:i], y[i+1:]], axis=0)
            dists = distance_matrix(new_X, self.X)
            indices = np.argsort(dists, axis=0)[:self.k]
            labels = new_y[indices]
            predictions.append(mode(labels))

        return predictions

if __name__ == "__main__":
    X = np.array(
        [[1, 2, 3],
        [1, 0, 1],
        [-1, 0, 1],
        [-1, 1, 1],
        [-2, 3, 0]]
    )

    y = np.array([1, 0, 0, 1, 0])

    target = np.array(
        [[1, 2, 3],
        [1, 0, 1], 
        [1, 1, 1],
        [1, -1, -1],
        [1, 1, 0]]
    )

    knn = KNeighborsClassifier(k = 3)
    knn.fit(X, y)
    print("MLFS", knn.predict(target))


    import sklearn.neighbors as sn
    clf = sn.KNeighborsClassifier(n_neighbors= 3)
    clf.fit(X, y)
    print("SKLearn", clf.predict(target))

