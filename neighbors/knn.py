import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from base import Predictor
from spatial.distances import distance_matrix

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

        return self

    def predict(self, X):
        distances = distance_matrix(self.X, X)
        indices   = np.argsort(distances, axis=0)[:self.k]
        labels    = self.y[indices]

        predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), arr=labels, axis=0)

        # predictions = np.zeros(X.shape[0])
        # for i, pred in enumerate(self.y[indices]):
        #     labels, counts = np.unique(pred, return_counts=True)
        #     predictions[i] = labels[np.argmax(counts)]

        return predictions

if __name__ == "__main__":
    X = np.array(
        [[1, 2, 3],
        [1, 0, 1],
        [-1, 0, 1]]
    )

    y = np.array([1, 1, 0])

    target = np.array(
        [[1, 2, 3], 
        [1, 0, 1]]
    )

    knn = KNeighborsClassifier(k = 2)
    knn.fit(X, y)
    print("MLKit", knn.predict(target))


    import sklearn.neighbors as sn
    clf = sn.KNeighborsClassifier(n_neighbors=2)
    clf.fit(X, y)
    print("SKLearn", clf.predict(target))

