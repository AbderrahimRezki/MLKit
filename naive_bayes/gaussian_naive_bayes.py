import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base import Predictor
import numpy as np
import pandas as pd


class GaussianNaiveBayesClassifier:
    def __init__(self, std_correction_factor=1e-7):
        self.priors = {}
        self.likelihoods = {}
        self.classes = []
        self.attributes = []
        self.n_attributes = 0

        # To prevent std being 0 (because it will appear in the denominator later)
        self.std_correction_factor = std_correction_factor

    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(X)

        self.priors = dict(zip(self.classes, probabilities))
        self.attributes = X.columns.to_list()
        self.n_attributes = len(self.attributes)

        """
        Likelihoods:
        {
            class_: {
                means: [],
                stds: []
            },
            ...
        }
        """

        for class_ in self.classes:
            self.likelihoods[class_] = {}
            conditional_distribution = X[y == class_]

            means = conditional_distribution.mean(axis=0).to_numpy()
            stds = conditional_distribution.std(axis=0).to_numpy()

            self.likelihoods[class_]["means"] = means
            self.likelihoods[class_]["stds"] = stds


    def predict(self, X):
        return np.apply_along_axis(self.__predict, arr=X, axis=1)

    def __predict(self, x): 
        predictions = {class_: None for class_ in self.classes}

        for class_ in self.classes:
            probabilities = self.__fill_probabilities(x, class_)
            predictions[class_] = np.multiply.reduce(probabilities)

        return max(predictions.items(), key=lambda pair: pair[1])[0]

    def __fill_probabilities(self, x, class_):
        probabilities = np.zeros(len(x) + 1)
        probabilities[-1] = self.priors[class_]

        means, stds = self.likelihoods[class_].values()
        stds += self.std_correction_factor
        probabilities[:-1] = self.__gaussian_fn(x, means, stds)

        return probabilities

    def __gaussian_fn(self, x, means, stds):
        return (1 / (stds * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x - means) ** 2 / stds ** 2 )


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    print(X.head())

    clf = GaussianNaiveBayesClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    print((y_pred == y).sum() / len(X))

 