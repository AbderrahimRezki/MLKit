import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base import Predictor
import pandas as pd
import numpy as np

class CategoricalNaiveBayesClassifier(Predictor):
    def __init__(self, smoothing_k = 1):
        self.priors = {}
        self.likelihoods = {}
        self.totals = {}
        self.classes = []
        self.attributes = []
        self.n_attributes = 0
        self.smoothing_k = smoothing_k

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(X)

        self.priors = dict(zip(self.classes, probabilities))
        self.attributes = X.columns.to_list()
        self.n_attributes = len(self.attributes)

        """
        Likelihoods:
        {
            class: {
                attr: {
                    val: proba,
                    ...
                },
                ...
            },
            ...
        }
        """

        for class_ in self.classes:
            self.likelihoods[class_] = {}

            conditional_distribution = X[y == class_]
            self.totals[class_] = len(conditional_distribution)
            
            for attr in self.attributes:
                conditional_attr = conditional_distribution.loc[:, attr]
                values, counts = np.unique(conditional_attr, return_counts=True)
                self.likelihoods[class_][attr] = dict(zip(values, counts / self.totals[class_]))

    def predict(self, X): 
        return np.apply_along_axis(self.__predict, arr=X, axis=1)

    def __predict(self, x):
        predictions = {class_: None for class_ in self.classes}

        for class_ in self.classes:
            probabilities = self.__fill_probabilities(x, class_)

            if (probabilities == 0).any():
                class_total = self.totals[class_]
                counts = probabilities * class_total

                probabilities = self.__get_smooth(counts, class_total)

            predictions[class_] = np.multiply.reduce(probabilities)
            assert predictions[class_] > 0, "Smoothing failed."

        return max(predictions.items(), key=lambda pair: pair[1])[0]

    def __fill_probabilities(self, x, class_):
        probabilities = np.zeros(len(x) + 1)
        probabilities[-1] = self.priors[class_]

        for i, attr in enumerate(self.attributes):
            value = x[i]
            probabilities[i] = self.likelihoods[class_][attr].get(value, 0)

        return probabilities

    def __get_smooth(self, counts, total):
        assert self.smoothing_k > 0, "Smoothing parameter must be strictly positive."
        return (counts + self.smoothing_k) / (total + self.n_attributes * self.smoothing_k)

        

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder
    
    import seaborn as sns
    titanic = sns.load_dataset('titanic')

    titanic = titanic[['survived', 'pclass', 'sex', 'embarked']].dropna()

    X = titanic.drop('survived', axis=1)
    y = titanic['survived']

    print(X.head())
 
    clf = NaiveBayesClassifier()
    clf.fit(X, y)

    X[1,0] = -1
    y_pred = clf.predict(X)
    print((y_pred == y).sum() / len(X))

