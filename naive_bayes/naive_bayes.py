import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base import Predictor
import pandas as pd
import numpy as np

class NaiveBayesClassifier(Predictor):
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = []
        self.attributes = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(X)

        self.priors = dict(zip(self.classes, probabilities))
        self.attributes = X.columns.to_list()

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
            total = len(conditional_distribution)
            
            for attr in self.attributes:
                conditional_attr = conditional_distribution.loc[:, attr]
                values, counts = np.unique(conditional_attr, return_counts=True)
                self.likelihoods[class_][attr] = dict(zip(values, counts / total))


    def predict(self, X): 
        predictions = [
            dict(zip(self.classes, np.repeat(1, len(self.classes)))) 
            for _ in range(len(X))
        ]

        final_pred = np.empty(len(X))

        for class_ in self.classes:
            for attr in self.attributes:
                conditional_attr = X.loc[:, attr]

                for i, value in enumerate(conditional_attr):
                    predictions[i][class_] *= self.likelihoods[class_][attr][value]


        for i, pred in enumerate(predictions):
            final_pred[i] = max(pred.items(), key=lambda x: x[1])[0]

        return final_pred


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

    y_pred = clf.predict(X)
    print((y_pred == y).sum() / len(X))

