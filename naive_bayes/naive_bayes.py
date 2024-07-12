import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np

class NaiveBayesClassifier:
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
        return np.apply_along_axis(self.__predict, arr=X, axis=1)

    def __predict(self, x):
        predictions = {class_: None for class_ in self.classes}

        for class_ in self.classes:
            to_multiply = np.zeros(len(x) + 1)
            for i, attr in enumerate(self.attributes):
                value = x[i]
                to_multiply[i] = self.likelihoods[class_][attr][value]

            to_multiply[-1] = self.priors[class_]
            predictions[class_] = np.multiply.reduce(to_multiply)
        
        return max(predictions.items(), key=lambda pair: pair[1])[0]



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

