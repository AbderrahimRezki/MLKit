import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base import Predictor
import pandas as pd
import numpy as np


class CategoricalNaiveBayesClassifier(Predictor):
    def __init__(self, smoothing_k = 1, smoothing = False):
        self.priors = {}
        self.likelihoods = {}

        self.classes = []
        self.attributes = []
        self.attribute_values = {}
        self.n_attributes = 0

        self.smoothing = smoothing
        self.smoothing_k = smoothing_k

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(X)

        self.priors = dict(zip(self.classes, probabilities))
        self.attributes = X.columns.to_list()
        self.attribute_values = {}
        self.n_attributes = len(self.attributes)

        for attr in self.attributes:
            attr_values = np.unique(X.loc[:, attr])
            self.attribute_values[attr] = attr_values.tolist()


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
                attr_values = self.attribute_values.get(attr, [])
                self.likelihoods[class_][attr] = dict(zip(attr_values, np.zeros(len(attr_values))))
                
                conditional_attr = conditional_distribution.loc[:, attr]
                values, counts = np.unique(conditional_attr, return_counts=True)

                self.likelihoods[class_][attr].update(dict(zip(values, counts / total)))

                if self.smoothing:
                    counts = list(self.likelihoods[class_][attr].values())
                    probabilities = self.__get_smooth(np.array(counts), total)

                    self.likelihoods[class_][attr].update(dict(zip(attr_values, probabilities)))


    def predict(self, X): 
        return np.apply_along_axis(self.__predict, arr=X, axis=1)

    def __predict(self, x):
        predictions = {class_: None for class_ in self.classes}

        for class_ in self.classes:
            probabilities = self.__fill_probabilities(x, class_)

            predictions[class_] = np.multiply.reduce(probabilities)
            assert predictions[class_] > 0, "Found a class with probability 0, you might need to set smoothing = True."

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
 
    clf = CategoricalNaiveBayesClassifier()
    clf.fit(X, y)

    y_pred = clf.predict(X)
    print("Accuracy on Titanic: ", (y_pred == y).sum() / len(X))



    # This data is designed specifically to fail the CNB when no smoothing is applied
    # Try it with and without smoothing
    import pandas as pd

    X = pd.DataFrame(np.concatenate((np.zeros(50), np.ones(50))), columns=["dummy"])
    y = np.concatenate((np.zeros(50), np.ones(50)))

    clf = CategoricalNaiveBayesClassifier(smoothing=True)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    print("Accuracy With Smoothing: ", (y_pred == y).sum()  / len(X))


