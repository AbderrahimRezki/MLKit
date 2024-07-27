import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from base import Predictor
from stats import mode
from metrics.information_based import entropy, rem, information_gain
from metrics.classification_metrics import accuracy_score
from utils import Node


class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = Node(data=X, target=y)

        stack = [(self.root, 1)]
        while stack:
            node, depth = stack.pop()
            node.label = mode(node.target)

            if 0 in node.data.shape or depth >= self.max_depth:
                continue
            
            best_split = self.__find_best_split(node.data, node.target)
            node.column = best_split

            column = node.data[:, best_split]
            values = np.unique(column)
            data = np.delete(node.data, best_split, 1)
            
            for value in values:
                mask = (column == value)
                
                child_data = data[mask]
                child_target = node.target[mask]

                if 0 in child_data.shape:
                    child = Node()
                    child.label = node.label
                else:
                    child = Node(data=child_data, target=child_target)
                    child.label = mode(child.target)

                    stack.append((child, depth + 1))
                
                node.children[value] = child


    def predict(self, X):
        return np.apply_along_axis(self.__predict, axis=1, arr=X)

    def __predict(self, x):
        current_node = self.root
        while current_node.children:
            col = current_node.column
            value = x[col]
            x = np.concatenate((x[:col], x[col+1:]), axis=0)

            temp = current_node.children.get(value, None)

            if temp is None:
                return current_node.label

            current_node = temp
        
        return current_node.label
        

    def __find_best_split(self, X, y):
        f = lambda x: information_gain(x, y)
        return np.apply_along_axis(f, axis=0, arr=X).argmax(axis=0)


if __name__ == "__main__":
    import pandas as pd

    data = {
        "ID": [1, 2, 3, 4, 5, 6, 7],
        "STREAM": [False, True, True, False, False, True, True],
        "SLOPE": ["steep", "moderate", "steep", "steep", "flat", "steep", "steep"],
        "ELEVATION": ["high", "low", "medium", "medium", "high", "highest", "high"],
        "VEGETATION": ["chapparal", "riparian", "riparian", "chapparal", "conifer", "conifer", "chapparal"]
    }

    df = pd.DataFrame(data) 

    X = df.drop("VEGETATION", axis=1)
    y = df["VEGETATION"]

    X.drop("ID", axis=1, inplace=True)
    columns = X.columns

    df_original = df.copy()
    df = df.to_numpy()
    X = X.to_numpy()

    clf = DecisionTree(5)
    clf.fit(X, y)
    print(clf.root)


    y_pred = clf.predict(X)

    print("Accuracy:", accuracy_score(y_pred, y))

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import OrdinalEncoder

    dt = DecisionTreeClassifier(max_depth=5)
    X[:, [0, 1, 2]] = OrdinalEncoder().fit_transform(X[:, [0, 1, 2]])
    dt.fit(X, y)

    y_pred = dt.predict(X)
    print("Accuracy:", accuracy_score(y_pred, y))