import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base import Predictor
import numpy as np
import pandas as pd

class Node:
    def __init__(self, right=None, left=None, column=None, label=None):
        self.right = right
        self.left = left
        self.label = label


class DecisionTreeClassifier(Predictor):
    def __init__(self, max_depth=10)
        self.root = None

    def fit(self, X, y):
        self.root = self.find_best_split(X, y)

    def find_best_split(self, X, y):
        pass
