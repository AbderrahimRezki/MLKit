import numpy as np

def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    probabilities = counts / len(column)
    log2probabilites = np.log2(probabilities)

    return -np.dot(probabilities, log2probabilites)

def rem(column, y):
    values, counts = np.unique(column, return_counts=True)
    weights   = counts / len(column)
    entropies = np.array([entropy(y[column == value]) for value in values])

    return np.dot(weights, entropies)

def information_gain(column, y):
    return entropy(y) - rem(column, y)
