import numpy as np

def mode(x):
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts, axis=0)]
