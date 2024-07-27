import numpy as np

def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray):
    return (y_pred == y_true).sum() / len(y_true)

def precision_score(y_pred: np.ndarray, y_true: np.ndarray, class_ = 1):
    positive_pred = (y_pred == class_)
    positive_target = (y_true == class_)
    negative_target = (y_true != class_)
    
    TP = positive_pred & positive_target 
    FP = positive_pred & negative_target

    return TP / (TP + FP)