import numpy as np


def confusion_matrix_binary(y_true, y_pred):
    """
    Computes confusion matrix values for binary classification.
    Returns: TP, FP, TN, FN
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    return TP, FP, TN, FN


def precision(TP, FP):
    """Precision = TP / (TP + FP)"""
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def recall(TP, FN):
    """Recall (Sensitivity) = TP / (TP + FN)"""
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def sensitivity(TP, FN):
    """Alias for recall"""
    return recall(TP, FN)


def specificity(TN, FP):
    """Specificity = TN / (TN + FP)"""
    if TN + FP == 0:
        return 0.0
    return TN / (TN + FP)


def fbeta_score(TP, FP, FN, beta=1):
    """
    F-beta score.
    beta=1 → F1 score
    """

    if beta <= 0:
        raise ValueError("beta must be positive")

    prec = precision(TP, FP)
    rec = recall(TP, FN)

    if prec + rec == 0:
        return 0.0

    beta_sq = beta ** 2

    return (1 + beta_sq) * (prec * rec) / ((beta_sq * prec) + rec)
