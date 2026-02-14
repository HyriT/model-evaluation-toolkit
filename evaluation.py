import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_roc_auc(y_true, y_proba):
    """
    Computes ROC curve and AUC score.
    Returns: fpr, tpr, thresholds, auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return fpr, tpr, thresholds, auc


def compute_classification_metrics(y_true, y_pred):
    """
    Computes Precision, Recall and F1 Score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1
