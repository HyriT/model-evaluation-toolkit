import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score

import evaluation


def test_confusion_matrix_manual():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])

    TP, FP, TN, FN = evaluation.confusion_matrix_binary(y_true, y_pred)

    assert TP == 2
    assert FP == 1
    assert TN == 2
    assert FN == 1


def test_confusion_matrix_sklearn():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])

    TP, FP, TN, FN = evaluation.confusion_matrix_binary(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    assert TP == tp
    assert FP == fp
    assert TN == tn
    assert FN == fn


def test_metrics_vs_sklearn():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])

    TP, FP, TN, FN = evaluation.confusion_matrix_binary(y_true, y_pred)

    assert evaluation.precision(TP, FP) == precision_score(y_true, y_pred)
    assert evaluation.recall(TP, FN) == recall_score(y_true, y_pred)
    assert evaluation.fbeta_score(TP, FP, FN, beta=1) == fbeta_score(y_true, y_pred, beta=1)
