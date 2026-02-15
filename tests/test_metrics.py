import numpy as np
import evaluation


def test_confusion_matrix():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    TP, FP, TN, FN = evaluation.confusion_matrix_binary(y_true, y_pred)

    assert TP == 2
    assert TN == 2
    assert FP == 0
    assert FN == 0


def test_precision_recall():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    TP, FP, TN, FN = evaluation.confusion_matrix_binary(y_true, y_pred)

    assert evaluation.precision(TP, FP) == 1.0
    assert evaluation.recall(TP, FN) == 1.0
    assert evaluation.specificity(TN, FP) == 1.0
