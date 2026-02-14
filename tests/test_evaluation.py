import numpy as np
import evaluation


def test_auc():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8])

    _, _, _, auc_score = evaluation.compute_roc_auc(y_true, y_proba)

    assert 0 <= auc_score <= 1


def test_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    metrics = evaluation.compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
