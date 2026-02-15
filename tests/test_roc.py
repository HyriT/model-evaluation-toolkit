import numpy as np
import evaluation


def test_auc_range():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8])

    _, _, _, auc_score = evaluation.compute_roc_auc(y_true, y_proba)

    assert 0 <= auc_score <= 1
