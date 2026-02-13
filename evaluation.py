import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# CONFUSION MATRIX

def confusion_matrix_binary(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    return TP, FP, TN, FN


# BASIC METRICS

def precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) > 0 else 0


def recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) > 0 else 0


def specificity(TN, FP):
    return TN / (TN + FP) if (TN + FP) > 0 else 0


def fbeta_score(TP, FP, FN, beta=1):
    prec = precision(TP, FP)
    rec = recall(TP, FN)

    if prec + rec == 0:
        return 0

    return (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)


# ROC & AUC

def compute_roc_auc(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# PRECISION-RECALL CURVE

def compute_pr_curve(y_true, y_proba):
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
    return precision_vals, recall_vals, thresholds


def plot_pr_curve(precision_vals, recall_vals):
    plt.figure()
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()


# THRESHOLD SWEEP

def threshold_sweep(y_true, y_proba, beta=2, resolution=100):
    thresholds = np.linspace(0, 1, resolution)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        TP, FP, TN, FN = confusion_matrix_binary(y_true, y_pred)

        results.append({
            "threshold": t,
            "precision": precision(TP, FP),
            "recall": recall(TP, FN),
            "specificity": specificity(TN, FP),
            "f_beta": fbeta_score(TP, FP, FN, beta)
        })

    return pd.DataFrame(results)


def plot_threshold_curves(df):
    plt.figure()
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["specificity"], label="Specificity")
    plt.plot(df["threshold"], df["f_beta"], label="F_beta")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Analysis")
    plt.legend()
    plt.show()


def find_best_threshold(df_metrics, metric="f_beta"):
    idx = df_metrics[metric].idxmax()
    return df_metrics.loc[idx]
