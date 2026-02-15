# Model Card – Breast Cancer Classification

## 1. Model Overview

This project implements a reusable evaluation toolkit for binary classification.
The final selected model is:

**Logistic Regression**

The model predicts whether a tumor is:
- 1 → Malignant
- 0 → Benign

---

## 2. Business Objective

The goal is to build a reusable evaluation module that:

- Computes confusion matrix metrics
- Computes Precision, Recall, Specificity
- Computes Fβ score
- Computes ROC curve and AUC
- Performs threshold analysis
- Supports model comparison

This toolkit can be reused for any binary classification problem.

---

## 3. Dataset

**Dataset:** Breast Cancer Wisconsin Dataset  
**Samples:** 569  
**Features:** 30 numerical features  
**Target:** Binary (Malignant / Benign)

### Class Distribution

- Malignant: ~62.7%
- Benign: ~37.3%

The dataset contains:
- No missing values
- Only numeric features

---

## 4. Data Split

The dataset was split using:

- 80% Training
- 20% Testing
- Stratified split to preserve class balance
- random_state = 42

---

## 5. Baseline Model

A DummyClassifier (most frequent strategy) was used as baseline.

Baseline accuracy: ~0.62

This means any meaningful model must perform significantly better than 62%.

---

## 6. Model Experiments

Two models were trained and compared:

### Logistic Regression
AUC ≈ 0.995

### Random Forest
AUC ≈ 0.993

Logistic Regression slightly outperformed Random Forest based on AUC,
so it was selected as the final model.

---

## 7. Final Evaluation Metrics (Threshold = 0.5)

Metrics computed from scratch using evaluation.py:

- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- ROC AUC

The model achieved very high precision and recall,
indicating strong classification performance.

---

## 8. Threshold Analysis

A threshold sweep was performed to evaluate:

- Precision vs Threshold
- Recall vs Threshold
- Specificity vs Threshold
- Fβ Score vs Threshold

This allows business-driven optimization,
for example prioritizing recall in medical diagnosis.

---

## 9. Limitations

- Dataset is relatively small (569 samples)
- No cross-validation implemented
- No hyperparameter tuning performed
- Results may vary on real-world clinical data

---

## 10. Ethical Considerations

- False Negatives (missed malignant cases) are critical
- Model should assist doctors, not replace medical judgment
- Threshold should be adjusted based on medical risk tolerance

---

## 11. Reproducibility

To reproduce results:

1. Install dependencies:
   numpy, pandas, matplotlib, scikit-learn

2. Run:
   demo.ipynb

3. Evaluation module:
   evaluation.py

---

## 12. Repository

GitHub Repository:
https://github.com/HyriT/model-evaluation-toolkit/tree/main
---

## 13. Team Members

- Nadja Brari – Core Metrics Implementation
- Hyrije Taga – ROC, PR & Threshold Analysis
- Silva Tabaku – Model Training & Comparison
- Oltjona Gjyriqi – Data Analysis & Model Card

---

