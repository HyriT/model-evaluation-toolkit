# Model Evaluation Toolkit

A reusable Python toolkit for evaluating binary classification models from scratch using custom implementations of machine learning evaluation metrics.

## Features

- Confusion Matrix
- Precision
- Recall (Sensitivity)
- Specificity
- F-beta Score
- ROC Curve & AUC
- Precision–Recall Curve
- Threshold Analysis

## Dataset

Breast Cancer Wisconsin Dataset  
- 569 samples  
- 30 numerical features  
- Binary classification:
  - `0` → Benign
  - `1` → Malignant

## Models Used

- Logistic Regression
- Random Forest
- DummyClassifier (baseline)

## Results

| Model | AUC Score |
|---|---|
| Logistic Regression | 0.995 |
| Random Forest | 0.994 |

## Installation

```bash
git clone https://github.com/HyriT/model-evaluation-toolkit.git
cd model-evaluation-toolkit
pip install -r requirements.txt
```

## Example Usage

```python
from src.metrics import precision, recall

precision_score = precision(tp=71, fp=3)
recall_score = recall(tp=71, fn=1)

print(precision_score)
print(recall_score)
```

## GitHub Repository

https://github.com/HyriT/model-evaluation-toolkit
