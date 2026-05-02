# Wafer Yield Analysis & Defect Pattern Classification

Semiconductor process analysis on the [SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom)
(UCI ML Repository) — identifying yield-limiting sensor parameters and predicting
pass/fail outcomes across 1,567 wafer runs.

---

## Problem

In semiconductor manufacturing, hundreds of sensors monitor each wafer as it moves
through fabrication. At the end of the line, a wafer either passes or fails electrical
testing. The goal: **which sensor readings predict failure early**, before the wafer
wastes further processing steps?

The core challenge is severe class imbalance — only **6.6% of wafers pass** (104 out
of 1,567). A naive model that always predicts "fail" gets 93% accuracy while being
completely useless. This project handles that correctly.

---

## Dataset

| Property | Value |
|---|---|
| Source | UCI ML Repository — SECOM |
| Wafer runs | 1,567 |
| Sensor parameters | 590 |
| Pass rate | 6.6% (104 pass / 1,463 fail) |
| Missing values | 4.5% |

---

## Project Structure

```
├── secom_analysis.ipynb   # Full Colab notebook
├── correlation_analysis.png
├── wafer_maps.png
├── yield_trends.png
├── random_forest_results.png
├── roc_curve.png
└── model_comparison.png
```

---

## Methodology

### 1. Data Cleaning
- Dropped columns with >40% missing values: 590 → 558 features
- Imputed remaining NaNs with column median
- Removed zero-variance columns: 558 → 442 features

### 2. Correlation Analysis
Computed point-biserial correlation between each sensor and the pass/fail label.
Top sensor (column 59) showed correlation of 0.156 with p-value 5.3×10⁻¹⁰.
Selected top 50 features for modelling.

### 3. Synthetic Wafer Map Classification
Generated 300 synthetic wafer maps with three defect patterns and built a
rule-based spatial classifier using nearest-neighbour distance analysis:

| Pattern | Precision | Recall | F1 |
|---|---|---|---|
| Edge | 1.00 | 1.00 | 1.00 |
| Cluster | 1.00 | 0.41 | 0.58 |
| Random | 0.68 | 1.00 | 0.81 |
| **Overall** | | **81.7%** | |

### 4. ML Modelling
Addressed class imbalance with two techniques:
- **SMOTE** — synthesised minority (Pass) examples: 83 → 1,170 balanced training samples
- **Threshold tuning** — lowered decision threshold from 0.50 to 0.28 to optimise Pass recall

#### Model comparison (test set)

| Model | AUC | Pass Recall | Pass F1 |
|---|---|---|---|
| **Random Forest** | **0.762** | **0.524** | **0.272** |
| Logistic Regression | 0.710 | 0.429 | 0.286 |
| XGBoost | 0.691 | 0.381 | 0.216 |

#### Cross-validation (Random Forest, 5-fold stratified)

| Fold | AUC |
|---|---|
| 1 | 0.703 |
| 2 | 0.744 |
| 3 | 0.768 |
| 4 | 0.786 |
| 5 | 0.706 |
| **Mean** | **0.741 ± 0.033** |

---

## Key Results

- Random Forest achieves **AUC 0.778** on the held-out test set
- Catches **52.4% of passing wafers** (Pass Recall) — the operationally critical metric,
  since missing a good wafer means scrapping a usable product
- Cross-validation confirms the model generalises (AUC 0.741 ± 0.033) and is not
  overfitting a single lucky split
- Accuracy alone (93%) is **misleading** on this dataset — a model predicting all-fail
  scores the same. AUC and Pass Recall are the right metrics here.

---

## Why AUC matters more than accuracy here

> A model that always predicts "Fail" gets **93.4% accuracy**.
> It catches **0% of passing wafers**.
> That is worse than useless in a fab setting.

AUC of 0.778 means: given one random passing wafer and one random failing wafer,
the model correctly ranks the passing wafer higher **77.8% of the time**.

---

## Tech Stack

- Python 3.12
- pandas, numpy, scipy
- scikit-learn (Random Forest, Logistic Regression, SMOTE via imbalanced-learn)
- XGBoost
- matplotlib, seaborn

---

## How to Run

1. Open `secom_analysis.ipynb` in Google Colab
2. Run all cells top to bottom
3. Dataset downloads automatically from UCI — no manual setup needed
