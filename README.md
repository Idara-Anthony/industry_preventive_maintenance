# Predictive Maintenance: Machine Failure Classification & False Discovery Rate Optimization

## Overview
This project builds a multi-label machine learning classifier to predict specific industrial machine failure modes using the **AI4I 2020 Predictive Maintenance Dataset**. In manufacturing environments, unscheduled downtime is costly, while false alarms unnecessarily halt production. To satisfy this industry constraint, the modeling pipeline prioritizes minimizing the **False Discovery Rate (FDR)** through physics-informed feature engineering, cost-sensitive learning, and decision-threshold tuning.

## Dataset & Target Failure Modes
* **Source:** AI4I 2020 Predictive Maintenance Dataset (`ai4i2020.csv`, 10,000 records).
* **Input Features:** Product `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`.
* **Supervised Targets:**
  * `TWF` — Tool Wear Failure
  * `HDF` — Heat Dissipation Failure
  * `PWF` — Power Failure
  * `OSF` — Overstrain Failure
  *(Note: `RNF` [Random Failures] is excluded from supervised modeling because it occurs randomly independent of sensor telemetry.)*

## Methodology
1. **Physics-Informed Feature Engineering:**
   * `Temp_diff [K]`: Temperature difference between process and ambient air (captures heat dissipation dynamics).
   * `Power [W]`: Mechanical power calculated from torque and rotational speed.
   * `Overstrain [Nm*min]`: Product of torque and accumulated tool wear.
2. **Handling Extreme Class Imbalance:**
   * Used `MultilabelStratifiedShuffleSplit` (80/20 train-test split) to preserve rare failure distributions (`<1.2%` positive rate per class).
   * Evaluated three `MultiOutputClassifier` architectures: **Logistic Regression**, **XGBoost**, and **LightGBM**.
3. **False Discovery Rate (FDR) Optimization:**
   * Evaluated models using **FDR** (`FDR = FP / (FP + TP) = 1 - Precision`), Macro F1-Score, PR-AUC, Hamming Loss, and multi-label confusion matrices.
   * Applied cost-sensitive probability threshold tuning (`0.50` -> `0.70`) on the top-performing **LightGBM** model to suppress false alarms.

## Key Results (LightGBM)

| Failure Mode | False Discovery Rate (FDR) | False Positives (False Alarms) |
| :--- | :---: | :---: |
| **TWF** (Tool Wear) | 0.00% | 0 |
| **HDF** (Heat Dissipation) | 0.00% | 0 |
| **PWF** (Power) | 5.88% – 7.14% | 1 |
| **OSF** (Overstrain) | 0.00% | 0 |
| **Overall Exact-Match Accuracy** | **99.10% – 99.30%** | **1 total across 2,000 test samples** |

## Tech Stack
* **Language:** Python 3
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `iterative-stratification`, `matplotlib`, `seaborn`

## How to Run
1. Place `ai4i2020.csv` in the same directory as the Jupyter Notebook.
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm iterative-stratification matplotlib seaborn
   ```
3. Open the notebook and run all cells sequentially from top to bottom.
