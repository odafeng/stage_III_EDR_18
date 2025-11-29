# Early Distant Recurrence Prediction in Stage III Colon Cancer
## Machine Learning Model Development and External Validation

This repository contains the analysis workflow for developing and validating a machine learning model to predict **18-month early distant recurrence (EDR-18)** in stage III colon cancer.

The workflow covers:
- Feature selection (LASSO and XGBoost)
- Model development with nested cross-validation
- Risk stratification and Kaplan–Meier analysis
- External validation in an independent cohort

> **⚠️ Note:**
> Raw clinical datasets are **not included** in this repository due to patient privacy.
> Users should provide their own datasets under the `./data/` directory.

---

## Repository Structure

```text
/
├── notebooks
│   ├── 1.LASSO_XGBoost_Feature_Selection.ipynb
│   ├── 2.XGBoost_StageIII.ipynb
│   └── 3.External_Validation.ipynb
├── data/              # An example data schema was included; user-provided data also go here
├── model/             # Directory for saving model artifacts
├── README.md
└── requirements.txt
```

---

## Notebook Overview

### 1. `1.LASSO_XGBoost_Feature_Selection.ipynb`

**Purpose:**
Perform feature selection using LASSO logistic regression and XGBoost feature importance.

**Key Steps:**
- Load derivation cohort from `./data/`.
- Fit **LASSO models** to identify sparse sets of predictors.
- Train **XGBoost models** and summarize feature importance across folds.
- Derive a parsimonious, clinically interpretable set of variables for downstream modeling.

### 2. `2.XGBoost_StageIII.ipynb`

**Purpose:**
Develop the final four-variable model for EDR-18 prediction in stage III colon cancer.

**Key Steps:**
- Load the derivation cohort from `./data/`.
- Implement **nested cross-validation** to:
    - Tune XGBoost hyperparameters.
    - Generate out-of-fold (OOF) predictions.
    - Calibrate predicted probabilities (e.g., isotonic regression).
- Define a fixed decision threshold based on the **OOF Youden index**.
- Construct **Kaplan–Meier curves** for:
    - All stage III patients.
    - Subgroups of interest (e.g., AJCC stage IIIB).
- Export model artifacts and OOF predictions for external use.

### 3. `3.External_Validation.ipynb`

**Purpose:**
Apply the finalized four-variable model to an independent external cohort.

**Key Steps:**
- Load the external validation cohort from `./data/`.
- Apply the saved preprocessing and model pipeline **without retraining**.
- Evaluate performance metrics:
    - **ROC-AUC**
    - **Brier score**
    - **Calibration performance**
- Perform **Cox regression** for high- vs low-risk groups.
- Generate Kaplan–Meier curves in the external cohort (overall and subgroups).

---

## Data Location and File Naming

The notebooks assume the following (or similar) structure for input data files:

```text
./data/
    stageIII_derivation.xlsx   # Derivation cohort (not included)
    stageIII_external.xlsx     # External validation cohort (not included)
```

> You may adjust filenames in the notebooks as needed. Since real clinical data cannot be shared, users must replace these with their own datasets containing equivalent variables.

---

## Environment and Installation

The analysis was developed using **Python 3.11**.

### 1. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate      # On macOS/Linux
# .venv\Scripts\activate       # On Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Notebooks

1. Place your derivation and external validation datasets under `./data/`.
2. Open the notebooks in JupyterLab, Jupyter Notebook, or VS Code.
3. Run the notebooks in the following order:
    1. `1.LASSO_XGBoost_Feature_Selection.ipynb`
    2. `2.XGBoost_StageIII.ipynb`
    3. `3.External_Validation.ipynb`
4. Review generated figures, metrics, and model outputs for reproduction or extension.

---

## Notes on Reproducibility and Privacy

- All paths in the notebooks are relative (e.g., `./data/...`) and do not include any hospital-specific directories.
- **Raw clinical data are not included and must not be committed to this repository.**
- Results may vary slightly if:
    - Random seeds are changed.
    - Library versions differ from those in `requirements.txt`.

---

## Citation

If you use or adapt this workflow in your own research, please cite the corresponding manuscript (once published) and this repository.

> **Huang SF, et al.** Ruling Out Early Distant Recurrence in Stage III Colon Cancer: A Parsimonious Machine Learning Model with External Validation [Manuscript in preparation]

# Early Distant Recurrence Prediction in Stage III Colon Cancer  
## Machine Learning Model Development and External Validation

This repository contains the analysis workflow for developing and validating a machine learning model to predict 18-month early distant recurrence (EDR-18) in stage III colon cancer.  
The workflow covers:

- Feature selection (LASSO and XGBoost)
- Model development with nested cross-validation
- Risk stratification and Kaplan–Meier analysis
- External validation in an independent cohort

> **Note:**  
> Raw clinical datasets are **not included** in this repository due to patient privacy.  
> Users should provide their own datasets under the `./data/` directory.

---

## Repository Structure

```text
/
├── 1.LASSO_XGBoost_Feature_Selection.ipynb
├── 2.XGBoost_StageIII.ipynb
├── 3.External_Validation.ipynb
├── data/              # (Not included; user-provided data go here)
├── README.md
└── requirements.txt
