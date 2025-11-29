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
