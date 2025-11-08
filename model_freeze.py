"""
Model Freeze Script - 凍結模型用於外部驗證

將訓練好的模型、Pipeline、閾值等完整保存，供外部驗證使用

校正方法: Sigmoid (Platt)
理由: 保持calibration slope≈1.0，避免Isotonic過度壓縮風險區分度
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import logit
import warnings
warnings.filterwarnings('ignore')

# ====================================
# 輔助函數
# ====================================

def compute_calibration_slope_intercept(y_true, y_pred_proba, eps=1e-10):
    y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1 - eps)
    logit_proba = logit(y_pred_proba_clipped)
    lr = LogisticRegression(penalty='none', max_iter=1000, solver='lbfgs')
    lr.fit(logit_proba.reshape(-1, 1), y_true)
    return lr.coef_[0][0], lr.intercept_[0]

def ensure_binary_01(X, binary_cols):
    X_out = X.copy()
    for col in binary_cols:
        if col in X_out.columns:
            X_out[col] = (X_out[col] > 0).astype(int)
    return X_out

class InteractionTransformer2(BaseEstimator, TransformerMixin):
    """2個核心交互項"""
    
    def __init__(self):
        self.interaction_specs = [
            ('num__LNR', 'ord__pN_ordinal', 'LNR × pN_ordinal'),
            ('bin__LVI', 'bin__PNI', 'LVI × PNI'),
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            for feat1, feat2, interaction_name in self.interaction_specs:
                X_out[interaction_name] = X[feat1] * X[feat2]
            return X_out
        else:
            raise ValueError("Requires DataFrame input.")
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        original_names = list(input_features)
        interaction_names = [spec[2] for spec in self.interaction_specs]
        return original_names + interaction_names

# ====================================
# 主程序
# ====================================

print("=" * 80)
print("模型凍結 - 準備外部驗證")
print("=" * 80)

# 載入訓練數據
data_dir = Path('/Users/huangshifeng/Desktop/stage_III_colon_surv/data')
train_df = pd.read_parquet(data_dir / 'train_preprocessed.parquet')
test_df = pd.read_parquet(data_dir / 'test_preprocessed.parquet')

# 定義特徵
continuous_features = ['Age', 'LNR', 'Tumor_Size_cm', 'Log_CEA_PreOp']
ordinal_features = ['pT_ordinal', 'pN_ordinal', 'Differentiation_ordinal']
binary_features = ['Tumor_Location', 'LVI', 'PNI', 'Tumor_Deposits', 'Mucinous_Any', 'MSI_High']
all_features = continuous_features + ordinal_features + binary_features

target = 'edr_18m'
X_train = train_df[all_features].copy()
y_train = train_df[target].copy()
X_test = test_df[all_features].copy()
y_test = test_df[target].copy()

print(f"\n訓練集: {len(y_train)} 例 (事件: {y_train.sum()})")
print(f"內部測試集: {len(y_test)} 例 (事件: {y_test.sum()})")

# ====================================
# 1. 訓練模型
# ====================================

print("\n" + "=" * 80)
print("步驟 1/5: 訓練最終模型")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
C_range = np.logspace(-4, 4, 20)

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), continuous_features),
        ('ord', SimpleImputer(strategy='most_frequent'), ordinal_features),
        ('bin', SimpleImputer(strategy='most_frequent'), binary_features)
    ],
    remainder='passthrough'
)
preprocessor.set_output(transform="pandas")

binary_cols_with_prefix = [f'bin__{c}' for c in binary_features]

# 創建可pickle的transformer函數
def post_binary_func(X):
    return ensure_binary_01(X, binary_cols_with_prefix)

post_binary_transformer = FunctionTransformer(
    post_binary_func,
    feature_names_out='one-to-one'
)

interaction_transformer = InteractionTransformer2()

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('post_bin', post_binary_transformer),
    ('interactions', interaction_transformer),
    ('classifier', LogisticRegressionCV(
        Cs=C_range, cv=cv, penalty='l2', solver='lbfgs',
        scoring='roc_auc', max_iter=3000, tol=1e-3,
        random_state=42, n_jobs=1, class_weight='balanced'
    ))
])

print("✓ 訓練未校正模型...")
pipeline.fit(X_train, y_train)

clf = pipeline.named_steps['classifier']
cv_auc = clf.scores_[1].mean(axis=0).max()
best_C = clf.C_[0]

print(f"✓ 最佳 C: {best_C:.4f}")
print(f"✓ CV ROC-AUC: {cv_auc:.4f}")

# ====================================
# 2. 獲取OOF閾值
# ====================================

print("\n" + "=" * 80)
print("步驟 2/5: 計算OOF閾值")
print("=" * 80)

y_train_oof = cross_val_predict(
    pipeline, X_train, y_train, cv=cv,
    method='predict_proba', n_jobs=1
)[:, 1]

precision_oof, recall_oof, thresholds_pr_oof = precision_recall_curve(y_train, y_train_oof)
f1_scores_oof = 2 * (precision_oof[:-1] * recall_oof[:-1]) / (precision_oof[:-1] + recall_oof[:-1] + 1e-10)
threshold_f1 = thresholds_pr_oof[np.argmax(f1_scores_oof)]

# Sensitivity ≥ 0.70：在所有recall≥0.70的點中，選precision最高的
target_sensitivity = 0.70
valid_indices = np.where(recall_oof[:-1] >= target_sensitivity)[0]
if len(valid_indices) > 0:
    # 找出滿足Sens≥70%且precision最高的點
    best_idx = valid_indices[np.argmax(precision_oof[:-1][valid_indices])]
    threshold_sens = thresholds_pr_oof[best_idx]
else:
    threshold_sens = threshold_f1

fpr_oof, tpr_oof, thresholds_roc_oof = roc_curve(y_train, y_train_oof)
youden_index_oof = tpr_oof - fpr_oof
threshold_youden = thresholds_roc_oof[np.argmax(youden_index_oof)]

print(f"✓ F1-Optimal:   {threshold_f1:.4f}")
print(f"✓ Sens≥70%:     {threshold_sens:.4f}")
print(f"✓ Youden Index: {threshold_youden:.4f}")

# ====================================
# 3. 訓練校正模型
# ====================================

print("\n" + "=" * 80)
print("步驟 3/5: 訓練校正模型 (Sigmoid)")
print("=" * 80)

print("✓ 在訓練集上用5-fold CV進行Sigmoid (Platt) 校正...")
print("  理由：保持slope≈1.0，避免Isotonic過度壓縮（slope會降至0.53）")

# 創建可pickle的post_binary函數
def post_binary_func_cal(X):
    binary_cols = [f'bin__{c}' for c in binary_features]
    return ensure_binary_01(X, binary_cols)

preprocessor_cal = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), continuous_features),
        ('ord', SimpleImputer(strategy='most_frequent'), ordinal_features),
        ('bin', SimpleImputer(strategy='most_frequent'), binary_features)
    ],
    remainder='passthrough'
)
preprocessor_cal.set_output(transform="pandas")

calibrated_pipeline = CalibratedClassifierCV(
    estimator=Pipeline([
        ('preprocessor', preprocessor_cal),
        ('post_bin', FunctionTransformer(post_binary_func_cal, feature_names_out='one-to-one')),
        ('interactions', InteractionTransformer2()),
        ('classifier', LogisticRegressionCV(
            Cs=C_range, cv=cv, penalty='l2', solver='lbfgs',
            scoring='roc_auc', max_iter=3000, tol=1e-3,
            random_state=42, n_jobs=1, class_weight='balanced'
        ))
    ]),
    method='sigmoid',
    cv=5
)
calibrated_pipeline.fit(X_train, y_train)

print("✓ Sigmoid校正完成")

# ====================================
# 4. 內部測試集驗證
# ====================================

print("\n" + "=" * 80)
print("步驟 4/5: 內部測試集驗證")
print("=" * 80)

y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_test_pred_calibrated = calibrated_pipeline.predict_proba(X_test)[:, 1]

test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
test_pr_auc = average_precision_score(y_test, y_test_pred_proba)
test_brier_before = brier_score_loss(y_test, y_test_pred_proba)
test_brier_after = brier_score_loss(y_test, y_test_pred_calibrated)

slope_before, intercept_before = compute_calibration_slope_intercept(y_test, y_test_pred_proba)
slope_after, intercept_after = compute_calibration_slope_intercept(y_test, y_test_pred_calibrated)

print(f"✓ 內部測試集效能:")
print(f"  ROC-AUC: {test_roc_auc:.4f}")
print(f"  PR-AUC:  {test_pr_auc:.4f}")
print(f"  Brier (校正前): {test_brier_before:.4f}")
print(f"  Brier (校正後): {test_brier_after:.4f}")
print(f"  校正斜率 (校正前): {slope_before:.4f}")
print(f"  校正斜率 (校正後): {slope_after:.4f}")

# 獲取特徵名和係數
X_train_transformed = pipeline.named_steps['interactions'].transform(
    pipeline.named_steps['post_bin'].transform(
        pipeline.named_steps['preprocessor'].transform(X_train.iloc[:1])
    )
)
feature_names = list(X_train_transformed.columns)
coef = pipeline.named_steps['classifier'].coef_[0]

# ====================================
# 5. 保存模型
# ====================================

print("\n" + "=" * 80)
print("步驟 5/5: 凍結並保存模型")
print("=" * 80)

freeze_dir = Path('/Users/huangshifeng/Desktop/stage_III_colon_surv/frozen_model')
freeze_dir.mkdir(parents=True, exist_ok=True)

# 5.1 保存Pipeline（未校正）
with open(freeze_dir / 'pipeline_uncalibrated.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print(f"✓ 已保存: pipeline_uncalibrated.pkl")

# 5.2 保存Pipeline（已校正）
with open(freeze_dir / 'pipeline_calibrated.pkl', 'wb') as f:
    pickle.dump(calibrated_pipeline, f)
print(f"✓ 已保存: pipeline_calibrated.pkl")

# 5.3 保存模型元數據
metadata = {
    'model_name': 'Stage III Colon Cancer EDR Prediction Model',
    'model_version': '1.0',
    'freeze_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'target_variable': target,
    'n_training_samples': len(y_train),
    'n_events': int(y_train.sum()),
    'n_features': len(all_features),
    'n_interactions': 2,
    'total_parameters': len(feature_names),
    'epv': float(y_train.sum() / (len(feature_names) + 1)),
    
    # 特徵定義
    'continuous_features': continuous_features,
    'ordinal_features': ordinal_features,
    'binary_features': binary_features,
    'all_features': all_features,
    'interaction_terms': ['LNR × pN_ordinal', 'LVI × PNI'],
    
    # 模型參數
    'best_C': float(best_C),
    'penalty': 'l2',
    'solver': 'lbfgs',
    'class_weight': 'balanced',
    
    # CV效能
    'cv_roc_auc': float(cv_auc),
    'cv_folds': 5,
    
    # 內部測試集效能
    'internal_test': {
        'n_samples': len(y_test),
        'n_events': int(y_test.sum()),
        'roc_auc': float(test_roc_auc),
        'pr_auc': float(test_pr_auc),
        'brier_uncalibrated': float(test_brier_before),
        'brier_calibrated': float(test_brier_after),
        'calibration_slope_uncalibrated': float(slope_before),
        'calibration_slope_calibrated': float(slope_after),
        'calibration_intercept_uncalibrated': float(intercept_before),
        'calibration_intercept_calibrated': float(intercept_after)
    },
    
    # 閾值（基於OOF）
    'thresholds': {
        'default': 0.5,
        'f1_optimal': float(threshold_f1),
        'sensitivity_70': float(threshold_sens),
        'youden_index': float(threshold_youden)
    },
    
    # 係數
    'coefficients': {
        name: float(coef_val) 
        for name, coef_val in zip(feature_names, coef)
    },
    
    # 使用說明
    'usage_notes': {
        'calibration': 'Use pipeline_calibrated.pkl (Sigmoid calibration) for clinical predictions',
        'calibration_method': 'Sigmoid (Platt) - preserves calibration slope≈1.0',
        'preprocessing': 'Input data must have the exact features listed in all_features',
        'missing_values': 'Continuous: median imputation, Others: most frequent imputation',
        'interactions': 'Automatically created by the pipeline',
        'recommended_threshold': 'f1_optimal for balanced performance, sensitivity_70 for screening'
    }
}

with open(freeze_dir / 'model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✓ 已保存: model_metadata.json")

# 5.4 保存特徵係數表
coefficients_df = pd.DataFrame({
    '特徵': feature_names,
    '係數': coef,
    '絕對值': np.abs(coef)
}).sort_values('絕對值', ascending=False)

coefficients_df.to_csv(freeze_dir / 'feature_coefficients.csv', index=False, encoding='utf-8-sig')
print(f"✓ 已保存: feature_coefficients.csv")

# 5.5 創建README
readme_content = f"""# Stage III Colon Cancer EDR Prediction Model (Frozen)

## 模型資訊

- **版本**: 1.0
- **凍結日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **目標變數**: {target} (18個月內早期疾病復發)
- **訓練樣本**: {len(y_train)} 例 (事件: {y_train.sum()})
- **EPV**: {metadata['epv']:.2f}

## 模型配置

### 特徵 (15個)
- **連續變數** (4): {', '.join(continuous_features)}
- **有序變數** (3): {', '.join(ordinal_features)}
- **二元變數** (6): {', '.join(binary_features)}
- **交互項** (2): LNR × pN_ordinal, LVI × PNI

### 模型參數
- **正則化**: L2 (Ridge)
- **最佳C**: {best_C:.4f}
- **類別權重**: 平衡

## 效能（內部測試集）

### 區分度
- **ROC-AUC**: {test_roc_auc:.4f}
- **PR-AUC**: {test_pr_auc:.4f}

### 校正品質
- **校正方法**: Sigmoid (Platt) - 保持風險區分度
- **Brier Score (校正前)**: {test_brier_before:.4f}
- **Brier Score (校正後)**: {test_brier_after:.4f}
- **校正斜率 (校正前)**: {slope_before:.4f}
- **校正斜率 (校正後)**: {slope_after:.4f} (理想值=1.0)

## 推薦閾值（基於OOF）

| 策略 | 閾值 | 用途 |
|------|------|------|
| F1-Optimal | {threshold_f1:.4f} | 平衡效能 |
| Sens≥70% | {threshold_sens:.4f} | 高敏感度篩檢 |
| Youden Index | {threshold_youden:.4f} | 最大化Sens+Spec |

## 檔案說明

1. **pipeline_uncalibrated.pkl**: 未校正的完整Pipeline
2. **pipeline_calibrated.pkl**: 已校正的完整Pipeline（推薦用於臨床預測）
3. **model_metadata.json**: 完整的模型元數據
4. **feature_coefficients.csv**: 特徵係數表
5. **README.md**: 本文件

## 使用方法

### Python範例

```python
import pickle
import pandas as pd

# 載入校正後的模型
with open('pipeline_calibrated.pkl', 'rb') as f:
    model = pickle.load(f)

# 準備外部驗證數據（必須包含所有13個基礎特徵）
# 注意：不需要手動創建交互項，Pipeline會自動處理
external_data = pd.DataFrame({{
    'Age': [...],
    'LNR': [...],
    'Tumor_Size_cm': [...],
    'Log_CEA_PreOp': [...],
    'pT_ordinal': [...],
    'pN_ordinal': [...],
    'Differentiation_ordinal': [...],
    'Tumor_Location': [...],
    'LVI': [...],
    'PNI': [...],
    'Tumor_Deposits': [...],
    'Mucinous_Any': [...],
    'MSI_High': [...]
}})

# 預測
proba = model.predict_proba(external_data)[:, 1]

# 應用閾值
f1_optimal_threshold = {threshold_f1:.4f}
predictions = (proba >= f1_optimal_threshold).astype(int)
```

## 外部驗證

請使用提供的 `external_validation.py` 腳本進行外部驗證。

## 注意事項

1. **特徵預處理**: Pipeline已包含完整預處理（填補、標準化、交互項），無需額外處理
2. **缺失值**: 連續變數用中位數填補，其他用眾數填補
3. **交互項**: 自動創建，無需手動計算
4. **推薦使用**: 使用 `pipeline_calibrated.pkl` 進行臨床預測

## 聯絡資訊

如有問題，請聯繫模型開發團隊。
"""

with open(freeze_dir / 'README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"✓ 已保存: README.md")

# ====================================
# 總結
# ====================================

print("\n" + "=" * 80)
print("模型凍結完成")
print("=" * 80)

print(f"""
【凍結模型資訊】
  版本: 1.0
  日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  訓練樣本: {len(y_train)} (事件: {y_train.sum()})
  特徵數: 15 (13基礎 + 2交互項)
  EPV: {metadata['epv']:.2f}

【內部測試集效能】
  ROC-AUC: {test_roc_auc:.4f}
  Brier (校正後): {test_brier_after:.4f}
  校正斜率 (校正前): {slope_before:.4f}

【已保存檔案】
  {freeze_dir}/
  ├── pipeline_uncalibrated.pkl    (未校正模型)
  ├── pipeline_calibrated.pkl      (已校正模型) ⭐推薦
  ├── model_metadata.json          (完整元數據)
  ├── feature_coefficients.csv     (係數表)
  └── README.md                    (使用說明)

【下一步】
  使用 external_validation.py 腳本載入外部數據進行驗證
  
✅ 模型已凍結，準備外部驗證！
""")

print("=" * 80)

