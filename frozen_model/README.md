# Stage III Colon Cancer EDR Prediction Model (Frozen)

## 模型資訊

- **版本**: 1.0
- **凍結日期**: 2025-11-07 22:21:44
- **目標變數**: edr_18m (18個月內早期疾病復發)
- **訓練樣本**: 253 例 (事件: 49)
- **EPV**: 3.06

## 模型配置

### 特徵 (15個)
- **連續變數** (4): Age, LNR, Tumor_Size_cm, Log_CEA_PreOp
- **有序變數** (3): pT_ordinal, pN_ordinal, Differentiation_ordinal
- **二元變數** (6): Tumor_Location, LVI, PNI, Tumor_Deposits, Mucinous_Any, MSI_High
- **交互項** (2): LNR × pN_ordinal, LVI × PNI

### 模型參數
- **正則化**: L2 (Ridge)
- **最佳C**: 0.0127
- **類別權重**: 平衡

## 效能（內部測試集）

### 區分度
- **ROC-AUC**: 0.6509
- **PR-AUC**: 0.3195

### 校正品質
- **校正方法**: Sigmoid (Platt) - 保持風險區分度
- **Brier Score (校正前)**: 0.2234
- **Brier Score (校正後)**: 0.1314
- **校正斜率 (校正前)**: 1.1119
- **校正斜率 (校正後)**: 0.9906 (理想值=1.0)

## 推薦閾值（基於OOF）

| 策略 | 閾值 | 用途 |
|------|------|------|
| F1-Optimal | 0.5059 | 平衡效能 |
| Sens≥70% | 0.4920 | 高敏感度篩檢 |
| Youden Index | 0.4983 | 最大化Sens+Spec |

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
external_data = pd.DataFrame({
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
})

# 預測
proba = model.predict_proba(external_data)[:, 1]

# 應用閾值
f1_optimal_threshold = 0.5059
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
