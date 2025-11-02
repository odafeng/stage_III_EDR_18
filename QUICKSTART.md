# 快速開始指南

## 專案簡介
這是一個第三期大腸癌存活預測的研究型專案，使用機器學習方法進行存活分析。

## 專案結構說明

```
Stage III Surv/
│
├── data/                          # 資料目錄
│   ├── raw/                       # 原始資料
│   │   └── raw_data.csv          # 原始資料檔案
│   └── processed/                 # 處理後的資料
│       ├── train_data.csv        # 訓練資料
│       ├── test_data.csv         # 測試資料
│       ├── train_features.csv    # 訓練特徵
│       └── test_features.csv     # 測試特徵
│
├── notebooks/                     # Jupyter notebooks (分析流程)
│   ├── 01_exploratory_data_analysis.ipynb    # 探索性資料分析
│   ├── 02_data_preprocessing.ipynb           # 資料預處理
│   ├── 03_feature_engineering.ipynb          # 特徵工程
│   ├── 04_model_training.ipynb               # 模型訓練
│   └── 05_model_evaluation.ipynb             # 模型評估
│
├── src/                           # 原始碼模組
│   ├── __init__.py               # 初始化檔案
│   ├── utils.py                  # 工具函數
│   ├── data_preprocessing.py     # 資料預處理類別
│   ├── feature_engineering.py    # 特徵工程類別
│   ├── model_training.py         # 模型訓練類別
│   └── model_evaluation.py       # 模型評估類別
│
├── models/                        # 訓練好的模型
│   ├── cox_ph.pkl                # Cox 比例風險模型
│   └── random_survival_forest.pkl # 隨機存活森林
│
├── results/                       # 結果輸出
│   ├── figures/                   # 圖表
│   │   ├── kaplan_meier_overall.png
│   │   ├── cox_risk_groups.png
│   │   ├── rsf_feature_importance.png
│   │   └── model_comparison.png
│   └── tables/                    # 表格
│       ├── cox_evaluation.txt
│       ├── rsf_evaluation.txt
│       └── model_comparison.csv
│
├── config/                        # 配置檔案
│   └── config.yaml               # 主配置檔案
│
├── requirements.txt               # Python 套件依賴
├── environment.yml                # Conda 環境配置
├── .gitignore                    # Git 忽略檔案
└── README.md                     # 專案說明文件
```

## 安裝步驟

### 方法 1: 使用 pip

1. 建立虛擬環境（建議）:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

2. 安裝套件:
```bash
pip install -r requirements.txt
```

### 方法 2: 使用 Conda

1. 建立並啟動環境:
```bash
conda env create -f environment.yml
conda activate colorectal-survival
```

## 使用流程

### 步驟 1: 資料準備
將您的原始資料放置在 `data/raw/raw_data.csv`

**資料格式建議:**
- 必要欄位:
  - `survival_time`: 存活時間（月）
  - `event`: 事件狀態 (0=censored, 1=death)
  
- 臨床特徵範例:
  - `age`: 年齡
  - `gender`: 性別
  - `tumor_location`: 腫瘤位置
  - `tumor_size`: 腫瘤大小
  - `lymph_nodes_examined`: 檢查的淋巴結數量
  - `lymph_nodes_positive`: 陽性淋巴結數量
  - `differentiation`: 分化程度
  - `CEA_level`: CEA 數值
  - `chemotherapy`: 是否接受化療
  - `radiation`: 是否接受放療

### 步驟 2: 執行分析流程

依序執行以下 Jupyter Notebooks:

#### 2.1 探索性資料分析
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```
- 了解資料分佈
- 檢查缺失值
- 視覺化變數關係

#### 2.2 資料預處理
```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```
- 處理缺失值
- 編碼類別變數
- 標準化數值特徵
- 分割訓練/測試集

#### 2.3 特徵工程
```bash
jupyter notebook notebooks/03_feature_engineering.ipynb
```
- 建立新特徵
- 特徵選擇
- 交互作用特徵

#### 2.4 模型訓練
```bash
jupyter notebook notebooks/04_model_training.ipynb
```
- 訓練 Cox 比例風險模型
- 訓練隨機存活森林
- 儲存模型

#### 2.5 模型評估
```bash
jupyter notebook notebooks/05_model_evaluation.ipynb
```
- 計算評估指標 (C-index, Brier Score)
- 繪製 Kaplan-Meier 曲線
- 比較模型效能

## 主要功能模組說明

### 1. 資料預處理 (data_preprocessing.py)
```python
from src.data_preprocessing import SurvivalDataPreprocessor

preprocessor = SurvivalDataPreprocessor()
df = preprocessor.load_data('data/raw/raw_data.csv')
df_clean = preprocessor.handle_missing_values(df)
train_df, test_df = preprocessor.split_data(df_clean)
```

### 2. 特徵工程 (feature_engineering.py)
```python
from src.feature_engineering import SurvivalFeatureEngineer

engineer = SurvivalFeatureEngineer()
df_engineered = engineer.apply_all_features(df)
```

### 3. 模型訓練 (model_training.py)
```python
from src.model_training import SurvivalModelTrainer

trainer = SurvivalModelTrainer()
cox_model = trainer.train_cox_ph(train_df)
trainer.save_all_models('models/')
```

### 4. 模型評估 (model_evaluation.py)
```python
from src.model_evaluation import SurvivalModelEvaluator

evaluator = SurvivalModelEvaluator()
c_index = evaluator.calculate_c_index(y_true_time, y_true_event, y_pred_risk)
evaluator.plot_kaplan_meier_curves(durations, events)
```

## 配置檔案 (config.yaml)

在 `config/config.yaml` 中可以調整:
- 資料路徑
- 分割比例
- 模型參數
- 評估指標

## 評估指標說明

1. **C-index (Concordance Index)**: 衡量模型預測順序的準確性
   - 範圍: 0.5-1.0
   - 0.5 = 隨機預測
   - 1.0 = 完美預測

2. **Brier Score**: 衡量預測機率的準確性
   - 範圍: 0-1
   - 越低越好

3. **Time-dependent AUC**: 特定時間點的預測準確性

4. **Kaplan-Meier 曲線**: 視覺化存活機率隨時間變化

## 常見問題

### Q1: 資料欄位名稱不符合怎麼辦？
A: 在 `config/config.yaml` 中修改欄位名稱，或在預處理階段重新命名欄位。

### Q2: 需要哪些最少的資料？
A: 至少需要存活時間 (survival_time) 和事件狀態 (event) 兩個欄位。

### Q3: 如何添加新的模型？
A: 在 `src/model_training.py` 中新增訓練函數，然後在 `04_model_training.ipynb` 中調用。

### Q4: 如何解讀結果？
A: 查看 `results/tables/model_comparison.csv` 比較不同模型，使用 C-index 作為主要指標。

## 進階使用

### 自訂特徵工程
在 `src/feature_engineering.py` 中添加新的特徵建立方法:

```python
def create_custom_feature(self, df):
    # 您的自訂特徵邏輯
    df['new_feature'] = ...
    return df
```

### 添加新模型
在 `src/model_training.py` 中添加新的訓練方法:

```python
def train_custom_model(self, X_train, y_train):
    model = YourModel()
    model.fit(X_train, y_train)
    self.models['custom'] = model
    return model
```

## 注意事項

1. 確保資料品質，處理好缺失值
2. 注意資料的臨床意義，避免資料洩漏
3. 使用交叉驗證來調整超參數
4. 解釋模型結果時要謹慎，考慮臨床可解釋性
5. 定期儲存模型和結果

## 參考文獻

建議閱讀相關存活分析文獻以更好地理解模型和方法。

## 技術支援

如有問題，請檢查:
1. Python 版本 (建議 3.8+)
2. 套件版本是否相容
3. 資料格式是否正確
4. 配置檔案是否正確設定

## 授權

[請添加您的授權資訊]

---

祝研究順利！🎯
