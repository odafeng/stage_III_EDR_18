# Stage III Colorectal Cancer Survival Prediction

## 專案概述
本專案旨在使用機器學習方法預測第三期大腸癌患者的存活率。

## 專案結構
```
Stage III Surv/
│
├── data/                      # 資料目錄
│   ├── raw/                   # 原始資料
│   └── processed/             # 處理後的資料
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/                       # 原始碼
│   ├── __init__.py
│   ├── data_preprocessing.py  # 資料預處理
│   ├── feature_engineering.py # 特徵工程
│   ├── model_training.py      # 模型訓練
│   ├── model_evaluation.py    # 模型評估
│   └── utils.py               # 工具函數
│
├── models/                    # 訓練好的模型
│
├── results/                   # 結果輸出
│   ├── figures/               # 圖表
│   └── tables/                # 表格
│
├── config/                    # 配置檔案
│   └── config.yaml
│
├── requirements.txt           # Python套件依賴
├── environment.yml            # Conda環境配置
└── README.md                  # 專案說明文件
```

## 安裝方式

### 使用 pip
```bash
pip install -r requirements.txt
```

### 使用 conda
```bash
conda env create -f environment.yml
conda activate colorectal-survival
```

## 使用方法

1. **資料探索**: 執行 `notebooks/01_exploratory_data_analysis.ipynb`
2. **資料預處理**: 執行 `notebooks/02_data_preprocessing.ipynb`
3. **特徵工程**: 執行 `notebooks/03_feature_engineering.ipynb`
4. **模型訓練**: 執行 `notebooks/04_model_training.ipynb`
5. **模型評估**: 執行 `notebooks/05_model_evaluation.ipynb`

## 研究目標

- 分析影響第三期大腸癌患者存活的關鍵因素
- 建立準確的存活預測模型
- 比較不同機器學習演算法的效能
- 提供臨床決策支援工具

## 模型評估指標

- C-index (Concordance Index)
- Time-dependent AUC
- Brier Score
- Kaplan-Meier 生存曲線
- Log-rank test

## 授權
[請添加授權資訊]

## 聯絡方式
[請添加聯絡資訊]
