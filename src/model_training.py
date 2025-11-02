"""
模型訓練模組
Model training for survival prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
import joblib
import os


class SurvivalModelTrainer:
    """存活預測模型訓練類別"""
    
    def __init__(self, config=None):
        """
        初始化訓練器
        
        Parameters:
        -----------
        config : dict
            配置字典
        """
        self.config = config
        self.models = {}
        
    def train_cox_ph(self, df, duration_col='survival_time', event_col='event'):
        """
        訓練Cox比例風險模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            訓練資料
        duration_col : str
            存活時間欄位
        event_col : str
            事件欄位
            
        Returns:
        --------
        CoxPHFitter : 訓練好的模型
        """
        print("正在訓練 Cox 比例風險模型...")
        
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df, duration_col=duration_col, event_col=event_col)
        
        self.models['cox_ph'] = cph
        print("Cox 模型訓練完成")
        print(cph.summary)
        
        return cph
    
    def train_random_survival_forest(self, X, y, n_estimators=100, random_state=42):
        """
        訓練隨機存活森林
        
        Parameters:
        -----------
        X : array-like
            特徵矩陣
        y : structured array
            存活資料 (event, time)
        n_estimators : int
            樹的數量
        random_state : int
            隨機種子
            
        Returns:
        --------
        RandomSurvivalForest : 訓練好的模型
        """
        print("正在訓練隨機存活森林...")
        
        rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state
        )
        
        rsf.fit(X, y)
        
        self.models['random_survival_forest'] = rsf
        print("隨機存活森林訓練完成")
        
        return rsf
    
    def train_xgboost_survival(self, X_train, y_train, params=None):
        """
        訓練XGBoost存活模型
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            訓練特徵
        y_train : pd.DataFrame
            訓練標籤
        params : dict
            模型參數
            
        Returns:
        --------
        model : 訓練好的模型
        """
        print("正在訓練 XGBoost 存活模型...")
        
        try:
            from xgbse import XGBSEDebiasedBCE
            
            if params is None:
                params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                }
            
            model = XGBSEDebiasedBCE(**params)
            model.fit(X_train, y_train)
            
            self.models['xgboost'] = model
            print("XGBoost 模型訓練完成")
            
            return model
        
        except ImportError:
            print("警告: 未安裝 xgbse 套件，跳過 XGBoost 存活模型")
            return None
    
    def save_model(self, model_name, filepath):
        """
        儲存模型
        
        Parameters:
        -----------
        model_name : str
            模型名稱
        filepath : str
            儲存路徑
        """
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"模型 '{model_name}' 已儲存至: {filepath}")
        else:
            print(f"錯誤: 找不到模型 '{model_name}'")
    
    def load_model(self, filepath):
        """
        載入模型
        
        Parameters:
        -----------
        filepath : str
            模型檔案路徑
            
        Returns:
        --------
        model : 載入的模型
        """
        model = joblib.load(filepath)
        print(f"模型已從 {filepath} 載入")
        return model
    
    def save_all_models(self, output_dir='models/'):
        """
        儲存所有訓練好的模型
        
        Parameters:
        -----------
        output_dir : str
            輸出目錄
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(output_dir, f'{model_name}.pkl')
            self.save_model(model_name, filepath)
        
        print(f"所有模型已儲存至: {output_dir}")


if __name__ == "__main__":
    # 測試模型訓練功能
    trainer = SurvivalModelTrainer()
    print("模型訓練模組已載入")
