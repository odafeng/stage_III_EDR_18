"""
資料預處理模組
Data preprocessing module for survival analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class SurvivalDataPreprocessor:
    """第三期大腸癌存活資料預處理類別"""
    
    def __init__(self, config=None):
        """
        初始化預處理器
        
        Parameters:
        -----------
        config : dict
            配置字典
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """
        載入原始資料
        
        Parameters:
        -----------
        filepath : str
            資料檔案路徑
            
        Returns:
        --------
        pd.DataFrame : 原始資料
        """
        print(f"正在載入資料: {filepath}")
        data = pd.read_csv(filepath)
        print(f"資料形狀: {data.shape}")
        return data
    
    def handle_missing_values(self, df, strategy='median'):
        """
        處理缺失值
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
        strategy : str
            處理策略 ('median', 'mean', 'mode', 'drop')
            
        Returns:
        --------
        pd.DataFrame : 處理後的資料
        """
        print(f"缺失值統計:\n{df.isnull().sum()}")
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'median':
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mean':
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'mode':
            for col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
        print(f"處理後缺失值: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical_variables(self, df, categorical_cols):
        """
        編碼類別變數
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
        categorical_cols : list
            類別變數欄位列表
            
        Returns:
        --------
        pd.DataFrame : 編碼後的資料
        """
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"已編碼 {len(categorical_cols)} 個類別變數")
        return df
    
    def normalize_features(self, df, numeric_cols):
        """
        標準化數值特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
        numeric_cols : list
            數值變數欄位列表
            
        Returns:
        --------
        pd.DataFrame : 標準化後的資料
        """
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        print(f"已標準化 {len(numeric_cols)} 個數值特徵")
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        分割訓練集與測試集
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
        test_size : float
            測試集比例
        random_state : int
            隨機種子
            
        Returns:
        --------
        tuple : (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['event'] if 'event' in df.columns else None
        )
        
        print(f"訓練集大小: {train_df.shape}")
        print(f"測試集大小: {test_df.shape}")
        
        return train_df, test_df
    
    def save_processed_data(self, train_df, test_df, train_path, test_path):
        """
        儲存處理後的資料
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            訓練資料
        test_df : pd.DataFrame
            測試資料
        train_path : str
            訓練資料儲存路徑
        test_path : str
            測試資料儲存路徑
        """
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"訓練資料已儲存至: {train_path}")
        print(f"測試資料已儲存至: {test_path}")


if __name__ == "__main__":
    # 測試預處理功能
    preprocessor = SurvivalDataPreprocessor()
    print("資料預處理模組已載入")
