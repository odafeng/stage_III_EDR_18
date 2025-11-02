"""
特徵工程模組
Feature engineering for survival analysis
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class SurvivalFeatureEngineer:
    """存活分析特徵工程類別"""
    
    def __init__(self):
        """初始化特徵工程器"""
        self.feature_importance = None
        
    def create_lymph_node_ratio(self, df):
        """
        建立淋巴結轉移比例特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
            
        Returns:
        --------
        pd.DataFrame : 新增特徵後的資料
        """
        if 'lymph_nodes_positive' in df.columns and 'lymph_nodes_examined' in df.columns:
            df['lymph_node_ratio'] = df['lymph_nodes_positive'] / (df['lymph_nodes_examined'] + 1e-6)
            print("已建立淋巴結轉移比例特徵")
        return df
    
    def create_age_groups(self, df, age_col='age'):
        """
        建立年齡分組特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
        age_col : str
            年齡欄位名稱
            
        Returns:
        --------
        pd.DataFrame : 新增特徵後的資料
        """
        if age_col in df.columns:
            df['age_group'] = pd.cut(
                df[age_col], 
                bins=[0, 50, 65, 75, 100], 
                labels=['<50', '50-65', '65-75', '>75']
            )
            df['age_group'] = df['age_group'].cat.codes
            print("已建立年齡分組特徵")
        return df
    
    def create_tumor_burden(self, df):
        """
        建立腫瘤負擔特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
            
        Returns:
        --------
        pd.DataFrame : 新增特徵後的資料
        """
        if 'tumor_size' in df.columns and 'lymph_nodes_positive' in df.columns:
            df['tumor_burden'] = df['tumor_size'] * (1 + df['lymph_nodes_positive'])
            print("已建立腫瘤負擔特徵")
        return df
    
    def create_high_risk_indicator(self, df):
        """
        建立高風險指標
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
            
        Returns:
        --------
        pd.DataFrame : 新增特徵後的資料
        """
        # 定義高風險條件
        conditions = []
        
        if 'lymph_nodes_positive' in df.columns:
            conditions.append(df['lymph_nodes_positive'] >= 4)
        
        if 'CEA_level' in df.columns:
            conditions.append(df['CEA_level'] > 5)
        
        if 'differentiation' in df.columns:
            conditions.append(df['differentiation'] == 'poor')
        
        if conditions:
            df['high_risk'] = sum(conditions).astype(int)
            print("已建立高風險指標特徵")
        
        return df
    
    def select_features(self, X, y, k=10, method='mutual_info'):
        """
        特徵選擇
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徵矩陣
        y : pd.Series
            目標變數
        k : int
            選擇特徵數量
        method : str
            選擇方法 ('f_classif', 'mutual_info')
            
        Returns:
        --------
        pd.DataFrame : 選擇後的特徵
        """
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"已選擇 {len(selected_features)} 個特徵:")
        print(selected_features)
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def create_interaction_features(self, df, feature_pairs):
        """
        建立交互作用特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
        feature_pairs : list of tuples
            特徵對列表
            
        Returns:
        --------
        pd.DataFrame : 新增特徵後的資料
        """
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        
        print(f"已建立 {len(feature_pairs)} 個交互作用特徵")
        return df
    
    def apply_all_features(self, df):
        """
        應用所有特徵工程
        
        Parameters:
        -----------
        df : pd.DataFrame
            資料框
            
        Returns:
        --------
        pd.DataFrame : 特徵工程後的資料
        """
        df = self.create_lymph_node_ratio(df)
        df = self.create_age_groups(df)
        df = self.create_tumor_burden(df)
        df = self.create_high_risk_indicator(df)
        
        print("所有特徵工程已完成")
        return df


if __name__ == "__main__":
    # 測試特徵工程功能
    engineer = SurvivalFeatureEngineer()
    print("特徵工程模組已載入")
