"""
工具函數模組
Utility functions for the project
"""

import yaml
import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path='config/config.yaml'):
    """
    載入配置檔案
    
    Parameters:
    -----------
    config_path : str
        配置檔案路徑
        
    Returns:
    --------
    dict : 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config):
    """
    建立專案所需的目錄
    
    Parameters:
    -----------
    config : dict
        配置字典
    """
    dirs = [
        config['output']['models_dir'],
        config['output']['figures_dir'],
        config['output']['tables_dir'],
        'data/raw',
        'data/processed'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("所有目錄已建立完成")


def save_results(results, filename, output_dir='results/tables/'):
    """
    儲存結果到檔案
    
    Parameters:
    -----------
    results : dict or pd.DataFrame
        結果資料
    filename : str
        檔案名稱
    output_dir : str
        輸出目錄
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if isinstance(results, pd.DataFrame):
        results.to_csv(filepath, index=False)
    elif isinstance(results, dict):
        pd.DataFrame([results]).to_csv(filepath, index=False)
    
    print(f"結果已儲存至: {filepath}")


def load_data(filepath):
    """
    載入資料
    
    Parameters:
    -----------
    filepath : str
        資料檔案路徑
        
    Returns:
    --------
    pd.DataFrame : 資料框
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    else:
        raise ValueError("不支援的檔案格式")


def get_project_root():
    """
    取得專案根目錄
    
    Returns:
    --------
    Path : 專案根目錄路徑
    """
    return Path(__file__).parent.parent


if __name__ == "__main__":
    # 測試配置載入
    config = load_config()
    print("配置載入成功:")
    print(config)
