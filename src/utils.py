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


def convert_data_types(df, verbose=True):
    """
    將資料框的欄位轉換為正確的資料型態
    適用於 Stage III 結腸癌存活預測研究
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始資料框
    verbose : bool
        是否顯示詳細訊息
        
    Returns:
    --------
    pd.DataFrame : 轉換後的資料框
    """
    df = df.copy()
    
    if verbose:
        print("=" * 60)
        print("開始資料型態轉換...")
        print("=" * 60)
    
    # 1. 日期欄位
    date_cols = ['Dx_Date', 'Radical_Op_Date', 'Last_FU_Date', 'Recurrence_Date']
    if verbose:
        print("\n【步驟 1】轉換日期欄位...")
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if verbose:
                print(f"  ✓ {col}")
    
    # 2. 有序類別變數
    if verbose:
        print("\n【步驟 2】轉換有序類別變數...")
    
    # pT Stage (實際資料: '1', '2', '3', '4A', '4B')
    if 'pT_Stage' in df.columns:
        df['pT_Stage'] = df['pT_Stage'].astype(str)
        T_stage = pd.CategoricalDtype(
            categories=['1', '2', '3', '4', '4A', '4B'], 
            ordered=True
        )
        df['pT_Stage'] = df['pT_Stage'].replace('nan', pd.NA)
        df['pT_Stage'] = df['pT_Stage'].astype(T_stage)
        if verbose:
            print(f"  ✓ pT_Stage (非 NaN: {df['pT_Stage'].notna().sum()})")
    
    # pN Stage (實際資料: '1A', '1B', '1C', '2A', '2B')
    if 'pN_Stage' in df.columns:
        df['pN_Stage'] = df['pN_Stage'].astype(str)
        N_stage = pd.CategoricalDtype(
            categories=['1A', '1B', '1C', '2A', '2B'], 
            ordered=True
        )
        df['pN_Stage'] = df['pN_Stage'].replace('nan', pd.NA)
        df['pN_Stage'] = df['pN_Stage'].astype(N_stage)
        if verbose:
            print(f"  ✓ pN_Stage (非 NaN: {df['pN_Stage'].notna().sum()})")
    
    # AJCC Substage (實際資料: '3A', '3B', '3C')
    if 'AJCC_Substage' in df.columns:
        df['AJCC_Substage'] = df['AJCC_Substage'].astype(str)
        AJCC_Stage = pd.CategoricalDtype(
            categories=['3A', '3B', '3C'], 
            ordered=True
        )
        df['AJCC_Substage'] = df['AJCC_Substage'].replace('nan', pd.NA)
        df['AJCC_Substage'] = df['AJCC_Substage'].astype(AJCC_Stage)
        if verbose:
            print(f"  ✓ AJCC_Substage (非 NaN: {df['AJCC_Substage'].notna().sum()})")
    
    # Differentiation (實際資料: 1, 2, 3, 4, 9 - 數字編碼)
    # 1=Well, 2=Moderate, 3=Poor, 4=Undifferentiated, 9=Unknown
    if 'Differentiation' in df.columns:
        diff_categories = pd.CategoricalDtype(
            categories=[1, 2, 3, 4, 9], 
            ordered=True
        )
        df['Differentiation'] = df['Differentiation'].astype(diff_categories)
        if verbose:
            print(f"  ✓ Differentiation (非 NaN: {df['Differentiation'].notna().sum()})")
    
    # ECOG
    if 'ECOG' in df.columns:
        ecog_categories = pd.CategoricalDtype(categories=[0, 1, 2, 3, 4], ordered=True)
        df['ECOG'] = df['ECOG'].astype(ecog_categories)
        if verbose:
            print(f"  ✓ ECOG")
    
    # 3. 二元變數 (實際資料已經是 0/1)
    if verbose:
        print("\n【步驟 3】轉換二元變數...")
    
    binary_cols = [
        'LVI', 'PNI', 'Tumor_Deposits', 'Mucinous_Gt_50', 
        'Mucinous_Any', 'Signet_Ring', 'Recurrence', 'Death'
    ]
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            if verbose:
                count_0 = (df[col] == 0).sum()
                count_1 = (df[col] == 1).sum()
                print(f"  ✓ {col} (0={count_0}, 1={count_1})")
    
    # 4. 無序類別變數
    if verbose:
        print("\n【步驟 4】轉換無序類別變數...")
    
    categorical_cols = [
        'Sex', 'Tumor_Location', 'Tumor_Location_Group', 'Histology',
        'MSI_Status', 'Op_Procedure', 'Visiting_Staff',
        'Recurrence_Type', 'Death_Cause'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            if verbose:
                print(f"  ✓ {col} ({df[col].nunique()} 類)")
    
    # 5. 數值變數
    if verbose:
        print("\n【步驟 5】確認數值變數...")
    
    numeric_cols = [
        'Age', 'BMI', 'LN_Total', 'LN_Positive', 'LNR', 
        'Tumor_Size_cm', 'CEA_PreOp', 'Log_CEA_PreOp', 
        'PreOp_Albumin', 'DFS_Months', 'OS_Months', 'Dx_Year'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if verbose:
                print(f"  ✓ {col}")
    
    # 6. 識別碼
    if verbose:
        print("\n【步驟 6】確認識別碼欄位...")
    
    id_cols = ['Patient_ID', 'Chart_No']
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            if verbose:
                print(f"  ✓ {col}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ 資料型態轉換完成！")
        print("=" * 60)
    
    return df


def validate_data_types(df):
    """
    驗證資料型態是否正確
    
    Parameters:
    -----------
    df : pd.DataFrame
        資料框
        
    Returns:
    --------
    pd.DataFrame : 驗證結果摘要
    """
    validation_results = []
    
    # 預期的資料型態
    expected_types = {
        'Patient_ID': 'object',
        'Chart_No': 'object',
        'Dx_Date': 'datetime64[ns]',
        'Radical_Op_Date': 'datetime64[ns]',
        'Last_FU_Date': 'datetime64[ns]',
        'Recurrence_Date': 'datetime64[ns]',
        'Age': 'float64',
        'BMI': 'float64',
        'LNR': 'float64',
        'Recurrence': 'Int64',
        'Death': 'Int64',
        'pT_Stage': 'category',
        'pN_Stage': 'category',
        'AJCC_Substage': 'category',
    }
    
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            is_valid = expected_type in actual_type or actual_type in expected_type
            validation_results.append({
                'Column': col,
                'Expected_Type': expected_type,
                'Actual_Type': actual_type,
                'Valid': '✓' if is_valid else '✗'
            })
    
    return pd.DataFrame(validation_results)


if __name__ == "__main__":
    # 測試配置載入
    config = load_config()
    print("配置載入成功:")
    print(config)
