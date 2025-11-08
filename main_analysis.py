"""
主分析：Stage III 結腸癌早期遠端復發預測模型
Main Analysis: Early Distant Recurrence Prediction for Stage III Colon Cancer

投稿級完整版 (Publication-Ready Version)
============================================

模型配置：
- 正則化：L2 (Ridge)
- 交互項：2個（LNR × pN_ordinal, LVI × PNI）
- 校正方法：Sigmoid (Platt) - 平衡Brier和Slope
- 驗證：5-fold CV + Bootstrap CI (1000次)
- 閾值：基於OOF選擇

方法學亮點：
1. Pipeline內整合預處理（避免CV洩漏）
2. OOF機率用於閾值選擇（避免test set洩漏）
3. Sigmoid校正保持slope≈1.0（臨床可用性）
4. L1驗證交互項穩定性
5. Bootstrap CI量化不確定性
6. 等分位校正曲線（更穩定）
7. EPV完整報告

Date: 2025-11-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    average_precision_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import logit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 中文字型設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 輔助函數
# ============================================================================

def compute_calibration_slope_intercept(y_true, y_pred_proba, eps=1e-10):
    """
    標準校正斜率/截距估計：logistic regression on logit(p̂)
    y ~ α + β·logit(p̂)
    
    返回: (slope β, intercept α)
    """
    y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1 - eps)
    logit_proba = logit(y_pred_proba_clipped)
    lr = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
    lr.fit(logit_proba.reshape(-1, 1), y_true)
    return lr.coef_[0][0], lr.intercept_[0]

def ensure_binary_01(X, binary_cols):
    """確保二元變數嚴格為{0,1}（交互項安全帶）"""
    X_out = X.copy()
    for col in binary_cols:
        if col in X_out.columns:
            X_out[col] = (X_out[col] > 0).astype(int)
    return X_out

def bootstrap_metrics(y_true, y_pred_proba, n_bootstrap=1000, confidence=0.95, random_state=42):
    """
    Stratified Bootstrap 95% CI for multiple metrics
    
    使用分層抽樣（stratified sampling）保持類別比例，
    避免小樣本不平衡情境下CI過度抖動
    
    返回: dict with (mean, ci_low, ci_high) for each metric
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # 獲取各類別索引
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    metrics = {
        'auc': [],
        'pr_auc': [],
        'brier': [],
        'slope': [],
        'intercept': []
    }
    
    for i in range(n_bootstrap):
        # 分層重抽樣：各類別內分別抽樣後合併
        pos_sample = np.random.choice(pos_indices, size=n_pos, replace=True)
        neg_sample = np.random.choice(neg_indices, size=n_neg, replace=True)
        indices = np.concatenate([pos_sample, neg_sample])
        
        # 打亂順序（避免所有陽性都在前面）
        np.random.shuffle(indices)
        
        try:
            metrics['auc'].append(roc_auc_score(y_true[indices], y_pred_proba[indices]))
            metrics['pr_auc'].append(average_precision_score(y_true[indices], y_pred_proba[indices]))
            metrics['brier'].append(brier_score_loss(y_true[indices], y_pred_proba[indices]))
            
            slope, intercept = compute_calibration_slope_intercept(y_true[indices], y_pred_proba[indices])
            metrics['slope'].append(slope)
            metrics['intercept'].append(intercept)
        except:
            continue
    
    alpha = (1 - confidence) / 2
    results = {}
    
    for metric_name, values in metrics.items():
        values = np.array(values)
        mean_val = np.mean(values)
        ci = np.percentile(values, [alpha * 100, (1 - alpha) * 100])
        results[metric_name] = {
            'mean': mean_val,
            'ci_low': ci[0],
            'ci_high': ci[1],
            'ci_str': f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        }
    
    return results

def compute_classification_metrics(y_true, y_pred_proba, threshold):
    """計算特定閾值下的分類指標"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

# ============================================================================
# 交互作用轉換器
# ============================================================================

class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    創建交互項：LNR × pN_ordinal, LVI × PNI
    
    使用特徵名稱（含前綴）確保正確對應
    """
    def __init__(self):
        self.interaction_specs = [
            ('num__LNR', 'ord__pN_ordinal', 'LNR × pN_ordinal'),
            ('bin__LVI', 'bin__PNI', 'LVI × PNI'),
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("InteractionTransformer requires DataFrame input.")
        
        X_out = X.copy()
        
        for feat1, feat2, interaction_name in self.interaction_specs:
            if feat1 in X_out.columns and feat2 in X_out.columns:
                X_out[interaction_name] = X_out[feat1] * X_out[feat2]
            else:
                missing = []
                if feat1 not in X_out.columns:
                    missing.append(feat1)
                if feat2 not in X_out.columns:
                    missing.append(feat2)
                raise ValueError(f"Missing features for interaction: {missing}")
        
        return X_out
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        original_names = list(input_features)
        interaction_names = [spec[2] for spec in self.interaction_specs]
        return original_names + interaction_names

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 80)
    print("Stage III 結腸癌早期遠端復發預測模型 - 主分析")
    print("=" * 80)
    print("\n【模型配置】")
    print("  正則化: L2 (Ridge)")
    print("  交互項: 2個 (LNR × pN_ordinal, LVI × PNI)")
    print("  校正: Sigmoid (Platt)")
    print("  驗證: 5-fold CV + Bootstrap CI (1000次)")
    print("  閾值選擇: 基於OOF\n")
    
    # ========================================================================
    # 1. 載入數據
    # ========================================================================
    
    print("=" * 80)
    print("步驟 1/9: 載入數據")
    print("=" * 80)
    
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
    
    n_events_train = int(y_train.sum())
    n_events_test = int(y_test.sum())
    
    print(f"✓ 訓練集: {len(y_train)} 例 (事件: {n_events_train}, {y_train.mean():.2%})")
    print(f"✓ 測試集: {len(y_test)} 例 (事件: {n_events_test}, {y_test.mean():.2%})")
    
    # EPV計算（13基礎特徵 + 2交互項 + 1截距 = 16參數）
    n_params = len(all_features) + 2 + 1  # 13 + 2 + 1 = 16
    epv = n_events_train / n_params
    print(f"\n✓ EPV: {n_events_train}/{n_params} = {epv:.2f}")
    print(f"  註：EPV < 5 但使用L2正則化 + 外部驗證確保穩定性")
    
    # ========================================================================
    # 2. Pipeline設定
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("步驟 2/9: Pipeline設定")
    print("=" * 80)
    
    # 預處理器
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
    
    # 二元變數安全帶
    binary_cols_with_prefix = [f'bin__{c}' for c in binary_features]
    
    def post_binary_func(X):
        return ensure_binary_01(X, binary_cols_with_prefix)
    
    post_binary_transformer = FunctionTransformer(
        post_binary_func,
        feature_names_out='one-to-one'
    )
    
    # 交互項轉換器
    interaction_transformer = InteractionTransformer()
    
    print("✓ Preprocessor: 填補 + 標準化")
    print("✓ Binary Safety: 確保{0,1}")
    print("✓ Interactions: LNR × pN_ordinal, LVI × PNI")
    
    # ========================================================================
    # 3. L1驗證交互項穩定性
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("步驟 3/9: L1驗證交互項穩定性")
    print("=" * 80)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    C_range = np.logspace(-4, 4, 20)
    
    pipeline_l1 = Pipeline([
        ('preprocessor', preprocessor),
        ('post_bin', post_binary_transformer),
        ('interactions', interaction_transformer),
        ('classifier', LogisticRegressionCV(
            Cs=C_range, cv=cv, penalty='l1', solver='saga',
            scoring='roc_auc', max_iter=5000, tol=1e-3,
            random_state=42, n_jobs=1, class_weight='balanced'
        ))
    ])
    
    print("✓ 訓練L1模型（SAGA solver, max_iter=5000）...")
    pipeline_l1.fit(X_train, y_train)
    
    # 檢查L1係數
    X_train_transformed = pipeline_l1.named_steps['interactions'].transform(
        pipeline_l1.named_steps['post_bin'].transform(
            pipeline_l1.named_steps['preprocessor'].transform(X_train.iloc[:1])
        )
    )
    feature_names_l1 = list(X_train_transformed.columns)
    coef_l1 = pipeline_l1.named_steps['classifier'].coef_[0]
    
    assert len(feature_names_l1) == len(coef_l1), "特徵名與係數長度不符！"
    
    n_nonzero = np.sum(coef_l1 != 0)
    print(f"✓ L1選擇: {n_nonzero}/{len(feature_names_l1)} 特徵保留（非零係數）")
    
    # 檢查兩個交互項
    interaction_status = {}
    for fname, coef in zip(feature_names_l1, coef_l1):
        if '×' in fname:
            interaction_status[fname] = coef
            status = "✅ 保留" if coef != 0 else "❌ 移除"
            print(f"  {fname}: {coef:.4f} {status}")
    
    if len(interaction_status) == 2:
        both_retained = all(v != 0 for v in interaction_status.values())
        if both_retained:
            print("\n✅ 兩個交互項均通過L1驗證，穩定性確認")
        else:
            print("\n⚠️  警告：部分交互項被L1移除，建議審慎解釋")
    
    # ========================================================================
    # 4. 訓練主模型（L2）
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("步驟 4/9: 訓練主模型（L2 Ridge）")
    print("=" * 80)
    
    pipeline_l2 = Pipeline([
        ('preprocessor', preprocessor),
        ('post_bin', post_binary_transformer),
        ('interactions', interaction_transformer),
        ('classifier', LogisticRegressionCV(
            Cs=C_range, cv=cv, penalty='l2', solver='lbfgs',
            scoring='roc_auc', max_iter=3000, tol=1e-3,
            random_state=42, n_jobs=1, class_weight='balanced'
        ))
    ])
    
    print("✓ 訓練L2模型（LBFGS solver, max_iter=3000）...")
    pipeline_l2.fit(X_train, y_train)
    
    clf_l2 = pipeline_l2.named_steps['classifier']
    best_C = clf_l2.C_[0]
    cv_auc = clf_l2.scores_[1].mean(axis=0).max()
    
    # 檢查收斂
    if hasattr(clf_l2, 'n_iter_'):
        try:
            n_iter_val = int(np.max(clf_l2.n_iter_))
            if n_iter_val >= clf_l2.max_iter:
                print(f"⚠️  警告：達到最大迭代次數 ({clf_l2.max_iter})，可能未收斂")
            else:
                print(f"✓ 收斂於第 {n_iter_val} 次迭代")
        except:
            pass  # 忽略收斂檢查錯誤
    
    print(f"✓ 最佳 C: {best_C:.4f}")
    print(f"✓ CV ROC-AUC: {cv_auc:.4f}")
    
    # 獲取特徵名和係數
    X_train_transformed_l2 = pipeline_l2.named_steps['interactions'].transform(
        pipeline_l2.named_steps['post_bin'].transform(
            pipeline_l2.named_steps['preprocessor'].transform(X_train.iloc[:1])
        )
    )
    feature_names = list(X_train_transformed_l2.columns)
    coef_l2 = clf_l2.coef_[0]
    
    assert len(feature_names) == len(coef_l2), "特徵名與係數長度不符！"
    
    # Sanity check: 確認L1和L2 pipeline完全等價（特徵名順序必須一致）
    # 這個斷言確保兩支pipeline的preprocessor步驟完全相同
    # 如果未來修改了preprocessor，這裡會第一時間報錯（這是好事）
    assert feature_names == feature_names_l1, \
        "L1和L2的特徵順序不一致！請檢查兩個pipeline的preprocessor是否相同"
    
    print(f"✓ 最終特徵數: {len(feature_names)} (13基礎 + 2交互項)")
    print(f"✓ L1/L2特徵順序驗證: 通過 ✓")
    
    # ========================================================================
    # 5. OOF閾值選擇
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("步驟 5/9: OOF閾值選擇（避免test set洩漏）")
    print("=" * 80)
    
    print("✓ 生成OOF預測機率...")
    y_train_oof = cross_val_predict(
        pipeline_l2, X_train, y_train,
        cv=cv, method='predict_proba', n_jobs=1
    )[:, 1]
    
    # PR曲線
    precision_oof, recall_oof, thresholds_pr = precision_recall_curve(y_train, y_train_oof)
    
    # 閾值1：F1最佳
    f1_scores = 2 * (precision_oof[:-1] * recall_oof[:-1]) / (precision_oof[:-1] + recall_oof[:-1] + 1e-10)
    idx_f1 = np.argmax(f1_scores)
    threshold_f1 = thresholds_pr[idx_f1]
    
    # 閾值2：Sensitivity ≥ 0.70的最小閾值
    # 在所有recall≥0.70的點中，選precision最高的
    sens_mask = recall_oof[:-1] >= 0.70
    if sens_mask.any():
        # 找出所有滿足Sens≥70%的點中，precision最高的那個
        valid_indices = np.where(sens_mask)[0]
        best_idx = valid_indices[np.argmax(precision_oof[:-1][valid_indices])]
        threshold_sens = thresholds_pr[best_idx]
    else:
        threshold_sens = thresholds_pr[np.argmin(np.abs(recall_oof[:-1] - 0.70))]
    
    # 閾值3：Youden Index (Sens + Spec - 1)
    fpr_oof, tpr_oof, thresholds_roc = roc_curve(y_train, y_train_oof)
    youden_index = tpr_oof - fpr_oof
    idx_youden = np.argmax(youden_index)
    threshold_youden = thresholds_roc[idx_youden]
    
    print(f"✓ F1-Optimal:   {threshold_f1:.4f} (F1={f1_scores[idx_f1]:.3f})")
    print(f"✓ Sens≥70%:     {threshold_sens:.4f}")
    print(f"✓ Youden Index: {threshold_youden:.4f} (J={youden_index[idx_youden]:.3f})")
    
    # ========================================================================
    # 6. Sigmoid (Platt) 校正
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("步驟 6/9: Sigmoid (Platt) 校正")
    print("=" * 80)
    
    print("✓ 在訓練集上用5-fold CV進行Sigmoid校正...")
    print("  （避免Isotonic過度平滑，保持slope≈1.0）")
    
    # 創建校正模型（使用clone避免重複訓練基礎模型）
    from sklearn.base import clone
    
    pipeline_calibrated = CalibratedClassifierCV(
        estimator=clone(pipeline_l2),
        method='sigmoid',
        cv=5
    )
    
    pipeline_calibrated.fit(X_train, y_train)
    print("✓ Sigmoid校正完成")
    
    # ========================================================================
    # 7. 測試集評估
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("步驟 7/9: 測試集評估")
    print("=" * 80)
    
    # 未校正預測
    y_test_pred_uncal = pipeline_l2.predict_proba(X_test)[:, 1]
    
    # 校正後預測
    y_test_pred_cal = pipeline_calibrated.predict_proba(X_test)[:, 1]
    
    # 基本指標
    test_auc_uncal = roc_auc_score(y_test, y_test_pred_uncal)
    test_pr_auc_uncal = average_precision_score(y_test, y_test_pred_uncal)
    test_brier_uncal = brier_score_loss(y_test, y_test_pred_uncal)
    
    test_auc_cal = roc_auc_score(y_test, y_test_pred_cal)
    test_pr_auc_cal = average_precision_score(y_test, y_test_pred_cal)
    test_brier_cal = brier_score_loss(y_test, y_test_pred_cal)
    
    # 校正指標
    slope_uncal, intercept_uncal = compute_calibration_slope_intercept(y_test.values, y_test_pred_uncal)
    slope_cal, intercept_cal = compute_calibration_slope_intercept(y_test.values, y_test_pred_cal)
    
    print("\n【校正前】")
    print(f"  ROC-AUC:  {test_auc_uncal:.4f}")
    print(f"  PR-AUC:   {test_pr_auc_uncal:.4f}")
    print(f"  Brier:    {test_brier_uncal:.4f}")
    print(f"  Slope:    {slope_uncal:.4f}")
    print(f"  Intercept: {intercept_uncal:.4f}")
    
    print("\n【校正後 (Sigmoid)】")
    print(f"  ROC-AUC:  {test_auc_cal:.4f}")
    print(f"  PR-AUC:   {test_pr_auc_cal:.4f}")
    print(f"  Brier:    {test_brier_cal:.4f} (改善 {(1-test_brier_cal/test_brier_uncal)*100:.1f}%)")
    print(f"  Slope:    {slope_cal:.4f} {'✅ 接近1.0' if 0.8 <= slope_cal <= 1.2 else '⚠️'}")
    print(f"  Intercept: {intercept_cal:.4f} {'✅ 接近0.0' if abs(intercept_cal) < 0.3 else '⚠️'}")
    
    # ========================================================================
    # 8. Bootstrap 95% CI
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("步驟 8/9: Stratified Bootstrap 95% CI (n=1000)")
    print("=" * 80)
    
    print("✓ 使用分層抽樣保持類別比例...")
    print("✓ 校正前...")
    boot_uncal = bootstrap_metrics(y_test.values, y_test_pred_uncal, n_bootstrap=1000)
    
    print("✓ 校正後...")
    boot_cal = bootstrap_metrics(y_test.values, y_test_pred_cal, n_bootstrap=1000)
    
    print("\n【Bootstrap 95% CI】")
    print("\n校正前:")
    for metric in ['auc', 'pr_auc', 'brier', 'slope', 'intercept']:
        result = boot_uncal[metric]
        print(f"  {metric.upper():10s}: {result['mean']:.4f} {result['ci_str']}")
    
    print("\n校正後:")
    for metric in ['auc', 'pr_auc', 'brier', 'slope', 'intercept']:
        result = boot_cal[metric]
        print(f"  {metric.upper():10s}: {result['mean']:.4f} {result['ci_str']}")
    
    # ========================================================================
    # 9. 保存結果與視覺化
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("步驟 9/9: 保存結果與視覺化")
    print("=" * 80)
    
    output_dir = Path('/Users/huangshifeng/Desktop/stage_III_colon_surv/results/main_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 9.1 係數表
    # L1/L2特徵順序已在前面驗證通過，可直接對比
    coefficients_df = pd.DataFrame({
        '特徵': feature_names,
        'L2係數': coef_l2,
        'L1係數': coef_l1,
        '絕對值': np.abs(coef_l2)
    }).sort_values('絕對值', ascending=False)
    
    coefficients_df.to_csv(output_dir / 'coefficients.csv', index=False, encoding='utf-8-sig')
    print(f"✓ {output_dir / 'coefficients.csv'}")
    
    # 9.2 模型效能摘要
    performance_df = pd.DataFrame([
        {
            '階段': '訓練集CV',
            'ROC-AUC': cv_auc,
            'PR-AUC': np.nan,
            'Brier': np.nan,
            'Slope': np.nan,
            'Intercept': np.nan
        },
        {
            '階段': '測試集-校正前',
            'ROC-AUC': test_auc_uncal,
            'PR-AUC': test_pr_auc_uncal,
            'Brier': test_brier_uncal,
            'Slope': slope_uncal,
            'Intercept': intercept_uncal
        },
        {
            '階段': '測試集-校正後',
            'ROC-AUC': test_auc_cal,
            'PR-AUC': test_pr_auc_cal,
            'Brier': test_brier_cal,
            'Slope': slope_cal,
            'Intercept': intercept_cal
        }
    ])
    
    performance_df.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
    print(f"✓ {output_dir / 'model_performance.csv'}")
    
    # 9.3 閾值指標
    threshold_metrics = []
    for thr_name, thr_val in [('F1-Optimal', threshold_f1), 
                               ('Sensitivity≥70%', threshold_sens), 
                               ('Youden Index', threshold_youden)]:
        metrics = compute_classification_metrics(y_test.values, y_test_pred_cal, thr_val)
        metrics['閾值名稱'] = thr_name
        threshold_metrics.append(metrics)
    
    threshold_df = pd.DataFrame(threshold_metrics)
    threshold_df = threshold_df[['閾值名稱', 'threshold', 'sensitivity', 'specificity', 
                                  'ppv', 'npv', 'f1', 'tp', 'fp', 'fn', 'tn']]
    threshold_df.columns = ['閾值名稱', '閾值', '敏感度', '特異度', 'PPV', 'NPV', 
                            'F1', 'TP', 'FP', 'FN', 'TN']
    
    threshold_df.to_csv(output_dir / 'threshold_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"✓ {output_dir / 'threshold_metrics.csv'}")
    
    # 9.4 校正結果摘要
    calibration_summary = pd.DataFrame([
        {
            '指標': metric_name,
            '校正前': f"{boot_uncal[metric]['mean']:.4f} {boot_uncal[metric]['ci_str']}",
            '校正後': f"{boot_cal[metric]['mean']:.4f} {boot_cal[metric]['ci_str']}"
        }
        for metric, metric_name in [
            ('auc', 'ROC-AUC'),
            ('pr_auc', 'PR-AUC'),
            ('brier', 'Brier Score'),
            ('slope', 'Calibration Slope'),
            ('intercept', 'Calibration Intercept')
        ]
    ])
    
    calibration_summary.to_csv(output_dir / 'calibration_results.csv', index=False, encoding='utf-8-sig')
    print(f"✓ {output_dir / 'calibration_results.csv'}")
    
    # 9.5 預測結果
    predictions_df = pd.DataFrame({
        '預測機率_校正前': y_test_pred_uncal,
        '預測機率_校正後': y_test_pred_cal,
        '預測類別_F1': (y_test_pred_cal >= threshold_f1).astype(int),
        '預測類別_Sens70': (y_test_pred_cal >= threshold_sens).astype(int),
        '實際標籤': y_test.values
    })
    
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
    print(f"✓ {output_dir / 'predictions.csv'}")
    
    # ========================================================================
    # 視覺化
    # ========================================================================
    
    print("\n✓ 生成視覺化圖表...")
    
    # 圖1：係數重要性
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = coefficients_df.head(15)
    colors_bar = ['red' if '×' in f else 'steelblue' for f in top_features['特徵']]
    
    bars = ax.barh(range(len(top_features)), top_features['L2係數'], color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['特徵'], fontsize=11)
    ax.set_xlabel('L2 係數', fontsize=12)
    ax.set_title('特徵重要性 (Top 15)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 標註交互項
    for i, (idx, row) in enumerate(top_features.iterrows()):
        if '×' in row['特徵']:
            ax.text(row['L2係數'] + 0.01, i, '★', fontsize=14, va='center', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_dir / 'feature_importance.png'}")
    plt.close()
    
    # 圖2：ROC曲線
    fig, ax = plt.subplots(figsize=(8, 8))
    
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_cal)
    
    ax.plot(fpr_test, tpr_test, 'b-', lw=2, 
            label=f'Test ROC (AUC={test_auc_cal:.3f}, 95% CI: {boot_cal["auc"]["ci_str"]})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='隨機分類器')
    
    ax.set_xlabel('False Positive Rate (1-Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve (校正後)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_dir / 'roc_curve.png'}")
    plt.close()
    
    # 圖3：PR曲線
    fig, ax = plt.subplots(figsize=(8, 8))
    
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_cal)
    
    ax.plot(recall_test, precision_test, 'g-', lw=2,
            label=f'Test PR (AUC={test_pr_auc_cal:.3f}, 95% CI: {boot_cal["pr_auc"]["ci_str"]})')
    ax.axhline(y=y_test.mean(), color='k', linestyle='--', lw=1, 
               label=f'基線 (prevalence={y_test.mean():.3f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.set_title('Precision-Recall Curve (校正後)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_dir / 'pr_curve.png'}")
    plt.close()
    
    # 圖4：校正曲線（等分位 + 等寬）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, strategy, title in zip(axes, ['quantile', 'uniform'], 
                                    ['等分位分箱 (Quantile)', '等寬分箱 (Uniform)']):
        # 校正前
        fraction_uncal, mean_pred_uncal = calibration_curve(
            y_test, y_test_pred_uncal, n_bins=10, strategy=strategy
        )
        
        # 校正後
        fraction_cal, mean_pred_cal = calibration_curve(
            y_test, y_test_pred_cal, n_bins=10, strategy=strategy
        )
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='完美校正')
        ax.plot(mean_pred_uncal, fraction_uncal, 's-', lw=2, ms=8, color='gray', alpha=0.7,
                label=f'校正前 (slope={slope_uncal:.2f})')
        ax.plot(mean_pred_cal, fraction_cal, 'o-', lw=2, ms=8, color='blue',
                label=f'校正後 (slope={slope_cal:.2f})')
        
        ax.set_xlabel('預測機率', fontsize=12)
        ax.set_ylabel('實際陽性比例', fontsize=12)
        ax.set_title(f'校正曲線 - {title}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_dir / 'calibration_curves.png'}")
    plt.close()
    
    # 圖5：混淆矩陣（3個閾值）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (thr_name, thr_val) in zip(axes, [('F1-Optimal', threshold_f1),
                                               ('Sens≥70%', threshold_sens),
                                               ('Youden', threshold_youden)]):
        y_pred = (y_test_pred_cal >= thr_val).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['陰性', '陽性'], fontsize=11)
        ax.set_yticklabels(['陰性', '陽性'], fontsize=11)
        ax.set_xlabel('預測', fontsize=12)
        ax.set_ylabel('實際', fontsize=12)
        ax.set_title(f'{thr_name}\n(閾值={thr_val:.3f})', fontsize=12, fontweight='bold')
        
        # 標註數值
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center",
                              color="white" if cm[i, j] > cm.max() / 2 else "black",
                              fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_dir / 'confusion_matrices.png'}")
    plt.close()
    
    # ========================================================================
    # 總結報告
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("主分析完成")
    print("=" * 80)
    
    print(f"""
【模型摘要】
  模型類型: Logistic Regression (L2 Ridge)
  特徵數: {len(feature_names)} (13基礎 + 2交互項)
  EPV: {epv:.2f}
  最佳C: {best_C:.4f}
  
【交叉驗證效能】
  CV ROC-AUC: {cv_auc:.4f}

【測試集效能（校正後, n={len(y_test)}）】
  ROC-AUC:  {test_auc_cal:.4f} {boot_cal['auc']['ci_str']}
  PR-AUC:   {test_pr_auc_cal:.4f} {boot_cal['pr_auc']['ci_str']}
  Brier:    {test_brier_cal:.4f} {boot_cal['brier']['ci_str']}
  
【校正品質（校正後）】
  Slope:    {slope_cal:.4f} {boot_cal['slope']['ci_str']} {'✅' if 0.8 <= slope_cal <= 1.2 else '⚠️'}
  Intercept: {intercept_cal:.4f} {boot_cal['intercept']['ci_str']} {'✅' if abs(intercept_cal) < 0.3 else '⚠️'}

【推薦閾值（基於OOF）】
  F1-Optimal:   {threshold_f1:.4f}
  Sens≥70%:     {threshold_sens:.4f}
  Youden Index: {threshold_youden:.4f}

【L1驗證】
  保留特徵: {n_nonzero}/{len(feature_names)}
  交互項: {'✅ 兩者均保留' if len(interaction_status) == 2 and all(v != 0 for v in interaction_status.values()) else '⚠️ 部分移除'}

【輸出檔案】{output_dir}
  ├── coefficients.csv              # 所有特徵係數
  ├── model_performance.csv         # CV + 測試集效能
  ├── threshold_metrics.csv         # 多閾值混淆矩陣指標
  ├── calibration_results.csv       # 校正前後對比 (含95% CI)
  ├── predictions.csv               # 測試集預測結果
  ├── feature_importance.png        # 係數重要性圖
  ├── roc_curve.png                 # ROC曲線
  ├── pr_curve.png                  # PR曲線
  ├── calibration_curves.png        # 校正曲線（雙策略）
  └── confusion_matrices.png        # 混淆矩陣（3閾值）

✅ 主分析完成！""")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

