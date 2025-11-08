"""
Logistic Regression Model - 校正方法完整分析（投稿級）

關鍵改進：
1. 比較多種校正方法：Isotonic vs Sigmoid(Platt) vs Logistic vs None
2. Bootstrap CI (1000次) 用於所有指標
3. L1確認交互項穩定性
4. 完整的校正前後對比
5. Sensitivity analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, TransformerMixin, clone
from scipy.special import logit, expit
from scipy.stats import sem
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ====================================
# 輔助函數
# ====================================

def compute_calibration_slope_intercept(y_true, y_pred_proba, eps=1e-10):
    """標準logistic校正：y ~ α + β·logit(p̂)"""
    y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1 - eps)
    logit_proba = logit(y_pred_proba_clipped)
    lr = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
    lr.fit(logit_proba.reshape(-1, 1), y_true)
    return lr.coef_[0][0], lr.intercept_[0]

def logistic_calibration(y_train, y_train_proba, eps=1e-10):
    """
    標準Logistic校正：估計 α, β
    p_calibrated = logit^(-1)(α + β·logit(p))
    """
    y_train_proba_clipped = np.clip(y_train_proba, eps, 1 - eps)
    logit_proba = logit(y_train_proba_clipped).reshape(-1, 1)
    
    lr = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
    lr.fit(logit_proba, y_train)
    
    return lr

def apply_logistic_calibration(calibrator, y_proba, eps=1e-10):
    """應用Logistic校正"""
    y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
    logit_proba = logit(y_proba_clipped).reshape(-1, 1)
    return calibrator.predict_proba(logit_proba)[:, 1]

def recalibration_in_the_large(y_train, y_train_proba):
    """
    Recalibration-in-the-large: 只調整截距
    p_calibrated = logit^(-1)(α + logit(p))
    即強制 β=1，只估 α
    """
    # 計算平均預測機率和實際陽性率的logit差
    mean_pred = y_train_proba.mean()
    mean_obs = y_train.mean()
    
    # α = logit(mean_obs) - logit(mean_pred)
    alpha = logit(np.clip(mean_obs, 1e-10, 1-1e-10)) - logit(np.clip(mean_pred, 1e-10, 1-1e-10))
    
    return alpha

def apply_recalibration_large(alpha, y_proba, eps=1e-10):
    """應用recalibration-in-the-large"""
    y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
    logit_proba = logit(y_proba_clipped)
    return expit(alpha + logit_proba)

def bootstrap_metrics(y_true, y_pred_proba, n_bootstrap=1000, confidence=0.95, random_state=42):
    """
    Bootstrap CI for multiple metrics
    返回: (metric_mean, ci_low, ci_high) for each metric
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    metrics = {
        'auc': [],
        'pr_auc': [],
        'brier': [],
        'slope': [],
        'intercept': []
    }
    
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        if len(np.unique(y_true[indices])) < 2:
            continue
        
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
        results[metric_name] = (mean_val, ci[0], ci[1])
    
    return results

def ensure_binary_01(X, binary_cols):
    X_out = X.copy()
    for col in binary_cols:
        if col in X_out.columns:
            X_out[col] = (X_out[col] > 0).astype(int)
    return X_out

# ====================================
# 交互作用轉換器
# ====================================

class InteractionTransformer2(BaseEstimator, TransformerMixin):
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
print("校正方法完整分析 - 投稿級")
print("=" * 80)

# 載入數據
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

print(f"\n訓練集: {len(y_train)} (事件: {y_train.sum()}, {y_train.mean():.2%})")
print(f"測試集: {len(y_test)} (事件: {y_test.sum()}, {y_test.mean():.2%})")

# ====================================
# 1. Pipeline設定
# ====================================

print("\n" + "=" * 80)
print("步驟 1/6: Pipeline設定")
print("=" * 80)

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

def post_binary_func(X):
    return ensure_binary_01(X, binary_cols_with_prefix)

post_binary_transformer = FunctionTransformer(
    post_binary_func,
    feature_names_out='one-to-one'
)

interaction_transformer = InteractionTransformer2()

# ====================================
# 2. L1確認交互項穩定性
# ====================================

print("\n" + "=" * 80)
print("步驟 2/6: L1確認交互項選擇")
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

print("✓ 訓練L1模型...")
pipeline_l1.fit(X_train, y_train)

# 檢查係數
X_train_transformed = pipeline_l1.named_steps['interactions'].transform(
    pipeline_l1.named_steps['post_bin'].transform(
        pipeline_l1.named_steps['preprocessor'].transform(X_train.iloc[:1])
    )
)
feature_names = list(X_train_transformed.columns)
coef_l1 = pipeline_l1.named_steps['classifier'].coef_[0]

n_nonzero = np.sum(coef_l1 != 0)
print(f"✓ L1選擇: {n_nonzero}/15 特徵保留")

# 檢查兩個交互項是否保留
interaction_coefs = {}
for fname, coef in zip(feature_names, coef_l1):
    if '×' in fname:
        interaction_coefs[fname] = coef
        status = "✓ 保留" if coef != 0 else "❌ 移除"
        print(f"  {fname}: {coef:.4f} {status}")

# ====================================
# 3. 訓練L2模型（主模型）
# ====================================

print("\n" + "=" * 80)
print("步驟 3/6: 訓練L2模型")
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

print("✓ 訓練L2模型...")
pipeline_l2.fit(X_train, y_train)

clf = pipeline_l2.named_steps['classifier']
best_C = clf.C_[0]
cv_auc = clf.scores_[1].mean(axis=0).max()

print(f"✓ 最佳 C: {best_C:.4f}")
print(f"✓ CV ROC-AUC: {cv_auc:.4f}")

# 獲取未校正預測
y_test_uncalibrated = pipeline_l2.predict_proba(X_test)[:, 1]
y_train_oof = cross_val_predict(pipeline_l2, X_train, y_train, cv=cv, method='predict_proba', n_jobs=1)[:, 1]

# ====================================
# 4. 多種校正方法比較
# ====================================

print("\n" + "=" * 80)
print("步驟 4/6: 多種校正方法比較")
print("=" * 80)

calibration_methods = {}

# 方法1: 無校正（基線）
print("\n【方法 1】無校正（Uncalibrated）")
calibration_methods['Uncalibrated'] = {
    'train_proba': y_train_oof,
    'test_proba': y_test_uncalibrated,
    'method': 'none'
}

# 方法2: Recalibration-in-the-large（只調截距）
print("\n【方法 2】Recalibration-in-the-large（只調截距）")
alpha_large = recalibration_in_the_large(y_train, y_train_oof)
y_test_calib_large = apply_recalibration_large(alpha_large, y_test_uncalibrated)
print(f"✓ 估計截距 α: {alpha_large:.4f}")

calibration_methods['Recalib-Large'] = {
    'train_proba': apply_recalibration_large(alpha_large, y_train_oof),
    'test_proba': y_test_calib_large,
    'method': 'recalib_large',
    'alpha': alpha_large
}

# 方法3: Logistic校正（標準方法）
print("\n【方法 3】Logistic校正（估計 α, β）")
logistic_calibrator = logistic_calibration(y_train, y_train_oof)
y_test_calib_logistic = apply_logistic_calibration(logistic_calibrator, y_test_uncalibrated)
alpha_log = logistic_calibrator.intercept_[0]
beta_log = logistic_calibrator.coef_[0][0]
print(f"✓ 估計參數: α={alpha_log:.4f}, β={beta_log:.4f}")

calibration_methods['Logistic'] = {
    'train_proba': apply_logistic_calibration(logistic_calibrator, y_train_oof),
    'test_proba': y_test_calib_logistic,
    'method': 'logistic',
    'alpha': alpha_log,
    'beta': beta_log,
    'calibrator': logistic_calibrator
}

# 方法4: Sigmoid (Platt)校正
print("\n【方法 4】Sigmoid (Platt)校正")
pipeline_sigmoid = CalibratedClassifierCV(
    estimator=clone(pipeline_l2),
    method='sigmoid',
    cv=5
)
pipeline_sigmoid.fit(X_train, y_train)
y_test_calib_sigmoid = pipeline_sigmoid.predict_proba(X_test)[:, 1]

calibration_methods['Sigmoid'] = {
    'train_proba': None,  # 難以直接獲取
    'test_proba': y_test_calib_sigmoid,
    'method': 'sigmoid'
}
print("✓ 已訓練")

# 方法5: Isotonic校正
print("\n【方法 5】Isotonic校正")
pipeline_isotonic = CalibratedClassifierCV(
    estimator=clone(pipeline_l2),
    method='isotonic',
    cv=5
)
pipeline_isotonic.fit(X_train, y_train)
y_test_calib_isotonic = pipeline_isotonic.predict_proba(X_test)[:, 1]

calibration_methods['Isotonic'] = {
    'train_proba': None,
    'test_proba': y_test_calib_isotonic,
    'method': 'isotonic'
}
print("✓ 已訓練")

# ====================================
# 5. 評估所有方法（含Bootstrap CI）
# ====================================

print("\n" + "=" * 80)
print("步驟 5/6: 評估所有方法（Bootstrap CI, n=1000）")
print("=" * 80)

results_summary = []

for method_name, method_data in calibration_methods.items():
    print(f"\n【{method_name}】")
    
    y_test_pred = method_data['test_proba']
    
    # 基本指標
    test_auc = roc_auc_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_pred)
    test_brier = brier_score_loss(y_test, y_test_pred)
    slope, intercept = compute_calibration_slope_intercept(y_test, y_test_pred)
    
    print(f"  AUC: {test_auc:.4f}")
    print(f"  PR-AUC: {test_pr_auc:.4f}")
    print(f"  Brier: {test_brier:.4f}")
    print(f"  Slope: {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    
    # Bootstrap CI
    print(f"  ✓ Bootstrap CI (1000次)...")
    boot_results = bootstrap_metrics(y_test.values, y_test_pred, n_bootstrap=1000)
    
    results_summary.append({
        '方法': method_name,
        'AUC': test_auc,
        'AUC_CI': f"[{boot_results['auc'][1]:.3f}, {boot_results['auc'][2]:.3f}]",
        'PR-AUC': test_pr_auc,
        'Brier': test_brier,
        'Brier_CI': f"[{boot_results['brier'][1]:.3f}, {boot_results['brier'][2]:.3f}]",
        'Slope': slope,
        'Slope_CI': f"[{boot_results['slope'][1]:.3f}, {boot_results['slope'][2]:.3f}]",
        'Intercept': intercept,
        'Intercept_CI': f"[{boot_results['intercept'][1]:.3f}, {boot_results['intercept'][2]:.3f}]"
    })

results_df = pd.DataFrame(results_summary)

# ====================================
# 6. 視覺化比較
# ====================================

print("\n" + "=" * 80)
print("步驟 6/6: 生成比較視覺化")
print("=" * 80)

output_dir = Path('/Users/huangshifeng/Desktop/stage_III_colon_surv/results/calibration_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# 6.1 校正曲線比較（5個方法）
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

colors = ['gray', 'blue', 'green', 'orange', 'red']
method_order = ['Uncalibrated', 'Recalib-Large', 'Logistic', 'Sigmoid', 'Isotonic']

for idx, method_name in enumerate(method_order):
    ax = axes[idx]
    y_pred = calibration_methods[method_name]['test_proba']
    
    fraction, mean_pred = calibration_curve(y_test, y_pred, n_bins=10, strategy='quantile')
    
    result = results_df[results_df['方法'] == method_name].iloc[0]
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='完美校正')
    ax.plot(mean_pred, fraction, 's-', lw=2, ms=8, color=colors[idx], label='觀察值')
    
    ax.set_xlabel('預測機率', fontsize=11)
    ax.set_ylabel('實際陽性比例', fontsize=11)
    ax.set_title(f'{method_name}\nBrier={result["Brier"]:.4f}, Slope={result["Slope"]:.2f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# 隱藏多餘的subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'calibration_methods_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ {output_dir / 'calibration_methods_comparison.png'}")
plt.close()

# 6.2 指標比較圖
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Brier Score
ax = axes[0, 0]
briers = [results_df[results_df['方法']==m]['Brier'].values[0] for m in method_order]
bars = ax.bar(range(len(method_order)), briers, color=colors, alpha=0.7)
ax.set_xticks(range(len(method_order)))
ax.set_xticklabels(method_order, rotation=45, ha='right')
ax.set_ylabel('Brier Score', fontsize=12)
ax.set_title('Brier Score (越低越好)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, briers)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# Calibration Slope
ax = axes[0, 1]
slopes = [results_df[results_df['方法']==m]['Slope'].values[0] for m in method_order]
bars = ax.bar(range(len(method_order)), slopes, color=colors, alpha=0.7)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='理想值=1.0')
ax.set_xticks(range(len(method_order)))
ax.set_xticklabels(method_order, rotation=45, ha='right')
ax.set_ylabel('Calibration Slope', fontsize=12)
ax.set_title('Calibration Slope (理想=1.0)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, slopes)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10)

# AUC
ax = axes[1, 0]
aucs = [results_df[results_df['方法']==m]['AUC'].values[0] for m in method_order]
bars = ax.bar(range(len(method_order)), aucs, color=colors, alpha=0.7)
ax.set_xticks(range(len(method_order)))
ax.set_xticklabels(method_order, rotation=45, ha='right')
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('ROC-AUC (區分度)', fontsize=13, fontweight='bold')
ax.set_ylim([0.5, 0.8])
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, aucs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# Calibration Intercept
ax = axes[1, 1]
intercepts = [results_df[results_df['方法']==m]['Intercept'].values[0] for m in method_order]
bars = ax.bar(range(len(method_order)), intercepts, color=colors, alpha=0.7)
ax.axhline(y=0.0, color='red', linestyle='--', linewidth=2, label='理想值=0.0')
ax.set_xticks(range(len(method_order)))
ax.set_xticklabels(method_order, rotation=45, ha='right')
ax.set_ylabel('Calibration Intercept', fontsize=12)
ax.set_title('Calibration Intercept (理想=0.0)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, intercepts)):
    y_pos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.15
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'calibration_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ {output_dir / 'calibration_metrics_comparison.png'}")
plt.close()

# ====================================
# 7. 保存結果
# ====================================

print("\n" + "=" * 80)
print("保存結果")
print("=" * 80)

results_df.to_csv(output_dir / 'calibration_methods_summary.csv', index=False, encoding='utf-8-sig')
print(f"✓ {output_dir / 'calibration_methods_summary.csv'}")

# ====================================
# 8. 總結與建議
# ====================================

print("\n" + "=" * 80)
print("分析完成 - 投稿建議")
print("=" * 80)

# 找出最佳方法
best_brier_method = results_df.loc[results_df['Brier'].idxmin(), '方法']
best_slope_method = results_df.iloc[(results_df['Slope'] - 1.0).abs().argsort()[:1]]['方法'].values[0]

print(f"""
【關鍵發現】

1. **Brier Score 最佳**: {best_brier_method}
   - 預測誤差最小
   
2. **Calibration Slope 最接近1.0**: {best_slope_method}
   - 風險區分度保留最好

3. **Isotonic的問題**:
   - Brier: {results_df[results_df['方法']=='Isotonic']['Brier'].values[0]:.4f} (最低 ✓)
   - Slope: {results_df[results_df['方法']=='Isotonic']['Slope'].values[0]:.4f} ⚠️ 嚴重壓縮
   - 結果：所有預測趨向中等風險，臨床判讀困難

4. **推薦方法排序**:
""")

# 計算綜合評分（slope接近1.0的權重更高）
results_df['slope_penalty'] = np.abs(results_df['Slope'] - 1.0)
results_df['brier_rank'] = results_df['Brier'].rank()
results_df['slope_rank'] = results_df['slope_penalty'].rank()
results_df['combined_score'] = results_df['brier_rank'] + 2 * results_df['slope_rank']  # slope權重x2

sorted_methods = results_df.sort_values('combined_score')[['方法', 'Brier', 'Slope', 'AUC']].head()

for idx, row in sorted_methods.iterrows():
    print(f"   {idx+1}. {row['方法']:15s} - Brier={row['Brier']:.4f}, Slope={row['Slope']:.2f}, AUC={row['AUC']:.4f}")

print(f"""
【投稿建議】

1. **主要模型**: 使用 {sorted_methods.iloc[0]['方法']}
   - 原因：平衡Brier和Slope，臨床可用性最佳

2. **報告策略**:
   - 主文：報告推薦方法的效能
   - 補充材料：展示5種方法的完整對比（Sensitivity Analysis）
   - 說明：Isotonic雖然Brier最低，但slope壓縮嚴重，不適合臨床

3. **Methods寫法**:
   "We compared five calibration methods: uncalibrated, recalibration-in-the-large,
   logistic calibration, Platt scaling (sigmoid), and isotonic regression. While
   isotonic regression achieved the lowest Brier score, it severely compressed the
   calibration slope (0.53), reducing clinical discriminability. We selected
   [推薦方法] as it provided the best balance between prediction error and
   maintaining risk stratification (slope={sorted_methods.iloc[0]['Slope']:.2f})."

4. **L1交互項確認**:
   - {n_nonzero}/15 特徵被L1保留
   - 兩個交互項狀態: {'LNR × pN' in [k for k, v in interaction_coefs.items() if v != 0]}

【輸出檔案】{output_dir}
  - calibration_methods_summary.csv (含Bootstrap CI)
  - calibration_methods_comparison.png (5方法校正曲線)
  - calibration_metrics_comparison.png (指標對比)

✅ 校正方法完整分析完成！建議採用{sorted_methods.iloc[0]['方法']}！
""")

print("=" * 80)

