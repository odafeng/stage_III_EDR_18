"""
模型評估模組
Model evaluation for survival prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns


class SurvivalModelEvaluator:
    """存活預測模型評估類別"""
    
    def __init__(self, config=None):
        """
        初始化評估器
        
        Parameters:
        -----------
        config : dict
            配置字典
        """
        self.config = config
        self.results = {}
        
    def calculate_c_index(self, y_true_time, y_true_event, y_pred_risk):
        """
        計算一致性指數 (C-index)
        
        Parameters:
        -----------
        y_true_time : array-like
            真實存活時間
        y_true_event : array-like
            真實事件狀態
        y_pred_risk : array-like
            預測風險分數
            
        Returns:
        --------
        float : C-index 值
        """
        c_index = concordance_index(y_true_time, -y_pred_risk, y_true_event)
        print(f"C-index: {c_index:.4f}")
        return c_index
    
    def calculate_brier_score(self, y_true, y_pred):
        """
        計算 Brier 分數
        
        Parameters:
        -----------
        y_true : array-like
            真實標籤
        y_pred : array-like
            預測機率
            
        Returns:
        --------
        float : Brier score
        """
        brier = brier_score_loss(y_true, y_pred)
        print(f"Brier Score: {brier:.4f}")
        return brier
    
    def calculate_time_dependent_auc(self, model, X_test, y_test, time_points):
        """
        計算時間相依的 AUC
        
        Parameters:
        -----------
        model : survival model
            訓練好的存活模型
        X_test : array-like
            測試特徵
        y_test : structured array
            測試標籤
        time_points : list
            時間點列表
            
        Returns:
        --------
        dict : 各時間點的 AUC
        """
        try:
            from sksurv.metrics import cumulative_dynamic_auc
            
            # 使用模型預測風險分數
            risk_scores = model.predict(X_test)
            
            # 計算時間相依 AUC
            auc_scores, mean_auc = cumulative_dynamic_auc(
                y_test, y_test, risk_scores, time_points
            )
            
            results = {f"AUC_at_{t}": score for t, score in zip(time_points, auc_scores)}
            results['mean_AUC'] = mean_auc
            
            print("時間相依 AUC:")
            for key, value in results.items():
                print(f"  {key}: {value:.4f}")
            
            return results
        
        except ImportError:
            print("警告: 無法計算時間相依 AUC (需要 scikit-survival 套件)")
            return {}
    
    def plot_kaplan_meier_curves(self, durations, events, groups=None, 
                                   title="Kaplan-Meier Survival Curves",
                                   save_path=None):
        """
        繪製 Kaplan-Meier 生存曲線
        
        Parameters:
        -----------
        durations : array-like
            存活時間
        events : array-like
            事件狀態
        groups : array-like, optional
            分組標籤
        title : str
            圖表標題
        save_path : str, optional
            儲存路徑
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if groups is None:
            # 單一曲線
            kmf = KaplanMeierFitter()
            kmf.fit(durations, events, label='Overall')
            kmf.plot_survival_function(ax=ax)
        else:
            # 多組曲線
            for group in np.unique(groups):
                mask = groups == group
                kmf = KaplanMeierFitter()
                kmf.fit(durations[mask], events[mask], label=f'Group {group}')
                kmf.plot_survival_function(ax=ax)
        
        ax.set_xlabel('時間 (月)')
        ax.set_ylabel('存活機率')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
        
        plt.show()
    
    def plot_risk_groups(self, risk_scores, durations, events, n_groups=3,
                          save_path=None):
        """
        根據風險分數分組並繪製生存曲線
        
        Parameters:
        -----------
        risk_scores : array-like
            風險分數
        durations : array-like
            存活時間
        events : array-like
            事件狀態
        n_groups : int
            分組數量
        save_path : str, optional
            儲存路徑
        """
        # 根據風險分數分組
        risk_groups = pd.qcut(risk_scores, q=n_groups, labels=['Low', 'Medium', 'High'])
        
        self.plot_kaplan_meier_curves(
            durations, 
            events, 
            groups=risk_groups,
            title=f"Kaplan-Meier Curves by Risk Groups (n={n_groups})",
            save_path=save_path
        )
    
    def plot_feature_importance(self, feature_names, importance_scores, 
                                 top_n=20, save_path=None):
        """
        繪製特徵重要性圖
        
        Parameters:
        -----------
        feature_names : list
            特徵名稱
        importance_scores : array-like
            重要性分數
        top_n : int
            顯示前幾個特徵
        save_path : str, optional
            儲存路徑
        """
        # 排序並選擇前 N 個特徵
        indices = np.argsort(importance_scores)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        # 繪圖
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x=top_scores, y=top_features, ax=ax)
        ax.set_xlabel('重要性分數')
        ax.set_ylabel('特徵')
        ax.set_title(f'Top {top_n} 特徵重要性')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, model_name, metrics, save_path=None):
        """
        生成評估報告
        
        Parameters:
        -----------
        model_name : str
            模型名稱
        metrics : dict
            評估指標字典
        save_path : str, optional
            儲存路徑
        """
        report = f"\n{'='*60}\n"
        report += f"模型評估報告: {model_name}\n"
        report += f"{'='*60}\n\n"
        
        for metric_name, value in metrics.items():
            report += f"{metric_name}: {value:.4f}\n"
        
        report += f"\n{'='*60}\n"
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"報告已儲存至: {save_path}")
        
        self.results[model_name] = metrics
        
        return report


if __name__ == "__main__":
    # 測試模型評估功能
    evaluator = SurvivalModelEvaluator()
    print("模型評估模組已載入")
