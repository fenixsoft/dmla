# LassoRegression 类定义
# 从文档自动提取生成

import numpy as np

class LassoRegression:
    """
    Lasso回归实现（L1正则化）
    使用坐标下降算法
    
    适用于：
    1. 需要自动特征选择
    2. 特征数量多，部分特征可能无关
    3. 追求稀疏、可解释的模型
    """
    
    def __init__(self, alpha=1.0, n_iterations=1000, tol=1e-4):
        self.alpha = alpha          # 正则化强度λ
        self.n_iterations = n_iterations  # 最大迭代次数
        self.tol = tol              # 收敛阈值
        self.coef_ = None
        self.intercept_ = None
    
    def soft_threshold(self, rho, lambda_):
        """
        软阈值函数（Lasso的核心操作）
        
        将参数"推向"零，可能精确到达零
        """
        if rho < -lambda_:
            return rho + lambda_
        elif rho > lambda_:
            return rho - lambda_
        else:
            return 0.0
    
    def fit(self, X, y):
        """
        训练模型（坐标下降）
        
        每次更新一个参数，轮流迭代直至收敛
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.mean(y)
        y_centered = y - self.intercept_
        
        # 数据标准化（加速收敛，保证公平惩罚）
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # 避免除零
        X_normalized = (X - X_mean) / X_std
        
        # 坐标下降迭代
        for iteration in range(self.n_iterations):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                # 计算当前特征的"部分残差"
                # 即：去掉第j个特征后的预测残差
                residual = y_centered - X_normalized @ self.coef_ + self.coef_[j] * X_normalized[:, j]
                
                # 计算rho（未正则化的梯度项）
                rho = X_normalized[:, j] @ residual / n_samples
                
                # 应用软阈值（Lasso的关键步骤）
                self.coef_[j] = self.soft_threshold(rho, self.alpha)
            
            # 检查收敛（在标准化空间中比较）
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        
        # 还原到原始尺度（迭代结束后执行一次）
        self.coef_ = self.coef_ / X_std
        
        return self
    
    def predict(self, X):
        """预测"""
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """R²得分"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
    
    def get_selected_features(self, threshold=0.01):
        """返回被选中的特征索引（非零系数）"""
        return np.where(np.abs(self.coef_) > threshold)[0]
