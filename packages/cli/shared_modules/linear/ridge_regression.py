# RidgeRegression 类定义
# 从文档自动提取生成

import numpy as np

class RidgeRegression:
    """
    岭回归实现（L2正则化）
    
    适用于：
    1. 特征之间存在共线性
    2. 参数估计不稳定
    3. 需要防止过拟合
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 正则化强度λ
        self.coef_ = None   # 特征系数
        self.intercept_ = None  # 截距
    
    def fit(self, X, y):
        """
        训练模型（闭式解）
        
        Parameters:
        X : ndarray, shape (n_samples, n_features)
            特征矩阵
        y : ndarray, shape (n_samples,)
            目标向量
        """
        n_samples = X.shape[0]
        X_augmented = np.column_stack([np.ones(n_samples), X])
        
        # 岭回归闭式解：β = (X^T X + λI)^(-1) X^T y
        # 注意：不对截距项正则化（I的第一行第一列为0）
        I = np.eye(X_augmented.shape[1])
        I[0, 0] = 0  # 截距项不参与正则化
        
        XtX = X_augmented.T @ X_augmented
        Xty = X_augmented.T @ y
        
        self.beta_ = np.linalg.solve(XtX + self.alpha * I, Xty)
        
        self.intercept_ = self.beta_[0]
        self.coef_ = self.beta_[1:]
        
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
