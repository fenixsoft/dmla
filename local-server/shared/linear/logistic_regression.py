# LogisticRegression 类定义
# 从文档自动提取生成

import numpy as np

class LogisticRegression:
    """
    手写逻辑回归实现
    使用梯度下降优化交叉熵损失
    """   
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr = learning_rate           # 学习率，控制梯度下降的步长
        self.n_iterations = n_iterations  # 迭代次数，梯度下降的最大迭代轮数
        self.coef_ = None                 # 特征系数（权重），训练后保存
        self.intercept_ = None            # 截距项，训练后保存
        self.loss_history = []            # 损失历史记录，用于可视化收敛过程，供后续可视化使用
    
    def sigmoid(self, z):
        """Sigmoid 函数"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def cross_entropy_loss(self, y, p):
        """交叉熵损失"""
        # 避免 log(0)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def fit(self, X, y):
        """
        训练模型（梯度下降）
        
        Parameters:
        X : ndarray, shape (n_samples, n_features)
            特征矩阵
        y : ndarray, shape (n_samples,)
            标签向量 (0 或 1)
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        # 梯度下降迭代
        for i in range(self.n_iterations):
            # 计算预测概率
            z = X @ self.coef_ + self.intercept_
            p = self.sigmoid(z)
            
            # 记录损失
            self.loss_history.append(self.cross_entropy_loss(y, p))
            
            # 计算梯度（交叉熵损失的简洁梯度）
            gradient_coef = (1 / n_samples) * (X.T @ (p - y))
            gradient_intercept = (1 / n_samples) * np.sum(p - y)
            
            # 更新参数
            self.coef_ -= self.lr * gradient_coef
            self.intercept_ -= self.lr * gradient_intercept
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        z = X @ self.coef_ + self.intercept_
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
