# KernelSVM 类定义
# 从文档自动提取生成

import numpy as np

class KernelSVM:
    """
    核SVM实现
    支持线性核、多项式核、RBF核
    """
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, degree=3, coef0=1):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0  # 多项式核的常数项
        
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.alpha_sv = None
    
    def _kernel(self, X1, X2):
        """计算核矩阵"""
        if self.kernel == 'linear':
            return X1 @ X2.T
        
        elif self.kernel == 'poly':
            return (X1 @ X2.T + self.coef0) ** self.degree
        
        elif self.kernel == 'rbf':
            # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2*x^T*x'
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * X1 @ X2.T
            return np.exp(-self.gamma * distances)
        
        else:
            raise ValueError(f"未知核函数: {self.kernel}")
    
    def fit(self, X, y, lr=0.01, n_iterations=500):
        """训练模型（简化版SMO思想）"""
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        
        # 计算核矩阵
        K = self._kernel(X, X)
        
        # 初始化
        self.alpha = np.zeros(n_samples)
        
        # 梯度上升优化
        for _ in range(n_iterations):
            for i in range(n_samples):
                # 梯度
                gradient = 1 - y[i] * np.sum(self.alpha * y * K[:, i])
                self.alpha[i] += lr * gradient
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
            
            # 约束修正
            self.alpha = self.alpha - np.mean(self.alpha * y) * y
            self.alpha = np.clip(self.alpha, 0, self.C)
        
        # 支持向量
        sv_mask = self.alpha > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_vector_labels_ = y[sv_mask]
        self.alpha_sv = self.alpha[sv_mask]
        
        # 计算b
        if len(self.support_vectors_) > 0:
            K_sv = self._kernel(self.support_vectors_, self.support_vectors_)
            margins = np.sum(self.alpha_sv * self.support_vector_labels_ * K_sv, axis=1)
            self.b = np.mean(self.support_vector_labels_ - margins)
        else:
            self.b = 0
        
        return self
    
    def decision_function(self, X):
        """决策函数"""
        K = self._kernel(X, self.support_vectors_)
        return K @ (self.alpha_sv * self.support_vector_labels_) + self.b
    
    def predict(self, X):
        """预测类别"""
        return np.sign(self.decision_function(X)).astype(int)
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
