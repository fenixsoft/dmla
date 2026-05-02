# SimpleSVM 类定义
# 从文档自动提取生成

import numpy as np

class SimpleSVM:
    """
    简化版软间隔SVM实现
    
    使用梯度上升优化对偶问题，支持软间隔（通过参数C控制）
    
    核心步骤：
    1. 预计算核矩阵 K = X @ X.T（线性核）
    2. 迭代更新拉格朗日乘子 alpha
    3. 根据alpha找出支持向量
    4. 计算超平面参数 w 和 b
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, C=1.0):
        self.lr = learning_rate       # 梯度上升的学习率
        self.n_iterations = n_iterations  # 迭代次数
        self.C = C                    # 软间隔惩罚系数
        self.alpha = None             # 拉格朗日乘子（训练后获得）
        self.w = None                 # 超平面法向量
        self.b = None                 # 超平面截距
        self.support_vectors_ = None  # 支持向量集合
    
    def fit(self, X, y):
        """
        训练SVM模型
        
        对偶问题的目标函数：
        max sum(alpha_i) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * x_i^T x_j)
        约束：0 <= alpha_i <= C, sum(alpha_i * y_i) = 0
        
        使用梯度上升迭代优化，每次更新一个alpha_i
        """
        n_samples, n_features = X.shape
        
        # 初始化拉格朗日乘子（全零）
        self.alpha = np.zeros(n_samples)
        
        # 预计算核矩阵（线性核：样本内积）
        # K[i,j] = x_i^T x_j，用于加速目标函数计算
        K = X @ X.T
        
        # 梯度上升优化对偶问题
        for iteration in range(self.n_iterations):
            for i in range(n_samples):
                # 计算alpha_i的梯度
                # 目标函数对alpha_i的偏导：1 - y_i * sum_j(alpha_j * y_j * K[j,i])
                gradient = 1 - y[i] * np.sum(self.alpha * y * K[:, i])
                
                # 梯度上升更新
                self.alpha[i] += self.lr * gradient
                
                # 投影到约束区间 [0, C]
                # 对应软间隔的约束：0 <= alpha_i <= C
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
            
            # 约束修正：确保 sum(alpha * y) = 0
            # 通过减去均值偏差来近似满足线性约束
            bias = np.mean(self.alpha * y)
            self.alpha = self.alpha - bias * y
            self.alpha = np.clip(self.alpha, 0, self.C)
        
        # 找出支持向量（alpha > 阈值的样本）
        sv_threshold = 1e-5
        sv_indices = self.alpha > sv_threshold
        self.support_vectors_ = X[sv_indices]
        sv_labels = y[sv_indices]
        sv_alpha = self.alpha[sv_indices]
        
        # 计算超平面参数 w = sum(alpha_i * y_i * x_i)
        # 只有支持向量参与计算（其他样本alpha=0）
        self.w = np.zeros(n_features)
        for i, (sv, label, a) in enumerate(zip(self.support_vectors_, sv_labels, sv_alpha)):
            self.w += a * label * sv
        
        # 计算截距 b
        # 使用支持向量计算：对于支持向量，y_i(w^T x_i + b) = 1（硬间隔）
        # 或 y_i(w^T x_i + b) = 1 - xi_i（软间隔）
        # 这里取所有支持向量的平均值
        if len(self.support_vectors_) > 0:
            self.b = np.mean(sv_labels - self.support_vectors_ @ self.w)
        else:
            self.b = 0
        
        return self
    
    def decision_function(self, X):
        """
        决策函数值：w^T x + b
        
        正值表示预测为正类，负值表示预测为负类
        绝对值大小反映样本到超平面的距离
        """
        return X @ self.w + self.b
    
    def predict(self, X):
        """
        预测类别标签
        
        sign(w^T x + b): +1 表示正类，-1 表示负类
        """
        return np.sign(self.decision_function(X)).astype(int)
    
    def score(self, X, y):
        """计算分类准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
