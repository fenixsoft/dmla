# Perceptron 类定义
# 从文档自动提取生成

import numpy as np

class Perceptron:
    """
    感知机实现
    
    使用错误驱动的权重更新规则：
    w = w + eta * y * x (当预测错误时)
    """
    def __init__(self, learning_rate=1.0, max_iterations=1000):
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.w = None  # 权重向量（包含偏置）
        self.errors_history = []  # 每轮迭代错误数
    
    def fit(self, X, y):
        """
        训练感知机
        
        Parameters:
        X : ndarray, shape (n_samples, n_features)
            输入特征矩阵
        y : ndarray, shape (n_samples,)
            标签向量，取值为 {1, -1}
        """
        n_samples, n_features = X.shape
        
        # 增广向量形式：添加常数1列（对应偏置）
        X_aug = np.column_stack([X, np.ones(n_samples)])
        
        # 初始化权重为零向量
        self.w = np.zeros(n_features + 1)
        
        # 训练循环
        for iteration in range(self.max_iter):
            errors = 0
            for i in range(n_samples):
                # 计算预测值
                prediction = np.sign(self.w @ X_aug[i])
                if prediction == 0:
                    prediction = -1  # 符号函数边界情况
                
                # 若预测错误，更新权重
                if prediction != y[i]:
                    self.w += self.lr * y[i] * X_aug[i]
                    errors += 1
            
            self.errors_history.append(errors)
            
            # 若所有样本正确分类，提前终止
            if errors == 0:
                print(f"在第 {iteration + 1} 轮迭代后收敛")
                break
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Parameters:
        X : ndarray, shape (n_samples, n_features)
        
        Returns:
        predictions : ndarray, shape (n_samples,)
            预测标签 {1, -1}
        """
        n_samples = X.shape[0]
        X_aug = np.column_stack([X, np.ones(n_samples)])
        predictions = np.sign(X_aug @ self.w)
        predictions[predictions == 0] = -1
        return predictions
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
