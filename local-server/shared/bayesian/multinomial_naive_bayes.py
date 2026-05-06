# MultinomialNaiveBayes 类定义
# 从文档自动提取生成

import numpy as np

class MultinomialNaiveBayes:
    """
    多项式朴素贝叶斯实现
    适用于离散特征（如文本词频）
    """
    
    def __init__(self, alpha=1.0):
        """
        Parameters:
        alpha : float, 拉普拉斯平滑参数
        """
        self.alpha = alpha  # 拉普拉斯平滑
        self.class_prior_ = None  # P(y)
        self.feature_prob_ = None  # P(x|y)
        self.classes_ = None
    
    def fit(self, X, y):
        """
        训练模型
        
        Parameters:
        X : ndarray, shape (n_samples, n_features)
            特征矩阵（词频/计数）
        y : ndarray, shape (n_samples,)
            类别标签
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # 计算先验概率 P(y)
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        self.class_prior_ = class_counts / n_samples
        
        # 计算条件概率 P(x|y)
        # 对于每个类别，计算每个特征在该类别文档中的总计数
        self.feature_prob_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            # 获取类别c的所有样本
            X_c = X[y == c]
            # 该类别每个特征的总计数 + 平滑
            feature_counts = X_c.sum(axis=0) + self.alpha
            # 归一化得到条件概率
            total_count = feature_counts.sum()
            self.feature_prob_[i] = feature_counts / total_count
        
        return self
    
    def predict_log_proba(self, X):
        """
        计算对数概率
        """
        # log P(y) + sum(log P(x|y))
        log_prior = np.log(self.class_prior_)
        log_likelihood = X @ np.log(self.feature_prob_.T)  # (n_samples, n_classes)
        return log_prior + log_likelihood
    
    def predict(self, X):
        """
        预测类别
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
