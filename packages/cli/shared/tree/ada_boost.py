# AdaBoost 类定义
# 从文档自动提取生成

import numpy as np

class AdaBoost:
    """
    AdaBoost 分类器
    
    核心思想：序列训练多个弱学习器，加权组合成强学习器
    """
    
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []   # 存储所有弱学习器
        self.alphas = []   # 存储所有学习器权重
    
    def fit(self, X, y):
        """
        训练 AdaBoost
        
        核心步骤（对应理论中的迭代流程）：
        1. 初始化样本权重
        2. 每轮迭代：训练弱学习器 → 计算错误率 → 计算学习器权重 → 更新样本权重
        3. 保存所有弱学习器及其权重
        """
        n_samples = X.shape[0]
        
        # 初始化权重：所有样本权重相等（对应理论中的 w_i^(1) = 1/n）
        weights = np.ones(n_samples) / n_samples
        
        self.stumps = []
        self.alphas = []
        
        for t in range(self.n_estimators):
            # 步骤一：训练弱学习器（决策桩）
            stump = DecisionStump()
            stump.fit(X, y, weights)
            
            # 步骤二：计算加权错误率 ε_t
            predictions = stump.predict(X)
            error = np.sum(weights[predictions != y])
            
            # 防止极端情况（错误率为 0 或 1）
            error = max(error, 1e-10)
            error = min(error, 1 - 1e-10)
            
            # 步骤三：计算学习器权重 α_t（对应理论中的公式）
            alpha = 0.5 * np.log((1 - error) / error)
            
            # 步骤四：更新样本权重（对应理论中的权重更新公式）
            # 预测正确的样本权重减小，预测错误的样本权重增大
            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / np.sum(weights)  # 归一化
            
            self.stumps.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        """
        加权投票预测
        
        对应理论中的 H(x) = sign(Σ α_t * h_t(x))
        """
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for stump, alpha in zip(self.stumps, self.alphas):
            scores += alpha * stump.predict(X)
        
        return np.sign(scores).astype(int)
    
    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)
