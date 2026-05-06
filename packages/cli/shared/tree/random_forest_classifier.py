# RandomForestClassifier 类定义
# 从文档自动提取生成

import numpy as np

class RandomForestClassifier:
    """
    随机森林分类器
    
    实现：
    1. Bootstrap采样（对应理论：样本随机）
    2. 多棵决策树训练（每棵树使用不同的Bootstrap样本和特征子集）
    3. 多数投票预测（对应理论：投票机制）
    
    参数:
        n_estimators : int, 默认值 100
            树的数量（对应理论中的B）
        max_depth : int, 默认值 10
            每棵树的最大深度
        max_features : str or int, 默认值 'sqrt'
            每次分裂时考虑的特征数量（对应理论中的m）
    """
    
    def __init__(self, n_estimators=100, max_depth=10, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
    
    def _bootstrap_sample(self, X, y):
        """
        Bootstrap采样（对应理论：有放回重采样）
        
        从原始数据集中有放回地抽取n个样本
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """
        训练随机森林
        
        核心步骤：
        1. 确定特征子集大小m
        2. 对每棵树：Bootstrap采样 → 训练决策树
        """
        n_features = X.shape[1]
        
        # 确定特征子集大小m（对应理论：分类用sqrt(d)，回归用d/3）
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap采样
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # 训练决策树（带特征随机）
            tree = DecisionTreeForRF(
                max_depth=self.max_depth,
                max_features=max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """
        多数投票预测（对应理论：硬投票）
        
        每棵树预测一个类别，选择得票最多的类别
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        result = []
        for i in range(X.shape[0]):
            values, counts = np.unique(predictions[:, i], return_counts=True)
            result.append(values[np.argmax(counts)])
        return np.array(result)
    
    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)
