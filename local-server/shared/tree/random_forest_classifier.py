# RandomForestClassifier 定义
# 从文档自动提取生成

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class DecisionTreeForRF:
    """
    用于随机森林的决策树

    与普通决策树的区别：每次分裂时只考虑随机选取的特征子集

    参数:
        max_depth : int, 默认值 10
            树的最大深度
        min_samples_split : int, 默认值 2
            分裂所需的最小样本数
        max_features : int or None
            每次分裂时考虑的特征数量
    """

    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def _gini(self, y):
        """计算Gini指数"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y, feature_indices):
        """
        寻找最佳分裂（只考虑指定特征子集）

        对应理论：特征随机——每个节点只从m个随机特征中选择最优分裂
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                n = len(y)
                gini = (np.sum(left_mask) / n) * self._gini(y[left_mask]) + \
                       (np.sum(right_mask) / n) * self._gini(y[right_mask])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        n_samples, n_features = X.shape

        # 检查终止条件
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}

        # 随机选择特征子集（对应理论：特征随机）
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)

        feature, threshold = self._best_split(X, y, feature_indices)

        if feature is None:
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        """训练决策树"""
        self.tree = self._build_tree(X, y, 0)
        return self

    def _predict_one(self, x, node):
        """预测单个样本"""
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        """批量预测"""
        return np.array([self._predict_one(x, self.tree) for x in X])


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
