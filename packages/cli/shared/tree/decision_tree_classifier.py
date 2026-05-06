# DecisionTreeClassifier 类定义
# 从文档自动提取生成

import numpy as np

class DecisionTreeClassifier:
    """
    CART 决策树分类器
    
    使用 Gini 指数作为分裂准则，构建二叉决策树。
    支持预剪枝策略：最大深度限制和叶节点最小样本数限制。
    
    参数:
        max_depth : int, 默认值 10
            树的最大深度，防止过拟合
        min_samples_split : int, 默认值 2
            分裂所需的最小样本数，防止学习孤例
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_gain_threshold=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_threshold = min_gain_threshold
        self.tree = None
    
    def _gini(self, y):
        """
        计算数据集的 Gini 指数
        
        Gini 指数衡量数据的不纯度，值越小越纯净。
        
        参数:
            y : ndarray
                目标变量数组
        
        返回:
            float : Gini 指数值
        """
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _gini_split(self, y_left, y_right):
        """
        计算分裂后的加权 Gini 指数
        
        加权平均两个子集的 Gini 指数，权重为样本数比例。
        
        参数:
            y_left : ndarray
                左分支的目标变量
            y_right : ndarray
                右分支的目标变量
        
        返回:
            float : 分裂后的加权 Gini 指数
        """
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * self._gini(y_left) + \
               (len(y_right) / n) * self._gini(y_right)
    
    def _best_split(self, X, y):
        """
        寻找最佳分裂特征和分割点
        
        遍历所有特征的所有候选分割点，选择 Gini 指数最小的分裂方案。
        候选分割点是特征的唯一值（CART 的标准策略）。
        
        参数:
            X : ndarray, shape (n_samples, n_features)
                特征矩阵
            y : ndarray, shape (n_samples,)
                目标变量
        
        返回:
            tuple : (最佳特征索引, 最佳分割点, 对应的 Gini 指数)
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            # 获取该特征的所有唯一值作为候选分割点
            # 使用相邻唯一值的中点作为候选阈值（标准 CART 算法策略）
            thresholds = np.unique(X[:, feature])
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            for threshold in thresholds:
                # 按阈值分裂数据
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # 忽略无效分裂（某分支为空）
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                gini = self._gini_split(y_left, y_right)
                
                # 更新最优分裂
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini
    
    def _build_tree(self, X, y, depth):
        """
        递归构建决策树
        
        核心步骤：
        1. 检查终止条件（深度限制、样本数限制、纯净度）
        2. 若满足终止条件，返回叶节点（多数类）
        3. 否则寻找最优分裂，创建内部节点
        4. 递归构建左右子树
        
        参数:
            X : ndarray
                特征矩阵
            y : ndarray
                目标变量
            depth : int
                当前深度
        
        返回:
            dict : 树节点（字典表示）
        """
        n_samples = len(y)
        
        # 检查预剪枝终止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            # 返回叶节点，预测值为多数类
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}
        
        # 寻找最优分裂
        feature, threshold, gini = self._best_split(X, y)
        
        # 若无法分裂或分裂增益不足，返回叶节点
        if feature is None or gini > self._gini(y) - self.min_gain_threshold:
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': values[np.argmax(counts)]}
        
        # 分裂数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """
        训练决策树
        
        参数:
            X : ndarray, shape (n_samples, n_features)
                特征矩阵
            y : ndarray, shape (n_samples,)
                目标变量
        
        返回:
            self : 训练后的模型实例
        """
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _predict_one(self, x, node):
        """
        预测单个样本
        
        从根节点开始，根据分裂条件选择分支，直到到达叶节点。
        
        参数:
            x : ndarray
                单个样本的特征向量
            node : dict
                当前树节点
        
        返回:
            int : 预测类别
        """
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
    
    def predict(self, X):
        """
        批量预测
        
        参数:
            X : ndarray, shape (n_samples, n_features)
                特征矩阵
        
        返回:
            ndarray : 预测类别数组
        """
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def score(self, X, y):
        """
        计算准确率
        
        参数:
            X : ndarray
                特征矩阵
            y : ndarray
                真实类别
        
        返回:
            float : 准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
