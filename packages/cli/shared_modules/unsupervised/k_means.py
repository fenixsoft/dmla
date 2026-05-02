# KMeans 类定义
# 从文档自动提取生成

import numpy as np

class KMeans:
    """
    K-means聚类算法实现
    
    参数:
        n_clusters : int, 簇的数量K
        max_iter : int, 最大迭代次数
        tol : float, 收敛阈值（中心变化小于此值时停止）
        n_init : int, 随机初始化的次数（取最优结果）
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        
        self.cluster_centers_ = None  # 簇中心
        self.labels_ = None           # 每个样本的簇分配
        self.inertia_ = None          # 目标函数值（距离平方和）
    
    def _init_centers(self, X):
        """
        随机初始化簇中心
        
        从数据中随机选择K个样本作为初始中心
        """
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _assign_clusters(self, X, centers):
        """
        分配步骤：将每个样本分配到最近的簇中心
        
        计算每个样本到所有中心的距离平方，返回最近的簇编号
        """
        distances = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            # 计算样本到第k个中心的距离平方（对应目标函数中的||x - μ||²）
            distances[:, k] = np.sum((X - centers[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)
    
    def _update_centers(self, X, labels):
        """
        更新步骤：重新计算每个簇的中心
        
        簇中心 = 簇内样本的均值（这就是"means"的含义）
        """
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                # 取簇内样本的均值作为新中心
                centers[k] = X[mask].mean(axis=0)
            else:
                # 空簇的罕见情况：随机重新初始化
                centers[k] = X[np.random.randint(len(X))]
        return centers
    
    def _compute_inertia(self, X, labels, centers):
        """
        计算目标函数值J
        
        J = 所有样本到其所属簇中心的距离平方和
        """
        inertia = 0
        for k in range(self.n_clusters):
            mask = labels == k
            inertia += np.sum((X[mask] - centers[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """
        训练K-means模型
        
        执行多次随机初始化，取目标函数最小的结果
        """
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        
        for init in range(self.n_init):
            # 初始化簇中心
            centers = self._init_centers(X)
            
            # 迭代直到收敛
            for i in range(self.max_iter):
                # 步骤2：分配样本到最近的簇
                labels = self._assign_clusters(X, centers)
                
                # 步骤3：更新簇中心
                new_centers = self._update_centers(X, labels)
                
                # 检查收敛：中心变化是否小于阈值
                if np.max(np.abs(new_centers - centers)) < self.tol:
                    break
                
                centers = new_centers
            
            # 计算本次初始化的目标函数值
            inertia = self._compute_inertia(X, labels, centers)
            
            # 保留最优结果
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
        
        # 存储最优结果
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def predict(self, X):
        """
        预测新样本所属的簇
        
        根据训练得到的簇中心，将新样本分配到最近的簇
        """
        return self._assign_clusters(X, self.cluster_centers_)
