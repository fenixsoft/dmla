# GaussianMixtureModel 类定义
# 从文档自动提取生成

import numpy as np

class GaussianMixtureModel:
    """
    高斯混合模型实现
    使用EM算法求解
    """
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol  # 收敛阈值
        
        self.weights_ = None   # 混合系数 (K,)
        self.means_ = None     # 均值 (K, n_features)
        self.covariances_ = None  # 协方差矩阵 (K, n_features, n_features)
        self.log_likelihood_history_ = []
    
    def _initialize(self, X):
        """初始化参数"""
        n_samples, n_features = X.shape
        K = self.n_components
        
        # 随机初始化均值（从数据中随机选择K个点）
        indices = np.random.choice(n_samples, K, replace=False)
        self.means_ = X[indices].copy()
        
        # 初始化协方差为数据协方差的对角线
        data_cov = np.cov(X.T)
        self.covariances_ = np.array([np.diag(np.diag(data_cov)) + 1e-6 * np.eye(n_features) 
                                       for _ in range(K)])
        
        # 初始化混合系数为均匀分布
        self.weights_ = np.ones(K) / K
    
    def _gaussian_pdf(self, X, mean, cov):
        """计算多元高斯概率密度"""
        n_features = X.shape[1]
        diff = X - mean
        
        # 加小值保证数值稳定
        cov_reg = cov + 1e-6 * np.eye(n_features)
        
        # 使用Cholesky分解计算行列式和逆
        try:
            L = np.linalg.cholesky(cov_reg)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            diff_L = np.linalg.solve(L, diff.T).T
            mahalanobis = np.sum(diff_L ** 2, axis=1)
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，使用标准方法
            sign, log_det = np.linalg.slogdet(cov_reg)
            cov_inv = np.linalg.inv(cov_reg)
            mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
        
        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)
        return log_prob
    
    def _e_step(self, X):
        """E步：计算责任度"""
        n_samples = X.shape[0]
        K = self.n_components
        
        # 计算每个成分的对数概率
        log_probs = np.zeros((n_samples, K))
        for k in range(K):
            log_probs[:, k] = (np.log(self.weights_[k] + 1e-10) + 
                               self._gaussian_pdf(X, self.means_[k], self.covariances_[k]))
        
        # 计算对数似然
        log_likelihood = np.sum(np.log(np.sum(np.exp(log_probs), axis=1)))
        
        # 计算责任度（使用log-sum-exp trick避免数值下溢）
        log_max = log_probs.max(axis=1, keepdims=True)
        log_sum = np.log(np.sum(np.exp(log_probs - log_max), axis=1, keepdims=True)) + log_max
        responsibilities = np.exp(log_probs - log_sum)
        
        return responsibilities, log_likelihood
    
    def _m_step(self, X, responsibilities):
        """M步：更新参数"""
        n_samples, n_features = X.shape
        K = self.n_components
        
        # 计算每个成分的有效样本数
        N_k = responsibilities.sum(axis=0) + 1e-10
        
        # 更新混合系数
        self.weights_ = N_k / n_samples
        
        # 更新均值
        self.means_ = (responsibilities.T @ X) / N_k[:, np.newaxis]
        
        # 更新协方差
        for k in range(K):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            self.covariances_[k] = (weighted_diff.T @ diff) / N_k[k]
            # 添加正则化
            self.covariances_[k] += 1e-6 * np.eye(n_features)
    
    def fit(self, X):
        """训练模型"""
        self._initialize(X)
        self.log_likelihood_history_ = []
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E步
            responsibilities, log_likelihood = self._e_step(X)
            self.log_likelihood_history_.append(log_likelihood)
            
            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"EM收敛于第{iteration}次迭代")
                break
            
            # M步
            self._m_step(X, responsibilities)
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict(self, X):
        """预测聚类标签"""
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """预测属于各成分的概率"""
        responsibilities, _ = self._e_step(X)
        return responsibilities
    
    def score(self, X):
        """计算对数似然"""
        _, log_likelihood = self._e_step(X)
        return log_likelihood
