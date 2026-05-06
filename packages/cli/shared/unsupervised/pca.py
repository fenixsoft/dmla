# PCA 类定义
# 从文档自动提取生成

import numpy as np

class PCA:
    """
    主成分分析（Principal Component Analysis）实现
    
    核心步骤（对应理论推导）：
    1. 数据中心化（减去均值）
    2. 计算协方差矩阵 S = X^T X / (n-1)
    3. 特征分解 S = V Λ V^T
    4. 选择前 k 个特征值对应的特征向量作为主成分
    5. 投影到主成分空间
    
    参数说明:
    n_components : int, 可选
        要保留的主成分数量。若为 None，保留所有成分
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        
        # 存储 PCA 结果
        self.components_ = None              # 主成分（特征向量矩阵）
        self.explained_variance_ = None      # 特征值（各主成分的方差）
        self.explained_variance_ratio_ = None  # 方差解释比例
        self.mean_ = None                    # 数据均值向量
    
    def fit(self, X):
        """
        训练 PCA 模型
        
        参数说明:
        X : ndarray, shape (n_samples, n_features)
            输入数据矩阵
        
        返回:
        self : PCA 对象实例
        """
        n_samples, n_features = X.shape
        
        # 步骤1：数据中心化（对应理论中的 x_i - x̄）
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # 步骤2：计算协方差矩阵（对应理论中的 S = 1/n Σ(x_i - x̄)(x_i - x̄)^T）
        # 使用 n-1 而非 n，得到无偏估计（与 sklearn 一致）
        cov_matrix = X_centered.T @ X_centered / (n_samples - 1)
        
        # 步骤3：特征分解（对应理论中的 S = VΛV^T）
        # np.linalg.eigh 专门用于对称矩阵，返回实数特征值
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 特征值和特征向量按降序排列（PCA 选择方差最大的方向）
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        
        # 存储特征值（对应理论中的 λ_j）
        self.explained_variance_ = eigenvalues
        
        # 步骤4：计算方差解释比例（对应理论中的 Σλ_j / Σλ_total）
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
        # 确定主成分数量
        if self.n_components is None:
            self.n_components = n_features
        
        # 步骤5：选择前 k 个主成分（对应理论中的 V_k）
        self.components_ = eigenvectors[:, :self.n_components].T
        
        return self
    
    def transform(self, X):
        """
        将数据投影到主成分空间
        
        参数说明:
        X : ndarray, shape (n_samples, n_features)
            输入数据
        
        返回:
        Z : ndarray, shape (n_samples, n_components)
            投影后的低维数据
        """
        # 中心化后投影（对应理论中的 Z = X̃ V_k）
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """训练并转换（一步完成）"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, Z):
        """
        从低维空间重构原始数据
        
        参数说明:
        Z : ndarray, shape (n_samples, n_components)
            低维表示
        
        返回:
        X_reconstructed : ndarray, shape (n_samples, n_features)
            重构的高维数据（加回均值）
        """
        # 重构公式（对应理论中的 X̂ = Z V_k^T + x̄）
        return Z @ self.components_ + self.mean_
