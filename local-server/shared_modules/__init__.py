from .svm.kernel_s_v_m import KernelSVM
from .bayesian.gaussian_mixture_model import GaussianMixtureModel
from .svm.simple_s_v_m import SimpleSVM
from .bayesian.simple_bayesian_network import SimpleBayesianNetwork
from .bayesian.multinomial_naive_bayes import MultinomialNaiveBayes
from .linear.ridge_regression import RidgeRegression
from .linear.lasso_regression import LassoRegression
from .linear.logistic_regression import LogisticRegression
# shared 模块包初始化
# 包含统计学习系列文档中可复用的类定义

from .linear import *