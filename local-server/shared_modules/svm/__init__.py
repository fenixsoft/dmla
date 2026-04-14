# 支持向量机模块
# 包含SVM、核方法等实现

from .simple_s_v_m import SimpleSVM
from .kernel_s_v_m import KernelSVM
__all__ = [ 'SimpleSVM', 'KernelSVM']