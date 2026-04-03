# 统计与概率论 - 任务列表

## 状态概览

- 总章节: 6
- 已完成: 6
- 进行中: 0
- 待开始: 0

---

## 章节写作任务（串行执行）

### 第1章：引言——不确定性的决策框架

- [x] 写作-引言                                          #owner: author
  - 文件: draft/chapters/01-introduction.md
  - 预计字数: 3000
  - 状态: ✅ 完成（约2800中文字）
  - 内容:
    - 概率统计在AI中的地位（三大支柱关系）
    - 从确定性编程到概率性思维的转变
    - 不确定性的来源分析
    - 学习路线图（Mermaid）
- [x] 校审-第1章                                         #owner: reviewer
  - 校审文件: reviews/review-01.md
  - 状态: ✅ 通过
  - 依赖: 写作-引言完成

### 第2章：概率基础——从随机变量到概率分布

- [x] 写作-概率基础                                      #owner: author
  - 文件: draft/chapters/02-probability-basics.md
  - 预计字数: 5000
  - 状态: ✅ 完成（约3350中文字）
  - 前置: 第1章完成
  - 内容:
    - 随机变量与概率分布（PMF/PDF/CDF）
    - 常见分布（伯努利、正态、多项、指数）及AI应用
    - 条件概率与独立性
    - 贝叶斯定理与直观理解
  - 代码:
    - NumPy分布采样示例（runnable）
    - 条件概率计算示例
    - 贝叶斯更新可视化
- [x] 校审-第2章                                         #owner: reviewer
  - 校审文件: reviews/review-02.md
  - 状态: ✅ 通过
  - 依赖: 写作-概率基础完成

### 第3章：统计推断——从数据估计参数

- [x] 写作-统计推断                                      #owner: author
  - 文件: draft/chapters/03-statistical-inference.md
  - 预计字数: 4000
  - 状态: ✅ 完成（约2860中文字）
  - 前置: 第2章完成
  - 内容:
    - 点估计（MLE与MAP）
    - 区间估计与置信区间
    - 贝叶斯推断vs频率学派
    - 假设检验思想
  - 代码:
    - NumPy实现MLE估计（runnable）
    - 贝叶斯更新过程可视化
    - 置信区间计算示例
- [x] 校审-第3章                                         #owner: reviewer
  - 校审文件: reviews/review-03.md
  - 状态: ✅ 通过
  - 依赖: 写作-统计推断完成

### 第4章：模型评估——偏差、方差与选择

- [x] 写作-模型评估                                      #owner: author
  - 文件: draft/chapters/04-model-evaluation.md
  - 预计字数: 4000
  - 状态: ✅ 完成（约2280中文字）
  - 前置: 第3章完成
  - 内容:
    - 偏差-方差分解（过拟合/欠拟合本质）
    - 交叉验证方法
    - 模型选择准则（AIC/BIC）
    - 泛化误差置信区间
  - 代码:
    - 偏差-方差分解可视化（runnable）
    - 交叉验证NumPy实现思路
    - 模型比较示例
- [x] 校审-第4章                                         #owner: reviewer
  - 校审文件: reviews/review-04.md
  - 状态: ✅ 通过
  - 依赖: 写作-模型评估完成

### 第5章：NumPy实践——概率统计的计算工具

- [x] 写作-NumPy实践                                     #owner: author
  - 文件: draft/chapters/05-numpy-practice.md
  - 预计字数: 3000
  - 状态: ✅ 完成（约1480中文字）
  - 前置: 第4章完成
  - 内容:
    - 随机数生成与采样
    - 分布的NumPy表示
    - 统计量计算
    - Monte Carlo模拟实例
  - 代码:
    - 随机数生成示例（runnable）
    - 统计量计算示例
    - Monte Carlo积分估计
- [x] 校审-第5章                                         #owner: reviewer
  - 校审文件: reviews/review-05.md
  - 状态: ✅ 通过
  - 依赖: 写作-NumPy实践完成

### 第6章：应用场景——概率在机器学习中的实践

- [x] 写作-应用场景                                      #owner: author
  - 文件: draft/chapters/06-applications.md
  - 预计字数: 4000
  - 状态: ✅ 完成（约2550中文字）
  - 前置: 第5章完成
  - 内容:
    - Naive Bayes分类器
    - 逻辑回归的概率解释
    - EM算法思想
    - 生成模型入门（GAN/VAE）
  - 代码:
    - Naive Bayes NumPy完整实现（runnable）
    - 逻辑回归概率视角可视化
- [x] 校审-第6章                                         #owner: reviewer
  - 校审文件: reviews/review-06.md
  - 状态: ✅ 通过
  - 依赖: 写作-应用场景完成

---

## 整合任务

- [ ] 整合发布                                          #owner: lead
  - 依赖: 所有章节校审通过
  - 操作:
    - 将 draft/chapters/ 文件移动到 docs/probability/
    - 更新 VuePress sidebar 配置
    - 确保文档可访问

---

## 备注

- **写作模式**: 章节模式（总字数 > 5000字）
- **串行执行**: 每章写完 → 校审通过 → 开始下一章
- **代码规范**: 仅使用NumPy，不使用SciPy
- **代码标记**: 使用 `runnable` 标记代码块
- **语言**: 注释和说明使用中文
- **风格**: 保持与前两部分（线性代数、微积分）一致