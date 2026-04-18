---
title: "感知机"
date: 2026-04-18
tags:
  - deep-learning
  - neural-networks
  - perceptron
---

# 感知机

在上一章的结尾，我们提到1969年《Perceptrons》一书对感知机提出了尖锐批评，指出其无法解决异或（XOR）问题，导致神经网络研究陷入长达十年的低谷。然而，批评本身恰恰揭示了感知机的本质特性：**线性可分性**（Linear Separability）。理解这一特性，不仅有助于掌握感知机的工作原理，更能深刻认识神经网络从单层到多层演进的内在逻辑。

**感知机**（Perceptron）由心理学家弗兰克·罗森布拉特（Frank Rosenblatt）于1957年提出，是第一个能够从数据中学习的神经网络模型。它在M-P模型的基础上引入了学习算法，能够自动调整权重，从而实现模式识别和分类任务。感知机的提出标志着神经网络从理论研究进入实践应用阶段，引发了第一次神经网络研究热潮。本章将详细介绍感知机的模型结构、几何解释、学习算法与收敛定理，并通过实验验证其学习能力与局限性。

## Rosenblatt感知机模型

### 模型结构

感知机是一个单层神经网络，由输入层和输出层组成，没有隐藏层。其基本结构如下：

![感知机结构示意图](assets/perceptron-structure.png)

*图：感知机的基本结构*

设输入向量为 $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$，权重向量为 $\mathbf{w} = (w_1, w_2, \ldots, w_n)^T$，偏置（bias）为 $b$。感知机的输出计算过程分为两步：

1. **线性组合**：计算输入的加权总和加上偏置
$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b$$

2. **激活函数**：通过阈值函数（符号函数）将线性输出转换为二值输出
$$y = \begin{cases} 1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}$$

这里使用符号函数 $\text{sign}(z)$ 作为激活函数，输出值为 $\{1, -1\}$（也可以使用 $\{1, 0\}$）。激活函数的作用是将连续的线性输出"离散化"为类别标签，实现分类决策。

偏置 $b$ 可以理解为阈值 $\theta$ 的负值。在上一章的M-P模型中，阈值条件是 $\sum w_i x_i \geq \theta$；若将阈值移到左侧，变为 $\sum w_i x_i - \theta \geq 0$，则 $b = -\theta$。这种形式在数学处理上更为方便，因为决策边界可以统一写成 $\mathbf{w}^T \mathbf{x} + b = 0$。

### 与M-P模型的关系

感知机继承了M-P模型的核心思想：加权求和、阈值决策、二值输出。两者的关键区别在于**学习能力**：

- M-P模型的权重和阈值需要人工设定，模型本身没有学习能力。
- 感知机引入了学习算法，能够根据训练数据自动调整权重和偏置。

这一进步源于Hebb学习规则的思想：权重可以根据神经元的活动进行调整。感知机将Hebb规则的"相关性学习"发展为"错误驱动学习"——只有当预测错误时才调整权重，调整的方向是减少预测错误。

### 感知机的数学表示

为了便于推导，通常将偏置 $b$ 合并入权重向量。定义增广输入向量 $\tilde{\mathbf{x}} = (x_1, x_2, \ldots, x_n, 1)^T$ 和增广权重向量 $\tilde{\mathbf{w}} = (w_1, w_2, \ldots, w_n, b)^T$，则感知机的输出可以简洁表示为：

$$y = \text{sign}(\tilde{\mathbf{w}}^T \tilde{\mathbf{x}})$$

这种表示方式将偏置视为一个"常数输入1"对应的权重，简化了数学推导。后续讨论中，若无特殊说明，我们使用增广向量形式，并省略"增广"标记，直接记为 $\mathbf{w}$ 和 $\mathbf{x}$。

## 几何解释与线性可分性

### 决策边界的几何意义

感知机的决策边界是超平面方程 $\mathbf{w}^T \mathbf{x} = 0$。在二维空间中，决策边界是一条直线；在三维空间中，是一个平面；在更高维空间中，是一个超平面。这个超平面将输入空间划分为两个区域：

- $\mathbf{w}^T \mathbf{x} > 0$：输出 $y = 1$，属于"正类"
- $\mathbf{w}^T \mathbf{x} < 0$：输出 $y = -1$，属于"负类"

决策边界的位置和方向由权重向量 $\mathbf{w}$ 决定。权重向量的方向垂直于决策边界（因为 $\mathbf{w}$ 是超平面的法向量），权重向量的长度决定了边界的"陡峭程度"。

### 线性可分性的定义

**线性可分**（Linearly Separable）是指存在一个超平面，能够完全分离两类数据点，使得所有正类样本位于超平面一侧，所有负类样本位于另一侧。

设训练数据集为 $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$，其中 $\mathbf{x}_i \in \mathbb{R}^n$，$y_i \in \{1, -1\}$。数据集 $D$ 线性可分的定义是：存在权重向量 $\mathbf{w}$，使得对所有样本都有：

$$y_i \cdot (\mathbf{w}^T \mathbf{x}_i) > 0$$

这个条件的几何含义是：正类样本（$y_i = 1$）落在超平面的"正侧"（$\mathbf{w}^T \mathbf{x}_i > 0$），负类样本（$y_i = -1$）落在超平面的"负侧"（$\mathbf{w}^T \mathbf{x}_i < 0$，即 $y_i \cdot \mathbf{w}^T \mathbf{x}_i = -1 \times (\text{负数}) = \text{正数} > 0$）。两类样本被超平面完全分开，没有任何样本落在边界上或跨越边界。

### 线性可分与不可分的例子

**线性可分例子**：AND逻辑运算

AND运算的真值表：
- $(0, 0) \rightarrow 0$（负类）
- $(0, 1) \rightarrow 0$（负类）
- $(1, 0) \rightarrow 0$（负类）
- $(1, 1) \rightarrow 1$（正类）

在二维平面上，三个负类点 $(0,0)$、$(0,1)$、$(1,0)$ 和一个正类点 $(1,1)$ 可以被一条直线分开。决策边界 $x_1 + x_2 = 1.5$（或 $x_1 + x_2 - 1.5 = 0$）将正类点与其他点分离。

**线性不可分例子**：XOR逻辑运算

XOR运算的真值表：
- $(0, 0) \rightarrow 0$（负类）
- $(0, 1) \rightarrow 1$（正类）
- $(1, 0) \rightarrow 1$（正类）
- $(1, 1) \rightarrow 0$（负类）

在二维平面上，四个点呈"对角分布"：正类点位于对角线 $(0,1)$—$(1,0)$ 上，负类点位于另一条对角线 $(0,0)$—$(1,1)$ 上。不存在任何直线能将两类完全分开——任何直线要么穿过正类点之间，要么穿过负类点之间。这就是"非线性可分"的典型例子。

![线性可分与XOR问题](assets/linear-separability.png)

*图：线性可分（AND）与线性不可分（XOR）的对比*

## 感知机学习算法

感知机学习算法的核心思想是"错误驱动"：只有当预测错误时才更新权重，更新的方向使得下一次预测更接近正确结果。算法流程如下：

### 算法步骤

1. **初始化**：权重向量 $\mathbf{w}$ 初始化为零向量或随机小值。

2. **迭代训练**：遍历训练数据，对每个样本 $(\mathbf{x}_i, y_i)$ 执行：
   - 计算预测值：$\hat{y}_i = \text{sign}(\mathbf{w}^T \mathbf{x}_i)$
   - 若预测错误（$\hat{y}_i \neq y_i$），更新权重：
     $$\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i$$
   其中 $\eta > 0$ 是学习率，控制更新幅度。

3. **终止条件**：当所有样本都正确分类，或达到最大迭代次数时停止。

### 权重更新的几何解释

权重更新公式 $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i$ 的几何含义如下：

- 当真实标签 $y_i = 1$ 但预测为 $-1$ 时（$\mathbf{w}^T \mathbf{x}_i < 0$），说明样本 $\mathbf{x}_i$ 落在了决策边界的"错误一侧"。更新规则 $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot \mathbf{x}_i$ 将权重向量向样本方向移动，使得 $\mathbf{w}^T \mathbf{x}_i$ 增大，更可能变为正值。

- 当真实标签 $y_i = -1$ 但预测为 $1$ 时（$\mathbf{w}^T \mathbf{x}_i > 0$），说明样本 $\mathbf{x}_i$ 也落在了"错误一侧"。更新规则 $\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \mathbf{x}_i$ 将权重向量远离样本方向，使得 $\mathbf{w}^T \mathbf{x}_i$ 减小，更可能变为负值。

直观理解：每次错误预测后，决策边界向错误分类的样本"旋转"，试图将该样本正确分类。

### 感知机收敛定理

**感知机收敛定理**（Perceptron Convergence Theorem）是Rosenblatt在1962年证明的重要理论结果：

> 如果训练数据集是线性可分的，感知机学习算法必能在有限步内收敛到一个解，使得所有样本都被正确分类。

定理的关键假设是"线性可分"。如果数据线性可分，存在至少一个权重向量 $\mathbf{w}^*$ 能正确分类所有样本。定理证明了感知机算法能在有限步内找到一个这样的解。

收敛步数的上限与数据的"分离程度"有关。定义**间隔**（Margin）为：

$$\gamma = \min_i \frac{y_i \cdot (\mathbf{w}^*^T \mathbf{x}_i)}{||\mathbf{w}^*||}$$

间隔 $\gamma$ 表示样本到决策边界最近距离。定理证明：如果数据线性可分且间隔为 $\gamma$，感知机算法最多需要 $O(R^2/\gamma^2)$ 步收敛，其中 $R$ 是样本的最大范数 $||\mathbf{x}_i||$。

这个结果有两层含义：

1. **保证收敛**：线性可分数据上，算法必然成功，不会无限循环。
2. **收敛速度**：间隔越大（两类数据分离越清晰），收敛越快；样本范数越大（数据分布越分散），收敛越慢。

然而，定理的前提"线性可分"也是算法的局限：如果数据非线性可分，算法可能无法收敛，权重会不断更新，永远存在错误分类的样本。

## XOR问题与感知机局限性

### XOR问题的几何分析

上一章已经介绍过XOR问题的定义和几何分布。这里从感知机的角度进一步分析为什么XOR无法被单层感知机解决。

设感知机模型为 $y = \text{sign}(w_1 x_1 + w_2 x_2 + b)$。XOR问题要求：
- $(0,0) \rightarrow -1$：需要 $w_1 \cdot 0 + w_2 \cdot 0 + b < 0$，即 $b < 0$
- $(0,1) \rightarrow +1$：需要 $w_1 \cdot 0 + w_2 \cdot 1 + b > 0$，即 $w_2 + b > 0$
- $(1,0) \rightarrow +1$：需要 $w_1 \cdot 1 + w_2 \cdot 0 + b > 0$，即 $w_1 + b > 0$
- $(1,1) \rightarrow -1$：需要 $w_1 \cdot 1 + w_2 \cdot 1 + b < 0$，即 $w_1 + w_2 + b < 0$

由前三个不等式得：
- 从 $w_2 + b > 0$ 和 $b < 0$ 得 $w_2 > -b > 0$
- 从 $w_1 + b > 0$ 和 $b < 0$ 得 $w_1 > -b > 0$

因此 $w_1 + w_2 + b > -b + (-b) + b = -b > 0$。但第四个不等式要求 $w_1 + w_2 + b < 0$，矛盾。证明不存在满足所有条件的权重，单层感知机无法解决XOR问题。

### XOR问题的深层含义

XOR问题不仅是一个反例，更揭示了感知机的本质局限：

1. **表达能力受限**：单层感知机的决策边界只能是线性超平面，无法表示非线性边界。任何非线性可分的数据分布都无法被单层感知机正确分类。

2. **特征提取缺失**：感知机直接对原始输入进行线性组合，没有"特征变换"的能力。XOR问题的本质是：两类样本的差异在于"是否恰有一个输入为1"，这需要检测两个输入的"组合特征"而非单独处理每个输入。

3. **多层网络的必要性**：如果增加一层隐藏层，先提取"组合特征"，再基于提取的特征做决策，就能解决XOR问题。这暗示了多层网络的必要性。

### 多层感知机解决XOR

如上一章练习题所示，一个两层感知机可以解决XOR问题：

第一层两个隐藏神经元：
- 神经元1：$h_1 = \text{sign}(x_1 + x_2 - 0.5)$（检测"至少一个为1"）
- 神经元2：$h_2 = \text{sign}(x_1 + x_2 - 1.5)$（检测"两个都为1"）

第二层输出神经元：
- $y = \text{sign}(h_1 - h_2)$（实现"至少一个为1"但"不是两个都为1"）

验证：
| 输入 $(x_1, x_2)$ | $h_1$ | $h_2$ | 输出 $y$ | XOR正确结果 |
|:----------------:|:-----:|:-----:|:--------:|:-----------:|
| $(0, 0)$ | 0 | 0 | 0 | 0 ✓ |
| $(0, 1)$ | 1 | 0 | 1 | 1 ✓ |
| $(1, 0)$ | 1 | 0 | 1 | 1 ✓ |
| $(1, 1)$ | 1 | 1 | 0 | 0 ✓ |

这个例子展示了多层网络的表达能力：隐藏层提取"组合特征"（$h_1$ 和 $h_2$），输出层基于提取的特征做最终决策。问题是：1969年时，多层网络的学习算法尚未发现。单层感知机有收敛定理保证能学习，但多层网络如何训练？这一问题直到1980年代反向传播算法提出后才解决。

## 实验：感知机实现与线性可分性验证

下面通过代码实现感知机，并验证其在线性可分数据上的学习能力。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    感知机实现
    
    使用错误驱动的权重更新规则：
    w = w + eta * y * x (当预测错误时)
    """
    def __init__(self, learning_rate=1.0, max_iterations=1000):
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.w = None  # 权重向量（包含偏置）
        self.errors_history = []  # 每轮迭代错误数
    
    def fit(self, X, y):
        """
        训练感知机
        
        Parameters:
        X : ndarray, shape (n_samples, n_features)
            输入特征矩阵
        y : ndarray, shape (n_samples,)
            标签向量，取值为 {1, -1}
        """
        n_samples, n_features = X.shape
        
        # 增广向量形式：添加常数1列（对应偏置）
        X_aug = np.column_stack([X, np.ones(n_samples)])
        
        # 初始化权重为零向量
        self.w = np.zeros(n_features + 1)
        
        # 训练循环
        for iteration in range(self.max_iter):
            errors = 0
            for i in range(n_samples):
                # 计算预测值
                prediction = np.sign(self.w @ X_aug[i])
                if prediction == 0:
                    prediction = -1  # 符号函数边界情况
                
                # 若预测错误，更新权重
                if prediction != y[i]:
                    self.w += self.lr * y[i] * X_aug[i]
                    errors += 1
            
            self.errors_history.append(errors)
            
            # 若所有样本正确分类，提前终止
            if errors == 0:
                print(f"在第 {iteration + 1} 轮迭代后收敛")
                break
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Parameters:
        X : ndarray, shape (n_samples, n_features)
        
        Returns:
        predictions : ndarray, shape (n_samples,)
            预测标签 {1, -1}
        """
        n_samples = X.shape[0]
        X_aug = np.column_stack([X, np.ones(n_samples)])
        predictions = np.sign(X_aug @ self.w)
        predictions[predictions == 0] = -1
        return predictions
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# 实验1：线性可分数据
print("=" * 50)
print("实验1：线性可分数据（AND逻辑）")
print("=" * 50)

# AND数据：三个负类，一个正类
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([-1, -1, -1, 1])  # 用 -1 表示类别0

model_and = Perceptron(learning_rate=1.0, max_iterations=100)
model_and.fit(X_and, y_and)

print(f"学习到的权重: w1={model_and.w[0]:.2f}, w2={model_and.w[1]:.2f}, b={model_and.w[2]:.2f}")
print(f"决策边界: {model_and.w[0]:.2f}*x1 + {model_and.w[1]:.2f}*x2 + {model_and.w[2]:.2f} = 0")
print(f"训练准确率: {model_and.score(X_and, y_and):.2%}")

# 实验2：线性不可分数据（XOR逻辑）
print("\n" + "=" * 50)
print("实验2：线性不可分数据（XOR逻辑）")
print("=" * 50)

# XOR数据
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([-1, 1, 1, -1])  # XOR: 两个为1或两个为0输出0，其他输出1

model_xor = Perceptron(learning_rate=1.0, max_iterations=100)
model_xor.fit(X_xor, y_xor)

print(f"训练准确率: {model_xor.score(X_xor, y_xor):.2%}")
print(f"说明: XOR问题非线性可分，感知机无法收敛到正确解")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1：AND问题的决策边界
def plot_decision_boundary(ax, X, y, model, title):
    # 绘制数据点
    colors = ['blue' if label == 1 else 'red' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolors='k', linewidth=2)
    
    # 绘制决策边界
    w1, w2, b = model.w
    if w2 != 0:
        x_line = np.linspace(-0.5, 1.5, 100)
        y_line = -(w1 * x_line + b) / w2
        ax.plot(x_line, y_line, 'g-', linewidth=2, label='决策边界')
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_decision_boundary(axes[0], X_and, y_and, model_and, 'AND问题（线性可分）')
plot_decision_boundary(axes[1], X_xor, y_xor, model_xor, 'XOR问题（非线性可分）')

# 图3：收敛过程对比
axes[2].plot(model_and.errors_history, 'b-', linewidth=2, label='AND（收敛）')
axes[2].plot(model_xor.errors_history, 'r-', linewidth=2, label='XOR（不收敛）')
axes[2].set_xlabel('迭代轮数')
axes[2].set_ylabel('错误样本数')
axes[2].set_title('收敛过程对比')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
```

### 实验结论

1. **线性可分数据（AND）**：感知机在有限步内收敛，找到正确的决策边界。权重更新停止时，所有样本都被正确分类。

2. **非线性可分数据（XOR）**：感知机无法收敛，错误数持续波动。无论迭代多少次，总存在错误分类的样本。这验证了感知机的局限性：只有线性可分数据才能被正确学习。

3. **收敛过程对比**：线性可分数据的错误数单调递减至零；非线性可分数据的错误数无法稳定减少，始终存在错误。

## 本章小结

本章详细介绍了Rosenblatt感知机模型，包括其结构、几何解释、学习算法与收敛定理。感知机是第一个可学习的神经网络模型，其核心贡献在于引入了"错误驱动"的学习机制，能够从数据中自动调整权重。

感知机的决策边界是线性超平面，这决定了其表达能力：只能解决线性可分的分类问题。感知机收敛定理保证了线性可分数据上的成功学习，但也暗示了非线性可分数据上的必然失败。XOR问题是这一局限的经典例证，揭示了单层网络表达能力的限制。

然而，XOR问题也指明了突破方向：通过增加隐藏层，构建多层网络，先提取组合特征再做决策，可以解决非线性问题。问题的关键在于如何训练多层网络——这一问题在下一章"多层感知机"中展开讨论，并在后续章节"反向传播算法"中得到解决。

感知机的意义不仅在于其本身的能力，更在于它开启了神经网络从理论到实践的进程。尽管存在局限，但感知机确立了"学习算法"的核心地位，为后续多层网络的发展奠定了基础。

## 练习题

1. 证明感知机权重更新规则 $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i$ 能够使错误分类样本的预测值朝正确方向移动。即证明更新后，$y_i \cdot (\mathbf{w}_{new}^T \mathbf{x}_i) > y_i \cdot (\mathbf{w}_{old}^T \mathbf{x}_i)$。
    <details>
    <summary>参考答案</summary>
    
    设更新前权重为 $\mathbf{w}$，样本 $(\mathbf{x}_i, y_i)$ 被错误分类，即 $y_i \cdot (\mathbf{w}^T \mathbf{x}_i) < 0$。
    
    更新后权重 $\mathbf{w}_{new} = \mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i$。
    
    计算更新后的预测值：
    $$\mathbf{w}_{new}^T \mathbf{x}_i = (\mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i)^T \mathbf{x}_i = \mathbf{w}^T \mathbf{x}_i + \eta \cdot y_i \cdot \mathbf{x}_i^T \mathbf{x}_i$$
    
    注意 $\mathbf{x}_i^T \mathbf{x}_i = ||\mathbf{x}_i||^2 > 0$（假设样本不为零向量），$\eta > 0$。
    
    因此：
    $$y_i \cdot (\mathbf{w}_{new}^T \mathbf{x}_i) = y_i \cdot (\mathbf{w}^T \mathbf{x}_i) + \eta \cdot y_i^2 \cdot ||\mathbf{x}_i||^2$$
    
    由于 $y_i^2 = 1$（标签为 $\pm 1$），$||\mathbf{x}_i||^2 > 0$，$\eta > 0$，所以：
    $$y_i \cdot (\mathbf{w}_{new}^T \mathbf{x}_i) = y_i \cdot (\mathbf{w}^T \mathbf{x}_i) + \eta \cdot ||\mathbf{x}_i||^2 > y_i \cdot (\mathbf{w}^T \mathbf{x}_i)$$
    
    这证明了更新后，$y_i \cdot (\mathbf{w}_{new}^T \mathbf{x}_i)$ 比更新前增大了 $\eta \cdot ||\mathbf{x}_i||^2$。如果更新足够多次，$y_i \cdot (\mathbf{w}^T \mathbf{x}_i)$ 最终会变为正值，样本将被正确分类。
    
    **关键洞察**：每次更新都将预测值朝正确方向移动一个固定步长 $\eta \cdot ||\mathbf{x}_i||^2$。这就是"错误驱动学习"的本质：只纠正错误，不优化正确。
    </details>

2. 设训练数据集包含两个正类样本 $\mathbf{x}_1 = (1, 1)^T$、$\mathbf{x}_2 = (2, 2)^T$ 和两个负类样本 $\mathbf{x}_3 = (-1, -1)^T$、$\mathbf{x}_4 = (-2, -2)^T$。判断该数据集是否线性可分。若是，手动计算感知机学习算法的收敛过程（学习率 $\eta = 1$，初始权重为零）。
    <details>
    <summary>参考答案</summary>
    
    **判断线性可分**：
    
    观察数据分布：正类样本在第一象限，负类样本在第三象限，两类完全分离。决策边界可以是直线 $x_1 - x_2 = 0$（即对角线）的垂直线，如 $x_1 + x_2 = 0$。因此数据线性可分。
    
    **手动计算收敛过程**：
    
    使用增广向量形式（添加偏置列），设初始权重 $\mathbf{w} = (w_1, w_2, b) = (0, 0, 0)$。
    
    增广输入向量：
    - $\tilde{\mathbf{x}}_1 = (1, 1, 1)^T$，$y_1 = 1$
    - $\tilde{\mathbf{x}}_2 = (2, 2, 1)^T$，$y_2 = 1$
    - $\tilde{\mathbf{x}}_3 = (-1, -1, 1)^T$，$y_3 = -1$
    - $\tilde{\mathbf{x}}_4 = (-2, -2, 1)^T$，$y_4 = -1$
    
    **第一轮迭代**：
    
    样本1：$\mathbf{w}^T \tilde{\mathbf{x}}_1 = 0 \cdot 1 + 0 \cdot 1 + 0 \cdot 1 = 0$，$\text{sign}(0)$ 视为 -1，预测错误。
    更新：$\mathbf{w} = \mathbf{w} + 1 \cdot y_1 \cdot \tilde{\mathbf{x}}_1 = (0, 0, 0) + (1, 1, 1) = (1, 1, 1)$
    
    样本2：$\mathbf{w}^T \tilde{\mathbf{x}}_2 = 1 \cdot 2 + 1 \cdot 2 + 1 \cdot 1 = 5 > 0$，$\text{sign}(5) = 1 = y_2$，预测正确，不更新。
    
    样本3：$\mathbf{w}^T \tilde{\mathbf{x}}_3 = 1 \cdot (-1) + 1 \cdot (-1) + 1 \cdot 1 = -1 < 0$，$\text{sign}(-1) = -1 = y_3$，预测正确，不更新。
    
    样本4：$\mathbf{w}^T \tilde{\mathbf{x}}_4 = 1 \cdot (-2) + 1 \cdot (-2) + 1 \cdot 1 = -3 < 0$，$\text{sign}(-3) = -1 = y_4$，预测正确，不更新。
    
    第一轮后，权重 $\mathbf{w} = (1, 1, 1)$，只有样本1错误。
    
    **第二轮迭代**：
    
    样本1：$\mathbf{w}^T \tilde{\mathbf{x}}_1 = 1 + 1 + 1 = 3 > 0$，$\text{sign}(3) = 1 = y_1$，预测正确。
    
    样本2、3、4同上，都正确。
    
    第二轮后，所有样本正确分类，算法收敛。
    
    **最终权重**：$\mathbf{w} = (1, 1, 1)$，决策边界 $x_1 + x_2 + 1 = 0$。
    
    验证：
    - $\mathbf{x}_1$：$1 + 1 + 1 = 3 > 0$ ✓
    - $\mathbf{x}_2$：$2 + 2 + 1 = 5 > 0$ ✓
    - $\mathbf{x}_3$：$-1 - 1 + 1 = -1 < 0$ ✓
    - $\mathbf{x}_4$：$-2 - 2 + 1 = -3 < 0$ ✓
    
    所有样本正确分类，算法在两轮迭代后收敛。
    </details>

3. 解释为什么感知机收敛定理的前提条件是"数据线性可分"。如果数据非线性可分，感知机算法会发生什么情况？
    <details>
    <summary>参考答案</summary>
    
    **定理前提的必要性**：
    
    感知机收敛定理的核心证明依赖于"存在正确解"这一前提。如果数据线性可分，存在权重向量 $\mathbf{w}^*$ 使得所有样本满足 $y_i \cdot (\mathbf{w}^*^T \mathbf{x}_i) > 0$。定理证明每次权重更新都使当前权重 $\mathbf{w}$ 向 $\mathbf{w}^*$ 的方向"移动"，最终在有限步内达到正确解。
    
    如果数据非线性可分，不存在任何权重向量能正确分类所有样本。每次更新虽然纠正了某个错误样本，但可能同时导致其他样本被错误分类。权重向量无法稳定收敛，而是在不同的错误解之间振荡。
    
    **非线性可分时的情况**：
    
    1. **无法收敛**：算法永远不会达到"所有样本正确分类"的状态。无论迭代多少次，总存在错误样本触发权重更新。
    
    2. **权重振荡**：权重向量可能在不同的错误解之间来回切换。例如，XOR问题中，权重可能在"倾向于正类点"和"倾向于负类点"之间振荡，无法稳定。
    
    3. **权重增长**：在没有收敛条件的情况下，权重值可能持续增长（绝对值增大），导致数值不稳定。
    
    4. **边界游走**：决策边界可能在不同的位置之间移动，试图"追逐"被错误分类的样本，但永远无法同时满足所有样本。
    
    **实际应用的处理方法**：
    
    对于非线性可分数据（实际中更常见），感知机的原始形式不适用。改进方法包括：
    
    1. **设置最大迭代次数**：强制停止迭代，返回"近似最优"解。
    
    2. **口袋算法**（Pocket Algorithm）：记录历史上正确率最高的权重，迭代结束后返回该权重。这是一种"近似最优"策略。
    
    3. **软边界感知机**：允许一定比例的错误样本，优化目标改为"最小化错误数"而非"零错误"。
    
    4. **多层网络**：使用隐藏层提取非线性特征，从根本上解决非线性可分问题。
    
    这些改进方法将感知机的思想扩展到更广泛的场景，但核心洞察——"线性可分是单层网络的必要条件"——仍然成立。
    </details>

4. 设计一个两层感知机解决OR逻辑运算，写出各层神经元的权重和阈值设置，并验证其正确性。OR运算定义：$(0,0)\rightarrow 0$，$(0,1)\rightarrow 1$，$(1,0)\rightarrow 1$，$(1,1)\rightarrow 1$。分析为何单层感知机就能解决OR问题，而XOR需要两层。
    <details>
    <summary>参考答案</summary>
    
    **单层感知机解决OR问题**：
    
    OR问题真值表：
    - $(0, 0) \rightarrow 0$（负类）
    - $(0, 1) \rightarrow 1$（正类）
    - $(1, 0) \rightarrow 1$（正类）
    - $(1, 1) \rightarrow 1$（正类）
    
    设感知机模型 $y = \text{sign}(w_1 x_1 + w_2 x_2 + b)$。
    
    选择权重 $w_1 = 1, w_2 = 1, b = -0.5$。
    
    验证：
    - $(0,0)$：$0 + 0 - 0.5 = -0.5 < 0$，输出 $-1$（类别0） ✓
    - $(0,1)$：$0 + 1 - 0.5 = 0.5 > 0$，输出 $1$ ✓
    - $(1,0)$：$1 + 0 - 0.5 = 0.5 > 0$，输出 $1$ ✓
    - $(1,1)$：$1 + 1 - 0.5 = 1.5 > 0$，输出 $1$ ✓
    
    决策边界 $x_1 + x_2 = 0.5$ 是一条直线，将原点（类别0）与其他三点（类别1）分开。OR数据线性可分，单层感知机足以解决。
    
    **为何XOR需要两层**：
    
    XOR和OR的关键区别在于数据分布的几何结构：
    
    - **OR**：三个正类点 $(0,1)$、$(1,0)$、$(1,1)$ 分布在"半个平面"，一个负类点 $(0,0)$ 在另一侧。存在直线将两类分开（如 $x_1 + x_2 = 0.5$）。这是**线性可分**分布。
    
    - **XOR**：两个正类点 $(0,1)$、$(1,0)$ 位于一条对角线，两个负类点 $(0,0)$、$(1,1)$ 位于另一条对角线。不存在任何直线将两类完全分开。这是**非线性可分**分布。
    
    **两层感知机解决XOR的原理**：
    
    两层感知机通过隐藏层"变换"特征空间，将非线性可分问题转化为线性可分问题。
    
    第一层隐藏神经元提取"组合特征"：
    - $h_1 = \text{sign}(x_1 + x_2 - 0.5)$：检测"是否至少有一个为1"
    - $h_2 = \text{sign}(x_1 + x_2 - 1.5)$：检测"是否两个都为1"
    
    隐藏层输出 $(h_1, h_2)$ 形成新的特征空间：
    | 输入 $(x_1, x_2)$ | XOR标签 | $h_1$ | $h_2$ | 新特征 $(h_1, h_2)$ |
    |:----------------:|:-------:|:-----:|:-----:|:-------------------:|
    | $(0, 0)$ | 0 | 0 | 0 | $(0, 0)$ |
    | $(0, 1)$ | 1 | 1 | 0 | $(1, 0)$ |
    | $(1, 0)$ | 1 | 1 | 0 | $(1, 0)$ |
    | $(1, 1)$ | 0 | 1 | 1 | $(1, 1)$ |
    
    在新特征空间 $(h_1, h_2)$ 中，正类点 $(1, 0)$ 和负类点 $(0, 0)$、$(1, 1)$ 分布为：
    - 负类：$(0, 0)$ 和 $(1, 1)$ 在两个极端
    - 正类：$(1, 0)$ 在中间
    
    第二层输出神经元在 $(h_1, h_2)$ 空间中做线性分类：
    $$y = \text{sign}(h_1 - h_2 - 0.5)$$
    
    验证：
    - $(0, 0)$：$0 - 0 - 0.5 < 0$，输出 $-1$ ✓
    - $(1, 0)$：$1 - 0 - 0.5 > 0$，输出 $1$ ✓
    - $(1, 1)$：$1 - 1 - 0.5 < 0$，输出 $-1$ ✓
    
    **核心洞察**：
    
    隐藏层的作用是**特征变换**：将原始输入空间映射到新的特征空间，在新的空间中问题变得线性可分。这揭示了多层网络的核心价值——通过层层变换，逐步将复杂问题简化，最终在顶层用简单的线性模型解决。
    
    OR问题在原始空间已线性可分，不需要变换；XOR问题在原始空间非线性可分，需要隐藏层变换后才能线性可分。这就是单层与两层的本质区别。
    </details>