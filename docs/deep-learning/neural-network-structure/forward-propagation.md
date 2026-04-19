---
title: "前向传播"
date: 2026-04-18
tags:
  - deep-learning
  - neural-networks
  - forward-propagation
---

# 前向传播

在前三章中，我们逐步构建了神经网络的理论框架：从生物神经元到 M-P 模型，从感知机到多层感知机，从线性决策边界到非线性表达能力。现在，我们将深入神经网络的核心计算机制 —— **前向传播**（Forward Propagation）。

前向传播是指信号从输入层经过各层神经元逐层传递到输出层的过程。理解前向传播的计算流程，是掌握神经网络工作原理的基础。本章将介绍信号流动过程、矩阵形式推导、计算图概念，以及批量计算与效率优化，并通过实验实现前向传播的计算图可视化。

## 信号流动过程

### 单个神经元的信号流动

回顾单个神经元的工作流程：

1. **接收输入**：神经元接收来自上一层神经元的输出信号，记为输入向量 $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$。

2. **加权求和**：每个输入信号乘以对应的权重，再加偏置：
$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
其中 $\mathbf{w} = (w_1, w_2, \ldots, w_n)^T$ 是权重向量，$b$ 是偏置。$z$ 称为**线性组合值**或**预激活值**（Pre-activation）。

3. **激活变换**：线性组合值通过激活函数变换，产生神经元输出：
$$a = f(z)$$
其中 $f$ 是激活函数（如 Sigmoid、ReLU 等），$a$ 称为**激活值**（Activation）或神经元输出。

这个三步流程 —— 输入→加权求和→激活变换 —— 构成了神经元的基本计算单元。将多个神经元组织成层，连接成网络，信号便从输入层流向输出层。

### 多层网络的信号流动

多层神经网络的前向传播是逐层计算的过程。设网络有 $L$ 层（包括输入层、隐藏层和输出层），第 $l$ 层的计算如下：

**输入到第 $l$ 层**：上一层（第 $l-1$ 层）的激活值向量 $\mathbf{a}^{l-1}$。

**线性组合**：
$$\mathbf{z}^l = \mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l$$

其中：
- $\mathbf{W}^l$ 是第 $l$ 层的权重矩阵，大小为 $n_l \times n_{l-1}$（$n_l$ 是第 $l$ 层神经元数）
- $\mathbf{b}^l$ 是第 $l$ 层的偏置向量，大小为 $n_l$
- $\mathbf{z}^l$ 是第 $l$ 层的预激活值向量

**激活变换**：
$$\mathbf{a}^l = f^l(\mathbf{z}^l)$$

其中 $f^l$ 是第 $l$ 层的激活函数，$\mathbf{a}^l$ 是第 $l$ 层的激活值向量。

**信号流动的全过程**：

设输入为 $\mathbf{x}$，则：
- 第 1 层（第一个隐藏层）：$\mathbf{z}^1 = \mathbf{W}^1 \mathbf{x} + \mathbf{b}^1$，$\mathbf{a}^1 = f^1(\mathbf{z}^1)$
- 第 2 层：$\mathbf{z}^2 = \mathbf{W}^2 \mathbf{a}^1 + \mathbf{b}^2$，$\mathbf{a}^2 = f^2(\mathbf{z}^2)$
- ...
- 第 $L$ 层（输出层）：$\mathbf{z}^L = \mathbf{W}^L \mathbf{a}^{L-1} + \mathbf{b}^L$，$\mathbf{a}^L = f^L(\mathbf{z}^L)$

输出 $\mathbf{a}^L$ 就是网络对输入 $\mathbf{x}$ 的预测结果。

### 符号约定总结

为保持符号一致性，本章使用以下约定：

| 符号 | 含义 | 形状 |
|:-----|:-----|:-----|
| $\mathbf{x}$ | 输入向量 | $n_0 \times 1$ |
| $\mathbf{W}^l$ | 第 $l$ 层权重矩阵 | $n_l \times n_{l-1}$ |
| $\mathbf{b}^l$ | 第 $l$ 层偏置向量 | $n_l \times 1$ |
| $\mathbf{z}^l$ | 第 $l$ 层预激活值向量 | $n_l \times 1$ |
| $\mathbf{a}^l$ | 第 $l$ 层激活值向量 | $n_l \times 1$ |
| $f^l$ | 第 $l$ 层激活函数 | 函数 |
| $n_l$ | 第 $l$ 层神经元数量 | 整数 |

注意：输入层记为第 0 层，$\mathbf{a}^0 = \mathbf{x}$；输出层记为第 $L$ 层。

## 矩阵形式推导

### 为什么使用矩阵形式

单个样本的前向传播可以直接使用向量运算。但当处理多个样本时，逐个计算效率低下。矩阵形式允许批量处理多个样本，显著提升计算效率。

此外，矩阵形式将网络计算表达为矩阵乘法和非线性变换的组合，便于推导反向传播公式，也便于利用 GPU 等硬件的矩阵运算优化能力。

### 批量输入的矩阵形式

设输入包含 $m$ 个样本，记为矩阵 $\mathbf{X} \in \mathbb{R}^{n_0 \times m}$，其中每一列是一个样本向量：
$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_m]$$

第 $l$ 层的计算推广为矩阵形式：

**预激活矩阵**：
$$\mathbf{Z}^l = \mathbf{W}^l \mathbf{A}^{l-1} + \mathbf{b}^l$$

其中：
- $\mathbf{A}^{l-1} \in \mathbb{R}^{n_{l-1} \times m}$ 是第 $l-1$ 层的激活值矩阵（每列一个样本的激活值）
- $\mathbf{Z}^l \in \mathbb{R}^{n_l \times m}$ 是第 $l$ 层的预激活值矩阵

**激活矩阵**：
$$\mathbf{A}^l = f^l(\mathbf{Z}^l)$$

激活函数 $f^l$ 对矩阵逐元素应用，即 $\mathbf{A}^l_{ij} = f^l(\mathbf{Z}^l_{ij})$。

**偏置广播机制**：

公式 $\mathbf{Z}^l = \mathbf{W}^l \mathbf{A}^{l-1} + \mathbf{b}^l$ 中，$\mathbf{W}^l \mathbf{A}^{l-1}$ 结果大小为 $n_l \times m$，$\mathbf{b}^l$ 大小为 $n_l \times 1$。两者形状不匹配，但通过"广播机制"（Broadcasting），$\mathbf{b}^l$ 自动扩展为 $n_l \times m$，每列都是相同的偏置向量。这相当于对所有样本添加相同的偏置。

### 矩阵形式的维度检查

矩阵运算的正确性可以通过维度检查验证。设第 $l-1$ 层有 $n_{l-1}$ 个神经元，第 $l$ 层有 $n_l$ 个神经元，批量大小为 $m$：

$$\mathbf{Z}^l = \mathbf{W}^l \mathbf{A}^{l-1} + \mathbf{b}^l$$

- $\mathbf{W}^l$: $n_l \times n_{l-1}$（权重矩阵）
- $\mathbf{A}^{l-1}$: $n_{l-1} \times m$（上一层激活值矩阵）
- $\mathbf{W}^l \mathbf{A}^{l-1}$: $n_l \times n_{l-1} \times n_{l-1} \times m = n_l \times m$（矩阵乘法结果）
- $\mathbf{b}^l$: $n_l \times 1$，广播后 $n_l \times m$
- $\mathbf{Z}^l$: $n_l \times m$（预激活矩阵）

维度一致性是验证前向传播实现正确性的重要方法。实践中常见错误（如权重矩阵维度错误）可以通过维度检查快速定位。

## 计算图概念

### 计算图的定义

**计算图**（Computational Graph）是一种表示计算过程的图形化方法。计算图中的节点表示运算（如加法、乘法、激活函数），边表示数据流动。计算图将复杂的计算过程分解为基本运算的组合，便于理解和实现。

神经网络的前向传播可以自然地表示为计算图。每个神经元包含两个节点：线性组合节点（$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$）和激活节点（$\mathbf{a} = f(\mathbf{z})$）。多层网络将这些节点按层连接，形成完整的计算图。

### 计算图的优势

1. **可视化计算过程**：计算图直观展示数据如何从输入流向输出，每步进行什么运算。

2. **支持自动求导**：计算图是自动微分（Automatic Differentiation）的基础。反向传播可以沿着计算图反向遍历，自动计算梯度。

3. **便于模块化实现**：计算图将计算分解为基本运算单元，便于模块化实现和组合。深度学习框架（如 TensorFlow、PyTorch）都基于计算图构建。

4. **支持优化**：计算图结构便于分析计算依赖关系，进行优化（如并行计算、内存优化）。

### 计算图示例

以单层感知机为例，计算图如下：

```
输入 x → [权重乘法] → z = Wx → [偏置加法] → z+b → [激活函数] → a = f(z+b)
        ↑                                              ↑
      权重 W                                         输出 a
```

对于多层网络，计算图是多个单层计算图的串联：

```
输入 X → [层1计算] → A1 → [层2计算] → A2 → ... → [层L计算] → 输出 Y
```

### 静态图与动态图

深度学习框架中，计算图有两种构建方式：

1. **静态图**（Static Graph）：先定义完整的计算图结构，再执行计算。TensorFlow 早期版本采用这种方式。优点是可以预先优化计算图；缺点是灵活性较低，难以处理动态结构（如条件分支）。

2. **动态图**（Dynamic Graph）：计算图在执行时动态构建。PyTorch 采用这种方式。优点是灵活性高，便于调试；缺点是无法预先优化，每次执行都要重新构建。

现代框架逐渐融合两种方式：TensorFlow 2.x 支持动态图模式，PyTorch 支持通过 JIT 编译优化动态图。

## 批量计算与效率优化

### 批量处理的意义

神经网络训练通常涉及大量样本。逐个样本处理（逐个前向传播、逐个反向传播）效率极低。**批量处理**（Batch Processing）将多个样本合并为一个矩阵，一次性计算，显著提升效率。

设批量大小为 $B$（Batch Size）。前向传播一次处理 $B$ 个样本，输出 $B$ 个预测结果。反向传播基于 $B$ 个样本的梯度平均值更新权重。

批量处理的优势：

1. **计算效率**：矩阵运算可以利用 GPU 等硬件的并行计算能力，处理 $B$ 个样本的时间远小于逐个处理 $B$ 个样本的时间之和。

2. **梯度稳定性**：基于多个样本的梯度平均值比单个样本的梯度更稳定，减少梯度波动，训练更平稳。

3. **内存利用**：批量处理可以更好地利用内存带宽，减少数据传输次数。

### 批量大小的选择

批量大小 $B$ 是重要的超参数。选择原则：

- **小批量（$B=16$-$64$）**：梯度噪声大，训练波动，但有利于跳出局部最优；适合内存受限场景。
- **中等批量（$B=128$-$512$）**：平衡效率和稳定性，是常用选择。
- **大批量（$B=1024$+）**：计算效率高，梯度稳定，但可能陷入局部最优；需要 GPU/TPU 等高性能硬件。

实际中，批量大小受限于硬件内存。GPU 显存大小决定最大批量。批量大小过大可能导致内存溢出。

### 计算效率优化策略

前向传播的计算效率优化策略包括：

1. **矩阵运算优化**：利用 GPU 的矩阵乘法加速（如 CUDA、cuBLAS）。矩阵乘法是前向传播的核心运算，GPU 可以高效并行执行。

2. **激活函数融合**：将线性组合和激活变换合并为单一操作，减少中间结果的存储和传输。

3. **内存复用**：复用中间结果的内存空间，减少内存分配开销。例如，$\mathbf{Z}^l$ 计算完成后，$\mathbf{A}^l$ 可以复用 $\mathbf{Z}^l$ 的内存。

4. **算子融合**：将多个连续运算合并为一个复合运算，减少计算图节点数量，降低执行开销。

5. **混合精度计算**：使用低精度浮点数（如 FP16）进行计算，减少内存占用和计算时间，同时保持足够的数值精度。

这些优化策略在现代深度学习框架中已广泛实现，用户无需手动优化，只需调用框架提供的算子即可享受优化效果。

## 实验：前向传播实现与计算图可视化

下面通过代码实现前向传播，并可视化计算图结构。

```python runnable
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    多层神经网络前向传播实现
    
    支持任意层数，任意激活函数
    """
    def __init__(self, layer_sizes, activations):
        """
        Parameters:
        layer_sizes : list of int
            各层神经元数量，如 [2, 4, 3, 1] 表示输入2，隐藏层4和3，输出1
        activations : list of str
            各层激活函数，如 ['sigmoid', 'relu', 'sigmoid']
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.num_layers = len(layer_sizes) - 1  # 不包括输入层
        
        # 初始化权重和偏置
        np.random.seed(42)
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.5
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
        
        # 存储中间结果（用于可视化）
        self.activations_history = []
        self.pre_activations_history = []
    
    def _apply_activation(self, Z, activation_name):
        """应用激活函数"""
        if activation_name == 'sigmoid':
            Z = np.clip(Z, -500, 500)
            return 1 / (1 + np.exp(-Z))
        elif activation_name == 'relu':
            return np.maximum(0, Z)
        elif activation_name == 'tanh':
            return np.tanh(Z)
        elif activation_name == 'linear':
            return Z
        else:
            raise ValueError(f"Unknown activation: {activation_name}")
    
    def forward(self, X):
        """
        前向传播
        
        Parameters:
        X : ndarray, shape (n_input, batch_size)
            输入数据
            
        Returns:
        Y : ndarray, shape (n_output, batch_size)
            输出结果
        """
        # 清空历史记录
        self.activations_history = [X]
        self.pre_activations_history = []
        
        A = X  # 当前激活值
        
        for i in range(self.num_layers):
            # 线性组合
            Z = self.weights[i] @ A + self.biases[i]
            self.pre_activations_history.append(Z)
            
            # 激活变换
            A = self._apply_activation(Z, self.activations[i])
            self.activations_history.append(A)
        
        return A
    
    def get_layer_info(self):
        """获取各层信息"""
        info = []
        for i in range(self.num_layers):
            info.append({
                'layer': i + 1,
                'input_size': self.layer_sizes[i],
                'output_size': self.layer_sizes[i+1],
                'activation': self.activations[i],
                'weight_shape': self.weights[i].shape,
                'bias_shape': self.biases[i].shape
            })
        return info


# 实验1：前向传播计算流程
print("=" * 50)
print("实验1：前向传播计算流程")
print("=" * 50)

# 创建网络：输入2 -> 隐藏层4(ReLU) -> 隐藏层3(Sigmoid) -> 输出1
layer_sizes = [2, 4, 3, 1]
activations = ['relu', 'sigmoid', 'sigmoid']
nn = NeuralNetwork(layer_sizes, activations)

# 单样本输入
X_single = np.array([[0.5], [0.3]])  # 形状 (2, 1)
Y_single = nn.forward(X_single)

print(f"输入: {X_single.T}")
print(f"输出: {Y_single.T}")
print()

# 查看中间结果
print("中间层激活值:")
for i, A in enumerate(nn.activations_history):
    layer_name = "输入" if i == 0 else f"第{i}层输出"
    print(f"  {layer_name}: shape {A.shape}")
print()

for i, Z in enumerate(nn.pre_activations_history):
    print(f"  第{i+1}层预激活值: shape {Z.shape}")

# 实验2：批量前向传播
print("\n" + "=" * 50)
print("实验2：批量前向传播")
print("=" * 50)

# 批量输入（10个样本）
batch_size = 10
X_batch = np.random.randn(2, batch_size)
Y_batch = nn.forward(X_batch)

print(f"批量输入: shape {X_batch.shape} (2特征, {batch_size}样本)")
print(f"批量输出: shape {Y_batch.shape} (1输出, {batch_size}样本)")
print(f"各样本输出值: {Y_batch.flatten()[:5]}...")  # 显示前5个

# 实验3：计算图可视化
print("\n" + "=" * 50)
print("实验3：计算图可视化")
print("=" * 50)

# 绘制计算图结构
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1：网络结构可视化
ax = axes[0]
layer_positions = np.linspace(0, 1, len(layer_sizes))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

# 绘制各层节点
for i, (pos, size) in enumerate(zip(layer_positions, layer_sizes)):
    y_positions = np.linspace(0.1, 0.9, size)
    for y in y_positions:
        circle = plt.Circle((pos, y), 0.05, color=colors[i], alpha=0.7)
        ax.add_patch(circle)
    
    # 标注层信息
    layer_name = "输入层" if i == 0 else f"第{i}层" if i < len(layer_sizes)-1 else "输出层"
    ax.text(pos, 0.95, f"{layer_name}\n({size}神经元)", ha='center', fontsize=10)

# 绘制连接线（简化：只显示部分连接）
for i in range(len(layer_sizes) - 1):
    for j in range(min(3, layer_sizes[i])):
        for k in range(min(3, layer_sizes[i+1])):
            y1 = np.linspace(0.1, 0.9, layer_sizes[i])[j]
            y2 = np.linspace(0.1, 0.9, layer_sizes[i+1])[k]
            ax.plot([layer_positions[i], layer_positions[i+1]], [y1, y2], 
                   'gray', alpha=0.3, linewidth=0.5)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('神经网络结构可视化')

# 图2：计算流程可视化
ax = axes[1]

# 创建计算流程图
steps = ['输入 X', 'W1·X+b1', 'ReLU(Z1)', 'W2·A1+b2', 'Sigmoid(Z2)', 'W3·A2+b3', 'Sigmoid(Z3)', '输出 Y']
step_positions = range(len(steps))
values = [X_single.flatten(), nn.pre_activations_history[0].flatten()[:2], 
          nn.activations_history[1].flatten()[:2], nn.pre_activations_history[1].flatten()[:2],
          nn.activations_history[2].flatten()[:2], nn.pre_activations_history[2].flatten()[:1],
          nn.activations_history[3].flatten()[:1], Y_single.flatten()]

for i, (step, val) in enumerate(zip(steps, values)):
    # 节点
    ax.barh(i, 1, color=colors[i % 4], alpha=0.6, height=0.6)
    ax.text(0.5, i, step, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 数值标注（只显示部分）
    if i < len(values):
        val_str = f"[{val[0]:.2f}...]" if len(val) > 1 else f"{val[0]:.3f}"
        ax.text(1.1, i, val_str, ha='left', va='center', fontsize=9, color='gray')

ax.set_xlim(0, 1.5)
ax.set_ylim(-0.5, len(steps) - 0.5)
ax.set_yticks(range(len(steps)))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlabel('')
ax.set_title('前向传播计算流程')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
plt.close()

# 输出网络信息
print("\n网络结构信息:")
for info in nn.get_layer_info():
    print(f"  第{info['layer']}层: 输入{info['input_size']} → 输出{info['output_size']}, "
          f"激活:{info['activation']}, 权重形状:{info['weight_shape']}")
```

### 实验结论

1. **前向传播流程清晰**：从输入逐层计算，每层包含线性组合和激活变换两个步骤。

2. **批量处理高效**：批量输入矩阵形状 $(n_{input}, batch\_size)$，输出矩阵形状 $(n_{output}, batch\_size)$，一次计算处理多个样本。

3. **计算图直观**：可视化展示了网络结构和计算流程，便于理解和调试。

## 本章小结

本章详细介绍了神经网络前向传播的计算机制，包括信号流动过程、矩阵形式推导、计算图概念和批量计算优化。核心要点如下：

1. **信号流动**：前向传播是逐层计算的过程。每层包含线性组合（$\mathbf{z}^l = \mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l$）和激活变换（$\mathbf{a}^l = f^l(\mathbf{z}^l$)）两个步骤。信号从输入层经过各隐藏层流向输出层。

2. **矩阵形式**：批量处理将多个样本合并为矩阵，利用矩阵运算并行计算。维度检查是验证实现正确性的重要方法。

3. **计算图**：计算图将计算过程图形化表示，便于理解、实现和自动求导。现代深度学习框架都基于计算图构建。

4. **效率优化**：批量处理利用 GPU 并行计算能力，显著提升效率。批量大小是重要超参数，需要平衡效率、稳定性和内存限制。

前向传播是神经网络的基础计算机制。理解前向传播，为后续学习反向传播、梯度下降、网络训练等核心内容奠定基础。下一章将进入第二章"深度神经网络建模"，首先介绍反向传播算法，揭示神经网络如何通过梯度学习自动调整参数。

## 练习题

1. 给定一个三层神经网络，输入维度 $n_0=3$，隐藏层神经元数 $n_1=5$，$n_2=4$，输出维度 $n_3=2$。批量大小 $B=10$。计算各层权重矩阵和预激活矩阵的形状，并进行维度检查验证。
    <details>
    <summary>参考答案</summary>
    
    **各层参数形状**：
    
    - 第 1 层权重矩阵 $\mathbf{W}^1$: $n_1 \times n_0 = 5 \times 3$
    - 第 1 层偏置向量 $\mathbf{b}^1$: $n_1 \times 1 = 5 \times 1$
    - 第 2 层权重矩阵 $\mathbf{W}^2$: $n_2 \times n_1 = 4 \times 5$
    - 第 2 层偏置向量 $\mathbf{b}^2$: $n_2 \times 1 = 4 \times 1$
    - 第 3 层权重矩阵 $\mathbf{W}^3$: $n_3 \times n_2 = 2 \times 4$
    - 第 3 层偏置向量 $\mathbf{b}^3$: $n_3 \times 1 = 2 \times 1$
    
    **各层预激活矩阵形状**（批量大小 $B=10$）：
    
    - 输入矩阵 $\mathbf{X}$（即 $\mathbf{A}^0$）: $n_0 \times B = 3 \times 10$
    
    - 第 1 层预激活 $\mathbf{Z}^1 = \mathbf{W}^1 \mathbf{A}^0 + \mathbf{b}^1$:
      - $\mathbf{W}^1$: $5 \times 3$
      - $\mathbf{A}^0$: $3 \times 10$
      - $\mathbf{W}^1 \mathbf{A}^0$: $5 \times 10$
      - $\mathbf{b}^1$（广播）: $5 \times 10$
      - $\mathbf{Z}^1$: $5 \times 10$
    
    - 第 1 层激活 $\mathbf{A}^1$: $5 \times 10$
    
    - 第 2 层预激活 $\mathbf{Z}^2 = \mathbf{W}^2 \mathbf{A}^1 + \mathbf{b}^2$:
      - $\mathbf{W}^2$: $4 \times 5$
      - $\mathbf{A}^1$: $5 \times 10$
      - $\mathbf{W}^2 \mathbf{A}^1$: $4 \times 10$
      - $\mathbf{Z}^2$: $4 \times 10$
    
    - 第 2 层激活 $\mathbf{A}^2$: $4 \times 10$
    
    - 第 3 层预激活 $\mathbf{Z}^3 = \mathbf{W}^3 \mathbf{A}^2 + \mathbf{b}^3$:
      - $\mathbf{W}^3$: $2 \times 4$
      - $\mathbf{A}^2$: $4 \times 10$
      - $\mathbf{W}^3 \mathbf{A}^2$: $2 \times 10$
      - $\mathbf{Z}^3$: $2 \times 10$
    
    - 第 3 层激活 $\mathbf{A}^3$（输出）: $2 \times 10$
    
    **维度检查总结**：
    
    | 变量 | 形状 | 验证 |
    |:-----|:-----|:-----|
    | $\mathbf{X}$ | $3 \times 10$ | 输入 |
    | $\mathbf{W}^1$ | $5 \times 3$ | $n_1 \times n_0$ ✓ |
    | $\mathbf{Z}^1$ | $5 \times 10$ | $n_1 \times B$ ✓ |
    | $\mathbf{A}^1$ | $5 \times 10$ | $n_1 \times B$ ✓ |
    | $\mathbf{W}^2$ | $4 \times 5$ | $n_2 \times n_1$ ✓ |
    | $\mathbf{Z}^2$ | $4 \times 10$ | $n_2 \times B$ ✓ |
    | $\mathbf{A}^2$ | $4 \times 10$ | $n_2 \times B$ ✓ |
    | $\mathbf{W}^3$ | $2 \times 4$ | $n_3 \times n_2$ ✓ |
    | $\mathbf{Z}^3$ | $2 \times 10$ | $n_3 \times B$ ✓ |
    | $\mathbf{Y}$ | $2 \times 10$ | $n_3 \times B$ ✓ |
    
    所有维度一致，前向传播矩阵运算正确。
    </details>

2. 解释计算图中的"静态图"与"动态图"的区别，并分析各自的优缺点及适用场景。
    <details>
    <summary>参考答案</summary>
    
    **静态图与动态图的区别**：
    
    | 特性 | 静态图 | 动态图 |
    |:-----|:-------|:-------|
    | 构建时机 | 先定义，后执行 | 执行时动态构建 |
    | 执行方式 | 预编译优化后执行 | 解释式执行 |
    | 灵活性 | 低（结构固定） | 高（可动态调整） |
    | 调试难度 | 较高（难以追踪） | 较低（可实时检查） |
    | 优化程度 | 高（可全局优化） | 低（难以预先优化） |
    
    **静态图优点**：
    
    1. **全局优化**：框架可以分析完整计算图，进行算子融合、内存优化、并行调度等优化，提升执行效率。
    
    2. **部署高效**：优化后的计算图适合部署到生产环境，执行稳定高效。
    
    3. **内存效率**：预先分析内存依赖，可以优化内存分配，减少内存占用。
    
    **静态图缺点**：
    
    1. **灵活性低**：计算图结构固定，难以处理条件分支、循环、动态形状等动态结构。
    
    2. **调试困难**：定义和执行分离，难以实时检查中间结果，调试体验较差。
    
    3. **学习曲线陡峭**：用户需要理解计算图概念，编程方式与普通 Python 不同。
    
    **动态图优点**：
    
    1. **灵活性高**：计算图在执行时构建，可以自由使用条件分支、循环等控制流，结构可以随数据变化。
    
    2. **调试便捷**：可以直接使用 Python 调试工具（如 print、断点），实时检查中间结果。
    
    3. **直观易学**：编程方式与普通 Python 一致，学习曲线平滑。
    
    **动态图缺点**：
    
    1. **优化受限**：每次执行都要重新构建计算图，难以进行全局优化，执行效率可能较低。
    
    2. **部署复杂**：动态结构难以直接部署，需要额外的转换步骤。
    
    3. **内存开销**：每次执行重新构建计算图，可能有额外的内存开销。
    
    **适用场景**：
    
    - **静态图适用**：
      - 生产部署：需要高效稳定的执行环境
      - 大规模训练：计算图结构固定，需要极致优化
      - 资源受限环境：需要内存和计算效率最大化
    
    - **动态图适用**：
      - 研究开发：快速实验，需要灵活调整网络结构
      - 复杂模型：包含条件分支、循环等动态结构（如 RNN 变体、强化学习）
      - 调试学习：新手学习神经网络，需要直观的调试体验
    
    **现代趋势**：
    
    现代框架正在融合两种方式：
    - TensorFlow 2.x 默认使用动态图（Eager Execution），但可以通过 `tf.function` 转换为静态图进行优化。
    - PyTorch 默认动态图，但支持 JIT 编译将动态图优化为静态图。
    
    这种融合让用户在开发时享受动态图的灵活性，在生产部署时享受静态图的高效性。
    </details>

3. 解释批量大小对训练效率、梯度稳定性和内存占用的影响。为何大批量训练可能导致"陷入局部最优"？
    <details>
    <summary>参考答案</summary>
    
    **批量大小对训练的影响**：
    
    | 批量大小 | 训练效率 | 梯度稳定性 | 内存占用 |
    |:---------|:---------|:-----------|:---------|
    | 小（16-64） | 低（串行开销大） | 低（噪声大） | 低 |
    | 中（128-512） | 中 | 中 | 中 |
    | 大（1024+） | 高（并行效率高） | 高（平滑稳定） | 高 |
    
    **详细分析**：
    
    **训练效率**：
    - 小批量：每次处理少量样本，GPU 并行能力利用不足，串行开销大，效率低。
    - 大批量：充分利用 GPU 并行能力，矩阵运算规模大，效率高。
    
    **梯度稳定性**：
    - 小批量：基于少量样本计算梯度，梯度噪声大，训练波动剧烈。
    - 大批量：基于大量样本平均梯度，梯度稳定平滑，训练平稳。
    
    **内存占用**：
    - 批量大小与内存占用线性相关。前向传播和反向传播都需要存储批量数据的中间结果。
    - 批量大小过大可能导致 GPU 显存溢出。
    
    **为何大批量导致"陷入局部最优"**：
    
    这个现象有几个解释：
    
    1. **梯度噪声的作用**：
    - 小批量梯度噪声大，相当于在梯度方向上添加随机扰动。这种扰动可能帮助跳出局部最优或平坦区域。
    - 大批量梯度平滑稳定，缺乏这种"随机探索"能力，容易沿着梯度方向稳定收敛到局部最优。
    
    2. **泛化差距**：
    - 研究发现大批量训练的模型往往在训练集上表现更好，但在测试集上表现较差（泛化差距大）。
    - 可能原因：小批量梯度噪声迫使模型学习更"鲁棒"的特征，而非精确拟合训练数据。大批量梯度稳定使模型精确拟合训练数据，可能过拟合。
    
    3. **收敛速度与精度权衡**：
    - 大批量每步梯度更准确，但可能"锁定"在某个方向，难以探索其他区域。
    - 小批量每步梯度噪声大，但可以探索更多区域，可能找到更好的全局最优。
    
    **缓解策略**：
    
    - 使用中等批量大小（如 128-512），平衡效率和稳定性。
    - 大批量训练时使用学习率 warm-up（先小学习率，逐渐增大），避免早期锁定。
    - 使用 SGD 而非 Adaptive 优化器，保留一定梯度噪声。
    - 使用学习率衰减，后期精细化搜索。
    
    **总结**：批量大小是重要超参数，需要在效率、稳定性、泛化能力之间权衡。实践中常用中等批量大小，既保证计算效率，又保留一定的梯度噪声帮助探索。
    </details>

4. 设网络第 $l$ 层使用 ReLU 激活函数 $f(z) = \max(0, z)$。写出 ReLU 函数的导数形式，并说明其在反向传播梯度计算中的作用。
    <details>
    <summary>参考答案</summary>
    
    **ReLU 函数定义**：
    
    $$f(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$
    
    **ReLU 导数**：
    
    $$f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$
    
    注意：ReLU 在 $z=0$ 处不可导，但实践中通常将导数定义为 0 或 1，不影响计算。
    
    **在反向传播中的作用**：
    
    反向传播中，梯度从输出层逐层传回输入层。激活函数的导数决定了梯度如何通过激活层传递。
    
    设第 $l$ 层的预激活值为 $\mathbf{z}^l$，激活值为 $\mathbf{a}^l = f(\mathbf{z}^l)$。反向传播中，从第 $l+1$ 层传来的梯度为 $\frac{\partial L}{\partial \mathbf{a}^l}$。通过激活函数传回第 $l$ 层的梯度为：
    
    $$\frac{\partial L}{\partial \mathbf{z}^l} = \frac{\partial L}{\partial \mathbf{a}^l} \cdot f'(\mathbf{z}^l)$$
    
    对于 ReLU，导数 $f'(\mathbf{z}^l)$ 是一个 0/1 矩阵：
    
    - 当 $\mathbf{z}^l > 0$ 时，导数为 1，梯度完整传递。
    - 当 $\mathbf{z}^l \leq 0$ 时，导数为 0，梯度被截断。
    
    **ReLU 导数的物理意义**：
    
    1. **梯度选择性传递**：ReLU 只传递"激活"神经元（$z>0$）的梯度，"未激活"神经元（$z \leq 0$）不接收梯度。
    
    2. **缓解梯度消失**：相比 Sigmoid（导数最大 0.25），ReLU 激活时导数为 1，梯度不会被衰减。这有效缓解了深层网络的梯度消失问题。
    
    3. **稀疏激活**：ReLU 使部分神经元输出为 0，网络呈现稀疏激活状态。反向传播中，只有激活神经元接收梯度更新，实现了某种"自动特征选择"。
    
    **潜在问题**：
    
    1. **ReLU 死亡问题**：如果神经元始终处于未激活状态（$z \leq 0$），导数始终为 0，梯度无法传递，权重永远不更新。这个神经元"死亡"。
    
    2. **解决方案**：
    - Leaky ReLU：$f(z) = \max(\alpha z, z)$（$\alpha$ 为小正数），未激活时导数为 $\alpha$，避免完全截断。
    - 参数初始化优化：使用 He 初始化，避免初始权重导致大量神经元未激活。
    - 学习率调整：避免学习率过大导致神经元"冲过"激活区域后死亡。
    
    **总结**：ReLU 的导数简单高效（0 或 1），激活时梯度完整传递，不激活时梯度截断。这种特性使 ReLU 成为深度网络的主流激活函数，但也需要注意 ReLU 死亡问题。
    </details>