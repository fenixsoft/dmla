# RNN 基础原理

行文至此，我们现在接触到的所有模型都遵循一个共同模式，模型对每次输入的处理都是独立的，不论是信息分类还是图像生成，本次处理都与上一次输入没有关系，用计算机的术语来讲就是操作都是无状态的。

现实世界中存在另一类数据，其核心特征是当前时刻的数据依赖于之前时刻的数据，数据之间存在时间依赖关系（Temporal Dependency）。这类数据被称为**序列数据**（Sequential Data），文本就是典型的序列数据，"狗 咬 人"这三个字的顺序决定了语义，"人 咬 狗"就完全是另一个意思了。除此之外，语音也是序列，声波随时间变化，前一秒的发音影响后一秒的理解；视频也是序列，每一帧图像有时间顺序；股票价格也是序列，今天的价格与昨天、上周的价格相关；天气预测也是序列，明天的气温与过去一周的温度变化相关。

处理序列数据需要一种具有状态、能够记住过去信息、根据历史做出当前决策的模型架构。1990 年，美国认知科学家杰弗里·埃尔曼（Jeffrey Elman）发表论文《[Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1)》，首次提出了**循环神经网络**（Recurrent Network，后来被称为 Elman Network）。埃尔曼的观点是人类理解语言时并非逐词独立处理，而是持续保持一种上下文状态，读完"猫"后，脑海中保留了"有只猫"的信息，读到"吃鱼"时，立即能关联到前面的"猫"。埃尔曼的论文标题本身就揭示了问题的关键 "Finding Structure in Time" —— 在时间中发现结构。

循环神经网络通过引入**循环连接**（Recurrent Connection）在时间维度上传递信息，让网络拥有记忆的能力，当前时刻的输出不仅取决于当前输入，还取决于之前所有时刻的输入。这个设计思想催生了 LSTM、GRU，注意力机制等关键设计，使神经网络在自然语言处理、语音识别、时间序列预测等领域取得了突破。本章将介绍 RNN 的基本原理、结构设计、训练方法以及主要问题。

## 序列建模

传统网络模型面对序列建模的障碍是与生俱来的，MLP 是全连接结构，所有输入位置天生具备对称性，位置 1 的神经元和位置 3 神经元并没有什么区别。输入数据打乱序列顺序后，MLP 的输出可能完全相同，因此它天然就无法理解时间顺序这个概念。序列数据的位置是时间意义，具有单向性（时刻 $t$ 在时刻 $t+1$ 之前），现有网络无法建模这种时间单向性，需要一种能够逐时刻处理序列、传递历史信息、捕捉时序依赖的新网络架构。

这个新网络架构需要满足四个目标。首先是逐时刻处理，每个时刻处理一个输入，而非一次性处理整个序列，这样可以自然适应变长序列。其次是信息传递，当前时刻能够利用之前时刻的信息，实现记忆能力。第三是变长适应性能，理论上应能够处理任意长度的序列，无需固定输入维度。最后是时序建模，捕捉数据的时间依赖关系，理解顺序的意义。RNN 通过**循环连接**（Recurrent Connection）巧妙地同时满足了这四个目标。循环连接是指网络在时刻 $t$ 的输出，不仅传递给下一层，还传递回网络自身，作为时刻 $t+1$ 的额外输入，这种设计巧妙地将"记忆"机制嵌入到网络结构中，如下图所示。

```nn-arch width=500
name: Recurrent Connection 架构
layout: horizontal

sections:
  - name: 时刻 t = 0
    layers: [x0, h0, y0]
    row_label: "h₀→h₁"
  - name: 时刻 t = 1
    layers: [x1, h1, y1]
    row_label: "h₁→h₂"
  - name: 时刻 t = 2
    layers: [x2, h2, y2]
    row_label: "h₂→h₃"

layers:
  - {id: x0, name: "x₀", type: input, size: "输入信息 0"}
  - {id: h0, name: "RNN h₀", type: rnn, size: 256, act: tanh}
  - {id: y0, name: "y₀", type: output, size: "输出信息 0"}
  - {id: x1, name: "x₁", type: input, size: "输入信息 1"}
  - {id: h1, name: "RNN h₁", type: rnn, size: 256, act: tanh}
  - {id: y1, name: "y₁", type: output, size: "输出信息 1"}
  - {id: x2, name: "x₂", type: input, size: "输入信息 2"}
  - {id: h2, name: "RNN h₂", type: rnn, size: 256, act: tanh}
  - {id: y2, name: "y₂", type: output, size: "输出信息 2"}
```
*图：Recurrent Connection 架构*

图中展示了循环连接的核心机制：$x_t$ 是时刻 $t$ 的输入（如第 $t$ 个词的向量表示），$h_t$ 是时刻 $t$ 的隐藏状态，同时也是时刻 $t+1$ 的另一项输入。网络在时刻 $t$ 的状态 $h_t$ 包含了时刻 $1$ 到 $t$ 所有输入的信息压缩，随着时间推移不断更新、累积信息。箭头表示信息在时间轴上流动的方向，当前隐藏状态流向下一时刻，实现记忆在时间维度上的传递。

### 数学表示

下面将循环连接的设计用数学语言描述出来。设 $h_{t-1}$ 是上一时刻的隐藏状态，即网络在当前操作前的历史记忆；$W_{hh}$ 是循环连接权重矩阵，控制历史信息如何影响当前状态；$x_t$ 是当前时刻的输入向量，即当前要处理的信息；$W_{xh}$ 是输入权重矩阵，控制当前输入如何影响隐藏状态；$b_h$ 是偏置向量，$\sigma$ 是激活函数（通常为 tanh）。那么循环连接计算当前时刻隐藏状态（当前记忆）的公式为：

$$[rnn_hs]h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

即新记忆 = 处理历史记忆 + 处理当前输入。每个时刻的隐藏状态由当前输入和上一时刻的隐藏状态共同决定。再设 $W_{hy}$ 是输出权重矩阵，将隐藏状态映射到输出空间；$b_y$ 是输出偏置，则循环神经网络的输出公式为：

$$y_t = W_{hy} h_t + b_y$$

公式表明网络的输出是由最后时刻的记忆 $h_t$ 所决定的。$h_{1}$ 到 $h_{t-1}$ 都影响着 $h_t$ 的计算，从而实现信息在时间维度上的传递，这条带有顺序关系的信息传递链路正是 RNN 区别于前馈网络的核心所在。

从 $h_{1}$ 到 $h_t$ 的计算不是直接依靠 $t$ 层的神经网络建模来完成的，因为每个序列的长度都不一样，$t$ 是一个不确定数字，无法用确定层数的神经网络来表达，实现上它是依靠同一个网络循环 $t$ 次来完成计算的，这也是"循环连接"与"循环神经网络"名字中"循环"二字的来由。在数学上也体现了这种递归嵌套的循环结构：

- 时刻 1 的隐藏状态 $h_0 = 0$，套公式 {{rnn_hs}} 可得 $h_1 = \sigma(W_{xh} x_1 + b_h)$
- 时刻 2 的隐藏状态嵌套了时刻 1 的信息 $h_2 = \sigma(W_{hh} h_1 + W_{xh} x_2 + b_h) = \sigma(W_{hh} \sigma(W_{xh} x_1 + b_h) + W_{xh} x_2 + b_h)$
- 时刻 3 的隐藏状态继续嵌套了时刻 1、时刻 2 的信息 $h_3 = \sigma(W_{hh} h_2 + W_{xh} x_3 + b_h) = \sigma(W_{hh} \sigma(W_{hh} \sigma(...) + ...) + W_{xh} x_3 + b_h)$
- ……

从 $h_3$ 的计算公式看出 $h_3$ 实际上依赖于 $x_1, x_2, x_3$ 的全部输入信息。信息通过嵌套的激活函数和权重矩阵压缩传递，历史信息被逐步编码到隐藏状态中。可以将隐藏状态 $h_t$ 视作一个包含了时刻 1 到 $t$ 所有输入的信息的压缩函数：

$$h_t = f(x_1, x_2, ..., x_t)$$

很容易证明这个压缩是有损的，因为 $h_t$ 是一个有着确定维度的向量，输入则是不确定的长度，显然 $h_t$ 无法完全无损地存储所有历史信息。RNN 通过训练学会如何有效压缩历史信息，保留对当前任务最有用的部分。将 RNN 信息压缩的行为类比成人类阅读过程更容易理解：$x_t$ 是当前读到的词，$h_t$ 是读完这个词后的理解状态。$h_t$ 不是记住所有历史词汇的逐字背诵，而是记住到目前为止，这段话讲了什么。这个状态会影响对下一个词的理解，譬如读到"猫吃鱼"时，脑海中保留了"猫"的信息，理解"吃"时自然关联到"猫"作为动作主体。

### 架构模式

根据输入输出形式不同，RNN 有多种网络架构模式，可适用于不同的任务场景，具体为：

- **一对一**（One-to-One）：最简单的形式，只有一个时刻的输入和输出，这种模式其实就退化为普通的神经网络，因为根本不需要序列建模能力。
- **一对多**（One-to-Many）：单个输入产生序列输出的模式。典型的应用场景如图像描述生成，输入一张图片的特征向量，输出一句描述文字序列。
- **多对一**（Many-to-One）：序列输入产生单个输出的模式。典型的应用场景如情感分析，输入一句话的词序列，输出情感分类标签；股票预测，输入历史价格序列，输出明天涨跌预测，等等。
- **多对多**（Many-to-Many）：序列输入产生序列输出的模式，输入输出长度相同。典型的应用场景包括视频分类，每一帧输入，每一帧输出分类标签。
- **编码器 - 解码器**（Encoder-Decoder）：输入序列和输出序列长度不同的模式。典型的应用场景包括机器翻译，输入英文句子（5 个词），输出中文翻译（7 个字）。编码器先将输入序列压缩为固定向量，解码器再从该向量生成输出序列。这是 [Seq2Seq](seq2seq.md) 模型的基础架构，将在下一篇文章详细介绍。

## 梯度传递与局限

在 [RNN 的数学表示](#数学表示)中我们分析了隐藏状态在前向传播中的信息传递，要训练好一个网络，还需要有反向传播的路径，即解决梯度如何在时间维度上流动的问题。RNN 的训练算法被称为**时间反向传播**（Backpropagation Through Time, BPTT），从名字可以看出它使用的依然是反向传播算法，只是要在时间维度上进行了额外处理。BPTT 的核心机制是时刻 $t$ 的损失函数 $L_t$ 对时刻 $k$（$k < t$）的参数梯度通过时间链式传播。设损失函数 $L$ 是所有时刻损失的和 $L = \sum_{t=1}^{T} L_t$，它对参数 $W_{hh}$ 的梯度为：

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

根据公式 {{rnn_hs}} 可知，$W_{hh}$ 在每个时刻 $k = 1, 2, ..., T$ 都参与了 $h_k$ 的计算，产生对 $h_k$ 的直接梯度 $\frac{\partial h_k}{\partial W_{hh}}$ 的同时，会通过 {{rnn_hs}} 影响后续时刻，形成间接影响链路 $h_k \rightarrow h_{k+1} \rightarrow ... \rightarrow h_T \rightarrow L_T$。因此 $L_T$ 对 $W_{hh}$ 的总梯度需要累加 $W_{hh}$ 在每个时刻产生的影响：

$$\frac{\partial L_T}{\partial W_{hh}} = \sum_{k=1}^{T} \frac{\partial L_T}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

根据多变量链式法则，$L_T$ 对 $W_{hh}$ 的梯度等于沿所有影响路径的梯度之和：

$$[bptt-eq]\frac{\partial L_T}{\partial W_{hh}} = \sum_{k=1}^{T} \frac{\partial L_T}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

整体公式可以理解为总梯度 = 影响链上的每一步贡献之和。公式中 $\frac{\partial L_T}{\partial h_T}$ 是损失对最终隐藏状态的梯度，表示调整最终状态对减少损失有多大帮助；$\frac{\partial h_T}{\partial h_k}$ 是时刻 $k$ 的隐藏状态对时刻 $T$ 的隐藏状态的梯度，表示早期状态对最终状态有多大影响，这是稍后讨论梯度消失问题的关键项；$\frac{\partial h_k}{\partial W_{hh}}$ 是隐藏状态对参数的梯度，表示调整参数对改变状态有多大帮助。

梯度消失是标准 RNN 训练中面临的首要问题。以杰弗里·埃尔曼论文中的场景为例：网络需要学习句子中相隔较远的两个词之间的关联，如 "The **cat**, which already ate a fish, ... , **was hungry**"。反向传播时，损失信号需要从 "was hungry" 传回到 "cat"，中间跨越多个时刻。即使人类处理远程依赖的能力远优于 RNN，距离过长的依赖关系也会给阅读造成负担。语言模型中的跨从句指代、时间序列预测中的远期事件影响、对话系统中的多轮上下文引用，所有需要长期依赖的任务都会遇到同样的障碍。下面我们从数学角度分析这一现象本质原因，将 BPTT 梯度传播公式 {{bptt-eq}} 的关键项 $\frac{\partial h_T}{\partial h_k}$ 展开后，得到一个链式乘积：

$$\frac{\partial h_T}{\partial h_k} = \frac{\partial h_T}{\partial h_{T-1}} \cdot \frac{\partial h_{T-1}}{\partial h_{T-2}} \cdot ... \cdot \frac{\partial h_{k+1}}{\partial h_k}$$

这个式子每一步的导数计算都要用到一次激活函数的导数，如果激活函数是 [tanh](../../deep-learning/neural-network-structure/activation-loss-functions.md#激活函数)，其导数为 $\tanh'(x) = 1 - \tanh^2(x) \in [0, 1]$，最大值为 1（当输入为 0 时），当输入过大或过小时，导数都趋近于 0。这意味着每当梯度经过一个 tanh 函数，就会被缩小一次。随着连乘项的增加式子也会趋近于 0。这意味着早期时刻（$k$ 较小）对后期时刻（$T$ 较大）的梯度贡献趋近于 0，网络无法有效学习长期依赖关系。例子中梯度从 "was hungry" 传回到 "cat" 时几乎消失殆尽，网络无法更新与 "cat" 相关的参数，也就无法学习到这个长期依赖关系。这正是标准 RNN 在长序列任务上表现不佳的根本原因。

梯度消失对 RNN 的实际应用造成严重限制，语言模型处理 "我出生于北京，...（50 个词）...，所以我的家乡是？" 时，需要记住 50 个词前的 "北京"；股票预测中某公司 30 天前发布财报，今天股价突变，需要关联 30 天前的信息；对话系统中用户 5 分钟前提到的实体，现在需要引用，需要跨多轮对话的记忆。标准 RNN 在这些场景下表现不好，因为梯度很难传递超过 10 个时刻的依赖关系。

梯度问题催生了后续架构的改进。1997 年，德国计算机科学家塞普·霍赫赖特（Sepp Hochreiter）和尤尔根·施密德胡伯（Jürgen Schmidhuber）提出了 **LSTM**（Long Short-Term Memory），引入门控机制选择性保留长期信息。2014 年，韩国学者曹庆铉（Kyunghyun Cho）提出了 **GRU**（Gated Recurrent Unit），简化 LSTM 的门控设计，计算效率更高，这些改进架构将在下一篇文章详细介绍。当然，最根本的革命性变化是彻底推倒 RNN 架构，2015 年，注意力机制的引入提供了另一种与 RNN 完全不同的思路，直接访问任意时刻的信息，绕过梯度传递的限制，这就是下一部分大语言模型中的内容了。

## RNN 序列预测实践

现在通过一个实验验证 RNN 的序列建模能力。实验任务预测正弦波序列的下一个值，输入序列为 $[\sin(0), \sin(0.1), \sin(0.2), ..., \sin(0.9)]$，预测目标是 $\sin(1.0)$。这个任务有两个特点，一是序列有明显的时序依赖，$\sin(t)$ 的值依赖于 $\sin(t-1), \sin(t-2)$ 等历史值，二是历史信息有助于预测，观察到上升或下降趋势后可以推断下一步走向。

实验结果显示训练过程在 50 个 epoch 内稳定收敛，损失函数持续下降，说明 RNN 能够有效学习正弦波的序列规律。测试集上的预测值与真实值接近，验证了模型的泛化能力。RNN 成功利用历史 20 个点的信息预测第 21 个点，这证明了其序列建模能力。

如果使用 MLP 处理相同任务（将序列拼接成向量输入），MLP 的预测误差通常更大。这是因为 MLP 无法有效捕捉时序依赖，它将 20 个点视为独立特征而非有时间顺序的序列。RNN 通过隐藏状态传递，能够理解正弦波的上升或下降趋势，从而做出更准确的预测。

```python runnable
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成正弦波序列数据
def generate_sine_data(num_samples, seq_len):
    """生成正弦波序列数据"""
    X = []
    Y = []
    
    for i in range(num_samples):
        # 随机起始相位
        start = np.random.rand() * 2 * np.pi
        
        # 生成序列
        t = np.linspace(start, start + seq_len * 0.1, seq_len + 1)
        sine_wave = np.sin(t)
        
        # 输入序列（前 seq_len 个点）
        X.append(sine_wave[:-1])
        # 目标（最后一个点）
        Y.append(sine_wave[-1])
    
    return np.array(X), np.array(Y)

# 定义 RNN 模型
class SinRNN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, 
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, hn = self.rnn(x)
        # 取最后时刻
        out = self.fc(out[:, -1, :])
        return out

# 生成数据
num_samples = 1000
seq_len = 20
X, Y = generate_sine_data(num_samples, seq_len)

# 转换为 PyTorch tensor
X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # (N, seq_len, 1)
Y_tensor = torch.FloatTensor(Y).unsqueeze(-1)  # (N, 1)

# 划分训练集和测试集
train_size = 800
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
Y_train, Y_test = Y_tensor[:train_size], Y_tensor[train_size:]

# 创建模型和优化器
model = SinRNN(hidden_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 训练
epochs = 50
train_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# 测试
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, Y_test)
    print(f"\n测试集损失: {test_loss.item():.4f}")

# 可视化预测效果
plt.figure(figsize=(12, 5))

# 子图1: 训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('训练损失曲线')
plt.grid(True, alpha=0.3)

# 子图2: 预测对比
plt.subplot(1, 2, 2)
# 取5个测试样本展示
for i in range(5):
    plt.plot(range(seq_len), X_test[i].numpy().flatten(), 
             'b-', alpha=0.5, label='输入序列' if i==0 else '')
    plt.scatter(seq_len, Y_test[i].numpy().flatten(), 
                c='green', marker='o', s=50, label='真实值' if i==0 else '')
    plt.scatter(seq_len, test_pred[i].numpy().flatten(), 
                c='red', marker='x', s=50, label='预测值' if i==0 else '')

plt.xlabel('时间步')
plt.ylabel('sin(t)')
plt.title('正弦波预测效果')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 本章小结

本文介绍了循环神经网络（RNN）的基本原理。RNN 通过循环连接在时间维度传递信息，实现序列建模能力，其隐藏状态 $h_t$ 是历史信息的压缩表示，随时间不断更新。BPTT 算法在时间维度展开反向传播，可以训练 RNN，但梯度消失问题导致早期时刻的信息难以传递到后期时刻，以至于标准 RNN 的应用场景受限。这催生了 LSTM、GRU 等后续架构改进，它们便是下一篇文章的主题内容。

## 练习题

1. 解释 RNN 为什么通常选择 tanh 作为隐藏状态的激活函数，而非 ReLU 或 Sigmoid。从梯度传播和输出范围两个角度分析。
    <details>
    <summary>参考答案</summary>

    **输出范围角度**：

    tanh 的输出范围为 $(-1, 1)$，以零为中心。相比之下，Sigmoid 输出范围为 $(0, 1)$，始终为正值。RNN 的隐藏状态 $h_t$ 需要作为下一时刻的输入参与 $W_{hh} h_t$ 的计算，如果 $h_t$ 始终为正（Sigmoid 的情况），则 $W_{hh} h_t$ 的各分量只会叠加，不会减损，导致隐藏状态趋向于越来越大的正值，数值不稳定。tanh 的零中心输出允许正负交替，使隐藏状态能在正负之间动态调整，数值更稳定。

    ReLU 的输出范围为 $[0, +\infty)$，同样存在始终非负的问题，且没有上界限制，在循环连接中更容易导致隐藏状态爆炸式增长（尽管有研究提出使用 ReLU 的 RNN 变体，但需要特殊的初始化和归一化技巧才能保持稳定）。

    **梯度传播角度**：

    tanh 在零点附近的导数接近 1（$\tanh'(0) = 1$），这意味着当隐藏状态值较小时，梯度几乎不会被衰减，有利于短距离的信息传递。Sigmoid 在零点的导数仅为 0.25（$\sigma'(0) = 0.25$），每一步梯度就被缩小 4 倍，加剧了梯度消失问题。

    但 tanh 也有局限：当输入较大时，$\tanh'(x) \to 0$，导数趋近于零。在长序列中，多个 tanh 导数连乘仍然会导致梯度消失，这正是标准 RNN 无法学习长期依赖的根本原因，也是 LSTM/GRU 引入门控机制的动机。
    </details>