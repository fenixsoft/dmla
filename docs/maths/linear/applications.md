---
title: "线性代数应用场景"
---

# 线性代数应用场景

线性代数不仅是理论工具，更是解决实际问题的关键。本章将展示线性代数在图像处理、文本分析、特征提取和推荐系统中的具体应用，帮助读者理解数学原理如何指导工程实践。

## 文本数据向量化

计算机无法理解原始文本，它需要数值形式来表达语义信息。文本向量化的核心思想是将文本（如词语、句子、文章）映射为多维向量空间中的点，在映射过程中，让每个维度对应某种语义特征或统计属性。通过向量化，原本抽象的语言信息被转化为可计算的数学对象。这不仅使得文本能够被机器学习模型处理，更重要的是，语义相近的文本在向量空间中也彼此靠近，从而实现相似度比较、聚类分析、检索匹配等关键任务。线性代数为此提供了完整的数学框架 —— 向量运算刻画语义关系，矩阵操作支持批量高效处理，它们是现在动辄百亿、千亿参数语言模型训练时，能处理数以 PB 计语料的关键前提。

### 经典 NLP 的代表：词袋模型

词袋模型（Bag of Words）在 1954 年由泽里格·哈里斯（Zellig Harris）提出，这是一种将文本转化为供机器学习算法处理的数据的算法，通过统计词频来捕捉文档的语义内容，使计算机能够"理解"文本的核心主题和关键词分布，早在 70 年前，线性代数的相关知识就被应用于 NLP 中。

词袋模型的核心思想是忽略文档中词语的顺序和语法结构，将文档视为词汇的"袋子" —— 只关注哪些词出现、出现多少次。它先构建一个包含所有文档词汇的词汇表，然后将每篇文档表示为一个固定长度的向量，向量的每个维度对应词汇表中的一个词，数值代表该词在文档中的出现频率。尽管只通过词频来理解文本主题肯定不够准确，但是这种将文档转化数值矩阵的方法足够简单，可作为当时机器资源受限、今天入门学习的样例，或者文本分类、聚类、相似度计算等的前置任务。

### 现代 NLP 的起点：词嵌入

区别于词袋模型、词频 - 逆文档频率（TF-IDF）这类从词频统计出发来理解语义的技术。2003 年，图灵奖得主约书亚·本吉奥（Yoshua Bengio）在文章《A Neural Probabilistic Language Model》中提出的**词嵌入**（Word Embedding）方法可以说是现代 NLP 技术的起源。以前基于词频统计的算法，任意两个词的 [One-Hot 向量](https://zh.wikipedia.org/wiki/%E7%8B%AC%E7%83%AD) 都是正交的，不同词之间没有任何关联，自然就无法捕捉"国王"与"女王"、"北京"与"中国"这样的语义关联关系。

词嵌入预先从大规模语料中学习，将语义信息压缩到几百维的稠密向量中，使得相似语义的词（如"开心"和"快乐"）在向量空间中距离更近，甚至能呈现有意义的线性关系（譬如："国王"的向量值 - "男人"的向量值 + "女人"的向量值 ≈ "女王"的向量值），这种分布式表示大幅提升了特征表达效率，也成为日后迁移学习的基础，为后续神经网络语言模型铺平了道路，从 Word2Vec、GloVe 到 ELMo、BERT，再到以 GPT 为代表的大语言模型，词嵌入技术不断演进 —— 从静态词向量到上下文相关的动态表示，因此词嵌入被誉为现代 NLP 的第一步。

今天，"词向量"（Word Vector）和"词嵌入"这两个术语常被混用，按原教旨主义来说，"词嵌入"强调的是过程，将离散的词语映射到连续向量空间的一种算法，而"词向量"强调的是结果，即运行词嵌入算法后得到的向量表示。下面这段代码模拟了已有训练好的词向量后，如何用它来表达词语间的关系。后续补全了机器学习的知识后，我们还会设计自己的分词器、预训练词向量。

```python runnable
import numpy as np

# 模拟词向量（实际应用中使用预训练模型）
word_vectors = {
    "机器学习": np.array([0.8, 0.2, 0.1]),
    "深度学习": np.array([0.9, 0.1, 0.2]),
    "人工智能": np.array([0.7, 0.3, 0.3]),
    "自然语言处理": np.array([0.5, 0.8, 0.2]),
    "计算机视觉": np.array([0.4, 0.3, 0.9]),
    "统计机器学习NLP": np.array([0.4, 0.9, 0.1]),
    "足球": np.array([0.1, 0.1, 0.8])
}

def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 计算词之间的相似度
print("词向量相似度：")
print(f"机器学习 vs 深度学习：{cosine_similarity(word_vectors['机器学习'], word_vectors['深度学习']):.3f}")
print(f"机器学习 vs 自然语言处理：{cosine_similarity(word_vectors['机器学习'], word_vectors['自然语言处理']):.3f}")
print(f"机器学习 vs 足球：{cosine_similarity(word_vectors['机器学习'], word_vectors['足球']):.3f}")

# 词向量运算：类比关系
# "国王" - "男人" + "女人" ≈ "女王"
print("\n词向量类比（简化示例）:")
result = word_vectors['机器学习'] - word_vectors['深度学习'] + word_vectors['自然语言处理']
print(f"机器学习 - 深度学习 + 自然语言处理 ≈ 统计机器学习NLP：{result.round(2)}")
```

## 图像数据的表示与操作

人类视觉系统每秒处理约 $10^{10}$ 位信息，而计算机要"看见"世界，必须将连续的光信号转化为离散的数值结构。图像数字化处理的起源可追溯到 1957 年，美国国家标准局的工程师罗素·科尔基（Russell Kirsch）在扫描仪上首次将一张婴儿照片转化为 $176 \times 176$ 的数字矩阵，这张 $5cm \times 5cm$ 的照片被分割成一个个小格子，每个格子记录一个亮度值 —— 这就是历史上第一张数字图像，也标志着图像从"视觉感知"到"数学对象"的根本转变。

一幅灰度图像本质上就是一个 $m \times n$ 的矩阵：矩阵的每个元素对应一个像素点，数值代表该点的亮度（通常 0 表示纯黑，255 表示纯白）。这种矩阵表示使得图像操作转化为矩阵运算 —— 裁剪对应切片、翻转对应行列重排、旋转对应矩阵转置或旋转操作。彩色图像则更进一步，采用红（R）、绿（G）、蓝（B）三个通道叠加来表示丰富的色彩，构成一个 $m \times n \times 3$ 的三维张量：第三个维度上的三个数值共同决定一个像素的最终颜色。这种张量表示不仅统一了灰度和彩色图像的数学描述，更为后续的卷积神经网络（CNN）奠定了基础 —— 从 LeNet-5（1998）识别手写数字，到 AlexNet（2012）突破 ImageNet 分类，再到 ResNet（2015）、Vision Transformer（2020）等现代架构，所有图像深度学习模型的输入层本质上都是处理这种矩阵或张量结构。

- 表示：灰度图像与矩阵

    ```python runnable
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # 创建一个简单的灰度图像矩阵
    image_matrix = np.array([
        [0, 50, 100, 150, 200, 250],
        [50, 100, 150, 200, 250, 200],
        [100, 150, 200, 250, 200, 150],
        [150, 200, 250, 200, 150, 100],
        [200, 250, 200, 150, 100, 50],
        [250, 200, 150, 100, 50, 0]
    ], dtype=np.uint8)

    print(f"图像矩阵形状：{image_matrix.shape}")  # (6, 6)
    print(f"像素值范围：[{image_matrix.min()}, {image_matrix.max()}]")

    # 显示图像
    plt.imshow(image_matrix, cmap='gray')
    plt.colorbar(label='Pixel Value')
    plt.title('Grayscale Image Matrix Visualization')
    plt.show()
    plt.close()
    ```

- 表示：彩色图像与张量

    ```python runnable
    import numpy as np
    import matplotlib.pyplot as plt

    color_image = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],      # Red, Green, Blue
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]], # Yellow, Magenta, Cyan
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]]  # White, Gray, Black
    ], dtype=np.uint8)

    print(f"Color image shape: {color_image.shape}")  # (3, 3, 3)

    # 分离 RGB 通道
    R = color_image[:, :, 0]
    G = color_image[:, :, 1]
    B = color_image[:, :, 2]

    print(f"Red channel:\n{R}")
    print(f"Green channel:\n{G}")
    print(f"Blue channel:\n{B}")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].imshow(color_image)
    axes[0].set_title('Original Color Image')
    axes[0].axis('off')

    axes[1].imshow(R, cmap='Reds')
    axes[1].set_title('Red Channel')
    axes[1].axis('off')

    axes[2].imshow(G, cmap='Greens')
    axes[2].set_title('Green Channel')
    axes[2].axis('off')

    axes[3].imshow(B, cmap='Blues')
    axes[3].set_title('Blue Channel')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()
    ```

- 操作：矩阵不仅能够方便表示图像，通过矩阵运算，可以轻松实现图像的常见操作：

    ```python runnable
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import requests
    from io import BytesIO

    # 加载图片
    response = requests.get("https://icyfenix.cn/images/logo-color.png")
    image_pil = Image.open(BytesIO(response.content))

    # 确保转换为 RGB 格式（PNG 可能带透明通道 RGBA）
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    image = np.array(image_pil)
    print(f"原图形状：{image.shape} (高度, 宽度, RGB通道)")

    # 对图片进行矩阵操作
    cropped = image[25:100, 25:100]               # 1. 裁剪：使用切片
    flipped_h = image[:, ::-1]                    # 2. 水平翻转：列反转
    flipped_v = image[::-1, :]                    # 3. 垂直翻转：行反转
    rotated_90 = np.rot90(image)                  # 4. 旋转 90 度（逆时针）
    rotated_180 = np.rot90(image, 2)              # 5. 旋转 180 度
    transposed = np.transpose(image, (1, 0, 2))   # 6. 转置（对角翻转）

    print(f"裁剪后形状：{cropped.shape}")
    print(f"旋转90度后形状：{rotated_90.shape}")
    print(f"转置后形状：{transposed.shape}")

    # 显示所有操作后的图片
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    titles = ['Original', 'Crop', 'Flip H', 'Flip V', 'Rotate 90°', 'Rotate 180°', 'Transpose']
    images = [image, cropped, flipped_h, flipped_v, rotated_90, rotated_180, transposed]

    for i, (ax, img, title) in enumerate(zip(axes.flat[:7], images, titles)):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    axes.flat[7].axis('off')  # 隐藏最后一个空白格子
    plt.tight_layout()
    plt.show()
    plt.close()
    ```

### 卷积操作的矩阵视角

卷积（Convolution）一词源于数学分析领域，最初描述的是两个函数通过"滑动叠加"产生新函数的运算过程。这一概念在信号处理中早已广泛应用 —— 从音频滤波到图像增强，本质都是让一个信号"穿过"另一个信号，提取或变换特定特征。1980 年，日本学者福岛邦彦（Kunihiko Fukushima）提出的神经认知机（Neocognitron）首次将卷积思想引入神经网络架构；1998 年，图灵奖得主杨立昆（Yann LeCun）发表经典的 LeNet-5 论文，正式确立了卷积神经网络（CNN）的基本范式：用小型矩阵（称为"卷积核"或"滤波器"）在图像上滑动，逐位置计算局部点积，生成特征图。这种局部运算的核心优势在于**参数共享**与**空间不变性** —— 同一个卷积核在图像各处重复使用，大幅减少参数量，同时天然适应图像中目标的平移。从 AlexNet（2012）的突破，到 VGG（2014）、ResNet（2015）的深化，再到今天视觉大模型的底层架构，卷积始终是图像特征提取的核心操作。

从矩阵视角看，卷积操作本质上是一种"局部点积"，即卷积核矩阵与图像局部区域的逐元素相乘再求和。假设有一个 $3 \times 3$ 的卷积核 $\mathbf{K}$ 和一个 $m \times n$ 的图像矩阵 $\mathbf{I}$，卷积在位置 $(i, j)$ 的计算可以表示为：

$$\text{Output}(i, j) = \sum_{p=0}^{2} \sum_{q=0}^{2} \mathbf{K}(p, q) \cdot \mathbf{I}(i+p, j+q)$$

这个公式描述了卷积的核心计算过程：当卷积核 $\mathbf{K}$ 位于图像位置 $(i, j)$ 时，双重求和遍历卷积核的每个元素（索引 $p, q$ 从 0 到 2），将卷积核元素 $\mathbf{K}(p, q)$ 与图像对应位置的像素 $\mathbf{I}(i+p, j+q)$ 相乘，再将所有乘积相加得到输出值。

如果对以代数表达的过程感到困惑，不妨这样直观理解：想象把一个小"窗口"（卷积核）放在图像上，窗口内每个格子都有一个权重，计算窗口覆盖区域内各像素与其权重的乘积之和，并将这一过程在图像上逐位置重复，最终生成一张新的"特征图" —— 其中每个像素值都反映了原图对应位置及其邻域与卷积核的"匹配程度"。不同卷积核的匹配设计会产生不同效果：当设计为边缘检测核（如拉普拉斯算子），输出值大的位置正是图像中边缘所在，突显图像边界；当设计为模糊核（均值滤波）时，输出值则平滑了原图的局部变化，抑制噪声；当设计为锐化核时增强细节，等等。这种"设计卷积核来实现特定图像操作"的思路，原本就是卷积在图像处理领域的传统应用。卷积神经网络作为机器学习中的重要角色之一，其革命性在于卷积核的值不再是人工预设，而是通过反向传播从数据中自动学习，让网络自己发现最有效的特征提取方式。

```python runnable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import requests
from io import BytesIO

# 加载真实图片
response = requests.get("https://icyfenix.cn/images/logo-color.png")
image_pil = Image.open(BytesIO(response.content))

# 转换为灰度图（卷积核通常作用于单通道）
if image_pil.mode != 'L':
    image_gray = image_pil.convert('L')

# 转换为 numpy 数组
image = np.array(image_gray, dtype=np.float32)
print(f"灰度图形状：{image.shape} (高度, 宽度)")
print(f"像素值范围：[{image.min():.1f}, {image.max():.1f}]")

# 边缘检测卷积核（拉普拉斯算子）
# 强调图像中亮度变化剧烈的区域，即边缘位置
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# 模糊卷积核（均值滤波）
# 平滑图像，抑制噪声和细节
blur_kernel = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

# 应用卷积操作
edge_result = convolve2d(image, edge_kernel, mode='same')
blur_result = convolve2d(image, blur_kernel, mode='same')

# 对边缘检测结果取绝对值并归一化（便于可视化）
edge_display = np.abs(edge_result)
edge_display = (edge_display / edge_display.max() * 255).astype(np.uint8)

# 模糊结果直接显示
blur_display = blur_result.astype(np.uint8)

print(f"\n卷积结果形状：{edge_result.shape}")

# 并排展示三张图：原图、边缘检测、模糊
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[0].axis('off')

axes[1].imshow(edge_display, cmap='gray')
axes[1].set_title('Edge Detection (Laplacian)')
axes[1].axis('off')

axes[2].imshow(blur_display, cmap='gray')
axes[2].set_title('Blur (Mean Filter)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
plt.close()
```

### 奇异值分解与数据压缩

奇异值是矩阵分解中产生的非负实数，它们揭示了矩阵的"能量分布"。具体来说，对于任意矩阵 $\mathbf{A}$（不限于方阵），其奇异值 $\sigma_1, \sigma_2, \ldots, \sigma_r$ 定义为矩阵 $\mathbf{A}^T\mathbf{A}$（或 $\mathbf{A}\mathbf{A}^T$）特征值的平方根：$\sigma_i = \sqrt{\lambda_i(\mathbf{A}^T\mathbf{A})}$ 。其中 $\lambda_i$ 是 $\mathbf{A}^T\mathbf{A}$ 的特征值。奇异值总是非负的，习惯上按从大到小排列：$\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r > 0$。

奇异值越大，说明矩阵在该方向上蕴含的信息量或者说"能量"越大；奇异值越小，说明该方向上的信息越微弱，往往对应噪声或次要细节。这一特性使奇异值成为数据压缩、降维和噪声过滤的关键指标 —— 只需保留较大的奇异值，就能在损失可控的前提下大幅减少数据存储量。下面用一个简单例子来演示奇异值的运用：

```python runnable
import numpy as np

# 定义一个简单的矩阵
A = np.array([
    [3, 2, 1],
    [1, 2, 3],
    [2, 1, 2]
])

print("矩阵 A：")
print(A)

# 计算奇异值
U, S, Vt = np.linalg.svd(A)

print(f"\n奇异值：{S.round(4)}")
print(f"奇异值之和（总能量）：{S.sum():.4f}")

# 分析奇异值的信息占比
print("\n奇异值信息占比分析：")
cumulative_energy = 0
for i, sigma in enumerate(S):
    energy_ratio = sigma**2 / (S**2).sum()  # 单个奇异值的能量占比
    cumulative_energy += energy_ratio
    print(f"σ_{i+1} = {sigma:.4f} → 能量占比 {energy_ratio:.2%}，累计 {cumulative_energy:.2%}")
```
从这个例子可以看出：第一个奇异值 $\sigma_1 = 4.89$ 占据了约 53% 的"能量"，前两个奇异值累计已覆盖近 90% 的信息。这意味着，如果我们只保留前两个奇异值，舍弃第三个，就能以约 22% 的存储量（2 个奇异值 vs 3 个）保留矩阵近 90% 的关键信息，这正是数据压缩的核心原理。

**奇异值分解**（Singular Value Decomposition，SVD）是图像压缩的基础，虽然图像矩阵包含大量数据，但其信息往往集中在少数几个主要方向上 —— 大奇异值对应主要特征，小奇异值对应细节噪声。通过保留前 $k$ 个最大奇异值，忽略其余较小的奇异值，可以用远少于原始数据的存储量重建一幅"近似"图像。压缩率取决于保留的奇异值数量 $k$：保留越多，图像质量越接近原图；保留越少，压缩率越高但细节损失越多。这种"保留主要能量、舍弃次要成分"的思路，与人类视觉系统对图像的认知方式天然契合 —— 人眼对图像的整体结构、主要轮廓敏感，而对细微纹理变化相对宽容。

## 特征与降维

机器学习领域有一句广为流传的名言："数据决定了模型的上限，算法只是逼近这个上限"。这句话出自 2009 年 Google 研究员吴恩达（Andrew Ng）在一次机器学习讲座中的总结，它强调了数据质量对模型性能的决定性作用。而特征工程（Feature Engineering）正是提升数据质量的核心手段 —— 它将原始数据转化为更能表达问题本质的特征表示，使模型更容易学习到数据中的规律。

上节的奇异值分解与本节将介绍的特征值、主成分分析等线性代数技术，都是特征工程中降维和特征提取的数学基础。它们帮助我们从数据矩阵中提炼出"主要成分" —— 那些携带最多信息、最能表征数据本质的方向和数值。

### 特征值与特征向量

特征值与特征向量的概念已经有很长的历史，1743 年，欧拉（Leonhard Euler）在研究旋转刚体的运动方程时，首次发现了这种特殊的数学结构 —— 某些方向上的向量在变换后保持方向不变，仅发生伸缩。这一发现当时并未引起广泛关注，但随后在微分方程、振动分析等领域反复出现。1904 年，德国数学家大卫·希尔伯特（David Hilbert）正式引入了"Eigen"这一术语（德语意为"自身的"、"固有的"），用以强调这些值和向量是矩阵固有的内在属性，而非偶然的数值巧合。中文将其译为"特征"，取"特有征象"之意。

从数学定义看，对于 $n \times n$ 方阵 $\mathbf{A}$，如果存在非零向量 $\mathbf{v}$ 和标量 $\lambda$，满足：$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$，则称 $\mathbf{v}$ 为 $\mathbf{A}$ 的**特征向量**，$\lambda$ 为对应的**特征值**。用一个具体例子来理解这个定义。考虑矩阵 $\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$，取向量 $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$。计算 $\mathbf{A}\mathbf{v}$：

$$\mathbf{A}\mathbf{v} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \cdot 1 + 1 \cdot 1 \\ 1 \cdot 1 + 2 \cdot 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix} = 3 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 3\mathbf{v}$$

结果 $\mathbf{A}\mathbf{v} = 3\mathbf{v}$ 正好符合定义的形式！这说明 $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ 是 $\mathbf{A}$ 的特征向量，对应的特征值 $\lambda = 3$。几何上，$\mathbf{v}$ 指向第一象限的 $45°$ 方向，矩阵 $\mathbf{A}$ 在这个方向上的作用只是把向量"拉长"了 3 倍，方向不变。再考虑另一个向量 $\mathbf{u} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$：

$$\mathbf{A}\mathbf{u} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 2 \cdot 1 + 1 \cdot (-1) \\ 1 \cdot 1 + 2 \cdot (-1) \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \end{bmatrix} = 1\mathbf{u}$$

同样满足 $\mathbf{A}\mathbf{u} = 1\mathbf{u}$！$\mathbf{u}$ 也是特征向量，对应特征值 $\lambda = 1$。几何上，$\mathbf{u}$ 指向第四象限的 $-45°$ 方向，矩阵 $\mathbf{A}$ 在这个方向上保持向量长度不变。这个 $2 \times 2$ 矩阵恰好有两个正交的特征方向：$45°$ 方向拉伸 3 倍，$-45°$ 方向保持不变。

![特征向量的几何可视化](./assets/eigenvectors_demo.png)

*图：特征向量的几何可视化*

图中蓝色向量 $\mathbf{v}$ 指向 $45°$ 方向，经矩阵 $\mathbf{A}$ 变换后（红色）被拉伸 3 倍，方向不变；绿色向量 $\mathbf{u}$ 指向 $-45°$ 方向，经变换后（橙色）长度保持不变。这正是特征向量的核心特性：在特征方向上，矩阵变换仅表现为缩放。以上例子揭示了一个深刻的事实：在一般情况下，矩阵作用于向量会产生复杂的变化 —— 既改变方向又改变长度；但在特征向量所指示的特殊方向上，矩阵变换"收敛"为最简单的形式：只进行缩放，方向保持不变。缩放的比例正是特征值 $\lambda$——若 $\lambda > 1$，向量被放大；若 $0 < \lambda < 1$，向量被压缩；若 $\lambda < 0$，方向反转后再缩放。

这一几何直觉在物理和工程中有着丰富的对应：振动系统中，特征向量指向"固有振动模式"的方向，特征值决定振动频率；量子力学里，测量算符的特征值就是可观测物理量的可能取值，特征向量对应各量子态；控制理论中，系统矩阵的特征值分布决定了系统是否稳定 —— 所有特征值位于单位圆内意味着系统收敛，任何一个超出则会导致发散。正是这种"捕捉系统本质行为"的能力，使特征值分解成为降维、压缩、稳定性分析等任务的数学核心。

### 主成分分析

**主成分分析**（Principal Component Analysis，PCA）是特征向量最重要的应用之一，也是降维技术的关键。这一方法由英国统计学家卡尔·皮尔逊（Karl Pearson）在 1901 年首次提出，当时他正在研究如何用"最佳拟合线"来描述数据点的主趋势 —— 这本质上是寻找数据分布中方差最大的方向。1933 年，美国数学家哈罗德·霍特林（Harold Hotelling）在此基础上进行了系统性的数学拓展，正式将特征值分解与数据统计分析联系起来，奠定了现代 PCA 的理论基础。此后，PCA 成为统计学、机器学习、信号处理等领域最广泛使用的降维工具。

PCA 的核心思想是：高维数据往往存在冗余 —— 多个维度之间可能高度相关，真正携带信息的"独立方向"远少于原始维度数。通过找到数据[协方差矩阵](https://en.wikipedia.org/wiki/Covariance_matrix)的特征向量（协方差矩阵是描述多变量数据统计依赖关系的工具，用以量化两个变量"一起变化"的程度，因此协方差矩阵的特征向量即主成分方向），我们可以将数据投影到这些新方向上，实现"以最少维度保留最多信息"的目标。具体而言，协方差矩阵 $\mathbf{C}$ 的特征值 $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_n$ 按大小排列，对应特征向量 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$。第一主成分 $\mathbf{v}_1$ 指向数据方差最大的方向 —— 在这个方向上，数据点分布最"分散"，蕴含最多信息；第二主成分 $\mathbf{v}_2$ 指向方差次大的方向，且与第一主成分正交（垂直），以此类推。每个特征值 $\lambda_i$ 正是该方向上的方差大小：$\lambda_i = \text{Var}($ 投影到 $\mathbf{v}_i$ 方向的数据 $)$ 。

![PCA 的几何视角](./assets/pca_geometric_view.png)

*图：PCA 的几何视角*

从几何视角看，PCA 相当于在数据点构成的"云团"中找到最佳的观察角度。想象三维空间中有一团呈扁平椭圆状的数据点 —— 虽然分布在三维空间，但主要信息其实集中在两个长轴方向上，第三个短轴方向数据变化很小（可能是噪声）。PCA 找到的主成分方向正是这个椭圆的长轴和短轴：投影到长轴方向保留了大部分信息，舍弃短轴方向则损失很小。这一几何特性使 PCA 天然适合数据压缩、可视化（高维数据降到 2D/3D 便于绘图）、噪声过滤（舍弃方差小的方向往往是噪声）、特征提取（为后续机器学习模型提供更精炼的输入）等任务。以下代码展示了主成分分析的过程：

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# 生成三维数据（有相关性）
np.random.seed(42)
n_samples = 200
mean = [0, 0, 0]
cov = [[3, 2, 1], [2, 2, 0.5], [1, 0.5, 1]]  # 三维协方差矩阵
data = np.random.multivariate_normal(mean, cov, n_samples)

print(f"原始数据形状：{data.shape}")

# 手动实现 PCA
# 1. 中心化
data_centered = data - data.mean(axis=0)

# 2. 计算协方差矩阵
cov_matrix = np.cov(data_centered.T)

# 3. 特征分解
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 4. 排序（按特征值降序）
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\n手动计算结果：")
print(f"特征值：{eigenvalues}")
print(f"特征向量：\n{eigenvectors}")

# 计算解释方差比例
explained_variance_ratio = eigenvalues / eigenvalues.sum()
print(f"解释方差比例：{explained_variance_ratio}")

# 降维到二维（投影到前两个主成分）
data_2d = data_centered @ eigenvectors[:, :2]
print(f"\n降维后形状：{data_2d.shape}")

# 可视化
plt.figure(figsize=(12, 5))

# 左图：三维原始数据（投影到XY平面显示）
ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.5)
origin = data.mean(axis=0)
for i, (length, vector) in enumerate(zip(eigenvalues[:2], eigenvectors.T[:2])):
    v = vector * 2 * np.sqrt(length)
    ax1.quiver(origin[0], origin[1], origin[2], v[0], v[1], v[2],
               color=f'C{i+1}', linewidth=2, label=f'PC{i+1}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
ax1.set_title('Original 3D Data and Principal Components')

# 右图：降维后的二维数据
plt.subplot(1, 2, 2)
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Dimensionality Reduced Data (2D)')
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()
plt.close()
```

## 推荐系统

推荐系统是现代互联网服务的核心组件之一，每天都在为数以亿计的用户筛选内容。从数学视角看，推荐系统本质上是预测缺失值的问题。譬如购物商城的用户 - 物品矩阵中存在大量空白 —— 用户尚未接触过的物品，系统需要根据已知信息（历史行为、物品属性、用户特征等）预测这些空白的可能值，并据此排序推荐。这一预测过程恰好落在线性代数的核心范畴：矩阵运算量化用户行为模式、向量相似度刻画偏好相近程度、矩阵分解揭示潜在的因子结构。无论是经典的协同过滤，还是现代的深度学习推荐模型，向量空间的表示与运算始终是理解用户偏好、发现物品关联的关键工具。本节将介绍两种最具代表性的推荐算法 —— 协同过滤与矩阵分解，展示线性代数如何支撑这一影响数十亿人日常生活的技术领域。

### 协同过滤

协同过滤（Collaborative Filtering）是推荐系统的经典算法。这一概念最早由施乐公司的帕洛阿尔托研究中心（Xerox PARC）在 1992 年的论文《Using Collaborative Filtering to Weave an Information Tapestry》中提出，他们设计了一个名为 Tapestry 的邮件过滤系统，允许用户对文档进行标注（如"感兴趣"、"有用"），系统则根据这些标注为其他用户推荐相关内容。这项工作正式确立了"基于用户行为相似性进行推荐"的基本范式 —— 其核心假设是：过去行为相似的用户，未来偏好也倾向于相似。

从数学视角看，协同过滤将推荐问题转化为矩阵运算问题：将用户对物品的评分、点击、购买等行为组织为一个**用户 - 物品矩阵**，矩阵的行代表用户，列代表物品，每个元素记录用户对物品的行为值（如评分）。这种矩阵表示使得"相似性"可以精确量化 —— 用户相似度转化为行向量的距离或夹角（余弦相似度），物品相似度转化为列向量的比较。当需要为某用户推荐物品时，系统先找到与其行为模式最相似的若干用户（称为"邻居"），然后聚合这些邻居对目标物品的评价，预测该用户可能的评分或兴趣程度，最终按预测值排序输出推荐列表。这种"借他人之经验为我所用"的策略，无需理解物品的语义内容，完全依赖行为数据的统计规律，使其天然适用于各类场景，譬如从淘宝的商品推荐到豆瓣的电影推荐，再到网易的音乐推荐，协同过滤始终是推荐系统的基础技术。下面的代码演示了协同过滤的核心流程：构建用户 - 物品评分矩阵、计算用户间的余弦相似度、基于相似用户的评分预测目标用户对未评分物品的兴趣程度。

```python runnable
import numpy as np

# 用户-物品评分矩阵
# 行：用户，列：物品，值：评分（0 表示未评分）
ratings = np.array([
    [5, 3, 0, 1, 0, 0],  # 用户 1
    [4, 0, 0, 1, 0, 2],  # 用户 2
    [1, 1, 0, 5, 4, 0],  # 用户 3
    [0, 0, 0, 4, 0, 5],  # 用户 4
    [0, 1, 5, 4, 0, 0],  # 用户 5
])

print("用户-物品评分矩阵：")
print(ratings)
print(f"矩阵形状：{ratings.shape}")

# 用户相似度（余弦相似度）
def cosine_similarity_matrix(matrix):
    """计算矩阵行之间的余弦相似度"""
    # 归一化
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    normalized = matrix / norms
    # 计算相似度
    return normalized @ normalized.T

user_sim = cosine_similarity_matrix(ratings)
print("\n 用户相似度矩阵：")
print(user_sim.round(3))
```

### 矩阵分解

前面的用户相似度矩阵方法通过直接比较用户之间的评分行为来寻找相似用户。矩阵分解则采用了另一种思路：将用户和物品映射到一个低维的隐因子空间，这是一个抽象的特征空间，其中的每个维度代表一个潜在的、不可直接观测的"因子"。例如在电影推荐中，这些隐因子可能对应"动作程度"、"浪漫程度"、"喜剧元素"等抽象特征 —— 虽然我们无法直接从数据中识别这些因子的具体含义，但算法会自动学习它们。每个用户在这个空间中有一个向量表示其对各因子的偏好程度，每个物品也有一个向量表示其在各因子上的属性强度。用户对物品的偏好程度，就可以通过这两个向量的内积来预测。矩阵分解的目标是找到两个低秩矩阵 $P$（用户矩阵）和 $Q$（物品矩阵），使得它们的乘积尽可能接近原始评分矩阵：

$$R \approx P \cdot Q$$

其中每个用户和物品都被表示为一个隐因子向量，用户对物品的预测评分等于两个向量的内积。这种方法具有如下优势：

- **降维压缩**：将高维稀疏矩阵压缩为低秩表示，减少存储和计算开销
- **发现隐含特征**：自动学习用户偏好和物品属性的潜在维度（如电影类型、用户口味倾向）
- **预测缺失值**：为用户未评分的物品生成预测评分，实现个性化推荐

```python runnable
import numpy as np
from scipy.sparse.linalg import svds

# 用户-物品评分矩阵（与前一节相同）
ratings = np.array([
    [5, 3, 0, 1, 0, 0],  # 用户 1
    [4, 0, 0, 1, 0, 2],  # 用户 2
    [1, 1, 0, 5, 4, 0],  # 用户 3
    [0, 0, 0, 4, 0, 5],  # 用户 4
    [0, 1, 5, 4, 0, 0],  # 用户 5
])

# 使用 SVD 进行矩阵分解
# 首先填充缺失值（用均值）
ratings_filled = ratings.copy()
ratings_filled[ratings_filled == 0] = ratings[ratings > 0].mean()

# SVD 分解
k = 2  # 隐因子数量
U, sigma, Vt = svds(ratings_filled, k=k)

print(f"U （用户矩阵） 形状：{U.shape}")
print(f"sigma （奇异值） 形状：{sigma.shape}")
print(f"Vt （物品矩阵） 形状：{Vt.shape}")

# 重构评分矩阵
predicted_ratings = U @ np.diag(sigma) @ Vt

print("\n 预测评分矩阵：")
print(predicted_ratings.round(2))

# 推荐给用户 1 的物品
user1_predicted = predicted_ratings[0]
user1_actual = ratings[0]
# 找出用户 1 未评分但预测评分高的物品
recommend_items = np.where(user1_actual == 0)[0]
recommended = sorted(recommend_items, key=lambda x: user1_predicted[x], reverse=True)
print(f"\n 推荐给用户 1 的物品：{recommended}")
```

## 本章小结

本章展示了线性代数从理论工具走向工程实践路径的一角。无论是文本、图像、推荐系统还是数据压缩，这些应用都印证了一个观点：线性代数不仅是抽象的数学理论，更是连接数据与算法的桥梁，为机器学习提供了从数据表示、特征工程到模型训练的完整数学框架。
