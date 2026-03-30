---
title: "线性代数应用场景"
---

# 线性代数应用场景

线性代数不仅是理论工具，更是解决实际问题的关键。本章将展示线性代数在图像处理、文本分析、特征提取和推荐系统中的具体应用，帮助读者理解数学原理如何指导工程实践。

## 文本数据向量化

计算机无法理解原始文本，它需要数值形式来表达语义信息。文本向量化的核心思想是将文本（如词语、句子、文章）映射为多维向量空间中的点，在映射过程中，让每个维度对应某种语义特征或统计属性。通过向量化，原本抽象的语言信息被转化为可计算的数学对象。这不仅使得文本能够被机器学习模型处理，更重要的是，语义相近的文本在向量空间中也彼此靠近，从而实现相似度比较、聚类分析、检索匹配等关键任务。线性代数为此提供了完整的数学框架——向量运算刻画语义关系，矩阵操作支持批量高效处理，它们是现在动辄百亿、千亿参数语言模型训练时，能处理数以 PB 计语料的关键前提。

### 经典 NLP 的代表：词袋模型

词袋模型（Bag of Words）在 1954 年由泽里格·哈里斯（Zellig Harris）提出，这是一种将文本转化为供机器学习算法处理的数据的算法，通过统计词频来捕捉文档的语义内容，使计算机能够"理解"文本的核心主题和关键词分布，早在 70 年前，线性代数的相关知识就被应用于 NLP 中。

词袋模型的核心思想是忽略文档中词语的顺序和语法结构，将文档视为词汇的"袋子"——只关注哪些词出现、出现多少次。它先构建一个包含所有文档词汇的词汇表，然后将每篇文档表示为一个固定长度的向量，向量的每个维度对应词汇表中的一个词，数值代表该词在文档中的出现频率。尽管只通过词频来理解文本主题肯定不够准确，但是这种将文档转化数值矩阵的方法足够简单，可作为当时机器资源受限、今天入门学习的样例，或者文本分类、聚类、相似度计算等的前置任务。

### 现代 NLP 的起点：词嵌入

区别于词袋模型、词频-逆文档频率（TF-IDF）这类从词频统计出发来理解语义的技术。2003 年，图灵奖得主约书亚·本吉奥（Yoshua Bengio）在文章《A Neural Probabilistic Language Model》中提出的**词嵌入（Word Embedding）**方法可以说是现代 NLP 技术的起源。以前基于词频统计的算法，任意两个词的 [One-Hot 向量](https://zh.wikipedia.org/wiki/%E7%8B%AC%E7%83%AD) 都是正交的，不同词之间没有任何关联，自然就无法捕捉"国王"与"女王"、"北京"与"中国"这样的语义关联关系。

词嵌入预先从大规模语料中学习，将语义信息压缩到几百维的稠密向量中，使得相似语义的词（如"开心"和"快乐"）在向量空间中距离更近，甚至能呈现有意义的线性关系（譬如："国王"的向量值 - "男人"的向量值 + "女人"的向量值 ≈ "女王"的向量值），这种分布式表示大幅提升了特征表达效率，也成为日后迁移学习的基础，为后续神经网络语言模型铺平了道路，从 Word2Vec、GloVe 到 ELMo、BERT，再到以 GPT 为代表的大语言模型，词嵌入技术不断演进——从静态词向量到上下文相关的动态表示，因此词嵌入被誉为现代 NLP 的第一步。

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

## 图像数据张量表示与操作

人类视觉系统每秒处理约 10^10 位信息，而计算机要"看见"世界，必须将连续的光信号转化为离散的数值结构。图像数字化处理的起源可追溯到 1957 年，美国国家标准局的工程师罗素·科尔基（Russell Kirsch）在扫描仪上首次将一张婴儿照片转化为 176×176 的数字矩阵，这张 5cm×5cm 的照片被分割成一个个小格子，每个格子记录一个亮度值——这就是历史上第一张数字图像，也标志着图像从"视觉感知"到"数学对象"的根本转变。

一幅灰度图像本质上就是一个 $m \times n$ 的矩阵：矩阵的每个元素对应一个像素点，数值代表该点的亮度（通常 0 表示纯黑，255 表示纯白）。这种矩阵表示使得图像操作转化为矩阵运算——裁剪对应切片、翻转对应行列重排、旋转对应矩阵转置或旋转操作。彩色图像则更进一步，采用红（R）、绿（G）、蓝（B）三个通道叠加来表示丰富的色彩，构成一个 $m \times n \times 3$ 的三维张量：第三个维度上的三个数值共同决定一个像素的最终颜色。这种张量表示不仅统一了灰度和彩色图像的数学描述，更为后续的卷积神经网络（CNN）奠定了基础——从 LeNet-5（1998）识别手写数字，到 AlexNet（2012）突破 ImageNet 分类，再到 ResNet（2015）、Vision Transformer（2020）等现代架构，所有图像深度学习模型的输入层本质上都是处理这种矩阵或张量结构。

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

卷积（Convolution）一词源于数学分析领域，最初描述的是两个函数通过"滑动叠加"产生新函数的运算过程。这一概念在信号处理中早已广泛应用——从音频滤波到图像增强，本质都是让一个信号"穿过"另一个信号，提取或变换特定特征。1980 年，日本学者福岛邦彦（Kunihiko Fukushima）提出的神经认知机（Neocognitron）首次将卷积思想引入神经网络架构；1998 年，图灵奖得主杨立昆（Yann LeCun）发表经典的 LeNet-5 论文，正式确立了卷积神经网络（CNN）的基本范式：用小型矩阵（称为"卷积核"或"滤波器"）在图像上滑动，逐位置计算局部点积，生成特征图。这种局部运算的核心优势在于**参数共享**与**空间不变性**——同一个卷积核在图像各处重复使用，大幅减少参数量，同时天然适应图像中目标的平移。从 AlexNet（2012）的突破，到 VGG（2014）、ResNet（2015）的深化，再到今天视觉大模型的底层架构，卷积始终是图像特征提取的核心操作。

从矩阵视角看，卷积操作本质上是卷积核矩阵与图像局部区域的逐元素相乘再求和——一种"局部点积"。假设有一个 $3 \times 3$ 的卷积核 $\mathbf{K}$ 和一个 $m \times n$ 的图像矩阵 $\mathbf{I}$，卷积在位置 $(i, j)$ 的计算可以表示为：

$$\text{Output}(i, j) = \sum_{p=0}^{2} \sum_{q=0}^{2} \mathbf{K}(p, q) \cdot \mathbf{I}(i+p, j+q)$$

这个公式描述了卷积的核心计算过程：当卷积核 $\mathbf{K}$ 位于图像位置 $(i, j)$ 时，双重求和遍历卷积核的每个元素（索引 $p, q$ 从 0 到 2），将卷积核元素 $\mathbf{K}(p, q)$ 与图像对应位置的像素 $\mathbf{I}(i+p, j+q)$ 相乘，再将所有乘积相加得到输出值。

如果对以代数表达的过程感到困惑，不妨这样直观理解：想象把一个小"窗口"（卷积核）放在图像上，窗口内每个格子都有一个权重，计算窗口覆盖区域内各像素与其权重的乘积之和，并将这一过程在图像上逐位置重复，最终生成一张新的"特征图"——其中每个像素值都反映了原图对应位置及其邻域与卷积核的"匹配程度"。不同卷积核的匹配设计会产生不同效果：当设计为边缘检测核（如拉普拉斯算子），输出值大的位置正是图像中边缘所在，突显图像边界；当设计为模糊核（均值滤波）时，输出值则平滑了原图的局部变化，抑制噪声；当设计为锐化核时增强细节，等等。这种"设计卷积核来实现特定图像操作"的思路，原本就是卷积在图像处理领域的传统应用。卷积神经网络作为机器学习中的重要角色之一，其革命性在于卷积核的值不再是人工预设，而是通过反向传播从数据中自动学习，让网络自己发现最有效的特征提取方式。

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

### 图像压缩与奇异值分解

矩阵分解技术为图像压缩提供了数学基础。奇异值分解（Singular Value Decomposition，SVD）最早由数学家欧拉（Euler）在 1846 年对二次型的研究中奠定基础，19 世纪 70 年代，贝尔特拉米（Beltrami）和若尔当（Jordan）分别独立发表了相关论文，确立了这一方法的理论框架。SVD 的核心思想是将任意矩阵 $\mathbf{A}$ 分解为三个矩阵的乘积：$\mathbf{A} = \mathbf{U}\mathbf{S}\mathbf{V}^T$，其中 $\mathbf{U}$ 和 $\mathbf{V}$ 是正交矩阵，$\mathbf{S}$ 是对角矩阵，对角元素（奇异值）按从大到小排列，代表矩阵各"方向"上的能量分布。

图像压缩的关键洞察在于：虽然图像矩阵包含大量数据，但其信息往往集中在少数几个主要方向上——大奇异值对应主要特征，小奇异值对应细节噪声。通过保留前 $k$ 个最大奇异值，忽略其余较小的奇异值，可以用远少于原始数据的存储量重建一幅"近似"图像。压缩率取决于保留的奇异值数量 $k$：保留越多，图像质量越接近原图；保留越少，压缩率越高但细节损失越多。这种"保留主要能量、舍弃次要成分"的思路，与人类视觉系统对图像的认知方式天然契合——人眼对图像的整体结构、主要轮廓敏感，而对细微纹理变化相对宽容。

```python runnable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 创建示例图像
image = np.random.rand(200, 200)

# SVD 分解
U, S, Vt = np.linalg.svd(image)

print(f"原始图像形状：{image.shape}")
print(f"奇异值数量：{len(S)}")

# 使用前 k 个奇异值重构图像
def compress_image(U, S, Vt, k):
    """使用前 k 个奇异值重构图像"""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# 不同压缩率
ks = [5, 20, 50, 100]
print("\n 压缩率分析：")
for k in ks:
    compressed = compress_image(U, S, Vt, k)
    error = np.linalg.norm(image - compressed, 'fro')
    compression_ratio = k * (200 + 200 + 1) / (200 * 200)
    print(f"k={k}: 压缩率={compression_ratio:.2%}, 重构误差={error:.4f}")
```

## 特征工程

### 特征值与特征向量

对于方阵 $\mathbf{A}$，如果存在非零向量 $\mathbf{v}$ 和标量 $\lambda$，使得：

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

则 $\mathbf{v}$ 称为 $\mathbf{A}$ 的**特征向量**，$\lambda$ 称为对应的**特征值**。

特征向量的几何意义：在这个方向上，矩阵变换只进行缩放，不改变方向。

```python
import numpy as np

# 定义矩阵
A = np.array([
    [4, 1],
    [2, 3]
])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("矩阵 A:\n", A)
print("\n 特征值：", eigenvalues)
print("特征向量（每列一个）:\n", eigenvectors)

# 验证：A @ v = λ * v
print("\n 验证特征向量：")
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i]
    Av = A @ v
    lambda_times_v = lambda_v * v
    print(f"特征值 {lambda_v:.2f}: A @ v ≈ λ * v? {np.allclose(Av, lambda_times_v)}")
```

### 主成分分析（PCA）

**PCA** 是特征向量最重要的应用之一。它通过找到数据方差最大的方向来实现降维。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 生成二维数据（有相关性）
np.random.seed(42)
n_samples = 200
mean = [0, 0]
cov = [[3, 2], [2, 2]]  # 协方差矩阵
data = np.random.multivariate_normal(mean, cov, n_samples)

print(f"原始数据形状：{data.shape}")

# 方法一：使用 sklearn
pca = PCA(n_components=2)
pca.fit(data)

print(f"\n 主成分方向（特征向量）:\n{pca.components_}")
print(f"解释方差比例：{pca.explained_variance_ratio_}")
print(f"特征值：{pca.explained_variance_}")

# 方法二：手动实现 PCA
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

print(f"\n 手动计算结果：")
print(f"特征值：{eigenvalues}")
print(f"特征向量：\n{eigenvectors}")

# 降维到一维
pca_1d = PCA(n_components=1)
data_1d = pca_1d.fit_transform(data)
print(f"\n 降维后形状：{data_1d.shape}")

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
origin = data.mean(axis=0)
for i, (length, vector) in enumerate(zip(pca.explained_variance_, pca.components_)):
    v = vector * 2 * np.sqrt(length)
    plt.arrow(origin[0], origin[1], v[0], v[1], head_width=0.3, head_length=0.2,
              fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2, label=f'PC{i+1}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('原始数据与主成分方向')
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(data_1d, np.zeros_like(data_1d), alpha=0.5)
plt.xlabel('主成分')
plt.title('降维后数据（一维）')
plt.grid(True)

plt.tight_layout()
plt.savefig('pca_demo.png', dpi=150)
plt.close()
print("PCA 演示图已保存")
```

### 特征脸简介

特征脸（Eigenface）是人脸识别的经典方法，将人脸图像表示为"特征脸"的线性组合：

```python
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载人脸数据集（如果网络不可用，可用随机数据模拟）
try:
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data  # (400, 4096) - 400 张 64x64 的人脸图像
except:
    # 模拟数据
    X = np.random.rand(100, 4096)

print(f"人脸数据形状：{X.shape}")  # （样本数，像素数）

# PCA 降维
n_components = 50
pca = PCA(n_components=n_components, whiten=True)
X_pca = pca.fit_transform(X)

print(f"降维后形状：{X_pca.shape}")
print(f"累计解释方差：{sum(pca.explained_variance_ratio_):.2%}")

# 可视化前几个"特征脸"
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    if i < n_components:
        ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
        ax.set_title(f'特征脸 {i+1}')
    ax.axis('off')
plt.suptitle('特征脸（Eigenfaces）')
plt.tight_layout()
plt.savefig('eigenfaces.png', dpi=150)
plt.close()
print("特征脸图已保存")
```

## 推荐系统

### 协同过滤

协同过滤是推荐系统的经典方法，基于用户行为相似性进行推荐。

```python
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

### 矩阵分解技术

矩阵分解将评分矩阵分解为用户矩阵和物品矩阵的乘积：

```python
import numpy as np
from scipy.sparse.linalg import svds

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

### 低秩近似

低秩近似用于数据压缩和降噪：

```python
import numpy as np

# 创建一个低秩矩阵（加噪声）
np.random.seed(42)
U_true = np.random.randn(10, 3)
V_true = np.random.randn(3, 5)
matrix = U_true @ V_true + 0.1 * np.random.randn(10, 5)

print(f"原始矩阵形状：{matrix.shape}")

# SVD 分解
U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

print(f"\n 奇异值：{S.round(3)}")

# 低秩近似
def low_rank_approximation(U, S, Vt, k):
    """保留前 k 个奇异值的低秩近似"""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# 不同秩的近似
print("\n 低秩近似误差：")
for k in [1, 2, 3, 5]:
    approx = low_rank_approximation(U, S, Vt, k)
    error = np.linalg.norm(matrix - approx, 'fro')
    print(f"秩 {k}: 重构误差 = {error:.4f}")
```

## 矩阵运算实践

矩阵分解是将矩阵表示为若干简单矩阵乘积的方法，在降维、特征提取、数值计算等领域有重要应用。NumPy 的 `np.linalg` 模块提供了常用的分解方法。

**分解类型：**

- **特征值分解（Eigendecomposition）**：将方阵分解为特征值和特征向量，适用于对称矩阵分析
- **奇异值分解（SVD）**：将任意矩阵分解为三个矩阵的乘积，是数据压缩和降维的基础
- **低秩近似**：通过保留前 k 个奇异值，用低秩矩阵近似原矩阵，实现数据压缩

```python runnable
import numpy as np

# 创建对称正定矩阵
A = np.random.rand(5, 5)
A = A @ A.T + np.eye(5) * 0.1  # 确保正定

# 1. 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值分解：")
print(f"特征值：{eigenvalues.round(4)}")
print(f"特征向量形状：{eigenvectors.shape}")

# 验证：A = V D V^-1
reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
print(f"重构误差：{np.linalg.norm(A - reconstructed):.6f}")

# 2. 奇异值分解（SVD）
U, S, Vt = np.linalg.svd(A)
print("\n 奇异值分解：")
print(f"U 形状：{U.shape}")
print(f"奇异值：{S.round(4)}")
print(f"Vt 形状：{Vt.shape}")

# 验证：A = U S Vt
reconstructed = U @ np.diag(S) @ Vt
print(f"重构误差：{np.linalg.norm(A - reconstructed):.6f}")

# 低秩近似
k = 3  # 保留前 3 个奇异值
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"\n 低秩近似误差 (k={k}): {np.linalg.norm(A - A_approx):.6f}")
```

### 线性方程组求解

求解线性方程组 `Ax = b` 是科学计算的基本问题。NumPy 提供了多种求解方法，适用于不同场景。

**求解方法：**

- **`np.linalg.solve`**：直接求解方阵方程组，数值稳定，是首选方法
- **逆矩阵法**：通过 `np.linalg.inv(A) @ b` 求解，计算逆矩阵成本高且数值不稳定，不推荐
- **`np.linalg.lstsq`**：最小二乘法，用于超定方程组（方程数多于未知数），寻找最优近似解

```python runnable
import numpy as np

# 求解 Ax = b
A = np.array([
    [3, 1, -1],
    [2, 4, 1],
    [-1, 2, 5]
])
b = np.array([4, 1, 1])

# 方法一：np.linalg.solve
x = np.linalg.solve(A, b)
print(f"解：{x}")
print(f"验证 Ax = b: {np.allclose(A @ x, b)}")

# 方法二：使用逆矩阵（不推荐，数值不稳定）
x2 = np.linalg.inv(A) @ b
print(f"逆矩阵法：{x2}")

# 方法三：最小二乘法（超定方程组）
A_over = np.random.rand(5, 3)
b_over = np.random.rand(5)
x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"最小二乘解：{x_lstsq}")
```

---

## 本章小结

本章展示了线性代数在机器学习中的实际应用：

1. **图像处理**：图像即矩阵，矩阵运算实现图像操作，SVD 实现压缩
2. **文本分析**：文本向量化（词袋、TF-IDF、词向量），余弦相似度计算
3. **特征提取**：特征值分解、PCA 降维、特征脸
4. **推荐系统**：协同过滤、矩阵分解、低秩近似

这些应用展示了线性代数不仅是理论工具，更是解决实际问题的基础。

下一章将回顾全文要点，并提供思考题和扩展阅读。