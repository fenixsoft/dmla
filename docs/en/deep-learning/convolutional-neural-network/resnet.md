# ResNet Residual Network

As we progress through the deep learning chapters, let's consider a question: can the depth of neural networks be increased indefinitely? VGG's experimental results showed that network depth can indeed improve accuracy. If depth can be increased without limit, does it mean that, as long as hardware computing power continues to advance, we could theoretically achieve networks of arbitrary accuracy even without better algorithms or model architectures? This sounds promising, but researchers quickly ran into problems in practice. Between 2014 and 2015, many teams attempted to train networks deeper than VGG-19 (19 layers) and GoogLeNet (22 layers), only to find that once depth exceeded a certain threshold (around 20-30 layers), accuracy stopped improving and actually declined. More puzzling was that this phenomenon was not caused by overfitting — overfitting typically manifests as decreasing training error alongside increasing test error, but here both training and test errors increased. This indicated that the network genuinely could not learn the patterns in the training data — it had truly 'hit a learning ceiling.'

In 2015, Kaiming He, then a researcher at Microsoft Research, proposed **ResNet** (Residual Network), which completely solved this problem. ResNet introduced **residual connections** (also called skip connections), enabling networks to train 150 layers or even 1000+ layers, while reducing ImageNet's Top-5 error rate to 3.57% — the first time a machine surpassed human-level visual performance (approximately 5.1%). The work was published as the paper "[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)" at CVPR 2016, winning the Best Paper Award that year. It has since become one of the most cited papers in deep learning history, with over 150,000 citations to date.

## Residual Learning Concept

Before diving into ResNet, let's understand the counterintuitive phenomenon mentioned earlier: as networks get deeper, accuracy doesn't just slow in its improvement — it hits a turning point and declines. This was not predicted by theory but was consistently observed in practical training.

He's team conducted comparative experiments in their paper: using the same dataset and training strategy, they trained a 20-layer network and a 56-layer network. The 56-layer network had higher error rates on both training and test sets compared to the 20-layer version. Intuitively, if a 20-layer network already learns well enough, adding 36 more layers (for a total of 56) should allow the new layers to learn to do nothing — i.e., identity mapping — so the 56-layer network should perform at least as well as the 20-layer version. But in practice, getting those 36 layers to learn a do-nothing identity mapping proved difficult. Convolutional layers are typically initialized with random small values, and during forward propagation, the input signal changes significantly after passing through multiple convolutional layers. Learning to keep the input unchanged requires precisely adjusting millions of parameters so that their combined effect equals identity mapping — an extremely challenging optimization task. This phenomenon, where both training and test errors increase along with network depth, is now known as the **degradation problem**.

ResNet's key innovation was changing what the network learns. In standard CNNs, each convolutional layer aims to learn a mapping function $H(x)$ from input features to output features from scratch. When the optimal solution is close to identity mapping $H(x) = x$, the network still needs to learn complex parameter combinations to express $x$ to $x$ from scratch. ResNet changed the learning target to the residual function $F(x) = H(x) - x$, making the model learn how much correction is needed beyond the identity mapping to arrive at the optimal mapping function.

This simple change effectively solved the difficulty of learning identity mappings. When the optimal mapping is exactly identity, ResNet only needs to learn $F(x) = 0$, making all weights tend toward zero. Since weights are typically initialized as small values near zero, the network starts close to the optimal solution and optimization converges easily. When the optimal mapping is close to but not exactly identity, the residual function only needs to learn small adjustments $F(x) \approx 0$, rather than the complete mapping function, which also significantly reduces optimization difficulty.

An analogy helps understand the idea behind residual learning. Imagine trying to copy a painting from a blank canvas (standard CNN learning identity mapping) — you need precise control over every stroke, color, and pressure, which is extremely difficult. But if you only need to add a few strokes to an almost-finished painting (ResNet learning residuals), the difficulty is greatly reduced. This is the essence of residual learning: not learning the complete target, but learning the difference between the target and a baseline. The mathematical expression of a residual block clearly illustrates this idea. Let $x$ be the input to the residual block, also the baseline signal, with a set of weights $W_i$ and a residual function $F$ consisting of several convolutional layers. $F$ learns how much correction is needed on top of $x$ to achieve the optimal result $y$. The formula is: output = baseline signal + correction:

$$[res_y] y = x + F(x, \{W_i\})$$

When the residual function learns $F(x) = 0$ (all convolutional weights approach zero), the output equals the input $y = x$, achieving identity mapping. This ensures that deeper networks are at least not worse than shallower ones — even if newly added layers learn nothing useful, they can at least maintain the input by learning $F(x) = 0$. The advantage of residual connections can also be understood from the perspective of gradient flow. During backpropagation, gradients can be directly transmitted from deep to shallow layers through the residual connection. Plugging formula {{res_y}} into the gradient formula gives:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$

The meaning of this formula is clear: input gradient = output gradient × (residual path gradient + shortcut path gradient). Even if $\frac{\partial F}{\partial x}$ approaches zero (the residual function suffers from vanishing gradients), $\frac{\partial L}{\partial x}$ still equals $\frac{\partial L}{\partial y}$, with gradients propagating through the $+1$ term of the residual connection without attenuation. This fundamentally solves the gradient flow problem in deep networks.

## Residual Block

There are two ways to implement residual connections, depending on whether the input and output dimensions match. The design presented in the ResNet paper is elegantly simple, achieving maximum effect with minimal changes. When the input $x$ and the residual output $F(x)$ have the same dimensions (same number of channels and spatial size), the residual connection simply adds the input element-wise. This is the simplest implementation and the type used in most residual blocks in ResNet, as shown in the diagram below:

```nn-arch width=620
name: Basic Residual Block (Same Dimensions)
layout: horizontal

layers:
  - {id: input, name: Input, type: input, size: "HxWxC"}

blocks:
  - name: Identity ResBlock
    type: residual
    style: arc
    main:
      - {id: conv1, name: Conv1, type: conv, kernel: 3, pad: 1, act: ReLU}
      - {id: conv2, name: Conv2, type: conv, kernel: 3, pad: 1, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

layers_after_blocks:
  - {id: output, name: Output, type: output, size: "HxWxC"}
```
*Figure: Basic Residual Block (Same Dimensions)*

When the input and output dimensions differ (e.g., channels change from 64 to 128, or spatial size changes from $56×56$ to $28×28$), the dimension mismatch prevents direct addition. In this case, a linear transformation is applied to the input $x$ on the shortcut path to match the dimensions of $F(x)$. ResNet uses a $1×1$ convolution for this transformation. The specific structure is shown below:

```nn-arch width=620
name: Basic Residual Block (Different Dimensions)
layout: horizontal

layers:
  - {id: input, name: Input, type: input, size: "HxWxC"}

blocks:
  - name: Projection ResBlock
    type: residual
    style: parallel
    main:
      - {id: conv1, name: Conv1, type: conv, kernel: 3, stride: 2, pad: 1, act: ReLU}
      - {id: conv2, name: Conv2, type: conv, kernel: 3, pad: 1, act: ReLU}
    skip:
      - {id: skip_conv, name: "Conv_Skip", type: conv, kernel: 1, stride: 2}
    merge: add
    act: ReLU

layers_after_blocks:
  - {id: output, name: Output, type: output, size: "H'xW'xC'"}
```

Conv_Skip is a $1×1$ convolutional layer used to adjust the number of channels. If the spatial size also needs to change (e.g., downsampling with Stride=2), a $1×1$ convolution with Stride=2 is used on the shortcut path for downsampling. According to the experimental data in the ResNet paper, the projection method using $1×1$ convolution performs slightly better than zero-padding for dimension matching, but the difference is small (approximately 0.2% accuracy gap). Considering computational efficiency, ResNet only uses projection when dimensions need to change; in all other cases, direct addition is employed.

Both types of residual blocks described above are collectively called **Basic Blocks**, the most fundamental residual block structure in the ResNet paper. They have a simple structure and a clear receptive field, each consisting of two $3×3$ convolutional layers. The stacked convolutions produce a combined receptive field of $5×5$ (the first convolution has a $3×3$ receptive field, and the second expands upon it, yielding a total receptive field of $5×5$). This design borrows VGG's "stacking small convolutions" approach, using multiple small convolutions in place of one large convolution, which both reduces parameter count and increases nonlinear activation, enhancing the network's expressive power. Basic Blocks are suitable for relatively shallow networks such as ResNet-18 and ResNet-34.

For deeper networks such as ResNet-50, ResNet-101, and ResNet-152, a different structure called the **Bottleneck Block** is used. This is an efficient structure designed by ResNet for deep networks, consisting of three convolutional layers: a $1×1$ dimensionality-reduction convolution, a $3×3$ main convolution, and a $1×1$ dimensionality-expansion convolution. The core idea of the Bottleneck Block is to reduce the number of channels using a $1×1$ convolution, decreasing the computational cost of the $3×3$ convolution, and then restore the channel count with another $1×1$ convolution. This "reduce-convolve-expand" design significantly reduces parameter count and computation while preserving the $3×3$ receptive field. The specific structure of the Bottleneck Block is shown below:

```nn-arch width=760
name: Bottleneck Residual Block
layout: horizontal

layers:
  - {id: input, name: Input, type: input, size: "HxWxC"}

blocks:
  - name: Bottleneck ResBlock
    type: residual
    style: arc
    main:
      - {id: conv1, name: Conv1, type: conv, kernel: 1, pad: 0, act: ReLU}
      - {id: conv2, name: Conv2, type: conv, kernel: 3, pad: 1, act: ReLU}
      - {id: conv3, name: Conv3, type: conv, kernel: 1, pad: 0, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

layers_after_blocks:
  - {id: output, name: Output, type: output, size: "HxWxC"}
```
*Figure: Bottleneck Residual Block (Same Dimensions)*

These residual blocks have two slightly counterintuitive characteristics worth noting. First, although Bottleneck Blocks are typically used in deeper networks, the network depth is largely due to the Bottleneck Blocks themselves. Network depth is defined as the number of layers with weight parameters, i.e., the total count of convolutional and fully connected layers. A Basic Block has 2 convolutional layers, while a Bottleneck Block has 3 — therefore, networks using Bottleneck Blocks have 50% more layers than those using Basic Blocks given the same number of blocks. If ResNet-34 were entirely converted to Bottleneck Blocks, it would have approximately 50 layers, which is the ResNet-50 structure. Second, compared to Basic Blocks, Bottleneck Blocks have more layers but fewer parameters (which is why they are called an advanced structure). Taking the example where both input and output have 256 channels, the parameter count of a Basic Block (two $3×3$ convolutions) is $1,180,160$, while that of a Bottleneck Block ($1×1$ → $3×3$ → $1×1$, with 64 intermediate channels) is $70,016$ — merely 5.9% of the Basic Block, a striking efficiency improvement. At the same time, the central receptive field of the Bottleneck Block still comes from the $3×3$ convolution, matching the effective receptive field of the Basic Block. This enables deep networks (such as ResNet-152) to stack more residual blocks while maintaining computational feasibility.

## Residual Network Architecture

The ResNet paper proposed several network configurations at varying depths, as shown in the table below, ranging from 18 to 152 layers, covering applications from lightweight to heavyweight.

| Network | Layers | Block Configuration | Block Type | Parameters | Top-1 Error (Single-Crop) |
|:----|:----|:-----------------|:---------|:--------|:------------------------|
| ResNet-18 | 18 | [2, 2, 2, 2] | Basic | ~11.7M | 30.3% |
| ResNet-34 | 34 | [3, 4, 6, 3] | Basic | ~21.8M | 26.2% |
| ResNet-50 | 50 | [3, 4, 6, 3] | Bottleneck | ~25.6M | 23.9% |
| ResNet-101 | 101 | [3, 4, 23, 3] | Bottleneck | ~44.5M | 22.6% |
| ResNet-152 | 152 | [3, 8, 36, 3] | Bottleneck | ~60.2M | 21.6% |

Taking the shallowest among them, ResNet-18, as an example, its complete network architecture can be clearly illustrated in the diagram below. This architecture reflects several key design decisions in ResNet. First, the initial layer uses a $7×7$ large convolution kernel with Stride=2 to reduce the spatial size from $224×224$ to $112×112$ in one step, then further reduces it to $56×56$ through pooling. Unlike VGG's "stacking $3×3$ convolutions" strategy, ResNet adopted a more aggressive downsampling approach. Second, each residual block performs downsampling once (Stride=2), halving the spatial size and doubling the number of channels. This design ensures that feature maps gradually shrink in the spatial dimension while expanding in the channel dimension, balancing information content and computational cost. Finally, [Global Average Pooling](cnn-basics.md#cnn-architecture-design-principles) replaces the multiple fully connected layers used in VGG and AlexNet, substantially reducing the parameter count (from approximately 100M in VGG-16 to approximately 21.8M in ResNet-34) while avoiding the overfitting problems that fully connected layers are prone to.

```nn-arch width=1400
name: ResNet-18 Architecture (Basic Block, [2, 2, 2, 2] Configuration)
layout: horizontal

layers:
  - {id: input, name: Input, type: input, size: "224x224x3"}
  - {id: conv1, name: Conv1, type: conv, kernel: 7, stride: 2, channels: 64, out: "112×112x64", act: ReLU}
  - {id: pool1, name: Pool1, type: pool, kernel: 3, stride: 2, out: "56x56x64"}

blocks:
  - name: ResBlock1
    type: residual
    main:
      - {id: rb1_conv1, name: conv1, type: conv, kernel: 3, channels: 64, act: ReLU}
      - {id: rb1_conv2, name: conv2, type: conv, kernel: 3, channels: 64, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

  - name: ResBlock2
    type: residual
    main:
      - {id: rb2_conv1, name: conv1, type: conv, kernel: 3, channels: 128, act: ReLU}
      - {id: rb2_conv2, name: conv2, type: conv, kernel: 3, channels: 128, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

  - name: ResBlock3
    type: residual
    main:
      - {id: rb3_conv1, name: conv1, type: conv, kernel: 3, channels: 256, act: ReLU}
      - {id: rb3_conv2, name: conv2, type: conv, kernel: 3, channels: 256, act: ReLU}
    skip: identity
    merge: add
    act: ReLU

  - name: ResBlock4
    type: residual
    main:
      - {id: rb4_conv1, name: conv1, type: conv, kernel: 3, channels: 512, act: ReLU}
      - {id: rb4_conv2, name: conv2, type: conv, kernel: 3, channels: 512, act: ReLU}
    skip: identity
    merge: add
    act: ReLU


layers_after_blocks:
  - {id: globalpool, name: GlobalPool, type: pool, kernel: global, out: "1x1x512"}
  - {id: fc, name: FC, type: fc, size: 1000}
  - {id: output, name: Output, type: output, size: 1000, act: Softmax}
```

After the ResNet paper was published, He's team continued to study the working principles of residual networks. In 2016, Kaiming He published another important paper, "[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)" (ECCV 2016), proposing a pre-activation version of the residual block that further optimized gradient flow. The original ResNet residual block used a post-activation structure: convolutional layers were followed by BN and ReLU, then the residual connection added the input to the output, followed by another ReLU. The problem with this structure is that the ReLU after the residual connection forces the output to be non-negative, limiting the expressive power of identity mapping. The pre-activation version improved this by moving BN and ReLU before the convolutional layers. The advantage is that the residual connection directly feeds into the output without passing through ReLU, preserving the purity of identity mapping.

The benefit of the pre-activation version can be understood from a gradient flow perspective. In the original version, the residual connection is followed by a ReLU, which truncates negative gradients during backpropagation (turning them to zero). In the pre-activation version, the residual connection directly passes through to the output, allowing gradients to propagate through this path without loss. He's experiments showed that the pre-activation version significantly outperformed the original on very deep networks (such as ResNet-1001), with more stable training and faster convergence.

Modern implementations of ResNet (such as PyTorch's torchvision library and Facebook's Detectron2 library) typically use the pre-activation version. Although for moderately deep networks like ResNet-50/101/152 the difference between the original and pre-activation versions is not large, the pre-activation version has become the standard paradigm for residual blocks in machine learning frameworks.

## Applications and Impact

ResNet's impact extends far beyond the ImageNet classification competition itself. The idea of residual learning — learning "improvements relative to a baseline" rather than "a complete mapping" — has become an important design paradigm in deep neural network architecture, widely adopted across almost all areas of deep learning, including computer vision, natural language processing, and generative models, as shown in the table below:

| Application Domain | Representative Model | Specific Role of Residual Connection |
|:-----------------|:------------------|:----------------------------------|
| Object Detection | Faster R-CNN, Mask R-CNN | Uses ResNet as backbone, replacing VGG-16 |
| Semantic Segmentation | DeepLab v3+, FCN-ResNet | Uses ResNet for multi-scale feature extraction |
| Natural Language Processing | Transformer, BERT | Residual connections throughout the architecture, ensuring gradient flow in deep networks |
| Generative Models | StyleGAN, DDPM | Residual blocks used in generators and denoising networks |
| Video Understanding | 3D ResNet, I3D | Extends 2D residual blocks to 3D for spatiotemporal information |
| Self-Supervised Learning | SimCLR, MoCo | ResNet as feature extractor for learning contrastive representations |

Among these, the most far-reaching influence is the residual connection in the Transformer architecture. The Transformer, the protagonist of the upcoming large language model section, adopts residual connections as a core design. Every multi-head attention layer and every feed-forward network layer adds its input to its output via a residual connection. This design allows Transformers to stack dozens of layers (e.g., BERT-Base has 12 layers, BERT-Large has 24 layers, GPT-3 has 96 layers) without encountering optimization difficulties. The success of the Transformer demonstrates that the value of residual learning extends far beyond image recognition, proving applicable to all kinds of neural network architectures.

## Chapter Summary

This chapter introduced ResNet, one of the most important network architectures in deep learning history. Through residual connections, it elegantly solves the difficulty of learning identity mappings, shifting the learning objective from learning the complete mapping $H(x)$ to learning the residual $H(x) = F(x) + x$. An even more important contribution of residual connections is the improvement of gradient flow. Residual connections provide a direct shortcut for gradients — during backpropagation, gradients can travel directly from deep layers to shallow layers without attenuation through the multiplicative chain of intermediate layer derivatives. This makes it possible to train networks with over 1000 layers without encountering vanishing gradients or the degradation problem.

ResNet-152 achieved a breakthrough result on ImageNet: a Top-5 error rate of 3.57% with a 6-model ensemble (multi-crop) and a Top-1 error rate of 21.6% (single-crop), surpassing human-level performance for the first time. More importantly, the idea of residual learning has been widely applied across all areas of deep learning. In object detection, Faster R-CNN and Mask R-CNN use ResNet as their backbone; in natural language processing, Transformers and BERT employ residual connections as a core design element; in generative models, StyleGAN and DDPM make extensive use of residual blocks throughout their networks. It is fair to say that residual connections have become an indispensable component of modern neural networks.

## Exercises

1. Compute and compare the parameter counts of a Basic Block and a Bottleneck Block. Given that both input and output channels are 256, calculate the parameter counts of both residual block types and analyze why the Bottleneck Block has fewer parameters.
    <details>
    <summary>Answer</summary>

    **Basic Block Parameter Calculation**:

    The Basic Block consists of two $3×3$ convolutional layers:
    - First $3×3$ convolution: $256 × 256 × 3 × 3 = 589,824$ weight parameters + $256$ bias parameters = $590,080$
    - Second $3×3$ convolution: $256 × 256 × 3 × 3 = 589,824$ weight parameters + $256$ bias parameters = $590,080$
    - **Total parameters**: $590,080 × 2 = 1,180,160$

    **Bottleneck Block Parameter Calculation**:

    The Bottleneck Block uses a "reduce-convolve-expand" structure with 64 intermediate channels:
    - $1×1$ dimensionality-reduction convolution: $256 × 64 × 1 × 1 = 16,384$ weights + $64$ biases = $16,448$
    - $3×3$ main convolution: $64 × 64 × 3 × 3 = 36,864$ weights + $64$ biases = $36,928$
    - $1×1$ dimensionality-expansion convolution: $64 × 256 × 1 × 1 = 16,384$ weights + $256$ biases = $16,640$
    - **Total parameters**: $16,448 + 36,928 + 16,640 = 70,016$

    **Efficiency Comparison**:

    The Bottleneck Block parameter count is only $\frac{70,016}{1,180,160} ≈ 5.9\%$ of the Basic Block. The Bottleneck Block uses $1×1$ convolutions to reduce the channel count from 256 to 64, so that the most computationally expensive $3×3$ convolution only needs to process $64×64$ channel mappings instead of $256×256$. This design enables deep networks (such as ResNet-152) to stack more residual blocks while maintaining computational feasibility.
    </details>

2. Demonstrate how residual connections improve gradient flow. Based on the formula $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (\frac{\partial F}{\partial x} + 1)$, explain why residual connections can prevent vanishing gradients.
    <details>
    <summary>Answer</summary>

    **Gradient Formula Analysis**:

    The output of a residual block is $y = x + F(x)$. During backpropagation, the gradient propagation formula is:

    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$

    This formula decomposes into two parts:
    - $\frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x}$: the gradient path through the residual function $F$
    - $\frac{\partial L}{\partial y} \cdot 1$: the direct shortcut gradient path through the residual connection

    **Key Insight**:

    Even if the residual function suffers from vanishing gradients ($\frac{\partial F}{\partial x} \to 0$), the $+1$ term from the shortcut path remains, and the gradient becomes:

    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot 1 = \frac{\partial L}{\partial y}$$

    This means gradients can propagate directly from deep layers to shallow layers without any attenuation.

    **Comparison with Standard CNN**:

    In a standard CNN (without residual connections), gradients must pass through successive multiplicative layers:

    $$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \cdot \prod_{i=1}^{n-1} \frac{\partial x_{i+1}}{\partial x_i}$$

    When each layer's derivative $\frac{\partial x_{i+1}}{\partial x_i} < 1$ (e.g., the maximum derivative of Sigmoid is only 0.25), the multiplicative product decays exponentially, and gradients from deep layers can hardly reach shallow layers.

    **ResNet Advantage**:

    The residual connection provides a "guaranteed" channel for each layer's gradient. Even if the gradient through $F$ vanishes, there is at least a direct shortcut gradient of $+1$. This allows ResNet to train 152-layer and even 1000-layer networks without severe vanishing gradient problems.
    </details>

3. ResNet-34 uses Basic Blocks with a configuration of [3, 4, 6, 3]. Calculate the total number of layers (layers with weight parameters) in this network, and explain why switching to Bottleneck Blocks would increase the layer count.
    <details>
    <summary>Answer</summary>

    **ResNet-34 Layer Count Calculation**:

    Network layer count is defined as the number of layers with weight parameters (convolutional layers + fully connected layers).

    - Initial layer: $7×7$ convolution (1 layer) + max pooling (no weights, not counted)
    - Residual block section:
      - Stage 1: 3 Basic Blocks × 2 convolutions = 6 layers
      - Stage 2: 4 Basic Blocks × 2 convolutions = 8 layers
      - Stage 3: 6 Basic Blocks × 2 convolutions = 12 layers
      - Stage 4: 3 Basic Blocks × 2 convolutions = 6 layers
    - Final layer: Global average pooling (no weights) + fully connected layer (1 layer)

    **Total layers**: $1 + 6 + 8 + 12 + 6 + 1 = 34$ layers

    **Effect of Switching to Bottleneck Blocks**:

    A Bottleneck Block consists of 3 convolutional layers ($1×1$ → $3×3$ → $1×1$), whereas a Basic Block has only 2. If ResNet-34's configuration [3, 4, 6, 3] were entirely converted to Bottleneck Blocks:

    - Residual block section: $(3 + 4 + 6 + 3) × 3 = 48$ layers
    - **Total layers**: $1 + 48 + 1 = 50$ layers

    This is precisely the layer configuration of ResNet-50. Although the layer count increases, the Bottleneck Block's parameter efficiency means that ResNet-50's parameter count (~25.6M) is not much larger than ResNet-34's (~21.8M), yet it achieves better performance (Top-1 error rate drops from 26.2% to 23.9%).

    **Design Trade-off**:

    Network depth and parameter count do not follow a simple proportional relationship. The Bottleneck Block uses a "dimensionality reduction" strategy to trade more layers for stronger expressive power while controlling parameter growth. This design finds a balance between computational resources and model performance in deep networks.
    </details>
