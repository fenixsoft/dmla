# VGG and GoogLeNet

The success of AlexNet in 2012 validated the effectiveness of deep convolutional neural networks on large-scale vision tasks. As a pioneering work, AlexNet's architecture design was inevitably somewhat rough — issues such as using large convolution kernels ($11 \times 11$, $5 \times 5$) for aggressive downsampling in its 8-layer network, and the severe imbalance in parameter counts between convolutional and fully connected layers (fully connected layers accounted for 94% of total parameters). Everyone wondered: where is the upper limit of deep neural networks' potential, and what is the source of performance improvement? Is it deeper or wider networks, or smaller convolution kernels? Is it improved training techniques or better data augmentation? Without understanding these questions, subsequent architecture design would be like blind men touching an elephant — each improvement might only coincidentally hit the right direction in some dimension.

In 2014, two landmark research works partially answered these questions from different directions. The Visual Geometry Group (VGG) at the University of Oxford demonstrated in their paper "[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)" that increasing network depth from 8 layers to 16-19 layers can significantly improve accuracy, later leading to the famous **VGGNet**. Shortly after, Christian Szegedy at Google proposed **GoogLeNet** (also known as Inception-v1) in the paper "[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)", achieving multi-scale feature fusion through the carefully designed Inception module, reaching a lower error rate than VGG with only 7 million parameters.

These two works represent two core exploration directions of CNN architecture design: VGG chose depth-first, stacking small convolution kernels to increase layer count, proving that deeper networks can learn more abstract features; GoogLeNet chose width-first, increasing effective width through parallel multi-scale branches, letting the network decide which scales of features are most valuable. Should networks be deeper or wider, and how to improve parameter efficiency — these explorations ushered in the golden age of modern CNN architecture design.

## Depth Exploration: VGGNet

The VGG group decided to answer the question "is deeper truly more effective for deep learning" with the simplest experimental method: they fixed all other factors of the model and only varied the depth to see what would happen. The core hypothesis of the experiment was that, keeping other hyperparameters such as kernel size and width (number of channels) unchanged, increasing network depth should improve accuracy. The logic behind this hypothesis was clear: deeper networks have more nonlinear transformation layers, each capable of learning more abstract features than the previous layer. From pixels to edges, from edges to textures, from textures to parts, from parts to objects — this hierarchical feature extraction is precisely the core advantage that distinguishes deep learning from traditional machine learning.

From the experimental results, we can see that more layers lead to lower error rates. VGG-19's Top-5 error rate of 7.0% is approximately 33% lower than VGG-11's roughly 10.4%, while the parameter count increase is relatively modest (133M → 144M), indicating that the accuracy improvement from increased depth is "worth it" — depth is an effective way to improve CNN performance.

| Configuration | Weight Layers | Block Structure | Parameters | Top-5 Error Rate |
|:----|:----------|:----------|:------|:------------|
| VGG-A | 11 layers | 1-1-2-2-2 | ~133M | ~10.4% |
| VGG-B | 13 layers | 2-2-2-2-2 | ~132M | ~9.9% |
| VGG-D (VGG-16) | 16 layers | 2-2-3-3-3 | ~138M | 7.3% |
| VGG-E (VGG-19) | 19 layers | 2-2-4-4-4 | ~144M | 7.0% |

Based on the hypothesis that "depth works," VGG adopted an extremely simple and highly regular network structure. Taking the most common VGG-16 as an example: all layers use $3 \times 3$ small convolution kernels, each Block is followed by a $2 \times 2$ max pooling layer; the number of channels doubles after each Block (64 → 128 → 256 → 512), and the spatial size halves after each pooling operation ($224 \times 224$ → $7 \times 7$). This copy-paste modular design is not particularly profound, yet it embodies sophisticated engineering wisdom. Its structure can be represented in the following architecture diagram:

```nn-arch width=900
name: VGG-16 Architecture (13 Conv Layers + 3 FC Layers)
layout: horizontal

sections:
  - name: Feature Extractor (5 Blocks, 13 Conv Layers)
    layers: ["Input", "block1", "block2", "block3", "block4", "block5"]
    row_label: "Flatten: 25088"
  - name: Classifier (3 FC Layers)
    layers: ["FC6", "FC7", "FC8", "Output"]

layers:
  - {name: "Input", type: input, size: "224×224×3"}
  - {id: block1, name: "2×Conv3×3", type: conv, kernel: 3, channels: 64, out: "112×112×64", pool: true}
  - {id: block2, name: "2×Conv3×3", type: conv, kernel: 3, channels: 128, out: "56×56×128", pool: true}
  - {id: block3, name: "3×Conv3×3", type: conv, kernel: 3, channels: 256, out: "28×28×256", pool: true}
  - {id: block4, name: "3×Conv3×3", type: conv, kernel: 3, channels: 512, out: "14×14×512", pool: true}
  - {id: block5, name: "3×Conv3×3", type: conv, kernel: 3, channels: 512, out: "7×7×512", pool: true}
  - {name: "FC6", type: fc, size: "4096-d", act: ReLU, dropout: true}
  - {name: "FC7", type: fc, size: "4096-d", act: ReLU, dropout: true}
  - {name: "FC8", type: fc, size: "1000-d"}
  - {name: "Output", type: output, size: 1000, act: Softmax}
```

*Figure: VGG-16 Architecture Diagram*

The most distinctive design decision in VGG's network architecture is the exclusive use of $3 \times 3$ small convolution kernels, completely abandoning the $11 \times 11$ and $5 \times 5$ large kernels found in AlexNet. This decision was based on three considerations:

1. **Equivalent receptive field, fewer parameters**: Intuitively, a larger receptive field means being able to "see" more input information, which helps capture global features. A $5 \times 5$ convolution kernel has a receptive field of $5 \times 5$, while a stack of two $3 \times 3$ convolutions also has a receptive field of $5 \times 5$. However, the parameter count of two $3 \times 3$ convolutions is only 72% of a single $5 \times 5$ convolution. Similarly, three $3 \times 3$ convolutions (receptive field $7 \times 7$) have only 55% of the parameters of a single $7 \times 7$ convolution.

2. **Stronger nonlinear representation capability**: Reduced parameter count is only a superficial advantage; the deeper benefit lies in the increased number of nonlinear transformations. Two $3 \times 3$ convolutions include two ReLU activation functions, meaning two nonlinear transformations. From a mathematical perspective, the function space representable by two $3 \times 3$ convolutions strictly contains the function space representable by a single $5 \times 5$ convolution — stacking small convolution kernels can fit more complex functional relationships.

3. **Naturally increases network depth**: Using multiple small convolution kernels to replace large ones naturally increases network depth. Greater depth means more feature hierarchies, from low-level features to high-level semantics, aligning with the core idea of hierarchical feature learning.

VGG's design of replacing large convolution kernels with $3 \times 3$ small ones was later adopted by many modern CNNs, and the $3 \times 3$ convolution kernel has since become a "standard component" of CNN architecture. However, VGG also inherited the design flaws of AlexNet and even made them worse: an enormous and extremely uneven parameter distribution, with fully connected layers becoming a "black hole" for parameters. Using the parameter count formula from [CNN Basics](cnn-basics.md) ($\text{params}_{conv} = k \times k \times C_{in} \times C_{out} + C_{out}$), we can compute the parameter count for each convolutional layer:

| Layer | Input Channels | Output Channels | Parameter Calculation | Parameters |
|:--|:--------|:--------|:----------|:------|
| Block1: Conv1 | 3 | 64 | $3 \times 3 \times 3 \times 64 + 64$ | 1,792 |
| Block1: Conv2 | 64 | 64 | $3 \times 3 \times 64 \times 64 + 64$ | 36,928 |
| Block2: Conv1 | 64 | 128 | $3 \times 3 \times 64 \times 128 + 128$ | 73,856 |
| Block2: Conv2 | 128 | 128 | $3 \times 3 \times 128 \times 128 + 128$ | 147,584 |
| Block3: Conv1 | 128 | 256 | $3 \times 3 \times 128 \times 256 + 256$ | 295,168 |
| Block3: Conv2 | 256 | 256 | $3 \times 3 \times 256 \times 256 + 256$ | 590,080 |
| Block3: Conv3 | 256 | 256 | $3 \times 3 \times 256 \times 256 + 256$ | 590,080 |
| Block4: Conv1 | 256 | 512 | $3 \times 3 \times 256 \times 512 + 512$ | 1,180,160 |
| Block4: Conv2, Conv3 | 512 | 512 | $(3 \times 3 \times 512 \times 512 + 512) \times 2$ | 4,719,616 |
| Block5: Conv1, Conv2, Conv3 | 512 | 512 | $(3 \times 3 \times 512 \times 512 + 512) \times 3$ | 7,079,424 |

The total parameter count for the convolutional layers is approximately 14.7 million. The final output size of the convolutional layers is $7 \times 7 \times 512 = 25,088$ dimensions, which is also the input to the fully connected layers. The total parameter count for the fully connected layers is approximately 123.6 million, as detailed below:

| Layer | Input Dim | Output Dim | Parameter Calculation | Parameters |
|:--|:--------|:--------|:----------|:------|
| FC6 | 25,088 | 4,096 | $25,088 \times 4,096 + 4,096$ | 102,764,544 |
| FC7 | 4,096 | 4,096 | $4,096 \times 4,096 + 4,096$ | 16,781,312 |
| FC8 | 4,096 | 1,000 | $4,096 \times 1,000 + 1,000$ | 4,097,000 |

Compared to AlexNet, VGG succeeded in halving the Top-5 error rate (from 15.3% to 7.3%), but at the cost of doubling the parameter count, with approximately 90% of parameters concentrated in the fully connected layers. This not only consumes a large amount of memory but also easily leads to overfitting. Solving this problem fell to GoogLeNet.

## Width Exploration: GoogLeNet

Google's research team thought about the development of deep networks from another perspective: if we can't make the network deeper, can we make it more powerful? This led to the Inception module, an innovative design that replaces serial stacking with multi-scale parallel structures.

In real vision tasks, the size of objects in images varies greatly. A single photo might contain both a small bird (occupying a few dozen pixels) and a large building (occupying hundreds of pixels). If only a fixed-size convolution kernel is used (e.g., $3 \times 3$), small objects might be overwhelmed within the kernel while large objects would have insufficient receptive fields. Different-sized convolution kernels can capture different features: $1 \times 1$ captures point features, $3 \times 3$ captures local textures, $5 \times 5$ captures larger-scale structures. Rather than artificially choosing one, it is better to let all sizes work in parallel and let the network itself decide which scales of features are most useful. Multi-scale parallel extraction and feature fusion decision-making are the core ideas of the Inception module. Its name "Inception" comes from a line in the movie *Inception*: "We need to go deeper," implying deep exploration — but here "deep" does not refer to stacking layers, but to the deep exploration of feature space.

The Inception module is the building block of GoogLeNet. Each Inception basic structure is a four-branch parallel network with concatenation at the end. Each branch is responsible for extracting features at different scales, as shown in the diagram below:

```nn-arch width=680
name: Inception Module (4 Parallel Branches)
layout: vertical

layers:
  - {id: input, name: "Input Feature Map", type: input, size: "H×W×C"}

blocks:
  - name: Inception
    type: parallel
    branches:
      # 1x1 conv branch
      - {id: branch_1x1, name: "1×1 Conv", type: conv, kernel: 1, channels: 64, act: ReLU}
      # 3x3 conv branch (1x1 reduce first)
      - [
          {id: branch_3x3_reduce, name: "1×1 Reduce", type: conv, kernel: 1, channels: 96, act: ReLU},
          {id: branch_3x3, name: "3×3 Conv", type: conv, kernel: 3, pad: 1, channels: 128, act: ReLU}
        ]
      # 5x5 conv branch (1x1 reduce first)
      - [
          {id: branch_5x5_reduce, name: "1×1 Reduce", type: conv, kernel: 1, channels: 16, act: ReLU},
          {id: branch_5x5, name: "5×5 Conv", type: conv, kernel: 5, pad: 2, channels: 32, act: ReLU}
        ]
      # pool branch
      - [
          {id: branch_pool, name: "3×3 Max Pool", type: pool, kernel: 3, stride: 1, pad: 1},
          {id: branch_pool_proj, name: "1×1 Conv", type: conv, kernel: 1, channels: 32, act: ReLU}
        ]
    merge: concat

layers_after_blocks:
  - {id: concat, name: "Channel Concat", type: note, label: "Concatenate" }  
  - {id: output, name: "Output Feature Map", type: output, size: "H×W×C'"}
```
*Figure: Inception Module*

From this structure diagram, we can clearly see that the Inception module contains four parallel branches, each with an independent design intent:

- **$1 \times 1$ convolution branch**: Directly extracts point-wise features, no spatial expansion, smallest receptive field.
- **$3 \times 3$ convolution branch**: First uses $1 \times 1$ to reduce dimensions (decrease channel count), then uses $3 \times 3$ convolution to extract medium-scale features.
- **$5 \times 5$ convolution branch**: Similarly reduces dimensions first, then uses $5 \times 5$ convolution to extract large-scale features.
- **Pooling branch**: Max pooling retains salient features, followed by $1 \times 1$ convolution to adjust the channel count.

The four outputs are concatenated along the channel dimension (i.e., the output channel count C' is the sum of the four branch channel counts), forming the final output feature map. The output spatial size remains unchanged (due to appropriate padding), only the channel count changes. This means subsequent modules can continue stacking in the same manner.

The contribution to parameter reduction comes from the $1 \times 1$ convolution in the Inception module, which creates a "bottleneck" structure. As mentioned in [CNN Basics](cnn-basics.md#cnn-architecture-design-principles), $1 \times 1$ convolution has three functions: cross-channel information fusion, increased nonlinearity, and channel dimensionality reduction. In the Inception module, dimensionality reduction is the key role, solving the parameter explosion problem caused by multi-scale parallel structures.

To give a specific example, suppose the input to the Inception module is $28 \times 28 \times 192$ (height 28, width 28, channels 192), the convolution kernel size is $5 \times 5$, and the output is $28 \times 28 \times 32$ (output channels 32, spatial size remains 28×28). Without dimensionality reduction, directly applying a $5 \times 5$ convolution to the 192-channel input gives $\text{params} = 5 \times 5 \times 192 \times 32 + 32 = 153,632$. If we first use a $1 \times 1$ convolution to reduce channels from 192 to 16 (the bottleneck channel count), then apply a $5 \times 5$ convolution to the 16-channel output, we get $\text{params}_{reduced} = (1 \times 1 \times 192 \times 16 + 16) + (5 \times 5 \times 16 \times 32 + 32) = 3,088 + 12,832 = 15,920$. The parameter count drops from 153,632 to 15,920, a reduction of approximately 90%, while the output size remains exactly the same at $28 \times 28 \times 32$.

The $1 \times 1$ convolution performs a linear combination along the channel dimension at each spatial position, acting as an information compressor that compresses 192-dimensional channel information down to 16 dimensions. Since channel information at the same spatial position is typically highly redundant — multiple channels may detect similar features — this compression does not lose key information while significantly reducing subsequent computational cost.

With an understanding of the Inception module's design principles, let's see how it is assembled into a complete classification network. GoogLeNet's structure is much more complex than the serial network structures of AlexNet and VGG. It consists of 9 Inception modules stacked sequentially, interspersed with several regular convolutional and pooling layers. The overall architecture can be divided into three parts: initial feature extraction, Inception stacking, and classification output, as shown below:

```nn-arch width=900
name: GoogLeNet Network Architecture
layout: horizontal

sections:
  - name: Initial Feature Extraction
    layers: [input, conv1, pool1, conv2, pool2]
  - name: Inception Stack
    layers: [Inception_3a, Inception_3b, pool3, Inception_4a, Inception_4b, Inception_4c, Inception_4d, Inception_4e, pool4, Inception_5a, Inception_5b]
  - name: Classification Output
    layers: [pool5, fc, output]

layers:
  - {id: input, name: Input, type: input, size: "224x224x3"}
  - {id: conv1, name: Conv1, type: conv, kernel: 7, stride: 2, channels: 64, out: "112x112x64", act: ReLU}
  - {id: pool1, name: Pool1, type: pool, kernel: 3, stride: 2, out: "56x56x64"}
  - {id: conv2, name: Conv2, type: conv, kernel: 3, channels: 192, act: ReLU}
  - {id: pool2, name: Pool2, type: pool, kernel: 3, stride: 2, out: "28x28x192"}

blocks:
  - name: Inception_3a
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc3a_1x1, name: "1x1", type: conv, kernel: 1, channels: 64, act: ReLU}
      - [{id: inc3a_3x3r, name: "reduce", type: conv, kernel: 1, channels: 96, act: ReLU},
         {id: inc3a_3x3, name: "3x3", type: conv, kernel: 3, channels: 128, act: ReLU}]
      - [{id: inc3a_5x5r, name: "reduce", type: conv, kernel: 1, channels: 16, act: ReLU},
         {id: inc3a_5x5, name: "5x5", type: conv, kernel: 5, channels: 32, act: ReLU}]
      - [{id: inc3a_pool, name: "pool", type: pool, kernel: 3},
         {id: inc3a_proj, name: "proj", type: conv, kernel: 1, channels: 32, act: ReLU}]
    merge: concat

  - name: Inception_3b
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc3b_1x1, name: "1x1", type: conv, kernel: 1, channels: 128, act: ReLU}
      - [{id: inc3b_3x3r, name: "reduce", type: conv, kernel: 1, channels: 128, act: ReLU},
         {id: inc3b_3x3, name: "3x3", type: conv, kernel: 3, channels: 192, act: ReLU}]
      - [{id: inc3b_5x5r, name: "reduce", type: conv, kernel: 1, channels: 32, act: ReLU},
         {id: inc3b_5x5, name: "5x5", type: conv, kernel: 5, channels: 96, act: ReLU}]
      - [{id: inc3b_pool, name: "pool", type: pool, kernel: 3},
         {id: inc3b_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - {id: pool3, name: Pool3, type: pool, kernel: 3, stride: 2}

  - name: Inception_4a
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4a_1x1, name: "1x1", type: conv, kernel: 1, channels: 192, act: ReLU}
      - [{id: inc4a_3x3r, name: "reduce", type: conv, kernel: 1, channels: 96, act: ReLU},
         {id: inc4a_3x3, name: "3x3", type: conv, kernel: 3, channels: 208, act: ReLU}]
      - [{id: inc4a_5x5r, name: "reduce", type: conv, kernel: 1, channels: 16, act: ReLU},
         {id: inc4a_5x5, name: "5x5", type: conv, kernel: 5, channels: 48, act: ReLU}]
      - [{id: inc4a_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4a_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - name: Inception_4b
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4b_1x1, name: "1x1", type: conv, kernel: 1, channels: 160, act: ReLU}
      - [{id: inc4b_3x3r, name: "reduce", type: conv, kernel: 1, channels: 112, act: ReLU},
         {id: inc4b_3x3, name: "3x3", type: conv, kernel: 3, channels: 224, act: ReLU}]
      - [{id: inc4b_5x5r, name: "reduce", type: conv, kernel: 1, channels: 24, act: ReLU},
         {id: inc4b_5x5, name: "5x5", type: conv, kernel: 5, channels: 64, act: ReLU}]
      - [{id: inc4b_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4b_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - name: Inception_4c
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4c_1x1, name: "1x1", type: conv, kernel: 1, channels: 128, act: ReLU}
      - [{id: inc4c_3x3r, name: "reduce", type: conv, kernel: 1, channels: 128, act: ReLU},
         {id: inc4c_3x3, name: "3x3", type: conv, kernel: 3, channels: 256, act: ReLU}]
      - [{id: inc4c_5x5r, name: "reduce", type: conv, kernel: 1, channels: 24, act: ReLU},
         {id: inc4c_5x5, name: "5x5", type: conv, kernel: 5, channels: 64, act: ReLU}]
      - [{id: inc4c_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4c_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - name: Inception_4d
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4d_1x1, name: "1x1", type: conv, kernel: 1, channels: 112, act: ReLU}
      - [{id: inc4d_3x3r, name: "reduce", type: conv, kernel: 1, channels: 144, act: ReLU},
         {id: inc4d_3x3, name: "3x3", type: conv, kernel: 3, channels: 288, act: ReLU}]
      - [{id: inc4d_5x5r, name: "reduce", type: conv, kernel: 1, channels: 32, act: ReLU},
         {id: inc4d_5x5, name: "5x5", type: conv, kernel: 5, channels: 64, act: ReLU}]
      - [{id: inc4d_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4d_proj, name: "proj", type: conv, kernel: 1, channels: 64, act: ReLU}]
    merge: concat

  - name: Inception_4e
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc4e_1x1, name: "1x1", type: conv, kernel: 1, channels: 256, act: ReLU}
      - [{id: inc4e_3x3r, name: "reduce", type: conv, kernel: 1, channels: 160, act: ReLU},
         {id: inc4e_3x3, name: "3x3", type: conv, kernel: 3, channels: 320, act: ReLU}]
      - [{id: inc4e_5x5r, name: "reduce", type: conv, kernel: 1, channels: 32, act: ReLU},
         {id: inc4e_5x5, name: "5x5", type: conv, kernel: 5, channels: 128, act: ReLU}]
      - [{id: inc4e_pool, name: "pool", type: pool, kernel: 3},
         {id: inc4e_proj, name: "proj", type: conv, kernel: 1, channels: 128, act: ReLU}]
    merge: concat

  - {id: pool4, name: Pool4, type: pool, kernel: 3, stride: 2}

  - name: Inception_5a
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc5a_1x1, name: "1x1", type: conv, kernel: 1, channels: 256, act: ReLU}
      - [{id: inc5a_3x3r, name: "reduce", type: conv, kernel: 1, channels: 160, act: ReLU},
         {id: inc5a_3x3, name: "3x3", type: conv, kernel: 3, channels: 320, act: ReLU}]
      - [{id: inc5a_5x5r, name: "reduce", type: conv, kernel: 1, channels: 32, act: ReLU},
         {id: inc5a_5x5, name: "5x5", type: conv, kernel: 5, channels: 128, act: ReLU}]
      - [{id: inc5a_pool, name: "pool", type: pool, kernel: 3},
         {id: inc5a_proj, name: "proj", type: conv, kernel: 1, channels: 128, act: ReLU}]
    merge: concat

  - name: Inception_5b
    type: parallel
    expand: "collapsed"
    branches:
      - {id: inc5b_1x1, name: "1x1", type: conv, kernel: 1, channels: 384, act: ReLU}
      - [{id: inc5b_3x3r, name: "reduce", type: conv, kernel: 1, channels: 192, act: ReLU},
         {id: inc5b_3x3, name: "3x3", type: conv, kernel: 3, channels: 384, act: ReLU}]
      - [{id: inc5b_5x5r, name: "reduce", type: conv, kernel: 1, channels: 48, act: ReLU},
         {id: inc5b_5x5, name: "5x5", type: conv, kernel: 5, channels: 128, act: ReLU}]
      - [{id: inc5b_pool, name: "pool", type: pool, kernel: 3},
         {id: inc5b_proj, name: "proj", type: conv, kernel: 1, channels: 128, act: ReLU}]
    merge: concat

layers_after_blocks:
  - {id: pool5, name: Pool5, type: pool, kernel: 7, stride: 1}
  - {id: fc, name: FC, type: fc, size: 1000, dropout: true}
  - {id: output, name: Output, type: output, size: 1000, act: Softmax}
```

*Figure: GoogLeNet Network Architecture Diagram*

From this architecture diagram, we can observe several key design features of GoogLeNet:

- **First, global average pooling replaces fully connected layers**: This is the most important reason for GoogLeNet's dramatic parameter reduction. VGG used three fully connected layers consuming 89% of its parameters, while GoogLeNet directly uses global average pooling to compress a $7 \times 7 \times 1024$ feature map into a $1 \times 1 \times 1024$ vector, followed by a simple classification layer. This design not only uses very few parameters but also avoids the overfitting problems easily caused by fully connected layers.

- **Second, auxiliary classifiers**: An auxiliary classifier is placed at each of two intermediate positions in the network (after Inception 4a and 4d). During training, the losses from the two auxiliary classifiers are combined with the main loss in a weighted sum:

    $$L_{total} = L_{main} + 0.3 \times L_{aux1} + 0.3 \times L_{aux2}$$

    GoogLeNet has 22 layers (including all Inception sublayers), and at this depth, the vanishing gradient problem becomes increasingly difficult to suppress. Rather than letting the gradient vanish as it propagates too far from the output layer, it is better to provide an exit along the way — inserting classifiers in the middle of the network to directly supervise intermediate features. This is equivalent to cutting a deep network into several shallower segments, allowing each segment to receive stronger gradient signals.

    During training, the auxiliary classifiers provide additional gradient signals for the intermediate layers during backpropagation, alleviating the vanishing gradient problem in deep networks. During inference, the auxiliary classifiers play no role — only the main classifier output is retained. Subsequent research has shown that with the introduction of techniques such as Batch Normalization, more effective means of suppressing vanishing gradients have emerged, making the role of auxiliary classifiers less critical. However, this design idea inspired later "residual connections" and multi-scale supervision techniques.

- **Third, hierarchical fusion of multi-scale features**: Each Inception module simultaneously extracts features at four scales: $1 \times 1$, $3 \times 3$, $5 \times 5$, and pooling. As the network deepens, multi-scale features from different levels are gradually fused, ultimately forming rich semantic representations.

GoogLeNet achieved a lower error rate with approximately 5% of VGG's parameter count. This was a triumph of architectural design — the Inception module, through multi-scale parallelism and $1 \times 1$ dimensionality reduction, achieved a revolutionary improvement in parameter efficiency. The parameter count comparison between GoogLeNet, AlexNet, and VGG-16 is shown in the following table:

| Network | Total Parameters | Conv Layer Params | FC Layer Params | Top-5 Error Rate |
|:----|:--------|:----------|:------------|:------------|
| AlexNet | 62M | 3.75M | 58.63M | 15.3% |
| VGG-16 | 138M | 14.7M | 123.3M | 7.3% |
| GoogLeNet | ~7M | ~4.6M | ~2.4M | 6.7% |

## Trade-off Between Network Depth and Width

The comparison between VGG and GoogLeNet raises a classic question: should neural network architecture design prioritize increasing depth or width? This question still has no definitive answer, but understanding the roles and trade-offs of both is key to mastering modern network design. **Depth** refers to the number of layers in the network — how many transformation layers from input to output; **Width** refers to the number of channels (number of features) per layer — how many different types of features each layer can extract simultaneously. Each has a different impact on network performance:
- **Increasing depth**: More layers, richer feature hierarchies, each layer abstracts more advanced semantics from the previous one. But the gradient propagation path is longer, making training more difficult.
- **Increasing width**: More channels per layer, richer feature types, can capture more diverse patterns simultaneously. But parameter count and computation grow faster, increasing memory pressure.

Both have their pros and cons; the key lies in finding the right balance for the task at hand. VGG and GoogLeNet represent two distinctly different design philosophies, and their choices and results provide valuable practical experience:

- **VGG chose the depth-first strategy**: VGG expanded AlexNet's 8 layers to 16-19 layers, with channel counts gradually increasing from 64 to 512. The core assumption of this design is that deeper networks can learn more abstract feature hierarchies, and experimental results validated its effectiveness. However, the cost of depth-first was enormous parameter counts — VGG's success proved the value of depth while also exposing the efficiency problem of fully connected layers.

- **GoogLeNet chose the width-first strategy**: GoogLeNet reached a depth of 22 layers, comparable to VGG-19, but through the four-branch parallel structure within each Inception module, it significantly increased the network's effective width. GoogLeNet simultaneously extracts features at four scales, learning more diverse patterns at each layer. The benefit of width-first is parameter efficiency — only 7 million parameters achieving a lower error rate than VGG — while the cost is higher design and computational complexity. The four-branch parallelism of the Inception module requires processing multiple branches simultaneously, naturally demanding higher parallelism during inference (parallelism is not computation; when parameters decrease, computation also decreases accordingly).

The ideas of depth-first and width-first have both continued to develop. On one hand, the strategy of stacking depth and parameter counts led major companies into a computational arms race, a phenomenon that peaked after the Transformer architecture emerged in 2017. On the other hand, modern network design has indeed increasingly emphasized efficiency optimization. Subsequent networks like MobileNet used depthwise separable convolutions to further reduce computation, and EfficientNet systematically balanced depth, width, and resolution through compound scaling. These innovations all continue the core idea of GoogLeNet: using smarter architectural design to replace brute-force resource stacking.

## Chapter Summary

This chapter introduced two landmark works from the 2014 ImageNet challenge, which broke through the design bottlenecks of AlexNet from different directions, ushering in the golden age of modern CNN architecture design. VGG and GoogLeNet pushed network depth to around 20 layers, but further deepening encountered new obstacles. When the number of layers exceeds a certain threshold, deeper networks actually perform worse — not because of overfitting, but because optimization becomes increasingly difficult. This problem was solved in 2015 by ResNet, where residual connections allow gradients to skip intermediate layers and propagate directly, making hundred-layer or even thousand-layer networks a reality. The next chapter will detail ResNet's design ideas and its profound impact on the field of deep learning.

## Exercises

1. VGG uses two $3 \times 3$ convolutional layers to replace one $5 \times 5$ convolutional layer, achieving the same receptive field with fewer parameters. Assuming the input channel count is $C_{in}$ and the output channel count is $C_{out}$, calculate the parameter ratio of the two schemes and explain the nonlinear expression advantage of stacking small convolution kernels.
    <details>
    <summary>Answer</summary>

    **Parameter Calculation**:

    - **Single $5 \times 5$ convolution**: params = $5 \times 5 \times C_{in} \times C_{out} + C_{out} = 25 C_{in} C_{out} + C_{out}$
    - **Two $3 \times 3$ convolutions**:
      - First layer: $3 \times 3 \times C_{in} \times C_{out} + C_{out} = 9 C_{in} C_{out} + C_{out}$
      - Second layer: $3 \times 3 \times C_{out} \times C_{out} + C_{out} = 9 C_{out}^2 + C_{out}$
      - Total params = $9 C_{in} C_{out} + 9 C_{out}^2 + 2 C_{out}$

    When $C_{in} = C_{out}$ (as in VGG's intra-block convolutional layers), the parameter ratio is:
    $$\frac{9 C^2 + 9 C^2 + 2C}{25 C^2 + C} = \frac{18 C^2 + 2C}{25 C^2 + C} \approx \frac{18}{25} = 72\%$$

    Two $3 \times 3$ convolutions use only **72%** of the parameters of a single $5 \times 5$ convolution, saving approximately 28% of parameters.

    **Nonlinear Expression Advantage**:

    Two $3 \times 3$ convolutions include two ReLU activation functions, meaning two nonlinear transformations. From the function space perspective:
    - Single $5 \times 5$ convolution: $f(x) = \text{ReLU}(W_5 x)$, one linear transformation + one nonlinearity
    - Two $3 \times 3$ convolutions: $f(x) = \text{ReLU}(W_3^{(2)} \cdot \text{ReLU}(W_3^{(1)} x))$, two linear transformations + two nonlinearities

    The function space of the second scheme strictly contains the first, enabling it to fit more complex feature relationships. For example, if the first layer detects "horizontal edges," the second layer can detect "specific combinations of horizontal edges" based on that — this hierarchical abstraction capability is the core advantage of deep networks.
    </details>
