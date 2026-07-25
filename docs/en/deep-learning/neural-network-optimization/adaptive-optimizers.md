# Adaptive Optimizers

The idea of adaptive learning rates stems from a simple observation: different parameters play different roles during training. Some parameters are updated frequently with large and stable gradients; others are updated sparsely with small gradients, occasionally even zero. If all parameters are given the same step size, the frequently updated parameters may take steps that are too large, causing oscillation and hindering convergence, while the sparsely updated parameters may take steps that are too small, making progress slow. It is like a principal assigning the same homework load to all students in the school — senior students find it boring, while junior students find it overwhelming.

**Adaptive Optimizers** are designed precisely to solve this "one-size-fits-all" problem. They automatically adjust the learning rate based on the historical gradients of each parameter, assigning an independent step size to every parameter. This "teach according to aptitude" strategy was first proposed by John Duchi in 2011, whose AdaGrad algorithm pioneered adaptive learning rates. Subsequently, Hinton proposed RMSprop in his 2012 Coursera course, addressing the issue of premature learning rate decay in AdaGrad. In 2015, Diederik Kingma and Jimmy Ba published the landmark paper "[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)", effectively combining momentum with adaptive learning rates, making Adam the most popular optimizer in deep learning. In 2019, Ilya Loshchilov and Frank Hutter discovered a theoretical flaw in Adam's weight decay implementation and proposed AdamW, further improving generalization. In 2024, Keller Jordan proposed Muon, which departs from the per-parameter adaptive learning rate approach and instead leverages the geometric structure of parameter matrices, using Newton-Schulz iteration to orthogonalize momentum updates, setting records in the NanoGPT and CIFAR-10 speed benchmarks. This chapter introduces these four adaptive optimizers along with the new-direction Muon, analyzing their design principles, advantages, disadvantages, and suitable application scenarios.

## AdaGrad

The momentum method and NAG introduced in the previous chapter solved the oscillation problem of gradient descent, but they still assign the same learning rate to all parameters. This is limiting in many scenarios. For example, in tasks such as natural language processing, the parameter updates in the word embedding layer are highly uneven: common words appear frequently with large gradients, while rare words appear occasionally with small or even zero gradients. With a uniform learning rate, the embedding vectors of common words update too quickly and tend to oscillate, while those of rare words update too slowly to learn effective representations. This motivated neural network researchers to find ways for each parameter to have its own learning rate.

**AdaGrad** (Adaptive Gradient Algorithm), proposed by John Duchi in 2011, is one of the earliest adaptive optimizers. The design philosophy of AdaGrad is straightforward: parameters that have been updated frequently have already learned a lot of information, so they should slow down; conversely, parameters updated sparsely still have a lot to learn, so they should take larger steps. This "teach according to aptitude" strategy allows each parameter to learn at its own optimal pace, avoiding the efficiency waste of a uniform learning rate.

The key to implementing this idea lies in measuring the magnitude of parameter updates. The metric AdaGrad chooses is the **accumulation of squared gradients**. There are two reasons for choosing squared gradients rather than gradients themselves: first, the squaring operation converts negative gradients to positive numbers, preventing positive and negative values from canceling out and distorting the accumulation (a parameter may frequently update in alternating positive and negative directions, but the accumulator should faithfully reflect the actual update activity); second, squaring amplifies large gradients and shrinks small ones, causing the accumulator for active parameters to grow faster and the learning rate to decay more significantly, achieving the effect of "the more active, the more cautious." AdaGrad chooses accumulation rather than moving average because its original design goal was to handle sparse gradients. Sparse parameters only receive gradients occasionally, so their accumulator grows slowly, preserving a large learning rate for a long time; frequent parameters, on the other hand, accumulate quickly, causing the learning rate to decay rapidly and preventing oscillation. This design of cumulative growth and decreasing learning rate essentially uses historical gradients as a "credit score" to endorse future updates — the more active the gradient, the larger the accumulation, the smaller the learning rate, indicating that the parameter has already learned enough and subsequent updates should be more cautious.

Let $\mathbf{G}_t$ be the accumulation of historical squared gradients, recording how many times and by how much the parameter has been updated. $(\nabla L_t)^2$ is the square of the current gradient, converting gradient values to non-negative numbers (negative gradients also need to be learned). The following formula expresses the gradient accumulation process:

$$[eq:adagrad-update] \mathbf{G}_t = \mathbf{G}_{t-1} + (\nabla L_t)^2$$

AdaGrad uses the learning rate hyperparameter divided by the square root of the accumulated gradients to achieve the effect that larger accumulation yields a smaller effective learning rate. To prevent division by zero at the beginning when there is no accumulated gradient yet, a small constant $\epsilon$ (typically $10^{-8}$) is added to the accumulation in practice. The effective learning rate of AdaGrad is expressed as $\frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}}$, where this denotes element-wise operations, meaning the learning rate for each component of the gradient is $\frac{\eta}{\sqrt{G_{t,i} + \epsilon}}$. This implies that within the same batch of parameters, some may have a large learning rate while others have a small one, depending entirely on their respective gradient histories. In summary, the weight update process is:

$$[eq:adagrad-sum]\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \cdot \nabla L_t$$

This adaptive adjustment of AdaGrad is particularly suitable for sparse gradient problems (such as word embeddings in natural language processing), where sparsely updated parameters receive larger learning rates to accelerate learning. However, AdaGrad's seemingly perfect strategy of accumulating historical gradients harbors a serious flaw. Since $\mathbf{G}_t$ is the accumulation of historical squared gradients, it only increases during training (squared gradients are non-negative), and $\sqrt{\mathbf{G}_t}$ also monotonically increases. This means the effective learning rate $\frac{\eta}{\sqrt{\mathbf{G}_t}}$ only ever decreases — AdaGrad merely regulates how fast the learning rate shrinks. With each parameter update, the learning rate becomes a little smaller. In the later stages of training, the learning rate can become extremely small (e.g., $10^{-6}$), causing parameters to almost stop updating and training to stagnate. This flaw puts AdaGrad in a dilemma: it is well-suited for short-term training or sparse gradient problems (frequently updated parameters get a small learning rate, sparse parameters get a large learning rate), but for long-term training, the learning rate decays too early, causing stagnation. To address this flaw, the improved version of AdaGrad, RMSprop, was introduced.

## RMSprop

**RMSprop** (Root Mean Square Propagation) was proposed by Geoffrey Hinton in his 2012 Coursera course. Interestingly, Hinton never formally published a paper on RMSprop — it was disseminated only as course notes, but its simplicity and effectiveness led to rapid adoption by the community. RMSprop's improvement over AdaGrad is to replace AdaGrad's accumulation with an exponentially weighted moving average. A moving average is a weighted average where new data has more weight and old data gradually decays, much like a window sliding along the data stream, allowing historical information to flow in and out. As a result, RMSprop retains gradient information from only the most recent approximately $\frac{1}{1-\gamma}$ steps.

Let $\mathbf{E}_t$ be the exponentially weighted moving average of squared gradients, and $\gamma$ be the decay coefficient (typically $0.9$), used as the weight for the historical accumulated squared gradients, controlling how much historical information is retained. The remaining $(1-\gamma)$ serves as the weight for the new squared gradients, ensuring the weights sum to 1. RMSprop's gradient accumulation formula is (compare with formula {{eq:adagrad-update}}):

$$[eq:rmsprop-update] \mathbf{E}_t = \gamma \mathbf{E}_{t-1} + (1 - \gamma)(\nabla L_t)^2$$

If $\gamma$ is set to $0.9$, it retains 90% of historical gradient information and incorporates 10% of the current gradient. Old history gradually fades out, preventing numerical inflation. The effective window is approximately $\frac{1}{1-0.9} = 10$ steps, retaining only the gradient information from the most recent 10 steps. Apart from this, RMSprop's weight update is identical to AdaGrad (see {{eq:adagrad-sum}}):

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\mathbf{E}_t + \epsilon}} \cdot \nabla L_t$$

RMSprop's use of an exponentially weighted moving average brings two benefits. First, the window effect: retaining only the gradient information from roughly the last $\frac{1}{1-\gamma}$ steps prevents the accumulator from only absorbing without discarding, so the learning rate does not monotonically decrease but instead exhibits mild fluctuations. Second, RMSprop can adapt more quickly to gradient changes — when gradients change sharply, $\mathbf{E}_t$ adjusts accordingly, making the learning rate more responsive. However, RMSprop also has a drawback (one that AdaGrad shares as well, and is not caused by RMSprop's improvement): it only uses squared gradients to adjust the learning rate without accumulating historical gradient directions, effectively forgoing the smoothing effect of [Momentum](gradient-descent.md#momentum). Is it possible to have both the stability of momentum and the flexibility of adaptive learning rates? This leads us to the widely used adaptive optimizer Adam.

## Adam

**Adam** (Adaptive Moment Estimation) was proposed by Diederik Kingma and Jimmy Ba at the International Conference on Learning Representations (ICLR) in 2015. The name Adam stems from its core mechanism: simultaneously estimating the [first moment](https://en.wikipedia.org/wiki/Moment_(mathematics)) (the mean) and the second moment (the uncentered variance) of gradients, combining momentum with adaptive learning rates. This dual-pronged strategy gives parameter updates both smooth directional stability and individualized step sizes, making Adam the most popular optimizer in deep learning today.

The key to this fusion is that Adam maintains two state variables simultaneously. The **first moment** $\mathbf{m}_t$ accumulates directional information from historical gradients, analogous to the velocity variable in momentum methods, smoothing the update path and suppressing oscillations. The **second moment** $\mathbf{v}_t$ accumulates the squares of historical gradients, similar to the moving average in RMSprop, adjusting the learning rate for each parameter. The two operate independently without interfering with each other: the first moment handles "which direction to go," while the second moment handles "how far to go." This division of labor makes Adam perform robustly across a wide range of tasks, with low sensitivity to hyperparameter choices. The default parameters ($\beta_1=0.9, \beta_2=0.999, \eta=0.001$) work well for most scenarios, which is an important reason for its widespread popularity.

Let $\mathbf{m}_t$ be the first moment estimate of the gradient (momentum), $\beta_1$ be the decay coefficient for the first moment (default $0.9$), and $\nabla L_t$ be the current gradient direction. The following formula expresses Adam's momentum accumulation process (consistent with the principle of [Momentum](gradient-descent.md#momentum)):

$$[eq:adam-m] \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla L_t$$

Let $\mathbf{v}_t$ be the second moment estimate of the gradient (accumulated gradients), $\beta_2$ be the decay coefficient for the second moment (default $0.999$), and $(\nabla L_t)^2$ be the square of the current gradient. The following formula expresses Adam's moving average process of squared gradients (identical to RMSprop's gradient accumulation formula {{eq:rmsprop-update}}):

$$ \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla L_t)^2$$

Adam initializes $\mathbf{m}_0 = 0, \mathbf{v}_0 = 0$. This seemingly natural choice, however, introduces a cold-start problem. Expanding the first moment formula {{eq:adam-m}}, $\mathbf{m}_t$ is essentially a weighted sum of historical gradients: $\mathbf{m}_t = (1-\beta_1)[\nabla L_t + \beta_1 \nabla L_{t-1} + \beta_1^2 \nabla L_{t-2} + ...]$. The sum of the weights should ideally be 1 to correctly reflect the weighted average of gradients. However, due to the initialization $\mathbf{m}_0 = 0$, the actual sum of weights is the geometric series $1 - \beta_1^t$. In early training, $t$ is small and $\beta_1^t$ is close to 1 (e.g., with $\beta_1=0.9$, at $t=1$, $\beta_1^t=0.9$), so the weight sum $1-\beta_1^t$ is far less than 1. The missing weight is occupied by the zero initialization, causing the estimate to be biased toward zero. The same flaw exists in the second moment formula. This is Adam's cold-start problem. Therefore, intermediate variables $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ are defined to correct the bias by dividing $\mathbf{m}_t$ and $\mathbf{v}_t$ by the weight sum to compensate for the missing portion:

$$ \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

Observing Adam's iteration process, in early training when $t$ is small, $\beta^t$ is close to 1, and the correction factor $\frac{1}{1 - \beta^t}$ reaches its maximum, amplifying the estimate to counteract the zero-initialization bias. In later training when $t$ is large, $\beta^t$ approaches 0, the correction factor approaches 1, and the correction effect disappears. To illustrate with concrete numbers, let $\beta_1 = 0.9$, $t = 1$, gradient $\nabla L_1 = 10$. The uncorrected $\mathbf{m}_1 = 0.9 \times 0 + 0.1 \times 10 = 1$. The corrected $\hat{\mathbf{m}}_1 = \frac{1}{1 - 0.9^1} = 10$, which exactly equals the actual gradient. Combining all four variables, Adam's weight update formula is:

$$[adam-w-update] \mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t$$

This formula can be understood as using the bias-corrected momentum $\hat{\mathbf{m}}_t$ to indicate the direction of progress (smoothed gradient), and the adaptive learning rate $\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}}$ to control the step size (individually adjusted per parameter). $\epsilon$ (typically $10^{-8}$) prevents division by zero, and $\eta$ is the global learning rate (typically $0.001$, smaller than SGD's default because the adaptive mechanism can make the effective step size for some parameters too large). Adam combines the advantages of momentum and adaptive learning rates and is widely used in computer vision, natural language processing, recommendation systems, and other fields, making it the default optimizer in deep learning research.

Adam provides four hyperparameters, but apart from the global learning rate, the other three can usually be left at their default values. In practice, if the loss oscillates, reduce the learning rate; if convergence is slow, increase the learning rate:

| Hyperparameter | Default | Role |
|:------|:------|:-----|
| $\eta$ | $0.001$ | Global learning rate, typically in the range $[10^{-4}, 10^{-2}]$ |
| $\beta_1$ | $0.9$ | First moment decay coefficient, controls momentum smoothing |
| $\beta_2$ | $0.999$ | Second moment decay coefficient, controls learning rate adaptivity |
| $\epsilon$ | $10^{-8}$ | Numerical stability constant, prevents division by zero |

## AdamW

Adam appears to have gathered all the advantages of momentum and adaptive learning rates, becoming the default optimizer in deep learning. However, a 2019 paper revealed a hidden theoretical flaw: the implementation of weight decay (L2 regularization) in Adam conflicts with adaptive learning rates, making the regularization effect unstable. This issue led to Adam's latest corrected version, AdamW.

Weight decay is a common technique for preventing overfitting. In SGD, it is essentially equivalent to [L2 regularization](../../statistical-learning/linear-models/regularization-glm.md#regularization-principle). A penalty term is added to the loss function to force parameters to remain small: $L_{total} = L_{data} + \lambda \|\mathbf{W}\|^2$. Correspondingly, the gradient becomes $\nabla L_{total} = \nabla L_{data} + 2\lambda \mathbf{W}$. In SGD, the implementation of weight decay is simple and intuitive:

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla L_{data} - 2\eta \lambda \mathbf{W}_t = \mathbf{W}_t(1 - 2\eta \lambda) - \eta \nabla L_{data}$$

Each step multiplies the weights by $(1 - 2\eta \lambda)$, gradually decaying them, treating all parameters equally. But Adam's implementation differs. The gradient $\nabla L_{total} = \nabla L_{data} + 2\lambda \mathbf{W}$ is accumulated into the first moment $\mathbf{m}_t$ and second moment $\mathbf{v}_t$, so the weight decay term $2\lambda \mathbf{W}$ is also scaled by the adaptive learning rate:

$$\Delta \mathbf{W}_{reg} = -\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot 2\lambda \mathbf{W}$$

This leads to a problem. When $\hat{\mathbf{v}}_t$ is large (the parameter's gradient history is active), the adaptive learning rate $\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}}$ is small, causing the weight decay term to be shrunk and the regularization effect to weaken. When $\hat{\mathbf{v}}_t$ is small (the parameter's gradient history is sparse), the adaptive learning rate is large, causing the weight decay term to be amplified. This clearly contradicts the original design intent of L2 regularization, which is to uniformly decay all weights. Recall Adam's mechanism: gradient updates are scaled by the adaptive learning rate so that frequently updated parameters slow down and sparsely updated parameters accelerate. However, the purpose of weight decay is to uniformly constrain all parameters, preventing any single parameter from becoming too large and causing overfitting — a goal entirely different from adaptive adjustment. Mixing the two means that regularization, which should treat all parameters equally, is also subjected to "teaching according to aptitude," which is logically contradictory.

**AdamW** (Adam with Decoupled Weight Decay) was proposed by Ilya Loshchilov in the 2019 paper "[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)". The paper reveals Adam's weight decay problem and provides a concise solution: separate weight decay from gradient updates. Let the gradient update be responsible for learning data patterns, and weight decay be responsible for controlling model complexity — the two should operate independently without interference. Compared to Adam (see {{adam-w-update}}), AdamW's update rule differs by the addition of a weight decay term:

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t - \eta \lambda \mathbf{W}_t$$

Here, the term $-\eta \lambda \mathbf{W}_t$ is the weight decay term, applied directly to the parameters without being scaled by the adaptive learning rate. The term $-\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \cdot \hat{\mathbf{m}}_t$ is the gradient update term, scaled by the adaptive learning rate. The overall formula can be understood as weight decay executing independently and gradient updates executing adaptively — the two are decoupled and do not interfere. In practice, AdamW can now replace Adam in almost all scenarios. Especially on tasks requiring strong regularization, such as large-scale language model training, AdamW has become the default optimizer for many Transformer-based models, such as BERT and GPT.

## Muon

A common feature of the optimizers introduced so far is that they assign an independent learning rate to each parameter, adjusting the step size based on gradient history. This per-parameter adaptive strategy has been highly successful, but neural network parameters are not truly independent scalars — there are correlations between parameters. The weights of a neural network's hidden layer are naturally two-dimensional matrices, where rows correspond to transformations of input features and columns correspond to combinations of output neurons. Flattening the matrix into a vector and adjusting the learning rate element by element effectively discards the geometric information of the matrix. In December 2024, Keller Jordan proposed the Muon optimizer in a blog post, taking a different path from the traditional per-parameter adaptive learning rate approach by leveraging the matrix's geometric structure through orthogonalization of the momentum update.

The name Muon is derived from the somewhat mouthful phrase "Momentum Orthogonalized by Newton-Schulz" (**M**oment**u**m **O**rthogonalized by **N**ewton-Schulz), yet its concrete steps are surprisingly simple, consisting of just the following two:

- **Step one: momentum accumulation.** Like momentum methods, Muon maintains a weighted average of historical gradients (a momentum buffer). When a new gradient arrives, it retains most of the old direction while incorporating a small portion of the new direction, thereby smoothing the update path and suppressing oscillations.

- **Step two: orthogonalization.** This is where Muon differs from all the previously discussed optimizers. Think of a matrix as a combination of multiple independent directions and their strengths. For example, a weight matrix might push very aggressively in some directions (large strength) while only lightly touching in others (small strength). Orthogonalization does exactly this: it retains only the directions themselves while leveling all their strengths to the same magnitude. It is like a team with loud members and quiet members — orthogonalization essentially gives everyone a microphone at the same volume, so all voices can be heard equally.

The reason orthogonalization is needed is that Jordan discovered, through experimentation, that the update matrices produced by SGD momentum and Adam for model hidden layers are often dominated by a few strong directions, while the magnitudes of the many remaining "rare directions" are negligible. These rare directions, though inconspicuous, may carry information crucial for learning. Orthogonalization amplifies these weak directions to the same intensity as the dominant ones, allowing them to participate equally in parameter updates, preventing the network from developing blind spots during learning. The standard approach to orthogonalization is to first perform [singular value decomposition](../../statistical-learning/unsupervised-learning/dimensionality-reduction.md#singular-value-decomposition) on the matrix, separating directions from strengths, then discard the strengths and keep only the directions. However, the computational cost of singular value decomposition is too high to run at every training step. Muon uses Newton-Schulz iteration to bypass singular value decomposition — this is the origin of the "N" in its name.

Newton-Schulz iteration exploits a mathematical fact: if a carefully constructed polynomial function is repeatedly applied to a matrix, the function acts on the "strength" part of the matrix rather than the "direction" part, gradually pushing all strengths toward 1, ultimately approximating orthogonalization. The entire process requires only matrix multiplication, without explicit matrix decomposition. Jordan's team chose a quintic polynomial and tuned its three coefficients through extensive experiments, achieving stable orthogonalization in just five iterations under BF16 precision. Before the iteration begins, the matrix is first scaled to unit length, ensuring all strengths lie between 0 and 1. Then, with each iteration, the polynomial is applied to the matrix once: initially small strengths are rapidly pulled up, gradually approaching 1. After five iterations, the matrix is already very close to the orthogonalized result. Compared to directly performing singular value decomposition (too slow) and another method called coupled Newton iteration (requires FP32 precision, inefficient on GPUs), Newton-Schulz iteration can run in native BF16 precision on modern GPUs, offering both speed and numerical stability — this is the key to its ability to be embedded in the training loop.

After completing these two steps, Muon uses the orthogonalized momentum as the final update, multiplied by the global learning rate and applied to the weights. The entire process can be summarized as: momentum accumulation provides a smooth direction, orthogonalization levels all direction strengths, and the learning rate controls the overall step size. However, since Muon leverages the two-dimensional geometric relationship of matrices, it is only applicable to the two-dimensional weight matrices of neural network hidden layers. For scalar parameters (such as biases), vector parameters (such as scaling factors in layer normalization), and input/output embedding layers, standard optimizers like AdamW are still needed. This hybrid optimizer strategy (Muon for hidden layers, AdamW for the rest) is the mainstream approach for language models in 2026.

Muon offers tremendous advantages in both resources and performance. In terms of performance, Muon excels across multiple benchmarks. It compresses the training time to reach 94% accuracy on CIFAR-10 by approximately 21%. In the NanoGPT speed benchmark (FineWeb dataset), Muon improves training speed by 1.35 times. When training a 1.5B parameter Transformer to GPT-2 XL level performance with 8 H100 GPUs, Muon takes only 10 hours, whereas AdamW requires 13.3 hours. Subsequently, large-scale projects such as the Moonlight 16B mixture-of-experts model (5.7T tokens of training data) have further validated Muon's feasibility in industrial-grade scenarios. In terms of resources, both the computational and memory overhead of Muon are very low. In typical language model training (e.g., Llama 405B, model dimension 16384, batch size 16M tokens), the additional floating-point operations from the five Newton-Schulz iterations account for less than 1% of the total. In terms of memory consumption, Muon maintains only one momentum buffer, saving approximately half the optimizer state memory compared to AdamW, significantly lowering the resource barrier for training environments.

Muon represents a new direction in optimizer design — moving from independently adjusting learning rates for each scalar parameter to directly leveraging the matrix structure of parameters and improving the update direction through geometric transformations. Of course, this matrix-native optimization is still in its early stages, covering only linear layers and not yet extended to more complex components such as attention mechanisms. However, it has already proven its value in speed benchmarks and has inspired a subsequent series of optimizer research based on matrix structure.

## Optimizer Selection Guide

At this point, we have studied eight optimizers: SGD, Momentum, NAG, AdaGrad, RMSprop, Adam, AdamW, and Muon. Each optimizer has its own characteristics. This section provides a straightforward "two-step" selection guide from the perspectives of both optimizer features and task types.

- Step one: first understand the characteristics of each optimizer:

    | Optimizer | Core Mechanism | Advantage | Disadvantage | Suitable Scenario |
    |:------|:--------|:-----|:-----|:--------|
    | SGD | Basic gradient descent | Simple, stable | Oscillation, slow | Simple tasks, fine-tuning |
    | Momentum | Momentum smoothing | Acceleration, suppresses oscillation | Requires learning rate tuning | General purpose |
    | NAG | Gradient at predicted position | Anticipates inflection points | Slightly more complex | Pursuing precision |
    | AdaGrad | Accumulates squared gradients | Sparse gradient friendly | Learning rate decays | Sparse data, short-term training |
    | RMSprop | Moving average of squared gradients | Stable learning rate | No momentum | Long-term training, RNN |
    | Adam | Momentum + adaptive | Robust, fast | Weight decay issue | General purpose, default choice |
    | AdamW | Adam + decoupled weight decay | Stable regularization | Additional second-moment memory overhead vs SGD | Default choice |
    | Muon | Momentum + Newton-Schulz orthogonalization | Leverages matrix structure, memory efficient | Only applies to 2D matrix parameters | Transformer hidden layers, large-scale training |

- Step two: select the optimizer based on task characteristics:

    | Task Type | Recommended Optimizer | Reason |
    |:--------|:----------|:-----|
    | General deep learning | AdamW | Strong robustness, default choice |
    | Computer vision | SGD + Momentum | Experiments show better generalization |
    | Natural language processing | AdamW | Sparse gradients, clear adaptive advantage |
    | RNN/LSTM | RMSprop / AdamW | Adapts to gradient scale differences |
    | Fine-tuning pretrained models | SGD + Momentum | Small learning rate fine-tuning, prevents damaging pretrained features |
    | Sparse data (recommendation systems) | AdamW | Sparse parameters receive large learning rates |
    | Large-scale language model training | Muon + AdamW | Muon for hidden layers to accelerate, AdamW for the rest |

## Adaptive Optimizer in Practice

Theoretical analysis has revealed the design principles of each optimizer. Next, we compare the convergence behavior of SGD, Momentum, NAG, AdaGrad, RMSprop, Adam, AdamW, and Muon on a quadratic loss function through code experiments. The experiment uses a long elliptical loss function (with large gradient differences across directions), starting from the same initial point, and observes the parameter path, loss curve, and effective learning rate changes of each optimizer. The code implements the complete update logic for all eight optimizers, including momentum accumulation, gradient squared accumulation, moving average, bias correction, gradient computation at the predicted position, weight decay decoupling, and Newton-Schulz orthogonalization. Note that Muon is designed specifically for two-dimensional matrix parameters; in this experiment it is applied to a two-dimensional vector (treated as a $2 \times 1$ matrix), where orthogonalization degenerates to direction normalization. Its matrix structure advantages can only be fully realized in actual neural network training.

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# Define loss function and gradient - 10x gradient difference to demonstrate SGD oscillation
def loss_function(W):
    """Quadratic loss function L = 0.5 * W^T A W"""
    A = np.array([[1, 0], [0, 10]])  # 10x gradient difference
    return 0.5 * np.dot(W, A @ W)

def gradient(W):
    """Gradient ∇L = A W"""
    A = np.array([[1, 0], [0, 10]])
    return A @ W

# Optimizer implementations
class SGD:
    def __init__(self, lr=0.15):  # learning rate > 0.1 causes oscillation (lr > 1/gradient difference)
        self.lr = lr
        self.path = []

    def step(self, W, grad):
        W_new = W - self.lr * grad
        self.path.append(W_new.copy())
        return W_new

class Momentum:
    def __init__(self, lr=0.05, momentum=0.9):  # smaller learning rate to avoid excessive oscillation
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(2)
        self.path = []

    def step(self, W, grad):
        self.v = self.momentum * self.v + self.lr * grad
        W_new = W - self.v
        self.path.append(W_new.copy())
        return W_new

class NAG:
    """Nesterov Accelerated Gradient - compute gradient at lookahead position"""
    def __init__(self, lr=0.05, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(2)
        self.path = []

    def step(self, W, grad_func):
        # NAG core: compute gradient at lookahead position, respond to inflection points early
        W_lookahead = W - self.momentum * self.v
        grad_lookahead = grad_func(W_lookahead)
        self.v = self.momentum * self.v + self.lr * grad_lookahead
        W_new = W - self.v
        self.path.append(W_new.copy())
        return W_new

class AdaGrad:
    def __init__(self, lr=1.0, eps=1e-8):  # large learning rate, fast initial convergence
        self.lr = lr
        self.eps = eps
        self.G = np.zeros(2)
        self.path = []

    def step(self, W, grad):
        self.G += grad ** 2  # accumulate squared gradients
        lr_adaptive = self.lr / np.sqrt(self.G + self.eps)  # decreasing learning rate
        W_new = W - lr_adaptive * grad
        self.path.append(W_new.copy())
        return W_new

class RMSprop:
    def __init__(self, lr=0.3, gamma=0.9, eps=1e-8):  # moderate learning rate
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.E = np.zeros(2)
        self.path = []

    def step(self, W, grad):
        self.E = self.gamma * self.E + (1 - self.gamma) * (grad ** 2)  # moving average
        lr_adaptive = self.lr / np.sqrt(self.E + self.eps)  # stable learning rate
        W_new = W - lr_adaptive * grad
        self.path.append(W_new.copy())
        return W_new

class Adam:
    def __init__(self, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(2)  # first moment (momentum)
        self.v = np.zeros(2)  # second moment (squared gradients)
        self.t = 0
        self.path = []

    def step(self, W, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad  # momentum accumulation
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)  # accumulate squared gradients

        # bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        W_new = W - self.lr / (np.sqrt(v_hat) + self.eps) * m_hat
        self.path.append(W_new.copy())
        return W_new

class AdamW:
    def __init__(self, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0
        self.path = []

    def step(self, W, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Decoupled weight decay: applied directly to parameters, not scaled by adaptive learning rate
        W_new = W - self.lr * self.weight_decay * W
        W_new = W_new - self.lr / (np.sqrt(v_hat) + self.eps) * m_hat
        self.path.append(W_new.copy())
        return W_new

class Muon:
    """Muon optimizer - accumulates momentum then orthogonalizes the update matrix via Newton-Schulz iteration"""
    def __init__(self, lr=0.3, momentum=0.9, ns_steps=5):
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.B = np.zeros((2, 1))  # momentum buffer (matrix form)
        self.path = []

    def _newtonschulz(self, G):
        """Newton-Schulz iteration for approximate orthogonalization (G ≈ U S V^T → U V^T)"""
        a, b, c = (3.4445, -4.7750, 2.0315)  # optimized coefficients for quintic polynomial
        X = G / (np.linalg.norm(G) + 1e-7)
        if G.shape[0] > G.shape[1]:
            X = X.T
        for _ in range(self.ns_steps):
            A = X @ X.T
            B_mat = b * A + c * A @ A
            X = a * X + B_mat @ X
        if G.shape[0] > G.shape[1]:
            X = X.T
        return X

    def step(self, W, grad):
        grad_mat = grad.reshape(-1, 1)
        self.B = self.momentum * self.B + grad_mat  # momentum accumulation
        O = self._newtonschulz(self.B)  # Newton-Schulz orthogonalization
        W_new = W - self.lr * O.flatten()
        self.path.append(W_new.copy())
        return W_new

# Run experiment
W_init = np.array([5.0, 5.0])  # starting point
n_iterations = 50

optimizers = {
    'SGD': SGD(lr=0.15),
    'Momentum': Momentum(lr=0.05, momentum=0.9),
    'NAG': NAG(lr=0.05, momentum=0.9),
    'AdaGrad': AdaGrad(lr=1.0),
    'RMSprop': RMSprop(lr=0.3, gamma=0.9),
    'Adam': Adam(lr=0.3),
    'AdamW': AdamW(lr=0.3, weight_decay=0.01),
    'Muon': Muon(lr=0.3, momentum=0.9)
}

results = {}
for name, opt in optimizers.items():
    W = W_init.copy()
    losses = []

    for t in range(n_iterations):
        loss = loss_function(W)
        losses.append(loss)
        grad = gradient(W)

        # NAG requires the gradient function, other optimizers receive gradient values
        if name == 'NAG':
            W = opt.step(W, gradient)
        else:
            W = opt.step(W, grad)

    results[name] = {
        'path': np.array(opt.path),
        'losses': losses,
        'final_W': W,
        'final_loss': loss_function(W)
    }

    print(f"{name:10s}: final position ({W[0]:.4f}, {W[1]:.4f}), final loss {loss_function(W):.6f}")

print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

colors = {'SGD': '#e74c3c', 'Momentum': '#3498db', 'NAG': '#e67e22',
          'AdaGrad': '#f39c12', 'RMSprop': '#9b59b6', 'Adam': '#2ecc71', 'AdamW': '#1abc9c',
          'Muon': '#e91e63'}

# Plot 1: parameter paths
ax1 = axes[0, 0]
W1_range = np.linspace(-6, 6, 100)
W2_range = np.linspace(-6, 6, 100)
W1_grid, W2_grid = np.meshgrid(W1_range, W2_range)
L_grid = 0.5 * (W1_grid**2 + 10 * W2_grid**2)

ax1.contour(W1_grid, W2_grid, L_grid, levels=[1, 5, 10, 25, 50, 100],
           colors='gray', alpha=0.5, linewidths=0.5)
ax1.contourf(W1_grid, W2_grid, L_grid, levels=[0, 1, 5, 10, 25, 50, 100, 200],
             cmap='Blues', alpha=0.3)

for name, result in results.items():
    path = result['path']
    ax1.plot(path[:, 0], path[:, 1], 'o-', color=colors[name],
             linewidth=2, markersize=3, alpha=0.7, label=name)

ax1.plot(W_init[0], W_init[1], 'ko', markersize=10, label='Start')
ax1.plot(0, 0, 'k*', markersize=15, label='Minimum')
ax1.set_xlabel('W1', fontsize=11)
ax1.set_ylabel('W2', fontsize=11)
ax1.set_title('Parameter Path Comparison', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)

# Plot 2: loss curves
ax2 = axes[0, 1]
for name, result in results.items():
    ax2.plot(result['losses'], color=colors[name], linewidth=2, label=name)

ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Loss Curve', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: effective learning rate (W1 direction) - adaptive optimizers only
ax3 = axes[1, 0]
adaptive_optimizers = ['AdaGrad', 'RMSprop', 'Adam', 'AdamW']

for name, result in results.items():
    if name in adaptive_optimizers:
        path = result['path']
        lr_eff = []
        for i in range(len(path) - 1):
            W1_change = path[i+1, 0] - path[i, 0]
            W1_grad = path[i, 0]  # grad_W1 ≈ W1
            lr_eff.append(np.abs(W1_change) / np.abs(W1_grad + 1e-8))
        ax3.plot(lr_eff[:min(30, len(lr_eff))], color=colors[name], linewidth=2, label=name, alpha=0.7)

ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('Effective Learning Rate (W1)', fontsize=11)
ax3.set_title('Adaptive Learning Rate', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: convergence speed comparison (loss decrease rate)
ax4 = axes[1, 1]

for name, result in results.items():
    losses = result['losses']
    loss_decrease = [losses[i] - losses[i+1] for i in range(len(losses)-1)]
    ax4.plot(loss_decrease[:min(30, len(loss_decrease))], color=colors[name], linewidth=2, label=name, alpha=0.7)

ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('Loss Decrease Per Step', fontsize=11)
ax4.set_title('Convergence Speed Comparison', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
```

## Chapter Summary

This chapter introduced the principles and applications of adaptive optimizers, demonstrating how the idea of assigning different learning rates to each parameter improves optimization efficiency, and presented Muon, a new optimizer that leverages matrix geometric structure. At this point, the content on neural network optimization is complete. We have mastered the principles and applications of gradient descent, momentum, NAG, adaptive optimizers, and Muon, and understand how to select and tune optimization algorithms. The next chapter moves into neural network stability, covering weight initialization, Dropout, batch normalization, and gradient problem diagnosis to address training stability issues.

## Practice Problems

1. Analyze why AdaGrad's learning rate decreases monotonically. Assume the gradient is constant at $g$, derive the effective learning rate $\eta_{eff} = \frac{\eta}{\sqrt{t \cdot g^2}}$ after $t$ steps. What is the impact of this property?
    <details>
    <summary>Reference Answer</summary>

    **AdaGrad Learning Rate Decay Derivation**:

    AdaGrad accumulates squared gradients:

    $$G_t = \sum_{i=1}^{t} g_i^2$$

    Assuming a constant gradient $g$ ($g_i = g$):

    $$G_t = \sum_{i=1}^{t} g^2 = t \cdot g^2$$

    Effective learning rate:

    $$\eta_{eff} = \frac{\eta}{\sqrt{G_t + \epsilon}} \approx \frac{\eta}{\sqrt{t \cdot g^2}} = \frac{\eta}{g \sqrt{t}}$$

    The effective learning rate is inversely proportional to $\sqrt{t}$, monotonically decreasing with the number of iterations.

    **Impact Analysis**:

    1. **Fast initial convergence**: Early in training, $t$ is small, $\eta_{eff}$ is large, parameters update quickly.

    2. **Slow later convergence**: Later in training, $t$ is large, $\eta_{eff}$ is small, parameter updates become extremely slow.

    3. **Long-term training stagnation**: When $t$ is very large (e.g., $t = 10^6$), $\eta_{eff} \approx \frac{\eta}{g \cdot 1000}$, the learning rate is minuscule, and training nearly stops.

    **Numerical Example**: Let $\eta = 0.1$, $g = 1$:

    | $t$ | $\eta_{eff}$ | Parameter Update Magnitude |
    |:---:|:-----------|:-----------|
    | 1 | 0.1 | Large |
    | 100 | 0.01 | Medium |
    | 10000 | 0.001 | Small |
    | $10^6$ | 0.0001 | Very small |

    **Conclusion**:

    AdaGrad's monotonically decreasing learning rate leads to:
    - Suitable for short-term training or sparse gradient problems (frequently updated parameters receive small learning rates, sparse parameters receive large learning rates)
    - Unsuitable for long-term training: learning rate decays too early, training stagnates in later stages
    - RMSprop solves this problem through a moving average

    **Improvement Direction**:

    RMSprop uses an exponentially weighted moving average:

    $$E_t = \gamma E_{t-1} + (1-\gamma) g^2$$

    With a constant gradient $g$:

    $$E_t = (1-\gamma) g^2 \sum_{i=0}^{t-1} \gamma^i = (1-\gamma) g^2 \frac{1-\gamma^t}{1-\gamma} \approx g^2$$

    $E_t$ converges to $g^2$ (a stable value) rather than growing indefinitely. The effective learning rate $\eta_{eff} = \frac{\eta}{\sqrt{g^2}} = \frac{\eta}{g}$ remains stable.

    **Summary**: AdaGrad's monotonically decreasing learning rate stems from the cumulative growth of squared gradient accumulation. This property makes AdaGrad suitable for short-term training and sparse gradients, but long-term training will stagnate. RMSprop uses a moving average to avoid accumulation, keeping the learning rate stable and suitable for long-term training.
    </details>

2. Explain the necessity of Adam's bias correction. Let $\mathbf{m}_0 = 0$, $\beta_1 = 0.9$, gradient $\nabla L_1 = 10$. Compute the uncorrected $\mathbf{m}_1$ and the corrected $\hat{\mathbf{m}}_1$, and analyze the difference.
    <details>
    <summary>Reference Answer</summary>

    **Bias Correction Calculation**:

    Adam first moment estimate:

    $$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L_t$$

    Let $\beta_1 = 0.9$, $m_0 = 0$, $\nabla L_1 = 10$:

    **Uncorrected**:

    $$m_1 = 0.9 \cdot 0 + 0.1 \cdot 10 = 1$$

    **Corrected**:

    $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{1}{1 - 0.9} = \frac{1}{0.1} = 10$$

    **Difference Analysis**:

    - Uncorrected $m_1 = 1$ (biased toward zero)
    - Corrected $\hat{m}_1 = 10$ (equal to the actual gradient)
    - Correction factor $\frac{1}{1-\beta_1^t} = 10$ amplifies $m_1$

    **Cause of Bias**:

    Adam initializes $m_0 = 0$, and the first moment estimate is a weighted average of historical gradients:

    $$m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \nabla L_i$$

    Sum of weights:

    $$\sum_{i=1}^{t} (1-\beta_1) \beta_1^{t-i} = (1-\beta_1) \frac{1-\beta_1^t}{1-\beta_1} = 1 - \beta_1^t$$

    When $t$ is small (early training), the weight sum $1 - \beta_1^t < 1$:

    - $t=1$: weight sum $= 0.1$
    - $t=2$: weight sum $= 0.19$
    - $t=10$: weight sum $= 0.65$

    The weight sum is less than 1, so the estimate is biased toward zero (because the initialization $m_0 = 0$ occupies the missing weight).

    **Correction Principle**:

    Bias correction counteracts the initialization bias:

    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

    Multiplying the estimate by $\frac{1}{1-\beta_1^t}$ compensates for the missing weight.

    When $t$ is large (late training), $\beta_1^t \to 0$:

    - $t=100$: $\beta_1^{100} \approx 0$
    - Correction factor $\frac{1}{1-\beta_1^{100}} \approx 1$
    - $\hat{m}_{100} \approx m_{100}$

    The correction effect disappears.

    **Numerical Example**: Assuming a constant gradient of 10:

    | $t$ | $m_t$ (uncorrected) | $\hat{m}_t$ (corrected) | Ratio |
    |:---:|:-------------|:-----------------|:---:|
    | 1 | 1 | 10 | 10x |
    | 5 | 4.1 | 6.9 | 1.7x |
    | 10 | 6.5 | 10 | 1.5x |
    | 100 | 9.99 | 10 | 1x |

    **Conclusion**:

    The necessity of Adam's bias correction:
    1. Early training (small $t$): initialization $m_0=0$ biases the estimate toward zero; correction amplifies the estimate, offsetting the bias.
    2. Late training (large $t$): weight sum approaches 1, bias disappears, correction effect diminishes.
    3. Bias correction ensures accurate gradient estimation in early training, avoiding the cold-start problem.

    **Practical Significance**:

    Without bias correction, Adam's learning rate in early training could be too small (because $m_t$ is biased toward zero), slowing parameter updates. Bias correction normalizes the early learning rate, accelerating convergence.

    This is why Adam's bias correction is a critical design — it solves the cold-start problem caused by initialization bias.
    </details>

3. Explain why AdamW's weight decay effect is more stable than Adam's. Analyze the problem of L2 regularization gradients being scaled by the adaptive learning rate in Adam.
    <details>
    <summary>Reference Answer</summary>

    **Weight Decay Implementation Problem in Adam**:

    L2 regularization adds a penalty term to the loss function:

    $$L_{total} = L_{data} + \lambda ||\mathbf{W}||^2$$

    The gradient becomes:

    $$\nabla L_{total} = \nabla L_{data} + 2\lambda \mathbf{W}$$

    Adam accumulates the gradient into the first moment $m_t$ and second moment $v_t$:

    $$m_t = \beta_1 m_{t-1} + (1-\beta_1)(\nabla L_{data} + 2\lambda \mathbf{W})$$
    $$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L_{data} + 2\lambda \mathbf{W})^2$$

    Parameter update:

    $$\Delta \mathbf{W} = -\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

    The weight decay term $2\lambda \mathbf{W}$ is included in $\hat{m}_t$ and is scaled by the adaptive learning rate $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$.

    **Problem Analysis**:

    When a parameter has an active gradient history ($\hat{v}_t$ is large):

    - Adaptive learning rate $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$ is small
    - Weight decay term $2\lambda \mathbf{W}$ is shrunk, regularization effect weakens

    When a parameter has a sparse gradient history ($\hat{v}_t$ is small):

    - Adaptive learning rate $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$ is large
    - Weight decay term $2\lambda \mathbf{W}$ is amplified, regularization effect strengthens

    This contradicts the original design intent of L2 regularization to "uniformly decay all weights." L2 regularization should decay all weights equally (multiplying each step by $1 - \eta \lambda$), but Adam makes the weight decay effect depend on the parameter's gradient history.

    **Numerical Example**:

    Let $\eta = 0.001$, $\lambda = 0.01$, $\epsilon = 10^{-8}$:

    | Parameter | $\hat{v}_t$ | Adaptive Learning Rate | Weight Decay Magnitude |
    |:----|:----------|:-----------|:-----------|
    | Active parameter | 100 | $\frac{0.001}{10} = 10^{-4}$ | $10^{-4} \cdot 0.02 \approx 0$ |
    | Sparse parameter | 1 | $\frac{0.001}{1} = 10^{-3}$ | $10^{-3} \cdot 0.02 = 2 \times 10^{-5}$ |

    Weight decay for active parameters is nearly zero, while it is larger for sparse parameters. The regularization effect is uneven.

    **AdamW's Decoupled Design**:

    AdamW separates weight decay from gradient updates:

    $$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \lambda \mathbf{W}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

    The weight decay term $-\eta \lambda \mathbf{W}_t$ is applied directly to the parameters, not scaled by the adaptive learning rate.

    Each step multiplies the weights by $(1 - \eta \lambda)$:

    $$\mathbf{W}_{t+1} = \mathbf{W}_t(1 - \eta \lambda) - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

    All weights decay equally, consistent with SGD's behavior.

    **AdamW vs Adam Weight Decay Comparison**:

    | Property | Adam | AdamW |
    |:-----|:-----|:------|
    | Weight decay implementation | Adds L2 term to gradient | Directly decays weights |
    | Decay effect | Affected by $\hat{v}_t$ (uneven) | Stable and uniform |
    | Hyperparameter coupling | $\eta$ and $\lambda$ coupled | Decoupled and independent |

    **Conclusion**:

    Adam's weight decay problem arises from L2 regularization gradients being scaled by the adaptive learning rate. The decay effect weakens for active parameters and strengthens for sparse parameters, making regularization uneven.

    AdamW decouples weight decay, applying it directly to parameters, resulting in a stable and uniform decay effect. Experiments show that AdamW generalizes better than Adam, especially on tasks requiring strong regularization.

    **Recommendation**: Prefer AdamW over Adam for more stable and reliable weight decay.
    </details>
