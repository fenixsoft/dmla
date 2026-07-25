# Batch Normalization

In the training of deep neural networks, there exists a subtle yet fatal problem: as the network depth increases, the input distribution of each layer keeps changing, leading to unstable gradient propagation. This phenomenon is called **Internal Covariate Shift**, which makes deep networks difficult to train, slow to converge, and highly sensitive to initialization.

In 2015, two Google researchers, Sergey Ioffe and Christian Szegedy, proposed **Batch Normalization** (BN) in their paper _[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)_. Inspired by the idea of data standardization in traditional machine learning, they extended it to every layer of neural networks, achieving real-time, adaptive standardization during training. BN makes deep network training more stable and faster, allows the use of larger learning rates, alleviates initialization sensitivity, and even provides a certain regularization effect. Starting from ResNet, BN has become a standard component of almost all modern deep networks.

This chapter will delve into the nature of the internal covariate shift problem, introduce the algorithmic principles and computation flow of BN, explore how BN improves training stability, explain its special application in convolutional neural networks, distinguish between training and inference modes, and verify BN's practical effects through experiments. Finally, we will discuss BN's limitations and its variants to help readers make informed choices in different scenarios.

## Internal Covariate Shift

Earlier, we addressed problems affecting training stability such as vanishing gradients, exploding gradients, and improper initialization. In this section, we discuss the internal covariate shift problem, which is not as easily observable as those mentioned above but fundamentally impacts the training efficiency of networks. Let us explain internal covariate shift with a concrete scenario:

Imagine we are training a neural network to predict housing prices, with input features including area, location, age, etc., and the output being the predicted price. This network has three hidden layers using the Sigmoid activation function. At a certain point in training, the pre-activation values (input to Sigmoid) of the second hidden layer are distributed between 0.3 and 0.6, with a mean around 0.45. These values, after passing through the Sigmoid function, happen to fall within its comfortable working range — Sigmoid is most sensitive near input 0, where the gradient is largest; while the range 0.3 to 0.6 corresponds to a moderate gradient, and training proceeds normally. However, after the parameters of the first hidden layer are updated, the input to the second hidden layer changes. The values that were distributed between 0.3 and 0.6 now become -5.5 to -5.0, with a mean of -5.3. The gradients of the Sigmoid function at these input values become very small. The third hidden layer suddenly finds that the distribution of its input data has completely changed, as if unrelated to the previous round, and the behavior of the activation function has also changed, making previously learned parameters no longer applicable. This is the internal covariate shift phenomenon: during network training, due to parameter updates in preceding layers, the input distribution of subsequent layers continuously changes, forcing each layer to constantly adapt to these changes instead of focusing on learning stable feature representations.

From the perspective of information flow, a deep network is like a multi-stage signal processing system. In traditional systems, we standardize the input signal to ensure it maintains stable statistical properties across processing units. However, in deep networks, since the parameters of each processing unit (hidden layer) are dynamically updated, the statistical properties of the signal change after each layer. Without control, these changes amplify layer by layer, ultimately causing the signal to deviate severely from the normal range in deep layers, either entering the saturation region of activation functions (causing vanishing gradients) or entering extreme regions (causing exploding gradients). This leads to the following four problems:

1. **Unstable training and difficult convergence.** Continuous changes in input distribution mean the gradient direction also changes, causing parameter updates to oscillate. Optimization in one direction may be interrupted by the next round of distribution changes, leading to slow convergence or even failure to converge.
2. **Learning rate forced to be low, limiting training acceleration.** In traditional optimization theory, a larger learning rate means larger steps, which can accelerate convergence. However, in networks with covariate shift, a large learning rate amplifies the problem. Large parameter updates cause drastic changes in input distribution, potentially destabilizing the entire network. For safety, only small learning rates can be used, sacrificing convergence speed for training stability.
3. **Abnormal activation function behavior, exacerbating gradient problems.** Distribution shifts can push activation values into dangerous regions. For Sigmoid and tanh functions, if input values enter the saturation region (e.g., when the absolute value of Sigmoid input exceeds 4, the output approaches 0 or 1 and the derivative approaches 0), gradients vanish; for the ReLU function, if input values are consistently negative (distribution shifts push most values into the negative range), many neurons die, output is constantly 0, and gradients completely vanish. Conversely, if distribution shifts make activation values extremely large, gradients may explode. Covariate shift and gradient problems reinforce each other.
4. **Increased initialization sensitivity, reduced fault tolerance.** Initialization is already critical for deep networks, and covariate shift further amplifies its impact. A good initialization keeps activation values within a reasonable range; but covariate shift can quickly disrupt this balance. A poor initialization (e.g., improper weight variance setting) causes activation values to deviate from the normal range from the start, and covariate shift amplifies this problem layer by layer. This makes the network highly sensitive to initialization, greatly reducing fault tolerance.

## BN Algorithm Principles

Since changes in the input distribution of each layer are the root cause of training difficulty, we need to introduce a mechanism in each layer that forces the input distribution to be standardized and stable. This idea is not actually new. In traditional machine learning, **data standardization** is already the most basic preprocessing step. For a feature vector $x$, we typically standardize it as:

$$\hat{x} = \frac{x - \mu}{\sigma}$$

where $\mu$ is the mean of the training data and $\sigma$ is the standard deviation. The standardized feature $\hat{x}$ has zero mean and unit variance, a distribution more conducive to optimization algorithms, with more stable gradient directions and allowing larger learning rates. The problem now is that in deep networks, the distribution of the "training data" for each layer (i.e., the output of the previous layer) changes dynamically. We cannot pre-compute fixed $\mu$ and $\sigma$ because these statistics evolve throughout training. Moreover, pure data standardization can introduce new problems: forcing all activation values to have zero mean and unit variance inevitably limits the network's representational capacity. The nonlinear characteristics of the Sigmoid function are stronger when inputs are far from zero; forcing inputs near zero could make the network overly linear.

BN's solution revolves around these two problems. First, it uses [Batch](../neural-network-structure/forward-propagation.md#batch-computation-and-efficiency-optimization) statistics instead of global statistics. Since fixed $\mu$ and $\sigma$ cannot be pre-computed, BN estimates statistics using the current Mini-Batch data each time parameters are updated. Although each Mini-Batch is only a subset of all data, its mean and variance are sufficient to reflect the general characteristics of the current distribution. Standardizing with these estimates achieves real-time stabilization of each layer's input during training. This means that no matter how the preceding layer's parameters change, the data received by the current layer is forced to have zero mean and unit variance, breaking the chain reaction of "parameter update → distribution change → training instability."

Second, it introduces learnable scaling and shifting parameters. Pure standardization would restrict all activation values to a fixed distribution, weakening the network's nonlinear representational capacity. BN adds two learnable parameters $\gamma$ (scale) and $\beta$ (shift) after standardization, upgrading "standardization" to "learnable distribution transformation." The network can autonomously decide whether standardization is needed and to what extent by learning appropriate $\gamma$ and $\beta$. If a layer's optimal distribution is indeed zero mean and unit variance, $\gamma$ and $\beta$ will learn the identity transformation; if a layer requires a specific distribution to activate nonlinear characteristics, $\gamma$ and $\beta$ will adjust the standardized data to the appropriate position and scale. This design ensures BN does not reduce the network's representational capacity — it merely provides a better starting point for optimization, and the network has the freedom to deviate from this starting point.

Combining these two ideas, BN embeds a distribution stabilizer in each layer of the network, computing Batch statistics in real-time during training, standardizing each layer's input, and fine-tuning the distribution with learnable parameters. This allows each layer to stop constantly adapting to distribution changes from preceding layers and instead focus on learning stable feature representations. BN operates independently on each feature dimension. Let the values of a certain feature in a Mini-Batch be $\{x_1, x_2, ..., x_m\}$, where $m$ is the Batch Size. BN's computation consists of three steps: computing statistics, standardization, and scaling and shifting. Each step has its specific purpose and design considerations.

- **Computing Batch Statistics**: Compute the mean and variance of this feature in the current Mini-Batch. Let $x_i$ be the value of the $i$-th sample on this feature in the Batch. The mean $\mu_B$ (computed by the formula below) is the center of the data, and the variance $\sigma_B^2$ (computed by the formula below) is the squared deviation of each sample from the mean, measuring the dispersion of data around the center:

    $$[batch_mu]\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$

    $$[batch_sigma]\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

- **Standardization**: Use the computed statistics to standardize each sample, moving the data to a standard position (zero mean) and standard scale (unit variance), just like unifying measurements from different units into the same metric:

    $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

    Here $x_i - \mu_B$ shifts the data so the center moves to zero, eliminating positional bias. $\sqrt{\sigma_B^2 + \epsilon}$ is the estimate of the standard deviation, where $\epsilon$ is a small constant (typically $10^{-5}$) to prevent division by zero when variance is zero. Dividing the shifted data by the standard deviation scales the data so its dispersion becomes unit variance. After standardization, $\hat{x}_i$ has the statistical properties of zero mean and unit variance. Regardless of the original distribution of $x_i$, the standardized values fall within a similar numerical range.

- **Scaling and Shifting**: If all activation values were standardized near zero, the network's nonlinear representational capacity might be weakened. To address this, BN introduces two learnable parameters $\gamma$ and $\beta$ after standardization for scaling and shifting:

    $$y_i = \gamma \hat{x}_i + \beta$$

    Here $\gamma$ is the scaling parameter, controlling the scale of the output values, and $\beta$ is the shifting parameter, controlling the position of the output values. Standardization provides a stable training foundation, while $\gamma$ and $\beta$ give the network the freedom to maintain the standardization effect or restore/adjust to a distribution more suitable for the current task.

    The existence of these two parameters makes BN not just a standardization operation, but a learnable distribution transformation. When $\gamma = \sigma_B$ and $\beta = \mu_B$, $y_i = \sigma_B \cdot \hat{x}_i + \mu_B = x_i$, and BN completely restores the original distribution; when $\gamma$ and $\beta$ learn other values, BN transforms the data to a new distribution that may be more suitable for the current task. This design preserves the network's representational flexibility while ensuring training stability.

## BN Algorithm Implementation

The three steps of BN are purely computational operations that can be implemented with simple Python code. The code below corresponds one-to-one with the theoretical formulas above: `np.mean` and `np.var` compute Batch statistics, `(x - mu) / np.sqrt(var + eps)` performs standardization, and `gamma * x_hat + beta` performs scaling and shifting. In actual deep learning frameworks, the BN implementation would be more complex, including backpropagation, moving average statistics maintenance, etc., but the core logic is consistent with this code.

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    """
    Batch Normalization forward pass
    
    Parameters:
        x: Input data, shape [batch_size, num_features]
        gamma: Scaling parameter, shape [num_features], learnable
        beta: Shifting parameter, shape [num_features], learnable
        eps: Small constant to prevent division by zero, default 1e-5
    
    Returns:
        y: BN output, same shape as x
        mu: Batch mean, for subsequent operations if needed
        var: Batch variance, for subsequent operations if needed
    """
    # Step 1: Compute batch statistics (corresponding to formulas mu_B and sigma_B^2)
    mu = np.mean(x, axis=0)      # Mean of each feature, computed along the batch dimension
    var = np.var(x, axis=0)      # Variance of each feature, computed along the batch dimension
    
    # Step 2: Standardization (corresponding to formula x_hat_i)
    x_hat = (x - mu) / np.sqrt(var + eps)  # Subtract mean and divide by standard deviation
    
    # Step 3: Scaling and shifting (corresponding to formula y_i)
    y = gamma * x_hat + beta     # Apply learnable parameters
    
    return y, mu, var
```


## BN and Backpropagation

As a layer in a neural network, BN also relies on backpropagation to update its parameters. Let the loss function be $l$, the input Batch be $\{x_1, ..., x_m\}$, and the output be $\{y_1, ..., y_m\}$. Given the gradient $\frac{\partial l}{\partial y_i}$ from the upstream layer, we need to compute the gradients of the learnable parameters $\gamma$ and $\beta$ for optimization updates, as well as the gradient of the input $x$ to pass to the preceding layer. Note that although $\mu_B$ and $\sigma_B$ used in the BN layer are intermediate nodes in backpropagation, they are not learnable parameters and do not require separate gradient updates; their effect on the gradient of $x$ is already accounted for in $\frac{\partial l}{\partial x_i}$ through the chain rule.

- **Gradients with respect to $\gamma$ and $\beta$**: Since $y_i = \gamma \hat{x}_i + \beta$, $\gamma$ and $\beta$ directly affect the output. The gradient of $\gamma$ is the sum of the output gradients multiplied by the standardized values (because $\gamma$ acts multiplicatively on $\hat{x}_i$), and the gradient of $\beta$ is simply the sum of the output gradients (because $\beta$ acts additively on the output):

    $$\frac{\partial l}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial l}{\partial y_i} \cdot \hat{x}_i$$

    $$\frac{\partial l}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \beta} =  \sum_{i=1}^{m} \frac{\partial l}{\partial y_i}$$

- **Gradient with respect to input $x$**: The effect of $x_i$ on output $y_j$ is multi-path. $x_i$ not only directly affects $\hat{x}_i$ and $y_i$, but also indirectly affects all $\hat{x}_j$ and $y_j$ by influencing $\mu_B$ and $\sigma_B$. The complete gradient derivation needs to consider all three paths simultaneously:

    $$\frac{\partial l}{\partial x_i} = \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial l}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \mu_B} \cdot \frac{\partial \mu_B}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial l}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \sigma_B} \cdot \frac{\partial \sigma_B}{\partial x_i}$$

    After a somewhat tedious but not difficult derivation (omitted here), the final gradient formula can be simplified to:

    $$\frac{\partial l}{\partial x_i} = \frac{\gamma}{m\sigma_B}\left(m \frac{\partial l}{\partial y_i} - \sum_{j=1}^{m} \frac{\partial l}{\partial y_j} - \hat{x}_i \sum_{j=1}^{m} \frac{\partial l}{\partial y_j} \hat{x}_j\right)$$

    Although this result looks somewhat complex, it ensures that gradients can be correctly propagated backward. The fact that the BN layer does not block gradient flow and that gradients can normally pass to the preceding layer is itself a key conclusion — it is one of the important reasons why BN can support deep network training. BN not only stabilizes the numerical distribution of forward propagation but also ensures smooth gradient flow during backpropagation.

## Training Stability

In this section, we delve into how BN specifically improves the training process of deep networks. BN's impact on training is multifaceted: it directly solves the covariate shift problem, indirectly provides a regularization effect, and changes the structural design choices of network architectures. First, BN's improvement to training stability can be understood from four dimensions, each directly corresponding to the covariate shift problems we analyzed earlier:

- First, BN achieves **distribution stability**. The input of each layer is forced to be standardized to zero mean and unit variance. No matter how the preceding layer's parameters are updated, the data received by the current layer always falls within a similar numerical range. This is like installing a voltage stabilizer for each layer — regardless of input fluctuations, after BN processing, everything becomes stable. The root problem of covariate shift (input distribution changing with training) is directly eliminated.
- Second, BN brings **gradient stability**. The gradient of an activation function is closely related to its input value: Sigmoid has very small gradients when inputs are far from zero (saturation region), and ReLU has zero gradients when inputs are negative (dead region). Standardization pulls activation values near zero, which happens to fall within the comfortable working range of Sigmoid and tanh, where gradients are moderate and stable; for ReLU, standardization also reduces the risk of consistently negative values, lowering the probability of neuron death. All these positively contribute to gradient stability, which means more stable parameter update directions and more reliable convergence.
- Third, BN allows **larger learning rates**. An important impact of covariate shift is limiting the choice of learning rate — a large learning rate causes drastic distribution changes that may crash training. BN eliminates this limitation. Even with large parameter updates, the input distribution of the next layer is still standardized to a stable range. Therefore, we can safely use larger learning rates to accelerate convergence. In the training of deep networks like ResNet, the learning rate typically starts at 0.1, which is almost unimaginable in networks without BN layers.
- Finally, BN reduces **initialization sensitivity**. The problem of improper initialization mainly stems from incorrect weight variance settings, which cause activation value distributions to deviate from the normal range and amplify layer by layer, making training difficult. BN performs standardization at each layer, effectively correcting the activation value distribution immediately after initialization. Even if initialization is not ideal, BN can pull it back to a reasonable range. This greatly improves the network's fault tolerance and reduces the effort spent on tuning initialization parameters.

Beyond direct stability improvements, BN also brings an interesting "side effect" — it indirectly provides a certain regularization effect. This effect stems from BN's reliance on Mini-Batch statistics. During training, BN uses the current Mini-Batch's mean $\mu_B$ and variance $\sigma_B^2$ for standardization. These two statistics are sample estimates of the data in the Batch, not the true population statistics. Different Batches contain different samples, so $\mu_B$ and $\sigma_B^2$ fluctuate across Batches. This fluctuation means that the same input $x_i$ may receive different standardized results $\hat{x}_i$ in different Batches.

Statistical fluctuation is essentially noise. From a regularization perspective, noise forces the network to learn more robust features because the standardization results of inputs have randomness — the network cannot over-rely on any specific numerical value and must learn representations that can tolerate this variation. This is conceptually similar to Dropout: Dropout introduces noise by randomly dropping neurons, while BN introduces noise through random standardization.

It is important to note that this noise only exists during training. During inference, global statistics (moving averages across all training Batches) are used, and the standardization result is deterministic. This means that during training, the network learns robust features, and during inference, these features are stably activated. This design — noise during training, no noise during inference — aligns perfectly with the purpose of regularization.

In practice, BN's regularization effect manifests as better generalization ability in networks that use BN, with a smaller gap between training loss and test loss. In some cases, BN's inherent regularization effect is sufficient and can even replace Dropout to some extent. Both are techniques that improve training, but their mechanisms and effects differ significantly:

- From the **regularization mechanism** perspective, BN's noise comes from the randomness of Batch statistics, with the noise amplitude depending on the distribution differences of samples within the Batch; Dropout's noise comes from randomly dropping neurons, with the noise amplitude directly controlled by the dropout probability $p$. BN's noise is passive, depending on the data; Dropout's noise is active and can be manually adjusted.
- From the **training stability** perspective, BN improves stability through standardization, making gradient propagation smoother; Dropout reduces stability by dropping neurons — only a portion of the network is used each time, effectively reducing model capacity. BN accelerates training, while Dropout may slow training (because each update uses a smaller network).
- From the **applicable scenarios** perspective, BN is a standard component in deep networks like CNNs, used by almost all modern vision networks; Dropout is more commonly used in fully connected networks, especially in classifier layers with large parameter counts.

The typical design of modern deep networks (such as ResNet and later architectures) is to have a convolutional layer followed by BN and ReLU, with fully connected layers optionally adding Dropout. This design has three reasons:
- First, BN already provides sufficient regularization, and additional Dropout offers limited benefit;
- Second, Dropout's regularization effect in convolutional layers is weak (adjacent pixels are highly correlated, so dropping some neurons loses limited information);
- Third, Dropout may disrupt BN's standardization effect — Dropout changes the activation value distribution, making the statistics computed by BN unstable;

In practice, whether additional Dropout is needed when the network already uses BN depends on the specific scenario. When the dataset is small and overfitting is severe, Dropout can be added to fully connected layers; when the dataset is large and BN's regularization is sufficient, Dropout can be omitted.

## Inference Mode

A unique aspect of BN is that it behaves differently during training and inference. This difference stems from practical constraints: during inference, the Batch Size may be very small or even 1, making it impossible to reliably compute Batch statistics. Therefore, different operational procedures must be designed for training and inference modes.

During training, BN uses the statistics of the current Mini-Batch for standardization (see {{batch_mu}} and {{batch_sigma}}), where $\mu_B$ and $\sigma_B^2$ are the mean and variance of the data in the current Batch. In addition to standardization, BN also maintains an estimate of global statistics during training. Global statistics are a weighted accumulation of historical statistics, with each Batch's contribution gradually incorporated, updated through moving averages:

$$[global_mu] \mu_{global} = \alpha \mu_{global} + (1 - \alpha) \mu_B$$
$$[global_sigma] \sigma_{global}^2 = \alpha \sigma_{global}^2 + (1 - \alpha) \sigma_B^2$$

Here the hyperparameter $\alpha$ is the decay coefficient, typically 0.9 or 0.99, controlling the proportion of new information incorporated. When $\alpha$ is close to 1 (e.g., 0.99), global statistics update slowly, relying more on long-term history; when $\alpha$ is smaller (e.g., 0.9), updates are faster, relying more on recent Batches. After sufficient training, global statistics converge to a stable value reflecting the overall distribution characteristics of the training data.

Global statistics themselves do not affect training; they are all prepared for the inference phase. Due to the following three characteristics of inference, global statistics are needed for proper operation:

- **Single-sample inference problem**: The most common inference scenario involves processing a single sample (Batch Size = 1). When the Batch has only one sample, $\mu_B = x_1$, $\sigma_B^2 = 0$, and the standardization formula $\hat{x}_1 = \frac{x_1 - x_1}{\sqrt{0 + \epsilon}} = 0$ causes all inputs to be standardized to zero, completely losing information.
- **Output stability requirement**: During inference, we need deterministic output — the same input should produce the same output. If Batch statistics were used, the output would vary with Batch composition. The output of sample A inferred alone might differ from the output when sample A is inferred together with sample B in a Batch. This is unacceptable for deployment and debugging.
- **Deployment consistency**: In production, models may be deployed in various scenarios such as real-time inference, batch processing, and distributed deployment. Using fixed global statistics ensures output consistency across all scenarios, facilitating testing and validation.

Based on the above reasons, the inference phase switches to using global statistics to compute $\hat{x}$ and $y$, where $\mu_{global}$ and $\sigma_{global}^2$ are obtained from formulas {{global_mu}} and {{global_sigma}}:

$$\hat{x} = \frac{x - \mu_{global}}{\sqrt{\sigma_{global}^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Deep learning frameworks (such as PyTorch) control BN behavior through mode switching: during training, call `model.train()` to use Batch statistics and update global statistics; during inference, call `model.eval()` to use global statistics. Mode switching may seem like a one-line command, but it is an easy place to make mistakes in practice — forgetting to switch to `eval()` mode during inference leads to unstable output; accidentally switching to `eval()` mode during training causes global statistics not to update. These errors can lead to inconsistency between training and inference results and degraded performance. A particularly noteworthy scenario is the validation phase during training. When validating model performance at the end of each epoch, you need to switch to `eval()` mode; after validation, before continuing training, you need to switch back to `train()` mode. Forgetting to switch back to `train()` after validation can impair training effectiveness.

## BN Training Practice

Theoretical analysis reveals the design principles and expected effects of BN. Now, let us verify these theories through concrete code experiments. The experiments will examine BN's practical impact from three perspectives: convergence speed, learning rate tolerance, and support for deep network training. The code below implements a complete BN layer, including forward propagation, backpropagation, global statistics maintenance, and other functions. We will use this implementation to build two comparison networks (one with BN, one without BN) and conduct comparative experiments on the same training task.

- Experiment 1: **BN's impact on convergence speed**. From the loss curves, we can clearly see that the network with BN has faster and more stable training loss reduction. The loss curve of the network without BN shows significant oscillations, caused by gradient instability due to covariate shift; the loss curve of the network with BN decreases smoothly, as BN's standardization stabilizes the input distribution of each layer, making gradient directions more consistent. Comparison of final test losses also shows that the BN network not only converges faster but also has better generalization performance (smaller gap between training loss and test loss).

- Experiment 2: **BN's tolerance for learning rates**. When the learning rate increases from 0.001 to 0.1, the network without BN significantly deteriorates at high learning rates, with the loss curve oscillating violently or even diverging; the network with BN can still train stably at high learning rates, with only slight changes in convergence speed. This confirms BN's core value: standardization makes the network insensitive to the magnitude of parameter updates, large learning rates no longer cause drastic distribution changes, and training can be safely accelerated.

- Experiment 3: **BN's support for deep networks**. When the network depth increases from 5 layers to 15 layers, the training difficulty of the network without BN rises significantly: the test loss of the 15-layer network is noticeably higher than that of the 5-layer network, and training may even crash (vanishing gradients prevent effective parameter updates). The network with BN demonstrates good adaptability to depth: from 5 layers to 15 layers, the test loss remains stable with no significant degradation. This is precisely why deep architectures like ResNet rely on BN — without BN, training deep networks is nearly impossible.


```python runnable
import numpy as np
import matplotlib.pyplot as plt

# Define activation function
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Batch Normalization implementation
class BatchNorm:
    def __init__(self, num_features, momentum=0.99, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.cache = None
    
    def forward(self, x, training=True):
        if training:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            self.cache = (x, x_hat, mu, var)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta
    
    def backward(self, dout, learning_rate=0.01):
        x, x_hat, mu, var = self.cache
        m = x.shape[0]
        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + self.eps)**(-1.5), axis=0)
        dmu = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / m + dmu / m
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        return dx

# Simple network (supports BN)
class SimpleNetwork:
    def __init__(self, layer_sizes, use_bn=True, grad_clip=5.0):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.use_bn = use_bn
        self.grad_clip = grad_clip
        self.weights = []
        self.biases = []
        self.bn_layers = []

        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            if use_bn and i < self.num_layers - 1:
                self.bn_layers.append(BatchNorm(layer_sizes[i+1]))
            else:
                self.bn_layers.append(None)

    def forward(self, X, training=True):
        self.activations = [X]
        self.pre_activations = []
        self.bn_outputs = []
        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            if self.bn_layers[i] is not None:
                z_bn = self.bn_layers[i].forward(z, training=training)
                self.bn_outputs.append(z_bn)
                a = relu(z_bn) if i < self.num_layers - 1 else z_bn
            else:
                self.bn_outputs.append(None)
                a = relu(z) if i < self.num_layers - 1 else z
            if i < self.num_layers - 1:
                a = np.clip(a, -10, 10)
            self.activations.append(a)
        return a

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        delta = self.activations[-1] - y
        delta = np.clip(delta, -5, 5)
        for i in range(self.num_layers - 1, -1, -1):
            if self.bn_layers[i] is not None:
                delta = self.bn_layers[i].backward(delta, learning_rate)
            grad_w = self.activations[i].T @ delta / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            if self.grad_clip is not None:
                grad_w = np.clip(grad_w, -self.grad_clip, self.grad_clip)
                grad_b = np.clip(grad_b, -self.grad_clip, self.grad_clip)
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            if i > 0:
                if self.bn_layers[i-1] is not None:
                    delta = (delta @ self.weights[i].T) * relu_derivative(self.bn_outputs[i-1])
                else:
                    delta = (delta @ self.weights[i].T) * relu_derivative(self.pre_activations[i-1])
                if np.isnan(delta).any():
                    delta = np.nan_to_num(delta, nan=0.0)

    def compute_loss(self, X, y, training=False):
        output = self.forward(X, training=training)
        if np.isnan(output).any() or np.isinf(output).any():
            return float('inf')
        return np.mean((output - y)**2)

print("Experiment 1: BN's Impact on Training Convergence")
print("-" * 40)

# Generate data
n_train = 200
n_test = 100
n_features = 50

X_train = np.random.randn(n_train, n_features)
y_train = np.sin(X_train[:, 0] * 2) + np.cos(X_train[:, 1]) + np.random.randn(n_train) * 0.1
y_train = y_train.reshape(-1, 1)

X_test = np.random.randn(n_test, n_features)
y_test = np.sin(X_test[:, 0] * 2) + np.cos(X_test[:, 1]) + np.random.randn(n_test) * 0.1
y_test = y_test.reshape(-1, 1)

# Network configuration
layer_sizes = [n_features, 128, 64, 32, 1]

# Mini-Batch configuration (using Mini-Batch SGD to demonstrate BN's effect on training stability)
batch_size = 32
n_batches = max(1, n_train // batch_size)

# Without BN
net_no_bn = SimpleNetwork(layer_sizes, use_bn=False, grad_clip=5.0)

# With BN
net_bn = SimpleNetwork(layer_sizes, use_bn=True, grad_clip=5.0)

# Training parameters
n_epochs = 200
learning_rate = 0.01

# Record training process
train_losses_no_bn = []
test_losses_no_bn = []
train_losses_bn = []
test_losses_bn = []

print("Starting training...")

for epoch in range(n_epochs):
    # Mini-Batch SGD training
    indices = np.random.permutation(n_train)
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_train)
        bi = indices[start:end]
        xb, yb = X_train[bi], y_train[bi]

        # Without BN
        net_no_bn.forward(xb, training=True)
        net_no_bn.backward(xb, yb, learning_rate)

        # With BN
        net_bn.forward(xb, training=True)
        net_bn.backward(xb, yb, learning_rate)

    # Loss computation (training=True to keep BN behavior consistent with training)
    train_loss_no = net_no_bn.compute_loss(X_train, y_train, training=True)
    test_loss_no = net_no_bn.compute_loss(X_test, y_test, training=True)
    train_loss_bn = net_bn.compute_loss(X_train, y_train, training=True)
    test_loss_bn = net_bn.compute_loss(X_test, y_test, training=True)

    train_losses_no_bn.append(train_loss_no)
    test_losses_no_bn.append(test_loss_no)
    train_losses_bn.append(train_loss_bn)
    test_losses_bn.append(test_loss_bn)

print(f"\nWithout BN:")
print(f"  Final training loss: {train_losses_no_bn[-1]:.4f}")
print(f"  Final test loss: {test_losses_no_bn[-1]:.4f}")

print(f"\nWith BN:")
print(f"  Final training loss: {train_losses_bn[-1]:.4f}")
print(f"  Final test loss: {test_losses_bn[-1]:.4f}")

# Visualize loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Without BN
ax1 = axes[0]
ax1.plot(train_losses_no_bn, label='Training Loss', linewidth=2, color='#3498db')
ax1.plot(test_losses_no_bn, label='Test Loss', linewidth=2, color='#e74c3c')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Without Batch Normalization', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# With BN
ax2 = axes[1]
ax2.plot(train_losses_bn, label='Training Loss', linewidth=2, color='#3498db')
ax2.plot(test_losses_bn, label='Test Loss', linewidth=2, color='#e74c3c')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('With Batch Normalization', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("Experiment 2: BN's Impact on Different Learning Rates")
print("-" * 40)

learning_rates = [0.001, 0.01, 0.05]
lr_results = {}

for lr in learning_rates:
    print(f"Learning rate = {lr}")
    
    net_no = SimpleNetwork(layer_sizes, use_bn=False, grad_clip=5.0)
    net_bn_lr = SimpleNetwork(layer_sizes, use_bn=True, grad_clip=5.0)
    
    no_bn_losses = []
    bn_losses = []
    
    for epoch in range(n_epochs):
        with np.errstate(over='ignore', invalid='ignore'):
            indices = np.random.permutation(n_train)
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_train)
                bi = indices[start:end]
                xb, yb = X_train[bi], y_train[bi]
                
                net_no.forward(xb, training=True)
                net_no.backward(xb, yb, lr)
                
                net_bn_lr.forward(xb, training=True)
                net_bn_lr.backward(xb, yb, lr)
        
        no_bn_losses.append(net_no.compute_loss(X_test, y_test, training=True))
        bn_losses.append(net_bn_lr.compute_loss(X_test, y_test, training=True))
    
    lr_results[lr] = {
        'no_bn': no_bn_losses,
        'bn': bn_losses,
        'no_bn_final': no_bn_losses[-1],
        'bn_final': bn_losses[-1]
    }
    
    print(f"  Without BN final test loss: {no_bn_losses[-1]:.4f}")
    print(f"  With BN final test loss: {bn_losses[-1]:.4f}")
    print()

# Visualize different learning rates
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = {'no_bn': '#e74c3c', 'bn': '#2ecc71'}

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]
    ax.plot(lr_results[lr]['no_bn'], label='Without BN', linewidth=2, color=colors['no_bn'])
    ax.plot(lr_results[lr]['bn'], label='With BN', linewidth=2, color=colors['bn'])
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Test Loss', fontsize=11)
    ax.set_title(f'Learning Rate = {lr}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("Experiment 3: BN's Impact on Deep Networks")
print("-" * 40)

depth_configs = [
    {'depth': 5, 'sizes': [n_features, 128, 64, 32, 16, 1]},
    {'depth': 10, 'sizes': [n_features, 128, 128, 64, 64, 32, 32, 16, 16, 8, 1]},
    {'depth': 15, 'sizes': [n_features] + [64]*14 + [1]}
]

depth_results = {}

for config in depth_configs:
    depth = config['depth']
    sizes = config['sizes']

    print(f"Network depth = {depth}")

    # Without BN (no gradient clipping, let gradient explosion happen naturally)
    try:
        net_no = SimpleNetwork(sizes, use_bn=False, grad_clip=None)
        no_bn_losses = []
        crashed = False

        with np.errstate(over='ignore', invalid='ignore'):
            for epoch in range(n_epochs):
                indices = np.random.permutation(n_train)
                for b in range(n_batches):
                    start = b * batch_size
                    end = min(start + batch_size, n_train)
                    bi = indices[start:end]
                    xb, yb = X_train[bi], y_train[bi]
                    net_no.forward(xb, training=True)
                    net_no.backward(xb, yb, 0.01)
                
                loss = net_no.compute_loss(X_test, y_test, training=True)
                if np.isnan(loss) or np.isinf(loss) or loss > 1e10:
                    crashed = True
                    break
                no_bn_losses.append(loss)

        if crashed:
            print(f"  Without BN training crashed (gradient explosion)")
            depth_results[depth] = {'no_bn': None, 'bn': None}
        else:
            depth_results[depth] = {'no_bn': no_bn_losses, 'bn': None}
            print(f"  Without BN final test loss: {no_bn_losses[-1]:.4f}")
    except Exception as e:
        print(f"  Without BN training failed: {e}")
        depth_results[depth] = {'no_bn': None, 'bn': None}

    # With BN
    try:
        net_bn_depth = SimpleNetwork(sizes, use_bn=True, grad_clip=5.0)
        bn_losses = []
        bn_crashed = False

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_train)
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_train)
                bi = indices[start:end]
                xb, yb = X_train[bi], y_train[bi]
                net_bn_depth.forward(xb, training=True)
                net_bn_depth.backward(xb, yb, 0.01)
            
            loss = net_bn_depth.compute_loss(X_test, y_test, training=True)
            if np.isnan(loss) or np.isinf(loss) or loss > 1e10:
                bn_crashed = True
                break
            bn_losses.append(loss)

        if bn_crashed:
            print(f"  With BN training crashed")
            depth_results[depth]['bn'] = None
        else:
            depth_results[depth]['bn'] = bn_losses
            print(f"  With BN final test loss: {bn_losses[-1]:.4f}")
    except Exception as e:
        print(f"  With BN training failed: {e}")
        depth_results[depth]['bn'] = None

    print()

# Visualize depth impact
fig, ax = plt.subplots(figsize=(10, 6))

depths = list(depth_results.keys())
no_bn_finals = [depth_results[d]['no_bn'][-1] if depth_results[d]['no_bn'] else None for d in depths]
bn_finals = [depth_results[d]['bn'][-1] if depth_results[d]['bn'] else None for d in depths]

# Detect crash values
finite_no_bn = []
explode_flags = []
for v in no_bn_finals:
    if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
        explode_flags.append(True)
        finite_no_bn.append(0)
    else:
        explode_flags.append(False)
        finite_no_bn.append(v)

finite_bn = []
for v in bn_finals:
    if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
        finite_bn.append(0)
    else:
        finite_bn.append(v)

x = range(len(depths))
width = 0.4

bars1 = ax.bar([i - width/2 for i in x], finite_no_bn,
               width, label='Without BN', color='#e74c3c', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], finite_bn,
               width, label='With BN', color='#2ecc71', alpha=0.7)

# Label crashed bars
for i, exploded in enumerate(explode_flags):
    if exploded:
        ax.text(i, ax.get_ylim()[1] * 0.95, 'Training Crashed',
                ha='center', va='top', fontsize=11, color='#e74c3c')

ax.set_xticks(x)
ax.set_xticklabels([f'Depth {d}' for d in depths])
ax.set_xlabel('Network Depth', fontsize=11)
ax.set_ylabel('Final Test Loss (Log Scale)', fontsize=11)
ax.set_title("BN's Impact on Networks of Different Depths", fontsize=12)
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.close()
```

## Limitations and Variants

BN performs excellently in most scenarios, but it is not without flaws. In specific scenarios, BN's design assumptions may break down, requiring alternative approaches or improved versions. Understanding the design motivations and application scenarios of these variants is key to flexibly using normalization techniques. BN's design relies on a key assumption: that the statistics of a Mini-Batch can estimate the overall distribution of the data. This assumption holds in most cases, but encounters problems in the following scenarios.

- **Dependence on Batch Size**. The quality of BN's statistic estimation directly depends on the Batch Size. When the Batch Size is very small (e.g., $m < 8$), the variance of the statistics is large and the estimation is unstable; in the extreme case of Batch Size = 1, the variance $\sigma_B^2 = 0$ and standardization completely fails. Batch Size is not always adjustable at will — in scenarios where GPU memory is limited or high-resolution image training is needed, it is a very real constraint that Batch Size cannot be increased.
- **Complexity in distributed training**. In multi-GPU or distributed training, each device processes a different Mini-Batch and computes its own Batch statistics. To maintain consistency, statistics need to be synchronized across all devices, increasing communication overhead and implementation complexity.
- **Inconsistency between training and inference**. Training uses Batch statistics while inference uses global statistics, and the standardization results of the two modes may differ. If the global statistics have not fully converged during training, inference results may deviate from expectations. This inconsistency can cause confusion during deployment debugging.
- **Inapplicability to sequence models**. As will be mentioned when discussing large language models in the next part, RNNs and Transformers process variable-length sequences, where the hidden state at each time step requires independent standardization. BN computes statistics across the Batch dimension, making it difficult to apply directly to sequence models. Different samples in the same Batch may have different sequence lengths, and the hidden state distributions at different time steps may also differ.

To address the above limitations, researchers have proposed various variants of batch normalization, each solving specific problems of BN from different perspectives and suitable for particular scenarios. The main variants include:

- **Batch Renormalization** (BrN): An improved approach for small Batch scenarios. The core idea of BrN is to correct Batch statistics when they deviate too much from global statistics, rather than relying entirely on them. Specifically, two correction factors $r$ and $d$ are introduced:

    $$\hat{x} = \frac{x - \mu_B}{\sigma_B} \cdot r + d$$

    where $r = clip(\frac{\sigma_B}{\sigma_{global}}, r_{min}, r_{max})$ limits the deviation of the standard deviation, and $d = clip(\frac{\mu_B - \mu_{global}}{\sigma_{global}}, d_{min}, d_{max})$ limits the deviation of the mean. When Batch statistics are close to global statistics, BrN behaves like BN; when the deviation is too large, the correction factors pull it back to a reasonable range. This design is more stable than BN in small Batch scenarios.

- **Layer Normalization** (LN): A standardization approach that does not depend on the Batch at all. LN computes statistics across all features of a single sample:

    $$\mu_L = \frac{1}{d}\sum_{j=1}^{d} x_j$$
    $$\sigma_L^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu_L)^2$$

    LN's statistics come from within a single sample and are completely independent of Batch Size. This makes LN naturally suitable for RNNs and Transformers, where the hidden state at each time step can be independently standardized, and training and inference behavior are consistent. LN is the [default normalization scheme for the Transformer architecture](../../language-models/architecture-basics/transformer-architecture.md#layer-normalization).

- **Group Normalization** (GN): A compromise between LN and BN. GN divides features into several groups, with each group independently standardized:

    $$\mu_G = \frac{1}{(C/G) \cdot H \cdot W}\sum_{c \in \text{group}_k} \sum_{p,q} x_{c,p,q}$$

    GN's statistics come from a portion of the features of a single sample and do not depend on Batch Size. The number of groups $G$ is a tunable parameter: when $G = 1$, GN is equivalent to LN (all features in one group); when $G = c$ (number of channels), GN is equivalent to Instance Normalization (each channel as a group). GN performs better than BN in small Batch CNN scenarios and is the recommended choice for memory-intensive tasks such as object detection and segmentation.

- **Instance Normalization** (IN): Each sample, each channel is standardized independently. IN's statistics come from a single channel of a single sample:

    $$\mu_I = \frac{1}{h \cdot w}\sum_{p,q} x_{p,q}$$

    IN has the finest granularity of standardization, preserving the most inter-sample and inter-channel differences. This property is particularly useful in image style transfer tasks, where style features are mainly reflected in channel-level statistical differences, and IN can effectively separate content and style. IN is not commonly used for general classification tasks, but has special value in generative models.

- **Local Response Normalization** (LRN): A cross-channel normalization operation primarily used in CNNs before the advent of BN, inspired by the lateral inhibition phenomenon in biological neural systems, where active neurons inhibit the activity of neighboring neurons. Let $a_{x,y}^i$ be the activation value (ReLU output) at position $(x, y)$ and channel $i$. The LRN output $b_{x,y}^i$ is:

    $$b_{x,y}^i = \frac{a_{x,y}^i}{\left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x,y}^j)^2\right)^\beta}$$

    where $\alpha$ is the scaling factor (default $\alpha = 10^{-4}$), $\beta$ is the exponent (default $\beta = 0.75$). The denominator accumulates the squared activation values of adjacent channels, suppressing channels with high activation values. The larger the activation value, the larger the denominator, and the relatively smaller the output. This encourages competition between different channels and increases feature diversity. LRN was the standard configuration for [AlexNet](../convolutional-neural-network/alexnet.md). Experiments in the AlexNet paper showed that LRN reduces the Top-1 error rate by 1.4% and the Top-5 error rate by 1.2%. However, subsequent research has shown that standard Batch Normalization (BN) is more effective than LRN for normalization. Modern AlexNet implementations typically use BN instead of LRN, or omit normalization entirely.

Different normalization schemes have their own applicable scenarios. When choosing a normalization scheme, the following three factors should be considered:

1. **Batch Size availability**: When Batch Size is sufficient ($\geq 16$), BN is preferred; when constrained, consider GN or LN.
2. **Architecture type**: CNNs commonly use BN or GN; RNNs and Transformers commonly use LN.
3. **Task characteristics**: Special tasks like style transfer may require IN.

It is worth noting that these schemes are not mutually exclusive. In some complex architectures, different parts may use different normalization methods — for example, the convolutional part of a Transformer might use GN while the attention part uses LN. Flexible combination is a sign of advanced design. The following table summarizes the characteristics and recommended uses of each scheme:

| Method | Applicable Scenarios | Batch Size Dependence | Source of Statistics |
|:------|:--------------------|:---------------------|:--------------------|
| BN | CNNs, Deep Networks | Strong (recommend $m \geq 16$) | Batch + feature dimension |
| LN | RNNs, Transformers | None | Single sample + all features |
| GN | Small Batch CNNs | None | Single sample + feature groups |
| IN | Style Transfer, Generative Models | None | Single sample + single channel |


## Chapter Summary

Batch Normalization is a landmark technology in the history of deep learning development. By standardizing Mini-Batch data at each layer, it solves the covariate shift problem that plagued deep network training for years, profoundly changing the practice of deep learning. The proposers of Batch Normalization, Sergey Ioffe and Christian Szegedy, wrote in their paper: "We hope that Batch Normalization will become a standard component of deep network training." A decade later, this vision has become a reality — from ResNet to Transformer, from computer vision to natural language processing, BN and its variants are everywhere. Mastering this technique is an important foundation for deep learning practitioners.

## Exercises

1. Given 4 sample values of a feature in a Mini-Batch as $\{2, 4, 6, 8\}$, let $\epsilon = 0$, $\gamma = 2$, $\beta = 1$. Follow the three steps of BN (computing statistics, standardization, scaling and shifting) to manually compute the BN output value for each sample.
    <details>
    <summary>Reference Answer</summary>

    **Step 1: Compute Batch Statistics**

    $$\mu_B = \frac{2 + 4 + 6 + 8}{4} = \frac{20}{4} = 5$$

    $$\sigma_B^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5$$

    **Step 2: Standardization** ($\epsilon = 0$, so $\sqrt{\sigma_B^2 + \epsilon} = \sqrt{5}$)

    $$\hat{x}_1 = \frac{2 - 5}{\sqrt{5}} = \frac{-3}{\sqrt{5}}, \quad \hat{x}_2 = \frac{4 - 5}{\sqrt{5}} = \frac{-1}{\sqrt{5}}, \quad \hat{x}_3 = \frac{6 - 5}{\sqrt{5}} = \frac{1}{\sqrt{5}}, \quad \hat{x}_4 = \frac{8 - 5}{\sqrt{5}} = \frac{3}{\sqrt{5}}$$

    **Step 3: Scaling and Shifting** ($y_i = \gamma \hat{x}_i + \beta = 2\hat{x}_i + 1$)

    $$y_1 = 2 \cdot \frac{-3}{\sqrt{5}} + 1 = 1 - \frac{6}{\sqrt{5}}, \quad y_2 = 2 \cdot \frac{-1}{\sqrt{5}} + 1 = 1 - \frac{2}{\sqrt{5}}$$

    $$y_3 = 2 \cdot \frac{1}{\sqrt{5}} + 1 = 1 + \frac{2}{\sqrt{5}}, \quad y_4 = 2 \cdot \frac{3}{\sqrt{5}} + 1 = 1 + \frac{6}{\sqrt{5}}$$

    Verification: After standardization, the mean is 0 and variance is 1; after scaling and shifting, the mean becomes $\beta = 1$ and the variance becomes $\gamma^2 = 4$, consistent with the computed results.
    </details>

1. Let $\gamma = \sigma_B$ and $\beta = \mu_B$. Substitute these into BN's scaling and shifting formula $y_i = \gamma \hat{x}_i + \beta$ to prove that the BN layer completely restores the original input $x_i$, and explain the significance of this property for the network's representational capacity.
    <details>
    <summary>Reference Answer</summary>
    Substituting $\gamma = \sigma_B$ and $\beta = \mu_B$ into the scaling and shifting formula (ignoring $\epsilon$ to simplify the derivation):

    $$y_i = \sigma_B \cdot \hat{x}_i + \mu_B = \sigma_B \cdot \frac{x_i - \mu_B}{\sigma_B} + \mu_B = x_i - \mu_B + \mu_B = x_i$$

    Q.E.D. When the learnable parameters take the above values, the BN layer is equivalent to an identity mapping, and the original information is completely preserved.

    **Significance**: This property ensures that BN does not reduce the network's representational capacity. Even if a layer does not need normalization, the network can "bypass" the BN operation by learning appropriate values for $\gamma$ and $\beta$. BN provides a better starting point for optimization (zero mean, unit variance), but the network has the freedom to deviate from this starting point. It expands the parameter space rather than shrinking it. This addresses the concern that "pure standardization would weaken nonlinear representational capacity."
    </details>

1. BN uses current Batch statistics during training while maintaining global statistics for inference. Assume the initial global mean $\mu_{global} = 0$, global variance $\sigma_{global}^2 = 1$, and decay coefficient $\alpha = 0.9$. If the statistics of the first 3 Batches are $(\mu_{B1}=2,\; \sigma_{B1}^2=3)$, $(\mu_{B2}=1,\; \sigma_{B2}^2=2)$, $(\mu_{B3}=-1,\; \sigma_{B3}^2=4)$ respectively, compute step by step the values of the global mean and global variance after processing the 3rd Batch.
    <details>
    <summary>Reference Answer</summary>
    Global mean update formula: $\mu_{global} \leftarrow \alpha \cdot \mu_{global} + (1 - \alpha) \cdot \mu_B$
    Global variance update formula: $\sigma_{global}^2 \leftarrow \alpha \cdot \sigma_{global}^2 + (1 - \alpha) \cdot \sigma_B^2$

    **1st Batch**:
    $$\mu_{global} = 0.9 \times 0 + 0.1 \times 2 = 0.2$$
    $$\sigma_{global}^2 = 0.9 \times 1 + 0.1 \times 3 = 0.9 + 0.3 = 1.2$$

    **2nd Batch**:
    $$\mu_{global} = 0.9 \times 0.2 + 0.1 \times 1 = 0.18 + 0.1 = 0.28$$
    $$\sigma_{global}^2 = 0.9 \times 1.2 + 0.1 \times 2 = 1.08 + 0.2 = 1.28$$

    **3rd Batch**:
    $$\mu_{global} = 0.9 \times 0.28 + 0.1 \times (-1) = 0.252 - 0.1 = 0.152$$
    $$\sigma_{global}^2 = 0.9 \times 1.28 + 0.1 \times 4 = 1.152 + 0.4 = 1.552$$

    Therefore, after the 3rd Batch, the global mean is $0.152$ and the global variance is $1.552$. It can be seen that since $\alpha = 0.9$ is relatively large, the global statistics rely more on historical accumulation and update more slowly. This is the characteristic of moving averages — they gradually approximate the true data distribution through the accumulation of many Batches.
    </details>

1. In the extreme case of Batch Size = 1, during BN training $\mu_B = x_1$ and $\sigma_B^2 = 0$. Starting from the standardization formula, analyze BN's behavior in this case, and explain why Batch statistics cannot be used during inference.
    <details>
    <summary>Reference Answer</summary>
    When Batch Size = 1, substituting into the standardization formula:

    $$\hat{x}_1 = \frac{x_1 - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} = \frac{x_1 - x_1}{\sqrt{0 + \epsilon}} = \frac{0}{\sqrt{\epsilon}} = 0$$

    All inputs are standardized to 0, and information is completely lost. The subsequent scaling and shifting $y_1 = \gamma \cdot 0 + \beta = \beta$ can only output a constant shift, and the original input feature $x_1$ has no effect on the output.

    **Why Batch statistics cannot be used during inference**:

    Inference scenarios typically involve single samples (Batch Size = 1), where Batch statistics would lead to: (1) the mean equals the only sample itself, $\mu_B = x_1$; (2) variance is always zero, $\sigma_B^2 = 0$; (3) after standardization, all inputs become 0, and the network output degenerates. Additionally, if multi-sample inference is used but the Batch composition is not fixed, the same input may have different standardization results in different Batches — violating the principle of inference determinism (the same input should produce the same output). Therefore, inference must use the global statistics accumulated during training to ensure stable output and information preservation.
    </details>

1. BN, Layer Normalization (LN), Group Normalization (GN), and Instance Normalization (IN) are four commonly used normalization methods. From the perspective of the scope of statistic computation, use a table to compare their differences, and explain why the Transformer architecture uses LN by default instead of BN.
    <details>
    <summary>Reference Answer</summary>
    **Comparison Table**:

    | Method | Scope of Statistic Computation | Depends on Batch | Applicable Architectures |
    |:------|:-----------------------------|:----------------|:------------------------|
    | BN | Across Batch, same feature dimension | Yes | CNN |
    | LN | Single sample, all features | No | RNN, Transformer |
    | GN | Single sample, feature groups | No | CNN (small Batch) |
    | IN | Single sample, single channel | No | Style Transfer, Generative Models |

    **Why Transformer uses LN by default instead of BN**:

    1. **Variable sequence length**: Different samples in the same Batch may have different sequence lengths. BN computing statistics across the Batch dimension encounters length alignment issues;
    2. **Batch Size is typically small**: Due to the massive parameter count of Transformers (especially the $O(n^2)$ complexity of the self-attention mechanism), the Batch Size in actual training is often limited, making BN's statistic estimation of poor quality;
    3. **Need for time step independence**: The hidden state at each time step requires independent standardization. LN computes statistics across all features of a single sample, naturally supporting variable-length sequences with consistent training/inference behavior. Therefore, LN has become the standard choice for Transformers.
    </details>
