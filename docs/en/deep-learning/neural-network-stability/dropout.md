# Dropout Regularization

In the [regularization](../../statistical-learning/linear-models/regularization-glm.md) chapter of linear models, we discussed the causes of overfitting and basic countermeasures, learning how L1 and L2 regularization constrain model complexity by penalizing parameter magnitudes. However, deep neural networks have their own unique characteristics — an enormous number of parameters (millions or even billions), long training times, and complex hierarchical structures. While regularization methods targeting network parameters are helpful, their effect is no longer sufficient to suppress overfitting. Deep networks require a more powerful regularization technique — one that directly intervenes in the network structure itself.

In 2012, Geoffrey Hinton's team proposed an extremely simple yet remarkably effective regularization method called **Dropout** in their paper _[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://arxiv.org/abs/1207.0580)_. Hinton was facing a practical problem at the time — his team was training deep networks on the ImageNet image classification task, and no matter how they tuned the hyperparameters, the model always overfit, performing nearly perfectly on the training set but poorly on the test set. One evening, Hinton came across a passage in a book: to prevent corruption, banks do not let a single person handle a large transaction alone; instead, multiple people work together, each responsible for only part of the process, so that the absence of any one person does not cause the system to collapse. Inspired by this, he devised a scheme where neurons are randomly dropped during training, forcing each neuron to learn to work independently without relying on the presence of other neurons. This idea seemed simple and crude, yet it achieved remarkable results. Dropout significantly improved ImageNet classification accuracy and quickly became a standard technique in deep learning.

## The Overfitting Challenge in Deep Neural Networks

The number of parameters in deep networks often far exceeds the number of training samples. Setting aside the large language models we will study later, around 2015 a typical image classification network might have 50 million parameters, while the training data consisted of only 100,000 images. This means the network theoretically has enough capacity to memorize the features of each image, rather than learning the general rules of image classification.

Traditional regularization methods (L1, L2 weight decay) provide some help for deep networks, but their effect is limited. L2 regularization constrains weight magnitudes by penalizing the squared values of parameters, essentially limiting parameter sensitivity — the smaller the parameter values, the milder the response to input changes. However, overfitting in deep networks arises not only from having too many parameters, but also from the **co-adaptation** between neurons, where certain neurons can only function when other specific neurons are present, forming complex dependencies. L2 regularization cannot break these dependencies because it only constrains individual parameters without intervening in the collaboration patterns between neurons. Dropout was designed precisely to address this deeper problem — by randomly dropping neurons, it forces each neuron to learn to work independently without relying on specific companions. This is a more structured form of regularization that does not adjust parameter values but instead directly intervenes in the network's topological structure.

## Dropout Principle

The core idea of Dropout is extremely simple: during training, randomly shut down a portion of neurons, letting the remaining neurons complete the task independently. Specifically, Dropout makes a random decision for each neuron during training, retaining the neuron with probability $p$ (allowing it to function normally) and dropping it with probability $1-p$ (forcing the output to zero). Let the original output of a neuron be $y$, and the random mask variable $r$ follow a Bernoulli distribution, determining whether the neuron is dropped. When $r=0$, $y_{drop}=0$, the neuron is shut off; when $r=1$, $y_{drop}=y$, the neuron outputs normally. The network output after Dropout is:

$$y_{drop} = r \cdot y$$

For a set of neurons $\{h_1, h_2, ..., h_n\}$ in a hidden layer, each neuron has its own independent random variable $r_i$:

$$h_i^{drop} = r_i \cdot h_i, \quad r_i \sim \text{Bernoulli}(p)$$

Dropped neurons ($r_i=0$) do not participate in forward propagation or backpropagation, and their gradient is zero — they temporarily disappear from the network. This temporary disappearance is key. A neuron is only dropped in the current training round but may be retained again in the next round, so all neurons are repeatedly used throughout the entire training process.

### Training vs. Inference

Dropout behaves completely differently during training and inference. The root of this difference lies in ensuring that the expected output of the network is consistent between training and inference. Suppose a neuron's output is $y$. During training, the [expectation](../../maths/probability/probability-basics.md#expectation) after Dropout is:

$$\mathbb{E}[y_{drop}] = \mathbb{E}[r \cdot y] = \mathbb{E}[r] \cdot y = p \cdot y$$

Since $r$ is a Bernoulli variable, its expectation is $\mathbb{E}[r]=p$. This means that during training, the average output of this neuron is only $p \cdot y$, which is $p$ times smaller than the original output $y$. During inference, we cannot continue dropping neurons, as that would make predictions unstable (each inference would yield a different result). Therefore, during inference, all neurons are retained ($r=1$) and the output is $y$. However, if we directly use $y$ as the inference output, it would be $1/p$ times larger than the training expectation, causing prediction bias. Hinton proposed two schemes to solve this problem:

- **Scheme 1 — Inference-time scaling**: During training, the output is $y_{drop} = r \cdot y$. During inference, the output is scaled down to $y_{test} = p \cdot y$. This way, the training expectation $\mathbb{E}[y_{drop}] = p \cdot y$ equals the inference output $y_{test} = p \cdot y$.
- **Scheme 2 — Training-time scaling (Inverted Dropout, more commonly used)**: During training, the output is amplified to $y_{drop} = \frac{r}{p} \cdot y$, so the expected output is:

$$\mathbb{E}[y_{drop}] = \mathbb{E}\left[\frac{r}{p} \cdot y\right] = \frac{\mathbb{E}[r]}{p} \cdot y = \frac{p}{p} \cdot y = y$$

During inference, the output is directly $y$ (no adjustment needed), and the training expectation equals the inference output. The advantage of Scheme 2 is that training is typically done once, while inference is repeated many times. Having no additional operations at inference time is more efficient overall. Frameworks like PyTorch and TensorFlow all adopt this Inverted Dropout approach.

### Dropout Rate Selection

The Dropout rate $1-p$ (drop probability) is a key hyperparameter that directly affects the regularization strength. The higher the drop rate, the stronger the regularization, but the harder it is for training to converge. Conversely, if the drop rate is too low, regularization is insufficient. This parameter should not be set uniformly across the network, as different types of layers have different sensitivities to Dropout and require targeted settings:

| Layer Type | Dropout Rate $1-p$ | Keep Probability $p$ | Reason |
|:-----------|:------------------:|:-------------------:|:-------|
| Fully connected layer | 0.5 | 0.5 | Many parameters, prone to overfitting, needs strong regularization |
| Convolutional layer | 0.1-0.25 | 0.75-0.9 | Relatively few parameters, convolution has built-in regularization (spatial sharing) |
| Input layer | 0.2 | 0.8 | Dropping input features may lose information, should not be too high |
| Output layer | 0 | 1.0 | Output layer needs accurate prediction, dropping affects stability |

Fully connected layers have far more parameters than convolutional layers. For example, a fully connected layer with $1000 \rightarrow 500$ neurons has 500,000 parameters, while a $3 \times 3$ convolutional layer (64 input channels, 128 output channels) has only about 70,000 parameters. Fully connected layers have higher parameter redundancy and are more prone to overfitting, thus requiring a higher Dropout rate.

Convolutional kernels share parameters across spatial positions (the same kernel sweeps across the entire image). This parameter sharing is itself a form of regularization, limiting the model's degrees of freedom. Therefore, convolutional layers typically only need a low Dropout rate (0.1-0.25), or even no Dropout at all.

Excessive dropping in the input layer leads to information loss. For instance, in image classification, if the input layer Dropout rate is 0.5, it means each image randomly loses half its pixels, making it very difficult for the model to learn complete features. The output layer should not use Dropout at all, because the output layer is responsible for the final prediction — dropping neurons here would cause prediction instability and affect model accuracy.

## Ensemble Learning Interpretation

After encountering Dropout's random dropping mechanism, a deeper question emerges: why does randomly dropping neurons improve generalization ability? Intuitively, dropping neurons should weaken the network's representational power. Ensemble learning theory attempts to provide an answer to this question. Dropout implicitly constructs a vast number of sub-networks during training, and during inference, it effectively averages the predictions of these sub-networks. This characteristic — training one model while obtaining the effect of multiple models — is the essence of Dropout.

Imagine a team project where 10 people originally work together, but because each person may be absent at any time, the team effectively evolves countless temporary group configurations. Alice and Bob are present today, but tomorrow it becomes Carol and Dave. Each time different members are absent, the team collaboration pattern changes. Over time, each configuration accumulates experience. Dropout does exactly the same thing to neural networks — each time a training sample passes through the network, a portion of neurons is randomly dropped, forming different "sub-network" structures.

Suppose the network has $n$ neurons, and each neuron is independently retained with probability $p$ (dropped with $1-p$). Theoretically, the total number of possible sub-network configurations is $2^n$, because each neuron has two states: "retained" or "dropped." A hidden layer with 100 neurons can theoretically produce $2^{100}$ different sub-networks — an astronomical number, far exceeding the number of atoms in the universe. In practice, we certainly cannot traverse all $2^n$ configurations. However, Dropout's randomness ensures that each training sample passes through a randomly sampled sub-network. Different samples, different training epochs, and different layers all have different masks. A model trained with 100,000 training samples over 100 epochs actually samples about 10 million different sub-network configurations. Although far fewer than $2^n$, the diversity covered is sufficient to force each neuron to learn to work independently in the "absence" of various companions.

In the statistical learning section, when discussing [Random Forest](../../statistical-learning/decision-tree-ensemble/random-forest.md), we encountered ensemble learning algorithms. Traditional ensemble learning requires training multiple independent models to improve generalization. For example, training 5 different neural networks and averaging their predictions during inference. The cost of this approach is obvious — training cost is 5 times, inference cost is 5 times, and storage cost is 5 times. In contrast, Dropout trains only one model yet achieves a similar ensemble effect to multiple models. The key advantage here is **weight sharing** — all sub-networks use the same set of parameters $\mathbf{W}$. The only difference between sub-networks is which neurons are activated, not the parameters themselves. This brings three significant benefits:

1. **Storage efficiency**: No need to store parameters for multiple models; one network's parameters suffice.
2. **Training efficiency**: No need to train multiple independent networks; a single training run is sufficient.
3. **Inference efficiency**: No dropping during inference; a single forward pass yields the "ensemble average" effect.

From a mathematical perspective, the output during inference is equivalent to the expected prediction over all possible sub-networks. Let the network output be $f(\mathbf{x}; \mathbf{W})$. After Dropout, the output is $f_{drop}(\mathbf{x}; \mathbf{W}, \mathbf{r}) = f(\mathbf{x}; \mathbf{W} \cdot (\mathbf{r} \odot \mathbf{x}))$, where $\mathbf{r}$ is the Dropout Mask determining whether each neuron is retained, and $\odot$ denotes element-wise multiplication. The ideal output during inference should be:

$$f_{test}(\mathbf{x}; \mathbf{W}) = \mathbb{E}_{\mathbf{r}}[f_{drop}(\mathbf{x}; \mathbf{W}, \mathbf{r})] = \frac{1}{2^n}\sum_{\mathbf{r}} f(\mathbf{x}; \mathbf{W} \odot \mathbf{r})$$

This formula looks elegant — averaging predictions over all $2^n$ masks. But in reality, $2^n$ is an astronomical number that cannot be exhaustively computed. Fortunately, for linear operations (such as matrix multiplication), the expectation can be computed directly:

$$\mathbb{E}[\mathbf{W} \cdot (\mathbf{r} \odot \mathbf{x})] = \mathbf{W} \cdot \mathbb{E}[\mathbf{r} \odot \mathbf{x}] = \mathbf{W} \cdot (p \cdot \mathbf{x}) = p \cdot \mathbf{W} \cdot \mathbf{x}$$

This is precisely the scaling scheme we discussed in "Training vs. Inference": retained neurons are amplified by $1/p$ during training, and the original value is directly used during inference. For non-linear networks (containing activation functions like ReLU, Sigmoid, etc.), this approximation has some error, but practice has proven it effective because most operations in deep networks are approximately linear (ReLU is a linear function in the positive region), and the diversity of random sampling compensates for the approximation error.

## Overfitting Prevention Mechanism

Ensemble learning provides a theoretical foundation for why Dropout works. In practice, Dropout's mechanism is more intuitive. Dropout prevents overfitting through three pathways: breaking neuron dependencies, reducing effective network complexity, and injecting noise to improve robustness. These three work together to form an effective regularization effect.

- **Breaking neuron co-adaptation**: During neural network training, neurons spontaneously form complex collaborative relationships. Some neurons can only function when other specific neurons are present — they rely on each other, forming implicit teams. This dependency is called neuron **co-adaptation**.

    For example: suppose there is a group of neurons $\{A, B, C\}$ in the network that work together to recognize the concept of "cat." Neuron $A$ detects ear shape, neuron $B$ detects eye position, and neuron $C$ synthesizes the information from the first two to make a judgment. If $C$ always receives reliable input from $A$ and $B$ during training, $C$ will "depend" on this collaboration pattern — its weights will be specifically optimized for the output features of $A$ and $B$. The problem is that this dependency is fragile: if the features of $A$ or $B$ vary slightly in the test data (e.g., the cat's ear shape is slightly different), $C$ may fail completely, leading to overall prediction failure.

    Dropout forcefully breaks these dependencies by randomly dropping neurons. During training, $A$, $B$, and $C$ can all be randomly dropped, so $C$ cannot stably rely on the input from $A$ and $B$. Over time, each neuron learns to "survive independently" — even when companions are absent, it can obtain information through other pathways. This is precisely Hinton's original intention in designing Dropout, simulating the "redundant collaboration" mechanism of bank workers.

- **Reducing effective network complexity**: Dropout dynamically reduces the network's "effective capacity" during training. Suppose a hidden layer has $N$ parameters (weights + biases), and the Dropout rate during training is 0.5. Then, in each training epoch, only about half the neurons participate in computation and parameter updates, and the "effective number of parameters" is approximately $p \cdot N$.

    This brings two benefits:

    - **Limiting model capacity**: Fewer effective parameters constrain the model's fitting ability, preventing it from "memorizing" noise details in the training data.
    - **Training many small networks**: Each sample passes through a different sub-network, which is equivalent to training multiple small models with fewer parameters.

    This is similar in spirit to L2 regularization, but different in approach. L2 limits complexity by penalizing parameter values (making parameters small), while Dropout limits complexity by reducing the number of parameters involved in computation (temporarily making some parameters "disappear"). The two are often used together in practice, complementing each other.

## Injecting Noise for Robustness

Dropout randomly sets neuron outputs to zero, which is equivalent to injecting random noise inside the network. This aligns with the idea of data augmentation, but at a different location: data augmentation injects noise at the input (e.g., image rotation, cropping), while Dropout injects noise inside the network (hidden layer activations).

The role of noise is to force the network to learn "robust" feature representations. Robustness means that the output remains stable even when the input or internal state is perturbed. For example, if the network has learned the feature "cats have triangular ears," and neurons are frequently dropped, the network must learn to still recognize cats through other clues (such as eyes, nose) even when some "ear detection" neurons fail. This "multi-cue, redundant backup" learning approach is precisely the source of robustness.

## Dropout Verification Practice

Theoretical analysis reveals Dropout's working mechanism, but ultimately its effectiveness needs experimental verification. The code below constructs a simulated regression task: using a small training set (100 samples) to train a deep network (64-32-1 architecture), comparing the training process with and without Dropout. We record the training loss and test loss at each epoch, plot the loss curves, and visually demonstrate how Dropout narrows the overfitting gap. The experiment also compares the effect of different Dropout rates (0.0, 0.2, 0.5, 0.7) and the impact of different training set sizes (50, 100, 200, 500) on Dropout effectiveness.

```python runnable
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Dropout Implementation
def dropout(x, p, training=True):
    """
    Dropout function
    x: neuron output
    p: keep probability
    training: whether in training mode
    """
    if not training or p == 1.0:
        return x, np.ones_like(x)
    mask = (np.random.rand(*x.shape) < p).astype(float)
    return x * mask / p, mask

# Multi-layer network (with Dropout support)
class NeuralNetwork:
    def __init__(self, layer_sizes, keep_probs=None, activation='relu'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Default keep probabilities
        if keep_probs is None:
            keep_probs = [0.0] * self.num_layers
        self.keep_probs = keep_probs
        
        # Activation function
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        
        # He initialization
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X, training=True):
        """Forward propagation"""
        self.activations = [X]
        self.pre_activations = []
        self.dropout_masks = []
        
        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            a = self.activation(z)
            
            # Apply Dropout (except last layer)
            if i < self.num_layers - 1:
                a, mask = dropout(a, self.keep_probs[i], training)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
            
            self.activations.append(a)
        
        return a
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward propagation"""
        m = X.shape[0]
        
        # Output layer error (MSE loss)
        delta = (self.activations[-1] - y) * self.activation_derivative(self.pre_activations[-1])
        
        # Backward propagation
        for i in range(self.num_layers - 1, -1, -1):
            # Compute gradients
            grad_w = self.activations[i].T @ delta / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            
            # Update parameters
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
            # Propagate error
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.pre_activations[i-1])
                # Dropout mask backward pass (gradient * mask)
                if self.keep_probs[i-1] > 0:
                    # Reuse mask generated during forward pass
                    delta = delta * self.dropout_masks[i-1] / self.keep_probs[i-1]
    
    def compute_loss(self, X, y, training=False):
        """Compute loss"""
        output = self.forward(X, training=training)
        return np.mean((output - y)**2)

print("Experiment 1: Overfitting vs Dropout Comparison")
print("-" * 40)

# Generate data (small training set, large test set, simulating overfitting scenario)
n_train = 100
n_test = 500
n_features = 20

# Training data
X_train = np.random.randn(n_train, n_features)
y_train = np.sin(X_train[:, 0] * 2) + np.cos(X_train[:, 1]) + np.random.randn(n_train) * 0.1
y_train = y_train.reshape(-1, 1)

# Test data
X_test = np.random.randn(n_test, n_features)
y_test = np.sin(X_test[:, 0] * 2) + np.cos(X_test[:, 1]) + np.random.randn(n_test) * 0.1
y_test = y_test.reshape(-1, 1)

# Network configuration
layer_sizes = [n_features, 64, 32, 1]

# Without Dropout
net_no_dropout = NeuralNetwork(layer_sizes, keep_probs=[1.0, 1.0], activation='relu')

# With Dropout (p=0.5)
net_dropout = NeuralNetwork(layer_sizes, keep_probs=[0.5, 0.5], activation='relu')

# Training parameters
n_epochs = 200
learning_rate = 0.01

# Record training process
train_losses_no_drop = []
test_losses_no_drop = []
train_losses_drop = []
test_losses_drop = []

print("Training...")
for epoch in range(n_epochs):
    # Without Dropout training
    net_no_dropout.forward(X_train, training=True)
    net_no_dropout.backward(X_train, y_train, learning_rate)
    
    train_loss_no = net_no_dropout.compute_loss(X_train, y_train, training=False)
    test_loss_no = net_no_dropout.compute_loss(X_test, y_test, training=False)
    
    train_losses_no_drop.append(train_loss_no)
    test_losses_no_drop.append(test_loss_no)
    
    # Dropout training
    net_dropout.forward(X_train, training=True)
    net_dropout.backward(X_train, y_train, learning_rate)
    
    train_loss_drop = net_dropout.compute_loss(X_train, y_train, training=False)
    test_loss_drop = net_dropout.compute_loss(X_test, y_test, training=False)
    
    train_losses_drop.append(train_loss_drop)
    test_losses_drop.append(test_loss_drop)

print(f"\nWithout Dropout:")
print(f"  Final training loss: {train_losses_no_drop[-1]:.4f}")
print(f"  Final test loss: {test_losses_no_drop[-1]:.4f}")
print(f"  Gap: {test_losses_no_drop[-1] - train_losses_no_drop[-1]:.4f}")

print(f"\nDropout (p=0.5):")
print(f"  Final training loss: {train_losses_drop[-1]:.4f}")
print(f"  Final test loss: {test_losses_drop[-1]:.4f}")
print(f"  Gap: {test_losses_drop[-1] - train_losses_drop[-1]:.4f}")

# Visualize loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Without Dropout
ax1 = axes[0]
ax1.plot(train_losses_no_drop, label='Training Loss', linewidth=2, color='#3498db')
ax1.plot(test_losses_no_drop, label='Test Loss', linewidth=2, color='#e74c3c')
ax1.fill_between(range(len(train_losses_no_drop)), train_losses_no_drop, test_losses_no_drop,
                 alpha=0.3, color='#f39c12', label='Overfitting Gap')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Without Dropout - Severe Overfitting', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Dropout
ax2 = axes[1]
ax2.plot(train_losses_drop, label='Training Loss', linewidth=2, color='#3498db')
ax2.plot(test_losses_drop, label='Test Loss', linewidth=2, color='#e74c3c')
ax2.fill_between(range(len(train_losses_drop)), train_losses_drop, test_losses_drop,
                 alpha=0.3, color='#2ecc71', label='Gap Reduced')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Dropout (p=0.5) - Overfitting Alleviated', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("Experiment 2: Effect of Different Keep Probabilities")
print("-" * 40)

keep_probs_list = [0.0, 0.2, 0.5, 0.7]
results = {}

for rate in keep_probs_list:
    keep_probs_config = [rate, rate] if rate > 0 else [0.0, 0.0]
    net = NeuralNetwork(layer_sizes, keep_probs=keep_probs_config, activation='relu')
    
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        net.forward(X_train, training=True)
        net.backward(X_train, y_train, learning_rate)
        
        train_loss = net.compute_loss(X_train, y_train, training=False)
        test_loss = net.compute_loss(X_test, y_test, training=False)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    results[rate] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_gap': test_losses[-1] - train_losses[-1]
    }
    
    print(f"Keep probability {rate:.1f}:")
    print(f"  Training loss: {train_losses[-1]:.4f}")
    print(f"  Test loss: {test_losses[-1]:.4f}")
    print(f"  Overfitting gap: {results[rate]['final_gap']:.4f}")
    print()

# Visualize different Dropout rates
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for idx, rate in enumerate(keep_probs_list):
    ax = axes[idx // 2, idx % 2]
    ax.plot(results[rate]['train_losses'], label='Training Loss', 
            linewidth=2, color=colors[idx])
    ax.plot(results[rate]['test_losses'], label='Test Loss', 
            linewidth=2, color=colors[idx], linestyle='--')
    
    gap = results[rate]['final_gap']
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'Keep Probability = {rate:.1f}\nOverfitting Gap = {gap:.4f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("Experiment 3: Effect of Training Set Size on Overfitting and Dropout")
print("-" * 40)

train_sizes = [50, 100, 200, 500]
n_epochs = 150

size_results = {}

for n_train in train_sizes:
    # Generate data
    X_train_small = np.random.randn(n_train, n_features)
    y_train_small = np.sin(X_train_small[:, 0] * 2) + np.cos(X_train_small[:, 1]) + \
                    np.random.randn(n_train) * 0.1
    y_train_small = y_train_small.reshape(-1, 1)
    
    # Without Dropout
    net_no = NeuralNetwork(layer_sizes, keep_probs=[1.0, 1.0], activation='relu')
    
    # With Dropout
    net_drop = NeuralNetwork(layer_sizes, keep_probs=[0.5, 0.5], activation='relu')
    
    no_drop_gaps = []
    drop_gaps = []
    
    for epoch in range(n_epochs):
        # Without Dropout
        net_no.forward(X_train_small, training=True)
        net_no.backward(X_train_small, y_train_small, learning_rate)
        
        train_loss_no = net_no.compute_loss(X_train_small, y_train_small, training=False)
        test_loss_no = net_no.compute_loss(X_test, y_test, training=False)
        no_drop_gaps.append(test_loss_no - train_loss_no)
        
        # With Dropout
        net_drop.forward(X_train_small, training=True)
        net_drop.backward(X_train_small, y_train_small, learning_rate)
        
        train_loss_drop = net_drop.compute_loss(X_train_small, y_train_small, training=False)
        test_loss_drop = net_drop.compute_loss(X_test, y_test, training=False)
        drop_gaps.append(test_loss_drop - train_loss_drop)
    
    size_results[n_train] = {
        'no_drop_gap': no_drop_gaps[-1],
        'drop_gap': drop_gaps[-1],
        'improvement': no_drop_gaps[-1] - drop_gaps[-1]
    }
    
    print(f"Training set size {n_train}:")
    print(f"  Without Dropout overfitting gap: {no_drop_gaps[-1]:.4f}")
    print(f"  Dropout overfitting gap: {drop_gaps[-1]:.4f}")
    print(f"  Dropout improvement: {size_results[n_train]['improvement']:.4f}")
    print()

# Visualize the effect of training set size
fig, ax = plt.subplots(figsize=(10, 6))

sizes = list(size_results.keys())
no_drop_gaps = [size_results[s]['no_drop_gap'] for s in sizes]
drop_gaps = [size_results[s]['drop_gap'] for s in sizes]
improvements = [size_results[s]['improvement'] for s in sizes]

x = range(len(sizes))
ax.bar(x, no_drop_gaps, width=0.4, label='Without Dropout', color='#e74c3c', alpha=0.7)
ax.bar([i + 0.4 for i in x], drop_gaps, width=0.4, label='Dropout', color='#2ecc71', alpha=0.7)

ax.set_xticks([i + 0.2 for i in x])
ax.set_xticklabels(sizes)
ax.set_xlabel('Training Set Size', fontsize=11)
ax.set_ylabel('Overfitting Gap (Test - Train Loss)', fontsize=11)
ax.set_title('Effect of Training Set Size on Overfitting', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add improvement annotations
for i, imp in enumerate(improvements):
    ax.annotate(f'Improvement {imp:.2f}', 
                xy=(i + 0.2, max(no_drop_gaps[i], drop_gaps[i]) + 0.02),
                ha='center', fontsize=10, color='#3498db')

plt.tight_layout()
plt.show()
plt.close()
```

## Practical Experience

The effectiveness of Dropout highly depends on correct usage. Using it in the wrong location, with incorrect hyperparameter settings, or in improper combination with [Batch Normalization](./batch-normalization.md) can weaken its effect or even produce negative results. This section summarizes practical experience to help readers apply Dropout effectively in real projects.

- **Dropout layer placement**: The position of the Dropout layer in the network directly affects its effectiveness. The principle for position selection is to use Dropout after layers with dense parameters that are prone to overfitting, and avoid using Dropout after layers that carry critical information or affect stability. Here are some examples:

    | Position | Suggestion | Reason |
    |:---------|:-----------|:-------|
    | After FC layer | Recommended | Many parameters, prone to overfitting, needs strong regularization |
    | After Conv layer | Optional | Fewer parameters, convolution has built-in regularization |
    | After LSTM layer | Caution | Temporal information may be disrupted |
    | Before BN layer | Avoid | BN statistics become unstable |
    | Input layer | Avoid | Dropping input features may lose information |
    | Output layer | Avoid | Affects prediction stability |

- **Dropout and weight decay**: Dropout and L2 weight decay can be used simultaneously — one constrains the effective structural complexity of the model, and the other constrains the magnitude of model parameters; they complement each other. Empirically, Dropout affects the effectiveness of the weight decay hyperparameter. When using Dropout, the weight decay coefficient $\lambda$ can be appropriately reduced — the higher the Dropout rate $1-p$, the lower the weight decay coefficient $\lambda$ should be.

- **Training tips**: Several empirical practices are worth noting during actual training. For Dropout rate tuning, start with $p=0.5$ and observe the gap between training and test losses — a large gap indicates serious overfitting, so lower the keep probability; a small gap may indicate over-regularization, so increase it back. Since Dropout reduces effective network complexity, training convergence typically requires more epochs — do not apply the same early stopping criteria used without Dropout but relax the iteration limit appropriately. Similarly, the random noise injected by Dropout adds jitter to the gradient direction, so the learning rate sometimes needs to be slightly higher than the conventional setting to maintain sufficient forward progress above this layer of noise.

## Summary

Dropout is one of the simplest yet most effective regularization techniques in deep learning. Its idea is straightforward: randomly shut down a portion of neurons during training, forcing the network to learn to survive independently. Behind this seemingly crude operation are three synergistic mechanisms: breaking neuron co-adaptation (each neuron no longer relies on specific companions), reducing effective network complexity (only part of the neurons are activated in each forward pass), and injecting noise for robustness (perturbing internal representations during training). Ensemble learning theory further reveals that Dropout is equivalent to training a vast number of sub-networks with shared weights, implicitly averaging predictions across all sub-networks during inference, achieving an ensemble effect of multiple models at the cost of a single model.

A remaining issue is that Dropout only addresses the overfitting problem, but deep network training faces another challenge — Internal Covariate Shift, where the input distribution of each layer changes as the preceding layers' parameters are updated, leading to unstable gradient propagation and slow convergence. The next chapter introduces [Batch Normalization](batch-normalization.md), which is designed precisely to address this problem by standardizing the input distribution of each layer, making training more stable and faster while also providing a regularization effect.

## Exercises

1. Suppose a neuron's output is $y = 2.5$ and the Dropout keep probability is $p = 0.6$. Calculate: (a) the expected output during training when using the training-time scaling scheme; (b) the inference output when using the inference-time scaling scheme.
    <details>
    <summary>Reference Answer</summary>

    (a) **Training-time scaling scheme**: During training, the output is $y_{drop} = \frac{r}{p} \cdot y$, where $r \sim \text{Bernoulli}(p)$.

    The expected output is:
    $$\mathbb{E}[y_{drop}] = \mathbb{E}\left[\frac{r}{p} \cdot y\right] = \frac{\mathbb{E}[r]}{p} \cdot y = \frac{p}{p} \cdot y = y = 2.5$$

    This means the expected output during training equals the original output.

    (b) **Inference-time scaling scheme**: During training, the output is $y_{drop} = r \cdot y$, and scaling is needed during inference.

    Training expectation:
    $$\mathbb{E}[y_{drop}] = \mathbb{E}[r \cdot y] = \mathbb{E}[r] \cdot y = p \cdot y = 0.6 \times 2.5 = 1.5$$

    Inference output (scaled down to match training expectation):
    $$y_{test} = p \cdot y = 0.6 \times 2.5 = 1.5$$

    **Summary**: Both schemes yield the same final expected output (1.5), differing only in when scaling is applied. The training-time scaling scheme amplifies during training (divides by $p$) and requires no operation during inference. The inference-time scaling scheme does not amplify during training and scales down during inference (multiplies by $p$). In practice, training-time scaling is more commonly used because inference occurs far more often than training, making it more efficient to have no extra computation during inference.
    </details>

1. Explain the behavioral differences of Dropout during the training and inference phases, and why this difference is necessary. Compare the advantages and disadvantages of the two scaling schemes from the perspective of expectation consistency.
    <details>
    <summary>Reference Answer</summary>

    **Behavioral differences between training and inference**:

    | Phase | Dropout Behavior | Reason |
    |:------|:-----------------|:-------|
    | Training | Randomly drop neurons | Break co-adaptation, enhance robustness |
    | Inference | All neurons retained | Need stable, deterministic predictions |

    Dropping during training is a necessary regularization technique, but dropping cannot continue during inference — otherwise, each prediction would yield a different result, and prediction quality would be unstable (dropped neurons could lead to incorrect predictions).

    **Expectation consistency principle**:

    Let a neuron's original output be $y$ and the keep probability be $p$. After random dropping during training, the expected output is $\mathbb{E}[y_{drop}] = p \cdot y$ (only $p$ times the original output). During inference, all neurons are retained and the output is $y$. The inconsistency between the two would lead to prediction bias.

    **Comparison of the two scaling schemes**:

    | Scheme | Training Operation | Inference Operation | Advantage | Disadvantage |
    |:------|:-----------------|:------------------|:---------|:------------|
    | Inference-time scaling | $y_{drop} = r \cdot y$ | $y_{test} = p \cdot y$ | Simple training computation | Scaling needed at every inference |
    | Training-time scaling | $y_{drop} = \frac{r}{p} \cdot y$ | $y_{test} = y$ | No extra inference operation | Amplification needed during training |

    **Pros and cons analysis**:

    Training is typically done once, while inference is repeated many times (a production model may be called millions of times). The training-time scaling scheme places the computational burden on the training phase (one-time cost), requiring no adjustment during inference (skipping scaling computation each time), making it more efficient overall. This is why frameworks like PyTorch and TensorFlow adopt training-time scaling.
    </details>

1. Derive the condition under which the expected output of a hidden layer containing $n$ neurons equals its original output under the Inverted Dropout (training-time scaling) scheme.
    <details>
    <summary>Reference Answer</summary>

    Suppose a hidden layer has $n$ neurons, with the original output vector $\mathbf{h} = [h_1, h_2, \ldots, h_n]$. Each neuron has an independent Dropout mask $r_i \sim \text{Bernoulli}(p)$.

    **Inverted Dropout formula**:
    $$h_i^{drop} = \frac{r_i}{p} \cdot h_i$$

    **Expectation derivation**:
    $$\mathbb{E}[h_i^{drop}] = \mathbb{E}\left[\frac{r_i}{p} \cdot h_i\right] = \frac{\mathbb{E}[r_i]}{p} \cdot h_i$$

    Since $r_i$ follows a Bernoulli distribution, its expectation is:
    $$\mathbb{E}[r_i] = p$$

    Substituting:
    $$\mathbb{E}[h_i^{drop}] = \frac{p}{p} \cdot h_i = h_i$$

    For the entire hidden layer vector:
    $$\mathbb{E}[\mathbf{h}^{drop}] = \mathbb{E}\left[\frac{\mathbf{r}}{p} \odot \mathbf{h}\right] = \frac{\mathbb{E}[\mathbf{r}]}{p} \odot \mathbf{h} = \frac{p}{p} \odot \mathbf{h} = \mathbf{h}$$

    **Conclusion**: By amplifying by $1/p$ during training, Inverted Dropout ensures that the expected output $\mathbb{E}[\mathbf{h}^{drop}] = \mathbf{h}$, equal to the original output. Therefore, during inference, directly outputting $\mathbf{h}$ is sufficient without any scaling adjustment.

    **Key assumption**: The above derivation assumes $r_i$ and $h_i$ are independent (the random mask does not depend on the neuron output value), which holds in standard Dropout.
    </details>
