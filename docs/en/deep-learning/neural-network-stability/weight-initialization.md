# Weight Initialization

**Weight Initialization** is the starting point of neural network training. A good initialization allows the training to converge quickly and stably; a poor initialization can lead to training stagnation, oscillation, or even collapse. This chapter will deeply analyze the importance of initialization, introduce two classic methods -- Xavier initialization and He initialization -- and verify the practical impact of initialization on training through experiments.

## The Symmetry Breaking Problem

Before introducing weight initialization methods, let us first consider what happens if we do not initialize weights at all and instead let all neurons start from the same parameter value (e.g., zero). Consider a simple two-layer fully-connected network where input $x$ passes through a hidden layer $h$ to reach the output $y$ (i.e., $h = \mathbf{W}_1 x$, $y = \mathbf{W}_2 h$), where $\mathbf{W}_1$ is the weight matrix from the input layer to the hidden layer and $\mathbf{W}_2$ is the weight matrix from the hidden layer to the output layer. If we initialize both matrices to zero (i.e., $\mathbf{W}_1 = 0, \quad \mathbf{W}_2 = 0$), then during [forward propagation](../neural-network-structure/backpropagation.md#forward-propagation-computation), we find that the outputs of all hidden layer neurons are $h_i = 0$ (because the weights are zero), and the outputs of all output layer neurons are also $y_i = 0$ (because $h = 0$ and the weights are zero). This means the network's expressive power completely degenerates -- the 100 hidden neurons intended to learn different features become 100 identical replicas.

The network cannot escape this predicament through training either, because the [backpropagation](../neural-network-structure/backpropagation.md#backward-propagation-gradient-computation) phase is equally problematic. The gradient of the hidden layer weights is $\frac{\partial L}{\partial \mathbf{W}_1} = \mathbf{W}_2^T \cdot \frac{\partial L}{\partial y} \cdot \mathbf{x}^T$. When $\mathbf{W}_2 = 0$, the gradient is directly zero, meaning the parameters cannot update and the network remains at its initial state forever. Even if we only initialize $\mathbf{W}_1$ to zero while keeping $\mathbf{W}_2$ non-zero, since $h = 0$, the error propagates identically to all hidden neurons regardless of the different values in $\mathbf{W}_2$'s components. The hidden layer neurons all receive the same gradient, and the updated weights remain identical. This phenomenon, where all neurons are forced to learn the same function and the network's multi-layer, multi-neuron structure becomes meaningless, is called the **Symmetry Breaking Problem**. Zero initialization strips neurons of their distinctiveness, degrading a carefully designed network structure into a single neuron.

## Weight Initialization

Since zero initialization suffers from the symmetry breaking problem, an intuitive solution naturally emerges: assign different random initial values to each neuron's weights to break the symmetry between neurons. We use two of the most common random distributions, the uniform distribution and the [normal distribution](../../maths/probability/probability-basics.md#normal-distribution), to attempt random initialization.

- **Uniform Distribution Initialization**: Randomly sample weights from the interval $[-a, a]$: $\mathbf{W}_{ij} \sim U[-a, a]$, each weight sampled independently. The variance of the uniform distribution is $\text{Var}(\mathbf{W}_{ij}) = \frac{a^2}{3}$.
- **Normal Distribution Initialization**: Sample weights from a zero-mean normal distribution: $\mathbf{W}_{ij} \sim N(0, \sigma^2)$. Most weights are concentrated within the range $[-\sigma, \sigma]$, with a few possibly larger or smaller. The mean is 0 and the variance is $\sigma^2$.

Random initialization successfully breaks the symmetry between neurons, allowing each neuron to learn different features. However, we must address a key question: how should the distribution parameters ($a$ or $\sigma$) be set? This question determines the success or failure of training. Using the normal distribution as an example, consider two extreme scenarios:

- **Parameters too large** (e.g., $\sigma = 10$): During forward propagation, weights multiply the input to produce enormous activation values. For activation functions like Sigmoid and tanh, receiving an input like 100 pushes the output very close to 1. The activations are constantly at the saturation boundary, gradients during backpropagation are near zero, and little parameter update information is transmitted.
- **Parameters too small** (e.g., $\sigma = 0.001$): During forward propagation, activation values decay layer by layer. The signal grows weaker like whispers as it propagates. During backpropagation, gradients also shrink layer by layer. By the time they reach the earlier layers, there is no parameter update information left, and training nearly stalls.

These phenomena are classic instances of the [vanishing gradient problem](../neural-network-structure/activation-loss-functions.md#vanishing-and-exploding-gradients). It is clear that good initialization must find a balance between two goals: "breaking symmetry so each neuron has a unique identity" and "maintaining signal strength to avoid gradient vanishing, allowing the error signal to propagate stably back to the input layer." These two goals appear contradictory. Finding the ideal random distribution parameters is precisely the problem that Xavier initialization and He initialization aim to solve.

### Xavier Initialization

By analyzing the signal propagation process through the network, we can derive the weight variance needed to maintain signal stability and precisely calculate the optimal initialization parameters. This method was first proposed in 2010 by Canadian computer scientist Xavier Glorot and his advisor Yoshua Bengio (2018 Turing Award winner). In their paper titled "[Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)", they systematically analyzed the causes of training difficulties in deep networks and introduced the famous Xavier initialization method. This paper revealed the deep connection between initialization and training stability, becoming a landmark work in the field of deep learning optimization.

Let us begin by analyzing the simplest case: what weight variance allows a signal to propagate through layer after layer of the network. Consider a linear neuron (temporarily ignoring the activation function, or assuming it is linear). Suppose this neuron receives $n_{in}$ inputs and produces one output $y = \sum_{i=1}^{n_{in}} w_i x_i$. To analyze the output variance, we need to make some reasonable assumptions. Let the weights $w_i$ and inputs $x_i$ satisfy the following conditions:

- Independent and identically distributed (each weight and each input is independently sampled)
- Zero mean ($E[w_i] = 0$, $E[x_i] = 0$)
- Fixed variance ($\text{Var}(w_i) = \text{Var}(w)$, $\text{Var}(x_i) = \text{Var}(x)$)

Under these assumptions, the variance of the output $y$ can be derived. First, by the definition of variance, $\text{Var}(y) = \text{Var}\left(\sum_{i=1}^{n_{in}} w_i x_i\right)$. Since $w_i$ and $x_i$ are independent, using the [variance of sum of independent variables](../../maths/probability/probability-basics.md) property ($\text{Var}[X + Y] = \text{Var}[X] + \text{Var}[Y]$), we obtain:

$$[eq:var-mul]\text{Var}(y) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i)$$

Furthermore, using the variance of product of independent variables property ($\text{Var}(XY) = \text{Var}(X) \cdot \text{Var}(Y) + \text{Var}(X) \cdot E[Y]^2 + \text{Var}(Y) \cdot E[X]^2$), combined with the zero-mean assumption ($E[w_i] = 0$ and $E[x_i] = 0$), we obtain:

$$[eq:var-sum]\text{Var}(w_i x_i) = \text{Var}(w) \cdot \text{Var}(x)$$

Combining {{eq:var-mul}} and {{eq:var-sum}} yields:

$$ \text{Var}(y) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

This formula reveals a key insight about signal forward propagation: the output variance equals the product of three factors. $n_{in}$ is the number of inputs -- more inputs lead to larger variance. $\text{Var}(w)$ is the weight variance -- more dispersed weights lead to larger variance. $\text{Var}(x)$ is the input variance -- more dispersed inputs lead to larger variance. To maintain the signal strength of the input variance, the first two factors should cancel each other out through multiplication. That is, the weight variance should be $\frac{1}{n_{in}}$ to preserve signal strength.

So far, we have only completed half of the analysis, considering only the forward propagation direction from input to output. Training a neural network also requires backpropagation. The output layer gradient $\delta_y$ is propagated back through the weight matrix to the input layer, forming the input layer gradient. The [propagation formula](../neural-network-structure/backpropagation.md#hidden-layer-gradient-propagation) is $\delta^k = (\mathbf{W}^{k+1})^T \delta^{k+1} \cdot \sigma'(\mathbf{z}^k)$. Since we are temporarily ignoring the activation function, its gradient is $1$, giving us:

$$\delta_x = \mathbf{W}^T \delta_y$$

Rewriting the matrix form, the gradient of the $i$-th neuron in the input layer is the weighted sum of the gradients of all neurons in the output layer:

$$\delta_{x_i} = \sum_{j=1}^{n_{out}} w_{ji} \delta_{y_j}$$

where $n_{out}$ is the number of output neurons (called Fan-Out), and $w_{ji}$ is the weight from input $i$ to output $j$. This structure is exactly the same as equation {{eq:var-mul}}. We similarly assume that the weights are independently and identically distributed, have zero mean, and fixed variance. Under these conditions, the variance of the input layer gradient $\delta_{x_i}$ can be derived using the same process as before, yielding:

$$\text{Var}(\delta_x) = n_{out} \cdot \text{Var}(w) \cdot \text{Var}(\delta_y)$$

To maintain the gradient strength unchanged ($\text{Var}(\delta_x) = \text{Var}(\delta_y)$), we need $n_{out} \cdot \text{Var}(w) = 1$. Therefore, the ideal weight variance should be $\text{Var}(w) = \frac{1}{n_{out}}$. A contradiction has emerged in our derivation: forward propagation requires $\text{Var}(w) = \frac{1}{n_{in}}$, while backpropagation requires $\text{Var}(w) = \frac{1}{n_{out}}$. In practice, $n_{in} \neq n_{out}$ in general, and reality dictates that we cannot satisfy both conditions simultaneously.

To address this, Xavier Glorot proposed an elegant compromise solution. Forward propagation requires $\text{Var}(w) = \frac{1}{n_{in}}$, while backpropagation requires $\text{Var}(w) = \frac{1}{n_{out}}$. These two conditions are like a tug-of-war -- one pulls left, the other pulls right. When $n_{in} \neq n_{out}$, we can only choose a compromise position. Xavier initialization adopts the harmonic mean of $\frac{1}{n_{in}}$ and $\frac{1}{n_{out}}$:

$$[eq:xavier-var] \text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$

Why choose the harmonic mean over the arithmetic mean? Glorot's reasoning is that the harmonic mean is more sensitive to smaller values. When $n_{in}$ and $n_{out}$ differ significantly, the harmonic mean leans toward the smaller value, avoiding excessively large variance that could lead to gradient explosion. Based on this variance formula, Xavier initialization has two specific implementations.

- **Xavier Uniform Initialization**: Sample from a uniform distribution. The variance of a uniform distribution $U[-a, a]$ is $\frac{a^2}{3}$. To make the variance equal to $\frac{2}{n_{in} + n_{out}}$, we need $\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}}$, giving $a = \sqrt{\frac{6}{n_{in} + n_{out}}}$.

    $$\mathbf{W}_{ij} \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

- **Xavier Normal Initialization**: Sample from a normal distribution. The variance of a normal distribution $N(0, \sigma^2)$ is $\sigma^2$, so the standard deviation is directly taken as $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$.

    $$\mathbf{W}_{ij} \sim N\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

The derivation of Xavier initialization has a key assumption: the activation function is linear. This assumption certainly does not hold in practice (activation functions are meant for non-linear expression). However, consider this: if Xavier initialization controls the weight variance, most activation values will not deviate far from zero and fall into the saturation region, but instead remain concentrated near zero. Let us revisit the shape of the [Sigmoid activation function](../../statistical-learning/linear-models/logistic-regression.md#sigmoid-function). When the input $x$ is near 0, the behavior of the Sigmoid function can be approximated using a Taylor expansion:

$$\sigma(x) \approx \sigma(0) + \sigma'(0) \cdot x = 0.5 + 0.25 \cdot x$$

This is a linear function. When the activation values fall near 0, Sigmoid approximates a linear transformation, and Xavier's linearity assumption is largely valid within this range. Similarly, the [tanh activation function](../neural-network-structure/activation-loss-functions.md#hyperbolic-tangent-function) is also approximately linear near 0:

$$\tanh(x) \approx \tanh(0) + \tanh'(0) \cdot x = 0 + 1 \cdot x = x$$

tanh's linear approximation near 0 is even more precise than sigmoid's (slope of 1 vs. sigmoid's slope of 0.25). Xavier initialization leverages this property by controlling the weight variance so that activation values fall near 0 (the linear region), thereby maintaining stable signal strength during propagation.

Of course, based on its underlying assumptions, Xavier initialization has notable limitations. On one hand, when inputs are far from 0, the non-linear characteristics of activation functions emerge. Xavier initialization can only try to keep activation values in the linear region but cannot completely avoid saturation. On the other hand, and more critically, Xavier initialization is not compatible with the [ReLU activation function](../neural-network-structure/activation-loss-functions.md#relu-and-its-variants). ReLU's characteristics are fundamentally different from Sigmoid and tanh. ReLU retains only positive values and sets all negative values to zero, meaning about half of the activation values are "killed" and the signal strength is halved. Xavier's linearity assumption completely fails for ReLU. If Xavier initialization is used for a ReLU network, activation values decay layer by layer, and the gradients of deep layers are nearly zero. To address ReLU's unique characteristics, a new initialization method is needed -- this is precisely the problem that He initialization solves.

### He Initialization

To address the incompatibility of Xavier initialization with ReLU, Chinese computer scientist Kaiming He proposed an initialization method specifically designed for ReLU in 2015. In his paper "[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)", he systematically analyzed the signal propagation characteristics of ReLU networks and proposed He initialization. This paper not only contributed to initialization theory but, more importantly, demonstrated for the first time the possibility of deep networks surpassing human performance on image classification tasks. Kaiming He's team achieved a Top-5 error rate of 3.57% on the ImageNet competition using deep residual networks (ResNet), lower than the human error rate of 5.1%. This milestone achievement proved the enormous potential of deep learning and established He initialization as a core method for ReLU networks.

The problem with Xavier initialization stems from ReLU killing half the signal in each layer; we need larger weight variance to compensate. Following a similar derivation to Xavier initialization, we consider signal propagation after ReLU activation and derive the weight variance needed to maintain signal stability. Let the input $x_i$ be the output of the previous layer's ReLU, with weights $w_i$ independently and identically distributed with zero mean. Forward propagation is divided into two steps: weighted summation ($z = \sum_{i=1}^{n_{in}} w_i x_i$) and ReLU activation ($y = \max(0, z)$). The variance derivation for the first step remains unchanged (see {{eq:var-sum}}), giving:

$$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

After ReLU activation $y = \max(0, z)$, for a zero-mean $z$, the variance of the ReLU output is half of the original variance:

$$\text{Var}(y) = \frac{1}{2} \text{Var}(z) = \frac{1}{2} \cdot n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

To maintain signal strength unchanged ($\text{Var}(y) = \text{Var}(x)$), the product of the first two terms must equal 1 ($\frac{1}{2} n_{in} \cdot \text{Var}(w) = 1$), giving:

$$\text{Var}(w) = \frac{2}{n_{in}}$$

The 2 in the numerator is a compensation factor that counteracts ReLU's signal attenuation (multiplying by $\frac{1}{2}$ requires $2$ to compensate). He initialization requires a larger weight variance than Xavier to offset ReLU's "kill half the signal" effect. Based on this variance formula, He initialization also has two implementations:

- **He Uniform Initialization**: Sample from a uniform distribution. The boundary of the uniform distribution is $a = \sqrt{\frac{6}{n_{in}}}$, so that the variance satisfies $\frac{a^2}{3} = \frac{2}{n_{in}}$.

$$\mathbf{W}_{ij} \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]$$

- **He Normal Initialization**: Sample from a normal distribution. The standard deviation is $\sigma = \sqrt{\frac{2}{n_{in}}}$, with most weights falling within $[-\sigma, \sigma]$.

$$\mathbf{W}_{ij} \sim N\left(0, \frac{2}{n_{in}}\right)$$

He initialization's variance formula only considers $n_{in}$, while Xavier considers the harmonic mean of $n_{in} + n_{out}$. This is because He initialization places greater emphasis on forward propagation signal compensation. ReLU's sparsity problem mainly occurs during forward propagation. During backpropagation, gradients only pass through positive-valued neurons, and the signal is also halved in the reverse direction. However, He initialization prioritizes forward propagation signal stability and therefore only compensates for the forward direction. When $n_{in} \approx n_{out}$, He initialization's variance is approximately twice that of Xavier.

## Initialization Method Experiments

Theoretical analysis tells us that Xavier initialization is suitable for Sigmoid/tanh, and He initialization is suitable for ReLU. In this section, let us verify this conclusion through code experiments.

The following experiment simulates a multi-layer neural network (similar to an MLP structure), comparing the performance of five initialization methods (zero initialization, small variance initialization, large variance initialization, Xavier initialization, He initialization) under three activation functions (Sigmoid, ReLU, tanh). The experiment measures three key indicators: the distribution of activation values per layer (verifying signal propagation), the gradient norm per layer (verifying gradient propagation), and the training loss curve (verifying convergence speed).

```python runnable
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Experiment: Impact of Different Initialization Methods on Training Stability")
print("=" * 60)
print()

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

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Define initialization methods
def zero_init(shape):
    """Zero initialization"""
    return np.zeros(shape)

def random_init(shape, scale=0.01):
    """Random initialization (small variance)"""
    return np.random.randn(*shape) * scale

def random_large_init(shape, scale=10):
    """Random initialization (large variance)"""
    return np.random.randn(*shape) * scale

def xavier_uniform_init(shape):
    """Xavier uniform initialization"""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def xavier_normal_init(shape):
    """Xavier normal initialization"""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.randn(*shape) * std

def he_normal_init(shape):
    """He normal initialization"""
    fan_in = shape[0]
    std = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * std

def he_uniform_init(shape):
    """He uniform initialization"""
    fan_in = shape[0]
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)

# Simple multi-layer network
class SimpleNetwork:
    def __init__(self, layer_sizes, activation='relu', init_method='he_normal'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Select activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        
        # Select initialization method
        init_funcs = {
            'zero': zero_init,
            'random_small': lambda s: random_init(s, 0.01),
            'random_large': lambda s: random_init(s, 10),
            'xavier_uniform': xavier_uniform_init,
            'xavier_normal': xavier_normal_init,
            'he_normal': he_normal_init,
            'he_uniform': he_uniform_init
        }
        self.init_func = init_funcs[init_method]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = self.init_func((layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Record activations and gradients per layer
        self.activations_history = []
        self.gradients_history = []
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.pre_activations = []
        
        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            a = self.activation(z)
            self.activations.append(a)
        
        return a
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward propagation"""
        m = X.shape[0]
        
        # Compute output layer error (simple MSE loss)
        delta = (self.activations[-1] - y) * self.activation_derivative(self.pre_activations[-1])
        
        # Store gradient norms
        gradients = []
        
        # Backward propagation
        for i in range(self.num_layers - 1, -1, -1):
            # Compute weight gradient
            grad_w = self.activations[i].T @ delta / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            
            gradients.append(np.linalg.norm(grad_w))
            
            # Update weights and biases
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
            # Propagate error to the previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.pre_activations[i-1])
        
        # Record gradient history (from first layer to last)
        self.gradients_history.append(gradients[::-1])
        self.activations_history.append([np.mean(np.abs(a)) for a in self.activations])

# Experiment 1: Activation value distribution for different initialization methods
print("Experiment 1: Activation Distribution for Different Initialization Methods")
print("-" * 40)

layer_sizes = [784, 512, 256, 128, 64, 10]  # Similar to MLP structure
n_samples = 100
X = np.random.randn(n_samples, 784) * 0.5  # Simulate standardized input

init_methods = ['zero', 'random_small', 'random_large', 'xavier_normal', 'he_normal']
activation_names = ['sigmoid', 'relu', 'tanh']

results = {}

for activation in activation_names:
    results[activation] = {}
    for init_method in init_methods:
        net = SimpleNetwork(layer_sizes, activation=activation, init_method=init_method)
        output = net.forward(X)
        
        # Record mean and variance of activation values per layer
        activation_stats = []
        for i, a in enumerate(net.activations):
            mean_val = np.mean(np.abs(a))
            std_val = np.std(a)
            activation_stats.append((mean_val, std_val))
        
        results[activation][init_method] = {
            'activations': net.activations,
            'pre_activations': net.pre_activations,
            'stats': activation_stats
        }
        
        print(f"{activation} + {init_method}:")
        for i, (mean_val, std_val) in enumerate(activation_stats):
            print(f"  {i}: mean={mean_val:.4f}, std={std_val:.4f}")
        print()

# Visualize activation distribution
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for row, activation in enumerate(activation_names):
    for col, init_method in enumerate(init_methods):
        ax = axes[row, col]
        
        stats = results[activation][init_method]['stats']
        layer_means = [s[0] for s in stats]
        layer_stds = [s[1] for s in stats]
        
        layers = range(len(stats))
        ax.bar(layers, layer_means, color='#3498db', alpha=0.7, label='Mean')
        ax.errorbar(layers, layer_means, yerr=layer_stds, fmt='o', color='#e74c3c', 
                   capsize=5, capthick=2, label='Std Dev')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Activation Value')
        ax.set_title(f'{activation} + {init_method}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("Experiment 2: Gradient Propagation for Different Initialization Methods")
print("-" * 40)

# Experiment 2: Gradient propagation for different initialization methods
layer_sizes = [784, 512, 256, 128, 10]
n_samples = 100
X = np.random.randn(n_samples, 784) * 0.5
y = np.random.randn(n_samples, 10) * 0.1  # Simulate target output

gradient_results = {}

for activation in ['sigmoid', 'relu']:
    gradient_results[activation] = {}
    for init_method in init_methods:
        if init_method == 'zero':
            continue  # Zero initialization yields zero gradients, skip
        
        net = SimpleNetwork(layer_sizes, activation=activation, init_method=init_method)
        
        # Train for 50 steps, record gradients
        for step in range(50):
            output = net.forward(X)
            net.backward(X, y, learning_rate=0.001)
        
        # Extract gradient norms per layer (take the last step)
        final_gradients = net.gradients_history[-1]
        gradient_results[activation][init_method] = {
            'gradients': final_gradients,
            'history': net.gradients_history
        }
        
        print(f"{activation} + {init_method}:")
        for i, g in enumerate(final_gradients):
            print(f"  Layer {i} gradient norm: {g:.6f}")
        print()

# Visualize gradient distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, activation in enumerate(['sigmoid', 'relu']):
    ax = axes[idx]
    
    for init_method in init_methods:
        if init_method == 'zero':
            continue
        
        gradients = gradient_results[activation][init_method]['gradients']
        layers = range(len(gradients))
        ax.plot(layers, gradients, 'o-', linewidth=2, markersize=8, label=init_method)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient Norm')
    ax.set_title(f'Gradient Propagation for {activation} Activation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.show()
plt.close()

print("\n" + "=" * 60)
print("Experiment 3: Impact of Initialization on Training Convergence")
print("-" * 40)

# Experiment 3: Impact of initialization on training convergence
layer_sizes = [784, 256, 128, 10]
n_samples = 500
n_epochs = 200

# Generate simple classification data
X_train = np.random.randn(n_samples, 784)
y_train = np.zeros((n_samples, 10))
y_train[:, 0] = 1  # All samples are class 0

convergence_results = {}

for activation in ['relu', 'sigmoid']:
    convergence_results[activation] = {}
    for init_method in ['random_small', 'xavier_normal', 'he_normal']:
        if activation == 'relu' and init_method == 'xavier_normal':
            continue  # Xavier is not suitable for ReLU
        
        if activation == 'sigmoid' and init_method == 'he_normal':
            continue  # He is not suitable for sigmoid
        
        net = SimpleNetwork(layer_sizes, activation=activation, init_method=init_method)
        losses = []
        
        for epoch in range(n_epochs):
            output = net.forward(X_train)
            loss = np.mean((output - y_train)**2)
            losses.append(loss)
            net.backward(X_train, y_train, learning_rate=0.01)
        
        convergence_results[activation][init_method] = {
            'losses': losses,
            'final_loss': losses[-1]
        }
        
        print(f"{activation} + {init_method}:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss reduction: {losses[0] - losses[-1]:.4f}")
        print()

# Visualize convergence curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, activation in enumerate(['relu', 'sigmoid']):
    ax = axes[idx]
    
    for init_method, data in convergence_results[activation].items():
        ax.plot(data['losses'], linewidth=2, label=init_method)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Convergence for {activation} Activation')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
```

## Bias Initialization

Thus far, we have discussed weight initialization. However, a neuron's parameters include both weights and biases. How should biases be initialized? The answer is that biases are typically initialized to zero, for the following reasons:

1. **Biases do not participate in signal strength transmission**: The weight matrix multiplied by the input determines how the signal flows; biases merely add a constant offset without changing the relative signal strength. Therefore, zero bias does not cause symmetry issues.
2. **Zero bias keeps the activation function in the linear region**: Sigmoid and tanh are approximately linear when the input is near 0, which is precisely the prerequisite for Xavier initialization to work. Zero bias makes the activation function's input $z = \mathbf{W}x + b$ approximately equal to $\mathbf{W}x$ (zero mean), helping activation values fall in the linear region.
3. **Non-zero bias may push activation values out of the linear region**: If the bias is initialized to a large positive value, Sigmoid activation values may directly saturate to 1, with gradients near zero. If the bias is initialized to a large negative value, activation values may directly saturate to 0, also with gradients near zero.

However, there are some special cases where non-zero bias initialization is needed, such as:

- **Positive bias for ReLU networks**: The ReLU activation function sets negative values to zero. If the weight initialization makes the variance of the weighted sum moderate (as in He initialization), most activation values may fall near zero, with half positive and half negative. After the negative portion is zeroed out, the network's initial output may be too sparse. Initializing biases to a small positive value (e.g., 0.01) can ensure that most neurons have non-zero output initially, helping to avoid the dead neuron problem.

- **LSTM forget gate bias**: LSTM (Long Short-Term Memory, covered in a later chapter) has a forget gate that controls how much of the previous time step's information is retained. When the forget gate output is close to 0, most historical information is forgotten; when the output is close to 1, most information is retained. If the forget gate bias is initialized to zero, the initial forget gate output may be close to 0.5 (the midpoint of Sigmoid), causing partial forgetting of historical information. Initializing the forget gate bias to 1 or larger makes the initial forget gate output close to 1, retaining more historical information and helping the network learn long-term dependencies in the early stages of training.

## Chapter Summary

This chapter started from "why initialization is crucial," systematically analyzed the impact of weight initialization on deep network training, and introduced two classic methods: Xavier initialization and He initialization. The challenge of initialization lies in finding a balance point. Zero initialization cannot break symmetry, forcing all neurons to learn the same function and degrading the network to a single neuron. Random initialization breaks symmetry, but improper variance selection leads to two extremes: variance too small causes signal decay layer by layer and slow training; variance too large causes activation saturation and gradient vanishing. Good initialization must find a balance between "breaking symmetry" and "maintaining signal strength."

Initialization is the starting point of training, determining whether the network can get off to a good start. However, the end goal of training is whether it can converge to the optimum. The greatest difficulty encountered here is overfitting. We previously covered the use of L1 and L2 norm regularization to address overfitting in the linear models section. The next chapter will introduce Dropout regularization, which forces the network to learn more robust features by randomly dropping neurons, improving training stability from another perspective.

## Exercises

1. Given a fully-connected layer with input dimension $n_{in} = 512$ and output dimension $n_{out} = 256$, calculate the parameter ranges for Xavier uniform initialization and Xavier normal initialization respectively.
    <details>
    <summary>Reference Answer</summary>

    **Xavier Uniform Initialization**:

    Using the formula $a = \sqrt{\frac{6}{n_{in} + n_{out}}}$, substituting the values:
    $$a = \sqrt{\frac{6}{512 + 256}} = \sqrt{\frac{6}{768}} = \sqrt{\frac{1}{128}} \approx 0.088$$

    Therefore, weights are sampled from $U[-0.088, 0.088]$.

    **Xavier Normal Initialization**:

    Using the formula $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$, substituting the values:
    $$\sigma = \sqrt{\frac{2}{512 + 256}} = \sqrt{\frac{2}{768}} = \sqrt{\frac{1}{384}} \approx 0.051$$

    Therefore, weights are sampled from $N(0, 0.051)$, with most weights falling within $[-0.051, 0.051]$.

    **Comparison**: The range of uniform initialization is approximately 1.73 times the standard deviation of normal initialization ($0.088 / 0.051 \approx 1.73$), which is consistent with the relationship between the uniform distribution variance formula $\frac{a^2}{3}$ and the normal distribution variance $\sigma^2$.
    </details>

2. Explain why Xavier initialization uses the harmonic mean of $\frac{1}{n_{in}}$ and $\frac{1}{n_{out}}$, i.e., $\frac{2}{n_{in} + n_{out}}$, instead of the arithmetic mean $\frac{1}{2}\left(\frac{1}{n_{in}} + \frac{1}{n_{out}}\right)$. Explain the advantage of the harmonic mean from the perspective of gradient stability.
    <details>
    <summary>Reference Answer</summary>

    Forward propagation requires $\text{Var}(w) = \frac{1}{n_{in}}$, while backpropagation requires $\text{Var}(w) = \frac{1}{n_{out}}$. When $n_{in} \neq n_{out}$, a compromise is needed.

    **Harmonic mean**: $\frac{2}{n_{in} + n_{out}}$

    **Arithmetic mean**: $\frac{1}{2}\left(\frac{1}{n_{in}} + \frac{1}{n_{out}}\right) = \frac{n_{in} + n_{out}}{2 n_{in} n_{out}}$

    The difference: the harmonic mean is more sensitive to smaller values. Let $n_{in} = 100$, $n_{out} = 1000$:

    - Harmonic mean: $\frac{2}{100 + 1000} = \frac{2}{1100} \approx 0.0018$
    - Arithmetic mean: $\frac{100 + 1000}{2 \times 100 \times 1000} = \frac{1100}{200000} = 0.0055$

    The harmonic mean is approximately $\frac{1}{3}$ of the arithmetic mean, leaning toward the smaller $\frac{1}{n_{out}} = 0.001$ (backpropagation requirement) rather than the larger $\frac{1}{n_{in}} = 0.01$ (forward propagation requirement).

    **Gradient stability perspective**: In the backpropagation gradient norm formula $\text{Var}(\delta_x) = n_{out} \cdot \text{Var}(w) \cdot \text{Var}(\delta_y)$, when $n_{out}$ is large and $\text{Var}(w)$ is also large, the gradient norm grows exponentially, leading to gradient explosion. The harmonic mean leans toward the smaller variance, effectively suppressing the risk of gradient explosion during backpropagation and prioritizing gradient stability.
    </details>

3. Consider a three-layer fully-connected network with dimensions $[784, 512, 128, 10]$ using the ReLU activation function. If Xavier initialization is mistakenly used instead of He initialization, analyze how the signal changes during forward propagation and calculate the factor by which the signal variance changes after three layers.
    <details>
    <summary>Reference Answer</summary>

    **Problem of using Xavier initialization for ReLU**:

    Xavier initialization assumes the activation function is approximately linear, and its derived variance formula $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$ does not account for ReLU's signal attenuation. However, ReLU zeroes out approximately half of the activation values, halving the signal variance.

    **Signal variance change analysis**:

    Let the first layer weights use Xavier initialization, with $n_{in} = 784$, $n_{out} = 512$:
    $$\text{Var}(w_1) = \frac{2}{784 + 512} = \frac{2}{1296} \approx 0.00154$$

    After forward propagation (weighted sum): $\text{Var}(z_1) = n_{in} \cdot \text{Var}(w_1) \cdot \text{Var}(x) = 784 \times 0.00154 \times \text{Var}(x) = 1.21 \cdot \text{Var}(x)$

    After ReLU activation: $\text{Var}(a_1) = \frac{1}{2} \text{Var}(z_1) = 0.61 \cdot \text{Var}(x)$ (signal attenuation)

    Similar attenuation occurs in the second and third layers, with each layer's signal variance being approximately $0.5 \sim 0.6$ times the previous layer.

    **Cumulative effect after three layers**:

    Rough estimate: $\text{Var}(a_3) \approx 0.61^3 \cdot \text{Var}(x) \approx 0.23 \cdot \text{Var}(x)$

    The signal variance decays to approximately 23% of its initial value. The activation values in deep layers tend toward zero, and gradient propagation is hindered.

    **Correct approach**: Use He initialization, where the factor of 2 in the variance formula $\text{Var}(w) = \frac{2}{n_{in}}$ compensates for ReLU's signal halving, maintaining stable signal strength.
    </details>
