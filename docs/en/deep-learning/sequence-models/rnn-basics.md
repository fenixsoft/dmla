# RNN Fundamentals

Thus far, all the models we have encountered follow a common pattern: each input is processed independently by the model, whether for information classification or image generation. The current processing has no relation to the previous input — in computer terms, the operations are stateless.

In the real world, there exists another type of data whose core characteristic is that the current data depends on previous data, with temporal dependencies between data points. This type of data is called **Sequential Data**. Text is a typical example: the order of the three words in "dog bites man" determines the meaning, while "man bites dog" means something completely different. Similarly, speech is a sequence — sound waves change over time, and the pronunciation of one moment affects the understanding of the next; video is also a sequence, where each frame has a temporal order; stock prices form a sequence, where today's price correlates with yesterday's and last week's prices; weather prediction is sequential, where tomorrow's temperature is related to the temperature changes of the past week.

Processing sequential data requires a model architecture that has state, can remember past information, and makes current decisions based on history. In 1990, American cognitive scientist Jeffrey Elman published the paper "[Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1)", which first proposed the **Recurrent Neural Network** (Recurrent Network, later called Elman Network). Elman's insight was that humans do not process words independently when understanding language; instead, they continuously maintain a contextual state. After reading "cat", the mind retains the information that "there is a cat", and upon reading "eats fish", it immediately associates back to the earlier "cat". The title of Elman's paper itself reveals the key — "Finding Structure in Time".

Recurrent Neural Networks introduce **Recurrent Connections** to transmit information along the temporal dimension, giving the network the ability to remember. The output at the current time step depends not only on the current input but also on all previous inputs. This design idea gave rise to key innovations such as LSTM, GRU, and the attention mechanism, enabling breakthroughs in natural language processing, speech recognition, time series prediction, and other fields. This chapter will introduce the basic principles, structural design, training methods, and main challenges of RNNs.

## Sequence Modeling

Traditional network models face an inherent obstacle when it comes to sequence modeling. MLP is a fully connected structure, where all input positions are naturally symmetric — the neuron at position 1 is no different from the neuron at position 3. If the sequence order of the input data is shuffled, the MLP output can remain exactly the same, making it inherently unable to understand the concept of temporal order. In sequential data, position has temporal meaning and is unidirectional (time step $t$ comes before time step $t+1$). Existing networks cannot model this temporal unidirectionality, so a new network architecture is needed — one that can process sequences step by step, transmit historical information, and capture temporal dependencies.

This new network architecture needs to satisfy four objectives. First, step-by-step processing: each time step processes one input rather than the entire sequence at once, naturally accommodating variable-length sequences. Second, information transmission: the current time step can utilize information from previous time steps, achieving memory capability. Third, variable-length adaptability: theoretically, it should be able to process sequences of arbitrary length without requiring a fixed input dimension. Finally, temporal modeling: it must capture the temporal dependencies in data and understand the meaning of order. RNNs elegantly satisfy all four objectives simultaneously through **Recurrent Connections**. A recurrent connection means that the output of the network at time step $t$ is not only passed to the next layer but also fed back to the network itself as an additional input at time step $t+1$. This design cleverly embeds the "memory" mechanism into the network structure, as shown in the figure below.

```nn-arch width=500
name: Recurrent Connection Architecture
layout: horizontal

sections:
  - name: Time Step t = 1
    layers: [x1, h1, y1]
    row_label: "h₁→h₂"
  - name: Time Step t = 2
    layers: [x2, h2, y2]
    row_label: "h₂→h₃"
  - name: Time Step t = 3
    layers: [x3, h3, y3]
    row_label: "h₃→h₄"

layers:
  - {id: x1, name: "x₁", type: input, size: "Input 1"}
  - {id: h1, name: "RNN h₁", type: rnn, size: 256, act: tanh}
  - {id: y1, name: "y₁", type: output, size: "Output 1"}
  - {id: x2, name: "x₂", type: input, size: "Input 2"}
  - {id: h2, name: "RNN h₂", type: rnn, size: 256, act: tanh}
  - {id: y2, name: "y₂", type: output, size: "Output 2"}
  - {id: x3, name: "x₃", type: input, size: "Input 3"}
  - {id: h3, name: "RNN h₃", type: rnn, size: 256, act: tanh}
  - {id: y3, name: "y₃", type: output, size: "Output 3"}
```
*Figure: Recurrent Connection Architecture*

The figure illustrates the core mechanism of recurrent connections: $x_t$ is the input at time step $t$ (e.g., the vector representation of the $t$-th word), $h_t$ is the hidden state at time step $t$, which also serves as an additional input at time step $t+1$. The network's state $h_t$ at time step $t$ contains a compressed representation of all input information from time step $1$ to $t$, continuously updating and accumulating information over time. The arrows indicate the direction of information flow along the temporal axis, where the current hidden state flows to the next time step, enabling memory transmission across time.

### Mathematical Representation

Now let us describe the design of recurrent connections in mathematical terms. Let $h_{t-1}$ be the hidden state from the previous time step, representing the network's historical memory before the current operation; $W_{hh}$ be the recurrent connection weight matrix, controlling how historical information influences the current state; $x_t$ be the input vector at the current time step, representing the information currently being processed; $W_{xh}$ be the input weight matrix, controlling how the current input influences the hidden state; $b_h$ be the bias vector, and $\sigma$ be the activation function (typically tanh). The formula for computing the current hidden state (current memory) via the recurrent connection is:

$$[rnn_hs]h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

That is, new memory = processing historical memory + processing current input. The hidden state at each time step is jointly determined by the current input and the hidden state from the previous time step. Then, let $W_{hy}$ be the output weight matrix, mapping the hidden state to the output space; $b_y$ be the output bias. The output formula of the recurrent neural network is:

$$y_t = W_{hy} h_t + b_y$$

The formula indicates that the network's output is determined by the final memory $h_t$. All hidden states from $h_{1}$ to $h_{t-1}$ influence the computation of $h_t$, thereby enabling information transmission across time. This information propagation chain with sequential relationships is the core distinction between RNNs and feedforward networks.

The computation from $h_{1}$ to $h_t$ is not accomplished by directly modeling with $t$ layers of a neural network, because each sequence has a different length and $t$ is an indeterminate number that cannot be expressed with a fixed number of network layers. In practice, it is accomplished by having the same network iterate $t$ times, which is the origin of the term "recurrent" in "recurrent connection" and "recurrent neural network". This recursive nested cyclic structure is also reflected mathematically:

- Initial hidden state $h_0 = 0$. Applying formula {{rnn_hs}} yields $h_1 = \sigma(W_{xh} x_1 + b_h)$
- Hidden state at time step 2 nests the information from time step 1: $h_2 = \sigma(W_{hh} h_1 + W_{xh} x_2 + b_h) = \sigma(W_{hh} \sigma(W_{xh} x_1 + b_h) + W_{xh} x_2 + b_h)$
- Hidden state at time step 3 further nests information from time steps 1 and 2: $h_3 = \sigma(W_{hh} h_2 + W_{xh} x_3 + b_h) = \sigma(W_{hh} \sigma(W_{hh} \sigma(...) + ...) + W_{xh} x_3 + b_h)$
- ……

From the computation formula of $h_3$, it can be seen that $h_3$ actually depends on all the input information $x_1, x_2, x_3$. Information is compressed and transmitted through nested activation functions and weight matrices, with historical information gradually encoded into the hidden state. The hidden state $h_t$ can be viewed as a compression function containing information from all inputs at time steps $1$ through $t$:

$$h_t = f(x_1, x_2, ..., x_t)$$

It is easy to prove that this compression is lossy, because $h_t$ is a vector of fixed dimension while the input has variable length; clearly $h_t$ cannot completely and losslessly store all historical information. RNNs learn to effectively compress historical information through training, retaining only the parts most useful for the current task. To better understand RNN information compression through analogy to human reading: $x_t$ is the word currently being read, and $h_t$ is the comprehension state after reading that word. $h_t$ does not memorize every historical word verbatim, but rather remembers what has been said so far. This state influences the understanding of the next word — for instance, when reading "cat eats fish", the mind retains the information about "cat", and understanding "eats" is naturally associated with "cat" as the subject of the action.

### Architecture Patterns

Depending on the input and output formats, RNNs have several architectural patterns suited to different tasks:

- **One-to-One**: The simplest form, with only one time step for input and output. This pattern effectively degenerates into an ordinary neural network, as it does not require sequence modeling capability at all.
- **One-to-Many**: A single input produces a sequence output. A typical application is image caption generation, where a feature vector of an image is input and a sequence of descriptive words is output.
- **Many-to-One**: A sequence input produces a single output. Typical applications include sentiment analysis, where a sequence of words in a sentence is input and a sentiment classification label is output; stock prediction, where a historical price sequence is input and a rise/fall prediction is output; and so on.
- **Many-to-Many**: A sequence input produces a sequence output, where the input and output have the same length. Typical applications include video classification, where each frame is input and a classification label is output for each frame.
- **Encoder-Decoder**: A pattern where the input sequence and output sequence have different lengths. Typical applications include machine translation, where an English sentence (5 words) is input and a Chinese translation (7 characters) is output. The encoder first compresses the input sequence into a fixed vector, and the decoder then generates the output sequence from that vector. This is the foundational architecture of the [Seq2Seq](seq2seq.md) model, which will be detailed in the next article.

## Gradient Propagation and Limitations

In the [mathematical representation of RNNs](#mathematical-representation), we analyzed the information flow of hidden states during forward propagation. To train a network, a path for backpropagation is also needed — that is, solving how gradients flow along the temporal dimension. The training algorithm for RNNs is called **Backpropagation Through Time** (BPTT). As the name suggests, it still uses the backpropagation algorithm, but with additional handling along the temporal dimension. The core mechanism of BPTT is that the gradient of the loss function $L_t$ at time step $t$ with respect to the parameters at time step $k$ ($k < t$) propagates through a chain along time. Let the total loss $L$ be the sum of losses at all time steps: $L = \sum_{t=1}^{T} L_t$. Its gradient with respect to the parameter $W_{hh}$ is:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

According to formula {{rnn_hs}}, $W_{hh}$ participates in the computation of $h_k$ at every time step $k = 1, 2, ..., T$, producing a direct gradient $\frac{\partial h_k}{\partial W_{hh}}$ with respect to $h_k$, while also influencing subsequent time steps through formula {{rnn_hs}}, forming an indirect influence chain $h_k \rightarrow h_{k+1} \rightarrow ... \rightarrow h_T \rightarrow L_T$. Therefore, the total gradient of $L_T$ with respect to $W_{hh}$ needs to accumulate the influence of $W_{hh}$ at each time step:

$$\frac{\partial L_T}{\partial W_{hh}} = \sum_{k=1}^{T} \frac{\partial L_T}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

According to the multivariate chain rule, the gradient of $L_T$ with respect to $W_{hh}$ equals the sum of gradients along all influence paths:

$$[bptt-eq]\frac{\partial L_T}{\partial W_{hh}} = \sum_{k=1}^{T} \frac{\partial L_T}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

The overall formula can be understood as: total gradient = sum of contributions from each step along the influence chain. In the formula, $\frac{\partial L_T}{\partial h_T}$ is the gradient of the loss with respect to the final hidden state, indicating how much adjusting the final state helps reduce the loss; $\frac{\partial h_T}{\partial h_k}$ is the gradient of the hidden state at time step $T$ with respect to the hidden state at time step $k$, indicating how much the early state influences the final state — this is the key term for understanding the vanishing gradient problem discussed shortly; $\frac{\partial h_k}{\partial W_{hh}}$ is the gradient of the hidden state with respect to the parameter, indicating how much adjusting the parameter helps change the state.

The vanishing gradient problem is the primary challenge in training standard RNNs. Using the scenario from Jeffrey Elman's paper as an example: the network needs to learn the association between two words far apart in a sentence, such as "The **cat**, which already ate a fish, ... , **was hungry**". During backpropagation, the loss signal needs to travel from "was hungry" back to "cat", spanning multiple time steps. Even though humans are far better at handling remote dependencies than RNNs, excessively long dependency relationships can still burden reading comprehension. Language model cross-clause anaphora, long-range event influences in time series prediction, and multi-turn context references in dialogue systems — all tasks requiring long-term dependencies face the same obstacle. Below, we analyze the essential cause of this phenomenon from a mathematical perspective. Expanding the key term $\frac{\partial h_T}{\partial h_k}$ from the BPTT gradient propagation formula {{bptt-eq}} yields a chain of products:

$$\frac{\partial h_T}{\partial h_k} = \frac{\partial h_T}{\partial h_{T-1}} \cdot \frac{\partial h_{T-1}}{\partial h_{T-2}} \cdot ... \cdot \frac{\partial h_{k+1}}{\partial h_k}$$

Each derivative computation in this expression involves the derivative of the activation function. If the activation function is [tanh](../../deep-learning/neural-network-structure/activation-loss-functions.md#activation-functions), its derivative is $\tanh'(x) = 1 - \tanh^2(x) \in (0, 1]$, with a maximum value of 1 (when the input is 0). When the input is too large or too small, the derivative approaches 0. This means that each time a gradient passes through a tanh function, it is reduced. As the number of multiplied terms increases, the expression also approaches 0. This implies that the gradient contribution from early time steps (small $k$) to later time steps (large $T$) approaches 0, and the network cannot effectively learn long-term dependencies. In the example, by the time the gradient propagates from "was hungry" back to "cat", it has almost completely vanished, and the network cannot update the parameters associated with "cat", making it impossible to learn this long-term dependency. This is the fundamental reason why standard RNNs perform poorly on long-sequence tasks.

The vanishing gradient problem imposes severe limitations on the practical application of RNNs. When a language model processes "I was born in Beijing, ... (50 words) ... , so my hometown is?", it needs to remember "Beijing" from 50 words earlier. In stock prediction, if a company released a financial report 30 days ago and the stock price suddenly changes today, the model needs to associate information from 30 days ago. In dialogue systems, an entity mentioned by the user 5 minutes ago needs to be referenced now, requiring memory across multiple turns of conversation. Standard RNNs perform poorly in these scenarios because gradients can hardly propagate dependencies beyond about 10 time steps.

The gradient problem spurred subsequent architectural improvements. In 1997, German computer scientists Sepp Hochreiter and Jurgen Schmidhuber proposed **LSTM** (Long Short-Term Memory), which introduced gating mechanisms to selectively retain long-term information. In 2014, Korean scholar Kyunghyun Cho proposed **GRU** (Gated Recurrent Unit), which simplified the gating design of LSTM for higher computational efficiency. These improved architectures will be detailed in the next article. Of course, the most revolutionary change was the complete overhaul of the RNN architecture. In 2015, the introduction of the attention mechanism provided a completely different approach, directly accessing information from any time step and bypassing the limitations of gradient propagation — this is the subject of the subsequent section on large language models.

## RNN Sequence Prediction Practice

Now let us verify the sequence modeling capability of RNNs through an experiment. The task is to predict the next value of a sine wave sequence. The input sequence is $[\sin(0), \sin(0.1), \sin(0.2), ..., \sin(1.9)]$, and the target is to predict $\sin(2.0)$. This task has two characteristics: first, the sequence has clear temporal dependencies, where the value of $\sin(t)$ depends on historical values $\sin(t-1), \sin(t-2)$, etc.; second, historical information aids prediction — by observing upward or downward trends, the next direction can be inferred.

The experimental results show that the training process converges stably within 50 epochs, with the loss function continuously decreasing, indicating that the RNN can effectively learn the sequential patterns of the sine wave. The predicted values on the test set are close to the true values, validating the model's generalization capability. The RNN successfully uses information from the previous 20 points to predict the 21st point, demonstrating its sequence modeling ability.

If an MLP were used for the same task (concatenating the sequence into a vector input), the prediction error would typically be larger. This is because the MLP cannot effectively capture temporal dependencies — it treats the 20 points as independent features rather than a sequence with temporal order. By passing information through hidden states, the RNN can understand the rising or falling trend of the sine wave and make more accurate predictions.

```python runnable
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate sine wave sequence data
def generate_sine_data(num_samples, seq_len):
    """Generate sine wave sequence data"""
    X = []
    Y = []
    
    for i in range(num_samples):
        # Random starting phase
        start = np.random.rand() * 2 * np.pi
        
        # Generate sequence
        t = np.linspace(start, start + seq_len * 0.1, seq_len + 1)
        sine_wave = np.sin(t)
        
        # Input sequence (first seq_len points)
        X.append(sine_wave[:-1])
        # Target (last point)
        Y.append(sine_wave[-1])
    
    return np.array(X), np.array(Y)

# Define RNN model
class SinRNN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, 
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, hn = self.rnn(x)
        # Take the last time step
        out = self.fc(out[:, -1, :])
        return out

# Generate data
num_samples = 1000
seq_len = 20
X, Y = generate_sine_data(num_samples, seq_len)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # (N, seq_len, 1)
Y_tensor = torch.FloatTensor(Y).unsqueeze(-1)  # (N, 1)

# Split into training and test sets
train_size = 800
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
Y_train, Y_test = Y_tensor[:train_size], Y_tensor[train_size:]

# Create model and optimizer
model = SinRNN(hidden_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training
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

# Testing
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, Y_test)
    print(f"\nTest set loss: {test_loss.item():.4f}")

# Visualize prediction results
plt.figure(figsize=(12, 5))

# Subplot 1: Training loss curve
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)

# Subplot 2: Prediction comparison
plt.subplot(1, 2, 2)
# Display 5 test samples
for i in range(5):
    plt.plot(range(seq_len), X_test[i].numpy().flatten(), 
             'b-', alpha=0.5, label='Input Sequence' if i==0 else '')
    plt.scatter(seq_len, Y_test[i].numpy().flatten(), 
                c='green', marker='o', s=50, label='True Value' if i==0 else '')
    plt.scatter(seq_len, test_pred[i].numpy().flatten(), 
                c='red', marker='x', s=50, label='Predicted Value' if i==0 else '')

plt.xlabel('Time Step')
plt.ylabel('sin(t)')
plt.title('Sine Wave Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Chapter Summary

This article introduced the basic principles of Recurrent Neural Networks (RNNs). RNNs use recurrent connections to transmit information along the temporal dimension, achieving sequence modeling capability. The hidden state $h_t$ is a compressed representation of historical information that is continuously updated over time. The BPTT algorithm unrolls backpropagation along the temporal dimension, enabling RNN training, but the vanishing gradient problem makes it difficult for information from early time steps to propagate to later time steps, limiting the applicability of standard RNNs. This spurred subsequent architectural improvements such as LSTM and GRU, which will be the subject of the next article.

## Exercises

1. Explain why RNNs typically choose tanh as the activation function for the hidden state rather than ReLU or Sigmoid. Analyze from the perspectives of both gradient propagation and output range.
    <details>
    <summary>Reference Answer</summary>

    **Output Range Perspective**:

    tanh has an output range of $(-1, 1)$, centered at zero. In contrast, Sigmoid has an output range of $(0, 1)$, always positive. The RNN hidden state $h_t$ serves as input for the next time step in the computation $W_{hh} h_t$. If $h_t$ is always positive (as with Sigmoid), the components of $W_{hh} h_t$ only accumulate and never diminish, causing the hidden state to grow increasingly large and positive, leading to numerical instability. tanh's zero-centered output allows alternating positive and negative values, enabling the hidden state to dynamically adjust between positive and negative, providing better numerical stability.

    ReLU's output range is $[0, +\infty)$, which also has the issue of always being non-negative, with no upper bound. In recurrent connections, this more readily leads to explosive growth of the hidden state (although some studies have proposed RNN variants using ReLU, they require special initialization and normalization techniques to maintain stability).

    **Gradient Propagation Perspective**:

    tanh has a derivative close to 1 near zero ($\tanh'(0) = 1$), meaning that when the hidden state values are small, the gradient is hardly attenuated, which benefits short-range information propagation. Sigmoid's derivative at zero is only 0.25 ($\sigma'(0) = 0.25$), so the gradient is reduced by a factor of 4 at each step, exacerbating the vanishing gradient problem.

    However, tanh also has limitations: when the input is large, $\tanh'(x) \to 0$, and the derivative approaches zero. In long sequences, the repeated multiplication of tanh derivatives still leads to vanishing gradients. This is the fundamental reason why standard RNNs cannot learn long-term dependencies and the motivation for introducing gating mechanisms in LSTM/GRU.
    </details>
