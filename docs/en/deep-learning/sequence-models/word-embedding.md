# Word Embedding and Representation Learning

The previous chapters introduced the basic structure and working principles of neural networks, where the inputs these models process are numerical vectors. The sequence models to be introduced later are closely related to Natural Language Processing (NLP), but the first problem NLP faces is: humans use symbols (text) rather than numerical values that neural networks excel at processing. How can discrete symbols be converted into continuous numerical values, enabling neural networks to understand and process natural language? The answer to this question is word embedding.

In 2003, Yoshua Bengio proposed a neural probabilistic language model in the paper "[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)", which can be considered the origin of modern NLP techniques. This method introduced distributed representations of words (later known as the concept of word embedding) for the first time. Word embedding is a technique that maps discrete symbols into a continuous vector space, and the resulting vectors are called **word vectors**. It not only solves the problem of converting symbols into numerical values but, more importantly, this mapping can capture semantic relationships between words. Semantically similar words are closer in the vector space, while semantically unrelated words are farther apart. This geometric property allows neural networks to "understand" the meaning of language, rather than merely memorizing permutations and combinations of symbols. This article starts from the most intuitive One-Hot encoding, analyzes its limitations, introduces the core idea of word embedding, and demonstrates the geometric properties and practical applications of word embedding through experiments.

## One-Hot Encoding and Bag of Words

The earliest and most intuitive approach to converting text into numerical values is **One-Hot encoding**. Assume the vocabulary has $V$ words, and each word is represented as a $V$-dimensional vector, where the position corresponding to that word is 1 and all other positions are 0. For example, suppose the vocabulary is `["spring", "summer", "autumn", "winter"]`, then the One-Hot encoding of each word is:

| Word | One-Hot Vector |
|:----:|:---------------|
| spring | $[1, 0, 0, 0]$ |
| summer | $[0, 1, 0, 0]$ |
| autumn | $[0, 0, 1, 0]$ |
| winter | $[0, 0, 0, 1]$ |

The **Bag of Words** model was proposed by Zellig Harris in 1954. It is an extension of One-Hot encoding. The Bag of Words model applies One-Hot encoding to each word in the vocabulary and then sums (or counts) these encoded vectors. Specifically, it ignores the order and grammatical structure of words in a document, treating the document as a "bag" of words, focusing only on which words appear and how many times they appear. It first builds a vocabulary containing all text words, then represents a piece of text as a fixed-length vector, where each dimension corresponds to a word in the vocabulary and the numerical value represents the frequency of that word in the text. Although understanding text topics solely through word frequency is certainly not accurate enough, this method of converting documents into numerical matrices is simple enough. By counting word frequencies to capture the semantic content of a document, it enables computers to "understand" the topic and keyword distribution of the text.

One-Hot encoding and the Bag of Words model are simple, intuitive, and easy to implement, but they have three serious drawbacks that make them difficult to meet the needs of practical NLP tasks.

- First, **dimensionality explosion**. In practice, the vocabulary size $V$ can easily reach tens or even hundreds of thousands. Chinese commonly uses about 3,000-5,000 characters, but the number of commonly used words can reach hundreds of thousands; English vocabulary is even more vast. Each word requires a $V$-dimensional vector for representation. When $V = 50,000$, a One-Hot vector for a single word requires 50,000 numerical values. This not only consumes a large amount of storage space but also results in an extremely high input dimensionality for neural networks, causing the number of parameters in the weight matrix to explode.

- Second, **sparsity**. A One-Hot vector has only one position set to 1, with the remaining $V-1$ positions all set to 0. This extremely sparse representation leads to low computational efficiency. Suppose a neural network receives a One-Hot vector as input, and the weight matrix from the input layer to the first hidden layer has dimensions $V \times d$, where $d$ is the hidden layer dimension. During forward propagation, the input vector is multiplied by the weight matrix $\mathbf{h} = \mathbf{W} \cdot \mathbf{x}$. Since $\mathbf{x}$ has only one non-zero position, only one row of the weight matrix actually needs to be extracted — the multiplication operations for the other $V-1$ rows are entirely wasted.

- Third, **inability to express semantic relationships**. This is the most fundamental flaw of One-Hot encoding and the Bag of Words model. Under One-Hot representation, the Euclidean distance between any two different words is $\sqrt{2}$, and the cosine similarity is 0. That is, the distance between "spring" and "summer" equals the distance between "spring" and "winter", and the similarity between "cat" and "dog" equals the similarity between "cat" and "car". This representation completely ignores semantic relationships between words. Neural networks cannot learn any semantic information from such a representation; they can only learn from contextual positional information, unable to utilize the semantic features of the words themselves.

## Word Embedding

One-Hot encoding represents each word as a $V$-dimensional sparse vector, while word embedding represents each word as a $d$-dimensional dense vector, where $d \ll V$. For example, with a vocabulary size $V = 50,000$ and embedding dimension $d = 300$, each word is compressed from a 50,000-dimensional sparse vector to a 300-dimensional dense vector. This compression is not merely dimensionality reduction; it is about learning a meaningful representation that brings semantically similar words closer in the vector space.

The idea of word embedding can be expressed using an embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$. Each row of the matrix corresponds to a word in the vocabulary and is a $d$-dimensional vector. Given the index $i$ of a word, its embedding vector is the $i$-th row of the embedding matrix: $\mathbf{e}_i = \mathbf{E}[i, :]$. From a mathematical perspective, word embedding can be understood as the matrix multiplication of a One-Hot vector and the embedding matrix. Let the One-Hot vector of word $w$ be $\mathbf{x} \in \mathbb{R}^V$ (only the $i$-th position is 1), then the embedding vector is:

$$\mathbf{e} = \mathbf{E}^T \mathbf{x} = \mathbf{E}[i, :]^T$$

Since $\mathbf{x}$ has only one non-zero position, the matrix multiplication essentially extracts the $i$-th row of the embedding matrix. Therefore, the embedding layer is implemented using direct index lookup, requiring no matrix multiplication — more efficient, with the same result. The key advantage of word embedding is that it is learnable. The embedding matrix $\mathbf{E}$ is a parameter of the neural network and is optimized along with the model through backpropagation. During training, the model automatically learns meaningful word vector representations. Words that frequently appear in similar contexts will have their embedding vectors gradually move closer, while semantically unrelated words will have their embedding vectors move farther apart. This learning mechanism enables word embedding to capture statistical patterns and semantic information of language.

### Geometric Properties of Word Embedding

The most fascinating property of word embedding is its geometric nature. In a well-trained word embedding space, semantically similar words are close together, while semantically unrelated words are far apart. Even more remarkably, the directions between word vectors can represent semantic relationships. The most famous example is:

$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$

::: info Note
The code example below uses manually constructed 3-dimensional vectors for demonstration purposes, aiming to intuitively show the principles of word vector arithmetic. In practice, word vectors typically have 100-300 dimensions and need to be trained on large-scale corpora using methods such as Word2Vec or GloVe.
:::

The meaning of this equation is: subtracting the "man" vector from the "king" vector and adding the "woman" vector yields a result close to the "queen" vector. This indicates that word embedding captures the semantic dimension of gender: the direction of $\vec{king} - \vec{queen}$ is similar to the direction of $\vec{man} - \vec{woman}$, both representing the semantic change from male to female. The standard method for measuring the similarity between two word vectors is [cosine similarity](../../maths/linear/vectors.md#dot-product-and-projection):

$$similarity(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \sqrt{\sum_{i=1}^{d} b_i^2}}$$

Cosine similarity measures the directional similarity of two vectors, with a range of $[-1, 1]$. A value of 1 indicates identical direction, 0 indicates orthogonality (unrelated), and -1 indicates opposite direction. Compared to Euclidean distance, cosine similarity focuses more on the direction of vectors rather than their magnitude, making it more suitable for measuring semantic similarity.

```python runnable
import numpy as np

# Simulate trained word embeddings (simplified example)
# In practice, these are obtained through training on large amounts of data
word_vectors = {
    "king": np.array([0.8, 0.2, 0.9]),
    "queen": np.array([0.7, 0.8, 0.85]),
    "man": np.array([0.9, 0.1, 0.3]),
    "woman": np.array([0.8, 0.7, 0.25]),
    "prince": np.array([0.75, 0.15, 0.7]),
    "princess": np.array([0.65, 0.75, 0.65]),
}

def cosine_similarity(v1, v2):
    """Compute cosine similarity"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Verify word vector arithmetic: king - man + woman \approx queen
result = word_vectors["king"] - word_vectors["man"] + word_vectors["woman"]
print("Word vector arithmetic: king - man + woman")
print(f"Result vector: {result.round(3)}")
print(f"Queen vector: {word_vectors['queen'].round(3)}")
print(f"Cosine similarity with queen: {cosine_similarity(result, word_vectors['queen']):.4f}")

# Compute similarity between result and each word
print("\nCosine similarity with each word:")
for word, vec in word_vectors.items():
    sim = cosine_similarity(result, vec)
    print(f"  {word}: {sim:.4f}")

# Verify semantic similarity
print("\nCosine similarity of semantically similar words:")
print(f"  king vs queen: {cosine_similarity(word_vectors['king'], word_vectors['queen']):.4f}")
print(f"  prince vs princess: {cosine_similarity(word_vectors['prince'], word_vectors['princess']):.4f}")
print(f"  man vs woman: {cosine_similarity(word_vectors['man'], word_vectors['woman']):.4f}")
```

This code demonstrates the geometric properties of word embedding using simplified word vectors. In practice, word vectors typically have hundreds of dimensions and need to be trained on large-scale corpora. GloVe and Word2Vec are two of the most famous pretrained word vector methods. The word vectors they produce through training on large-scale text exhibit rich semantic relationships.

### Word Embedding in Practice

Having understood the principles of word embedding, let us practice using the embedding layer with PyTorch. `nn.Embedding` encapsulates the storage and lookup operations of the embedding matrix, with two key parameters: `num_embeddings` represents the vocabulary size $V$ (the number of rows in the embedding matrix), and `embedding_dim` represents the embedding dimension $d$ (the number of columns in the embedding matrix).

The input to the embedding layer is word indices (integer tensor), and the output is the corresponding embedding vectors. The input can be a tensor of any shape, and the output shape gains an additional dimension (the embedding dimension) at the end. Sequence models such as LSTM and GRU, which will be introduced later, typically use an embedding layer as their first layer.

```python runnable
import torch
import torch.nn as nn

# Create embedding layer
vocab_size = 1000  # Vocabulary size
embedding_dim = 64  # Embedding dimension

embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

print(f"Embedding matrix shape: {embedding_layer.weight.shape}")
print(f"Number of parameters: {vocab_size * embedding_dim:,}")

# Embedding of a single word
word_idx = torch.tensor([42])
embedding = embedding_layer(word_idx)
print(f"\nInput shape: {word_idx.shape}")
print(f"Output shape: {embedding.shape}")

# Embedding of a batch of words
batch_indices = torch.tensor([[1, 42, 100], [200, 300, 999]])
batch_embeddings = embedding_layer(batch_indices)
print(f"\nBatch input shape: {batch_indices.shape}")
print(f"Batch output shape: {batch_embeddings.shape}")

# Example of joint training with downstream task
class TextClassifier(nn.Module):
    """Simple text classification model"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # Simple average pooling
        pooled = embedded.mean(dim=1)  # (batch_size, embedding_dim)
        hidden = self.relu(self.fc(pooled))
        return self.output(hidden)

# Create model
model = TextClassifier(vocab_size=1000, embedding_dim=64, hidden_dim=32, num_classes=3)
print(f"\nModel structure:\n{model}")

# Forward pass test
sample_input = torch.randint(0, 1000, (4, 10))  # batch_size=4, seq_len=10
output = model(sample_input)
print(f"\nInput shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

The code above demonstrates the basic usage of `nn.Embedding` and how to train it jointly with downstream tasks. The embedding layer serves as the first layer of the model, converting word indices into dense vectors, and the subsequent layers perform computations based on these vectors. During training, the parameters of the embedding matrix are optimized together with the entire model, learning word vector representations suitable for the current task.

### Pretrained Word Embeddings

Although the embedding layer can be trained jointly with downstream tasks, when the amount of data is small, a randomly initialized embedding matrix struggles to learn high-quality word vector representations. **Pretrained word embeddings** are an effective solution to this problem. Pretrained word embeddings are trained on large-scale corpora (such as Wikipedia, Common Crawl), containing rich semantic information, and can be directly loaded into the embedding layer. The two most famous pretrained word embedding methods are Word2Vec and GloVe:

- **Word2Vec**: Proposed by Google in 2013, the main idea is that words appearing in similar contexts have similar meanings. By predicting context words or the center word, the model learns word vectors capable of capturing semantic information. Word2Vec includes two training approaches:
    - Skip-Gram: Given a center word, predict the context words. The training objective is to maximize the predicted probability of context words.
    - CBOW (Continuous Bag of Words): Given context words, predict the center word. The training objective is to minimize the prediction error of the center word.

- **GloVe** (Global Vectors for Word Representation): Proposed by Stanford University in 2014, based on global word co-occurrence statistics. GloVe constructs a word co-occurrence matrix, counting how often any two words appear together within a window, and then learns word vectors through matrix factorization. GloVe combines global statistical information with local context information and performs better than Word2Vec on many tasks.

There are also two ways to use pretrained word embeddings: one is to freeze the embedding layer — after loading pretrained word vectors, freeze the embedding layer parameters so they do not participate in training, suitable for scenarios where the downstream task has a small amount of data and the pretrained word vectors are of high quality; the other is to fine-tune the embedding layer — load pretrained word vectors as initial values and allow the embedding layer parameters to update during training, suitable for scenarios where the downstream task has sufficient data and needs to learn task-specific semantics.

## Summary

Word embedding is a foundational technique for neural networks to process natural language, solving the core problem of converting discrete symbols into continuous numerical values. From One-Hot encoding to word embedding, it is not merely dimensionality compression but a qualitative change from symbolic representation to semantic representation. Word embedding lays the foundation for subsequent sequence models. Models such as LSTM and Transformer all use word embedding as input representation, learning contextual dependencies of sequences on top of word embedding. Understanding word embedding is the starting point for understanding modern natural language processing techniques.

## Exercises

1. Assume a vocabulary size of 10000 and an embedding dimension of 300. Calculate the number of parameters required to store all word vectors under One-Hot encoding and word embedding representations. How much memory is needed if using FP32 (4 bytes) storage?
   <details>
   <summary>Answer</summary>

   **One-Hot encoding**:
   - Dimension per word vector: 10000
   - Total parameters: $10000 \times 10000 = 100,000,000$ (100 million)
   - Memory usage: $100,000,000 \times 4 = 400,000,000$ bytes ≈ 400 MB

   **Word embedding**:
   - Dimension per word vector: 300
   - Total parameters: $10000 \times 300 = 3,000,000$ (3 million)
   - Memory usage: $3,000,000 \times 4 = 12,000,000$ bytes ≈ 12 MB

   Word embedding uses only 3% of the parameters of One-Hot encoding, with memory usage dropping from 400 MB to 12 MB.
   </details>

2. Given two word vectors $\mathbf{a} = [0.8, 0.6]$ and $\mathbf{b} = [0.6, 0.8]$, compute their cosine similarity and Euclidean distance. If $\mathbf{a}$ represents "cat" and $\mathbf{b}$ represents "dog", do these vectors indicate that the words are semantically similar or different?
   <details>
   <summary>Answer</summary>

   **Cosine similarity**:
   $$\cos(\mathbf{a}, \mathbf{b}) = \frac{0.8 \times 0.6 + 0.6 \times 0.8}{\sqrt{0.8^2 + 0.6^2} \times \sqrt{0.6^2 + 0.8^2}} = \frac{0.48 + 0.48}{1.0 \times 1.0} = 0.96$$

   **Euclidean distance**:
   $$d(\mathbf{a}, \mathbf{b}) = \sqrt{(0.8-0.6)^2 + (0.6-0.8)^2} = \sqrt{0.04 + 0.04} = \sqrt{0.08} \approx 0.28$$

   A cosine similarity of 0.96, close to 1, indicates that the two vectors have almost identical directions and are highly semantically similar. This aligns with the fact that "cat" and "dog" are both pets and animals.
   </details>
