# Language Models and Tokenization

Text is a continuous stream of characters, but language models process discrete sequences of tokens. The process of splitting continuous text into discrete tokens is called **tokenization**, which serves as the bridge between language models and raw text. The quality of tokenization directly affects model capability -- too coarse results in a large vocabulary and inability to handle new words, while too fine leads to overly long sequences and fragmented semantics.

Research on tokenization dates back to the 1990s. In 1994, American software engineer Philip Gage first proposed the BPE (Byte Pair Encoding) algorithm in his article "[A New Algorithm for Data Compression](https://dl.acm.org/doi/10.5555/177910.177914)", ushering in an era of subword tokenization over word-level tokenization. Prior to this, word-level tokenization in machine translation systems suffered from a severe Out-of-Vocabulary (OOV) problem, where words unseen during training could not be processed correctly. Applying BPE to tokenization changed this landscape, allowing models to represent arbitrary text by combining a limited set of subwords, and it became the foundation of modern LLM tokenization.

Before introducing tokenization algorithms in this chapter and actually training a large language model in the next chapter, we first need to understand what a language model is and what problem it aims to solve. We will trace the historical evolution of language models from statistical methods to neural approaches, then dive into the tokenization algorithms used by modern LLMs, and finally explore the trade-offs in vocabulary design.

## A Brief History of Language Models

A **language model** defines a [probability distribution](../../maths/probability/probability-basics.md#common-probability-distributions) over sequences of natural language. Given a text sequence $w_1, w_2, ..., w_T$, the language model assigns a probability $P(w_1, w_2, ..., w_T)$ to that sequence, measuring how likely it is to occur in natural language. Through the chain rule, this [joint probability](../../maths/probability/probability-basics.md#conditional-probability-and-joint-probability) can be decomposed into a product of step-by-step [conditional probabilities](../../maths/probability/probability-basics.md#conditional-probability-and-joint-probability):

$$P(w_1, ..., w_T) = \prod_{t=1}^{T} P(w_t | w_1, ..., w_{t-1})$$

Therefore, modeling $P(w_{t+1} | w_1, w_2, ..., w_t)$ -- predicting the next word given the preceding context -- has become the most common training objective for language models. Autoregressive models such as GPT, Qwen, DeepSeek, and Claude are all trained in this manner. However, this is not the only way to define a language model. Masked language models like BERT model $P(w_t | w_{-t})$ by predicting masked words, while in classic applications such as speech recognition, language models are used to score candidate sequences rather than predict word by word. Regardless of the approach, language model design requires a deep understanding of language itself. To accurately estimate sequence probabilities, a model must master grammar, semantics, common sense, world knowledge, and even reasoning ability.

### N-Gram Language Models

Before the rise of deep learning, **N-Gram language models** were the dominant approach to language modeling, with their foundation laid by Claude Shannon in his 1948 work on information theory. The core idea is the Markov assumption: the next word depends only on the preceding $n-1$ words, not on the entire history. Let $w_1, ..., w_{t-1}$ be all the words before time $t$ (the full history), and $w_{t-n+1}, ..., w_{t-1}$ be the preceding $n-1$ words (the recent history). The next word $w_t$ satisfies the conditional probability:

$$P(w_t | w_1, ..., w_{t-1}) \approx P(w_t | w_{t-n+1}, ..., w_{t-1})$$

That is, the probability of the next word $w_t$ given the full history ($w_1$ to $w_{t-1}$) is approximately equal to the probability given only the preceding $n-1$ words ($w_{t-n+1}$ to $w_{t-1}$). Taking Bigram as an example ($n=2$), the next word depends only on the previous word, and by the definition of conditional probability:

$$P(w_t | w_{t-1}) = \frac{Count(w_{t-1}, w_t)}{Count(w_{t-1})}$$

Thus, directly counting the occurrences of the pair $(w_{t-1}, w_t)$ in the training corpus and dividing by the count of $w_{t-1}$ yields the probability of the next word $w_t$. The following code implements a simple Bigram language model to demonstrate the basic workings of N-Gram models. As shown by the results, the model can provide reasonable probability estimates for word pairs that appear in the training corpus, but for unseen pairs (such as "love Guangzhou"), the model outputs a probability of zero for that pair.

```python runnable
# N-Gram Language Model Demonstration
from collections import defaultdict

class BigramModel:
    """Simple Bigram Language Model"""
    
    def __init__(self):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.vocab = set()
    
    def train(self, sentences):
        """Train the model from a list of sentences"""
        for sentence in sentences:
            tokens = ['<s>'] + sentence.split() + ['</s>']
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                self.vocab.add(w1)
                self.vocab.add(w2)
    
    def probability(self, w1, w2):
        """Compute P(w2 | w1)"""
        if self.unigram_counts[w1] == 0:
            return 0
        return self.bigram_counts[w1][w2] / self.unigram_counts[w1]
    
    def sentence_probability(self, sentence):
        """Compute the probability of a sentence"""
        tokens = ['<s>'] + sentence.split() + ['</s>']
        prob = 1.0
        for i in range(len(tokens) - 1):
            p = self.probability(tokens[i], tokens[i + 1])
            if p == 0:
                return 0  # Unseen combination encountered
            prob *= p
        return prob

# Training corpus
corpus = [
    "I love Beijing",
    "I love Shanghai",
    "Beijing is capital",
    "Shanghai is city",
    "I love China"
]

model = BigramModel()
model.train(corpus)

# Testing
print("Vocabulary:", model.vocab)
print("\nConditional Probability P(love|I):", model.probability("I", "love"))
print("Conditional Probability P(Beijing|love):", model.probability("love", "Beijing"))
print("Conditional Probability P(Shanghai|love):", model.probability("love", "Shanghai"))

# Compute sentence probability
test_sentence = "I love Beijing"
print(f"\nProbability of sentence '{test_sentence}':", model.sentence_probability(test_sentence))

test_sentence2 = "I love Guangzhou"
print(f"Probability of sentence '{test_sentence2}':", model.sentence_probability(test_sentence2))
```

The N-Gram model is simple and intuitive, but it suffers from two fundamental flaws: the sparsity problem and the inability to capture long-range dependencies. These flaws ultimately led to the birth of neural language models.

- **Sparsity problem**: This is exactly the problem exposed by the code above. No matter how large the training corpus, it cannot cover all possible word combinations. When encountering combinations that never appeared during training, the model outputs zero probability, making the entire sentence's probability zero. Taking Bigram as an example, suppose the vocabulary size is $|V|$; the number of possible word pairs is $|V|^2$. Even with a vocabulary of only 10,000 words, the number of word pairs reaches 100 million.

- **Inability to capture long-range dependencies**: The N-Gram model only looks at the preceding $n-1$ words and cannot capture dependencies further away. Consider this example: "I was born in Beijing, so ___ is very familiar to me." To predict the blank (e.g., "Tiananmen"), the model needs to understand the relationship between "Beijing" at the beginning of the sentence and the blank. However, in an N-Gram model, if $n=3$, the model can only see at most "so ___ is," completely unable to leverage the earlier information. Increasing $n$ can partially alleviate this problem but exacerbates the sparsity issue -- the larger $n$ is, the more possible N-Gram combinations exist, making it harder for the training corpus to cover them. In practice, N-Gram models typically use at most $n \leq 5$.

### Neural Language Models

In 2003, Canadian computer scientist Yoshua Bengio first proposed the **neural language model** in his paper "[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)", ushering in a new paradigm for language modeling. This work later earned him the 2018 Turing Award, and Bengio, along with Geoffrey Hinton and Yann LeCun, is collectively recognized as the "Big Three" of deep learning for their pioneering contributions.

Neural language models effectively address the sparsity problem by representing each word as a distributed representation rather than a discrete symbol. Word vectors capture semantic similarity between words, allowing the model to generalize to unseen but semantically similar word combinations. For instance, if the model has never seen "cats eat fish" during training but has seen "dogs eat meat," it can still infer that "cats eat fish" is plausible through the similarity of word vectors. Additionally, compared to the discrete probability distribution of N-Gram models, the distribution of neural language models is smooth. This is because the neural network output uses Softmax, assigning non-zero probability to all words, thus avoiding the problem of a single rare word combination causing the entire sentence's probability to become zero.

Regarding the inability to capture long-range dependencies, the original neural language model proposed by Bengio used a feedforward neural network with a fixed input window size. Although it performed slightly better than N-Gram models (which only support sequences of up to 5 words), it was still far from practical for production use. Later, architectures such as Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Sequence-to-Sequence (Seq2Seq) models further improved the capabilities of neural language models. It was not until the Transformer architecture emerged that the long-range dependency problem was finally resolved, marking the beginning of the modern large language model era.

### Autoregressive Language Models

From the earliest N-Gram models to today's Transformers, the thread of "predicting the next word based on preceding context" runs throughout. Language models trained with this next-word prediction objective have a formal name: **autoregressive language models**, also known as **causal language models (CLMs)**, which constitute the largest branch of neural language models. The formal definition of a CLM is: given a text sequence $\mathbf{w} = (w_1, w_2, ..., w_T)$, the language model decomposes it into a product of conditional probabilities:

$$P(\mathbf{w}) = \prod_{t=1}^{T} P(w_t | w_1, w_2, ..., w_{t-1})$$

This definition states that, from the CLM perspective, the probability of a sentence is the product of sequentially guessing each word from left to right, where each guess depends on all previously observed words. For example, for the sentence "Today the weather is nice," the model first computes the probability of "the" following "Today," then the probability of "weather" following "Today the," and so on. Multiplying these conditional probabilities yields the probability of the entire sentence. This is the origin of the term "autoregressive" -- each step's prediction depends on the outputs of all previous steps, generating text progressively. The training objective of a CLM is to maximize the likelihood of all sequences in the training corpus:

$$\mathcal{L} = \sum_{\mathbf{w} \in \mathcal{D}} \sum_{t=1}^{T} \log P(w_t | w_{<t})$$

This training objective means using large amounts of real text as reference answers, allowing the model to repeatedly practice the skill of guessing the next word given preceding context. During training, parameters are continuously adjusted to make the model increasingly accurate at predicting the next word at each position in real text. The logarithm in the objective function is an engineering technique that converts multiplication into addition, preventing numerical underflow from multiplying many small probabilities and also facilitating gradient-based optimization.

## Tokenization Algorithms

The input to a language model must consist of discrete tokens, but raw text is a continuous stream of characters that needs to be segmented into tokens from a finite vocabulary. The component that transforms a character stream into a sequence of discrete tokens that the model can process is called a **tokenizer**. The mainstream tokenization algorithms used by modern LLMs include BPE, WordPiece, and Unigram, among others. Choosing a different tokenizer largely determines several design decisions for the model:

- **Vocabulary size**: The larger the vocabulary, the higher the coverage, but also the more model parameters, since the model's output layer is a $|V| \times d$ matrix.
- **Sequence length**: The finer the tokenization, the longer the sequence produced from the same character stream, increasing the model's computational cost. For the attention mechanism, which has $O(n^2)$ complexity, sequence length imposes significant pressure.
- **Out-of-vocabulary handling**: No matter how large the vocabulary, there will always be words unseen during training. A good tokenization scheme should be able to handle this situation.

Tokenization can be categorized by granularity into three types: word-level, subword-level, and character-level. The most intuitive tokenization scheme is **word-level tokenization**, which splits text by words, with each word becoming a token in the vocabulary. For example, the raw text "I love Beijing Tiananmen" would be tokenized as `["I", "love", "Beijing", "Tiananmen"]`. Word-level tokenization seems intuitive and reasonable from a human perspective, but from the perspective of the three design decisions above, at least two of them are difficult to address:

- **Vocabulary size**: English has hundreds of thousands of words, and Chinese vocabulary is even more difficult to exhaust. A comprehensive vocabulary might require millions of entries, leading to an extremely large parameter count in the model's output layer.
- **Out-of-vocabulary problem**: No matter how large the vocabulary, new words will always appear, such as names, places, technical terms, and internet neologisms. Word-level tokenization has no way to handle these and can only replace them with `<UNK>`, losing all semantic information.

Therefore, modern LLMs choose the most complex of the three granularities: **subword-level tokenization**, a compromise between word-level and character-level tokenization. Subword tokenization keeps common words intact while splitting rare words into meaningful subword units. This is similar to how students memorize English words by roots and affixes. For instance, the word "unhappiness" can be split into `["un", "happiness"]` or `["un", "happy", "ness"]`, so that even if the model has never seen the word "unhappiness" before, it can understand its meaning through the combination of subwords.

Compared to word-level tokenization, subword tokenization offers significant advantages. With a fixed-size vocabulary, it achieves nearly unlimited coverage by combining a limited set of subwords to represent arbitrary text. Subwords typically carry semantic meaning (such as prefixes, suffixes, and roots), allowing the model to generalize to new words and thus solve the out-of-vocabulary problem. As will be discussed later in [Vocabulary Design Trade-offs](#vocabulary-design-trade-offs), the vocabulary sizes of mainstream LLMs typically range from 30,000 to 100,000, far smaller than what word-level tokenization would require.

### BPE

**Byte Pair Encoding (BPE)** was originally a data compression algorithm proposed by American software engineer Philip Gage in his 1994 article "[A New Algorithm for Data Compression](https://dl.acm.org/doi/10.5555/177910.177914)". Gage's original idea was to design an algorithm that finds the most frequent adjacent byte pair in text, replaces it with a byte not present in the original text, and repeats this process to progressively compress the text. In 2015, Philipp Sennrich of the University of Edinburgh adapted BPE into a subword tokenization algorithm in his paper "[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)", replacing the goal of text compression with building a subword vocabulary, enabling machine translation models to represent arbitrary text through combinations of a limited set of subwords -- thus solving the long-standing OOV problem in NLP.

The assumption of the BPE algorithm is that symbol combinations in language are not uniformly distributed. Certain character pairs frequently appear together, such as "th," "er," and "in" in English; certain substrings repeatedly form word roots and affixes, such as "ing," "tion," and "ment." If these high-frequency combinations are treated as units, the sequence length can be shortened while keeping the vocabulary at a moderate size. BPE exploits this non-uniform distribution, automatically discovering symbol pairs worth merging through a data-driven approach. The difference between compression and tokenization lies only in the objective: compression algorithms care about whether the text becomes shorter after replacement, while tokenization algorithms care about whether the merged vocabulary can cover new words. Both can share the same iterative merging mechanism, differing only in the evaluation criterion.

The BPE algorithm can be divided into two stages: training (learning merge rules from the corpus) and tokenization (applying the rules to new text). The training stage consists of the following steps:

1. **Initialize the vocabulary**: Split each word in the training corpus into a character sequence, and append a special end-of-word marker `</w>` to each word to distinguish between inner-word subwords and word-ending subwords. Also count the frequency of each word. Suppose "low" appears 5 times and "lower" appears 2 times in the corpus. After initialization:
   - `l o w </w>`: 5
   - `l o w e r </w>`: 2

   The vocabulary at this point is the set of all characters that have appeared: `{l, o, w, e, r, </w>}`.

2. **Count adjacent symbol pair frequencies**: Traverse each word in the frequency table and count the occurrences of all adjacent symbol pairs, weighted by word frequency. For the example above, the pair `(l, o)` appears 7 times, `(o, w)` appears 7 times, `(w, </w>)` appears 5 times, `(w, e)` appears 2 times, `(e, r)` appears 2 times, `(r, </w>)` appears 2 times.

3. **Merge the most frequent symbol pair**: Find the most frequent adjacent pair, merge it into a new symbol, add it to the vocabulary, and record this merge rule. In the example above, `(l, o)` and `(o, w)` are tied for the highest frequency. Assuming `(l, o)` is merged first, the vocabulary gains `lo`, and the word frequency table is updated to:
   - `lo w </w>`: 5
   - `lo w e r </w>`: 2

4. **Repeat iteratively**: Recompute adjacent symbol pair frequencies and continue merging until the vocabulary reaches the target size or the predetermined number of merges. Each merge produces a new rule, and rules are ordered by merge sequence, forming an ordered list.

In the tokenization stage, merge rules are applied to the input text strictly in the order learned during training. The order of application is important: rules learned earlier correspond to higher-frequency patterns and must be applied first to ensure tokenization results are consistent with training. If the training process first merged `(l, o)` and then `(lo, w)`, then when tokenizing "lower," the result would first become `["lo", "w", "e", "r"]` and then `["low", "e", "r"]`. The following code implements a simplified BPE algorithm to demonstrate the complete flow of both training and tokenization stages.

```python runnable
# BPE Algorithm Demonstration
from collections import defaultdict

class SimpleBPE:
    """Simplified BPE Algorithm Demonstration"""
    
    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.merges = []  # Merge rules
    
    def train(self, corpus):
        """Train BPE from a corpus"""
        # Count word frequencies
        word_freqs = defaultdict(int)
        for word in corpus.split():
            word_freqs[' '.join(list(word))] += 1
        
        print("Initial word frequencies:")
        for word, freq in sorted(word_freqs.items(), key=lambda x: -x[1])[:10]:
            print(f"  {word}: {freq}")
        
        # Iterative merging
        for i in range(self.num_merges):
            # Count adjacent symbol pair frequencies
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += freq
            
            if not pairs:
                break
            
            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            print(f"\nIteration {i+1}: Merge {best_pair} (frequency: {pairs[best_pair]})")
            
            # Merge
            new_symbol = ''.join(best_pair)
            self.merges.append(best_pair)
            
            # Update word frequency table
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(' '.join(best_pair), new_symbol)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
        
        print("\nFinal vocabulary:")
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word.split())
        print(f"  {sorted(vocab)}")
    
    def tokenize(self, word):
        """Tokenize a single word"""
        symbols = list(word)
        for pair in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    symbols = symbols[:i] + [''.join(pair)] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

# Training corpus (simplified example)
corpus = "low low low low low lower lower newest newest newest newest newest newest wider wider wider new new"

print("=== BPE Training Process ===\n")
bpe = SimpleBPE(num_merges=10)
bpe.train(corpus)

print("\n=== Tokenization Test ===")
test_words = ["low", "lower", "newest", "wider", "newer"]
for word in test_words:
    tokens = bpe.tokenize(word)
    print(f"'{word}' -> {tokens}")
```

The advantage of BPE is that the algorithm itself is logically clear -- it only requires frequency counting and iterative merging, making it very easy to implement. It is also entirely data-driven, relying on no language-specific rules or dictionaries, so the same algorithm can be applied to any language. When handling unseen words, BPE is naturally immune to OOV problems -- any new word can fall back to character-level splitting, so there is no situation where encoding fails. Additionally, vocabulary size can be flexibly controlled by adjusting the number of merges, striking a balance between sequence length and model parameters.

However, BPE also has its limitations. BPE merging follows a greedy strategy, considering only the current highest-frequency symbol pair at each step, without regard to the impact of merging on subsequent choices. A high-frequency but semantically insignificant merge may crowd out more valuable merge opportunities. For instance, in English corpora, the pair `(t, h)` has an extremely high frequency, and BPE will prioritize merging it, even though "th" is not an independent semantic unit. Meanwhile, some subwords with slightly lower frequency but richer semantic meaning (such as "ment" or "tion") may be delayed in merging.

Another problem is that the tokenization result is deterministic and unique. Given a trained set of merge rules, each word has only one tokenization, and the model always sees the same subword segmentation during training, unable to learn the semantic associations that different segmentations of the same word might provide. Furthermore, BPE's merge rules depend entirely on frequency statistics in the training corpus, making them sensitive to corpus distribution. If the corpus has an excessively high proportion of English, high-frequency Chinese subwords may not receive sufficient merge opportunities, leading to Chinese being overly segmented into individual characters -- a particular concern for multilingual models in vocabulary design.

### Byte-level BPE

Classic BPE uses characters as the initial unit for building the vocabulary. But what exactly is a "character"? The ASCII character set has only 128 symbols, far from sufficient to cover the world's writing systems. The Unicode standard defines over 140,000 characters, covering Chinese, Arabic, emoji, and more. If Unicode characters were used as the initial vocabulary, the base vocabulary would already be very large. Moreover, Unicode character encoding lengths are not uniform (1 to 4 bytes), which introduces unnecessary additional complexity to the BPE merging process.

Starting with GPT-2, a more elegant solution was adopted: **byte-level BPE**. Instead of using Unicode characters as the initial unit, it uses UTF-8 encoded bytes as the initial unit. UTF-8 is a variable-length encoding where ASCII characters occupy 1 byte, common Chinese characters occupy 3 bytes, and emoji occupy 4 bytes. Using bytes as the initial vocabulary, the base vocabulary size is fixed at 256 (a byte has 256 possible values). Regardless of the input language or symbol, everything can be encoded as a byte sequence, and BPE then merges from this unified byte sequence. This approach has many benefits:

- **No need for `<UNK>` token**: Any text can be encoded as a byte sequence; there are no characters that cannot be processed.
- **Cross-lingual consistency**: All languages start from the same level playing field at the byte level. When BPE merges by frequency, it naturally allocates reasonable subword entries for each language, without being distorted by differences in Unicode character table sizes.
- **Compact base vocabulary**: 256 base tokens are far smaller than the Unicode character table, leaving more room for merging semantically valuable subwords.

Note that byte-level BPE and character-level tokenization are two easily confused concepts. Character-level tokenization treats each Unicode character directly as a token, so the Chinese word "learning" (`学习`) would be split into two tokens: `["学", "习"]`. In byte-level BPE, characters are first encoded as bytes and then combined into subwords through merge rules. The Chinese word "learning" (`学习`) might be merged into a single token `["学习"]` or split into `["学", "习"]`, depending on the word's frequency in the corpus. Byte-level BPE is essentially still a subword tokenization algorithm, but it builds merge rules at a finer byte granularity, combining the coverage of character-level tokenization with the semantic integrity of subword tokenization. After GPT-2, byte-level BPE became the mainstream choice for modern LLMs, with models such as ChatGLM, DeepSeek, Llama, and Qwen all using byte-level BPE for their tokenizers.

### Unigram

BPE's tokenization result is deterministic and unique -- the model always sees the same subword segmentation during training. **Unigram** is another subword tokenization method proposed to address this shortcoming, introduced by Taku Kudo in his 2018 paper "[Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://aclanthology.org/P18-1007/)". In contrast to BPE's bottom-up progressive merging approach, Unigram adopts a top-down progressive deletion strategy. It first constructs a large candidate subword set (containing all possible substrings from the corpus) and then iteratively removes the subwords that contribute least to the likelihood of the training corpus until the vocabulary is reduced to the target size. Unigram is a probabilistic model defined over subwords. Given a subword set $V$, the probability of each subword $x$ is determined by its relative frequency in the corpus:

$$P(x) = \frac{count(x)}{\sum_{y \in V} count(y)}$$

A word $w$ typically has more than one possible tokenization. Unigram defines the probability of a word as the sum of probabilities over all possible tokenizations:

$$P(w) = \sum_{s \in S(w)} \prod_{x \in s} P(x)$$

where $S(w)$ is the set of all possible tokenizations of the word $w$. In each iteration, Unigram computes the decrease in training corpus likelihood that would result from removing each subword, removes the subword with the smallest decrease (i.e., the one that contributes least to the likelihood), and repeats this process until the vocabulary is reduced to the preset size. This likelihood-based pruning strategy ensures that Unigram retains the subwords that are most valuable for encoding the training corpus, rather than merely retaining the most frequent ones.

Unigram also has a unique advantage in supporting **subword regularization**. Since a word can have multiple tokenizations with different probabilities, during training, the tokenizer can randomly sample among them rather than always choosing the one with the highest probability. This allows the model to see different subword combinations for the same word during training, thereby enhancing training robustness.

## Vocabulary Design Trade-offs

After selecting a tokenization algorithm, the next step is to determine the vocabulary size, which is a critical design decision for model training that directly impacts model efficiency and capability. Larger and smaller vocabularies each have their pros and cons, requiring trade-offs across multiple factors. A larger vocabulary provides more complete subword coverage, shorter sequences, and higher computational efficiency, but also increases output layer parameters and may lead to insufficient learning for low-frequency subwords. A smaller vocabulary has fewer model parameters and more thorough learning for high-frequency subwords, but results in longer sequences, higher computational cost, and potentially fragmented semantics. The following table shows vocabulary sizes of some modern open-source models:

| Model | Vocabulary Size | Source | Country |
| --- | --- | --- | --- |
| Yi | 64,000 | 01.AI | China |
| Qwen2 | 151,643 | Alibaba Cloud | China |
| DeepSeek-V3 | 129,280 | DeepSeek | China |
| ChatGLM | 151,329 | Zhipu AI | China |
| Mistral | 32,000 | Mistral AI | France |
| Llama 3 | 128,000 | Meta | United States |

Vocabulary design must also consider the language itself. For mixed-language scenarios, multilingual models need to balance coverage across different languages. English, for example, is an alphabetic script where words are naturally separated by spaces, and subwords are typically roots and affixes, allowing a moderate vocabulary size to provide adequate coverage. Chinese, on the other hand, is a logographic script where words are not separated by spaces and individual characters themselves carry meaning. There is a well-known internet meme about the phrase "Nanjing Mayor Jiang Bridge" (`南京市长江大桥`), which can be segmented as "Nanjing City / Yangtze River / Bridge" or "Nanjing Mayor / Jiang / Bridge." This ambiguity means that Chinese LLMs often require larger vocabularies.

## Chapter Summary

The core task of a language model is to estimate the probability distribution of text sequences. From N-Gram models to neural networks, the modeling paradigm has undergone a shift from counting statistics to continuous representations, but the thread of "predicting the next word based on preceding context" has remained constant throughout. N-Gram models are limited by the Markov assumption, unable to handle unseen word combinations or capture long-range dependencies. Neural language models, through distributed word vector representations, naturally solve the sparsity problem and have gradually developed into two main branches: autoregressive models represented by the GPT series and masked language models represented by BERT.

With the foundational prerequisites for training a Transformer-based language model now in place, the next section will focus on training a small-scale but functional language model from scratch.

## Exercises

1. In BPE tokenization, the order of merge rules is critical. Suppose two rules were learned sequentially during training: Rule 1 merges `(e, r)`, and Rule 2 merges `(er, </w>)`. Now tokenize the word "lower," whose initial character sequence is `l o w e r`. Write the intermediate results after applying each rule during tokenization, and explain what happens to the tokenization result if the order of the two rules is swapped (Rule 2 applied first, then Rule 1).
    <details>
    <summary>Reference Answer</summary>

    **Original order (Rule 1 → Rule 2)**:

    1. Initial: `['l', 'o', 'w', 'e', 'r']`
    2. Apply Rule 1, merge `(e, r)` → `er`: `['l', 'o', 'w', 'er']`
    3. Apply Rule 2, merge `(er, </w>)` → `er</w>`: `['l', 'o', 'w']` (`er` is at the word boundary, merging with `</w>`, but the original character sequence of "lower" does not contain `</w>`. This is merely for illustrating the merge mechanism. In practice, whether the initial sequence includes `</w>` depends on the implementation. In this example, without `</w>`, Rule 2 does not match, and the final result is `['l', 'o', 'w', 'er']`)

    A more accurate demonstration (with `</w>` added):

    1. Initial: `['l', 'o', 'w', 'e', 'r', '</w>']`
    2. Apply Rule 1, merge `(e, r)` → `er`: `['l', 'o', 'w', 'er', '</w>']`
    3. Apply Rule 2, merge `(er, '</w>')` → `er</w>`: `['l', 'o', 'w', 'er</w>']`

    **Swapped order (Rule 2 → Rule 1)**:

    1. Initial: `['l', 'o', 'w', 'e', 'r', '</w>']`
    2. Apply Rule 2 first, looking for the `(er, '</w>')` pair. In the current sequence, `e` and `r` are still separate symbols, and `er` as a symbol does not exist, so Rule 2 **does not match** and is skipped.
    3. Apply Rule 1, merge `(e, r)` → `er`: `['l', 'o', 'w', 'er', '</w>']`

    The final result is `['l', 'o', 'w', 'er', '</w>']`, which differs from the original order's result of `['l', 'o', 'w', 'er</w>']`.

    **Key conclusion**: BPE tokenization must apply merge rules strictly in the order learned during training. Rules learned earlier correspond to higher-frequency patterns and must be applied first to ensure tokenization results are consistent with training. Swapping the order may cause some rules to fail to match, producing different tokenization results, which in turn causes the model to see subword sequences during inference that differ from those seen during training, affecting generation quality.
    </details>

1. In the Unigram tokenization model, suppose the vocabulary $V = \{\text{ab}, \text{a}, \text{b}, \text{c}\}$, with subword probabilities $P(\text{ab}) = 0.4$, $P(\text{a}) = 0.3$, $P(\text{b}) = 0.2$, $P(\text{c}) = 0.1$. For the input word "abc," list all possible tokenizations, compute the probability of each tokenization, and identify the tokenization with the highest probability.
    <details>
    <summary>Reference Answer</summary>

    **List all tokenizations**:

    All possible tokenizations of "abc":

    1. `["a", "b", "c"]`
    2. `["ab", "c"]`

    Note: `["a", "bc"]` is invalid because "bc" is not in the vocabulary; `["abc"]` is also invalid because "abc" is not in the vocabulary.

    **Compute the probability of each tokenization**:

    The Unigram model assumes subwords are independent, and the probability of a tokenization is the product of the probabilities of its subwords.

    1. $P([\text{a}, \text{b}, \text{c}]) = P(\text{a}) \times P(\text{b}) \times P(\text{c}) = 0.3 \times 0.2 \times 0.1 = 0.006$
    2. $P([\text{ab}, \text{c}]) = P(\text{ab}) \times P(\text{c}) = 0.4 \times 0.1 = 0.04$

    **Word probability** (sum of probabilities of all tokenizations):

    $$P(\text{abc}) = 0.006 + 0.04 = 0.046$$

    **Tokenization with the highest probability**: `["ab", "c"]` (probability 0.04), far higher than `["a", "b", "c"]` with probability 0.006.

    **Further discussion**: Unigram's subword regularization exploits precisely this property of multiple tokenizations. During training, rather than always selecting the highest-probability tokenization `["ab", "c"]`, it samples according to the probability distribution: approximately 87% of the time it selects `["ab", "c"]`, and approximately 13% of the time it selects `["a", "b", "c"]`. This allows the model to see different subword combinations for the same word during training, enhancing robustness to tokenization variations. BPE, by contrast, can only produce a single tokenization result and lacks this regularization capability.
    </details>

1. The table below lists the vocabulary sizes of several open-source models. Assume the model's hidden dimension $d = 4096$, and compute the number of parameters in the output layer for each model (the output layer is a $|V| \times d$ matrix). If Qwen2's vocabulary were reduced from 151,643 to 32,000 (matching Mistral's), how many parameters would be saved in the output layer? What percentage of Qwen2's original output layer parameters would these savings represent? Considering changes in sequence length, discuss the potential negative impacts of reducing vocabulary size.

    | Model | Vocabulary Size |
    |------|---------|
    | Yi | 64,000 |
    | Qwen2 | 151,643 |
    | Mistral | 32,000 |
    | Llama 3 | 128,000 |

    <details>
    <summary>Reference Answer</summary>

    **Output layer parameters for each model** ($|V| \times d$, $d = 4096$):

    | Model | Vocabulary Size | Output Layer Parameters |
    |------|---------|-------------|
    | Yi | 64,000 | $64000 \times 4096 = 262,144,000$ (approx. 262 million) |
    | Qwen2 | 151,643 | $151643 \times 4096 = 621,129,728$ (approx. 621 million) |
    | Mistral | 32,000 | $32000 \times 4096 = 131,072,000$ (approx. 131 million) |
    | Llama 3 | 128,000 | $128000 \times 4096 = 524,288,000$ (approx. 524 million) |

    **Parameter savings from Qwen2 vocabulary reduction**:

    Original output layer parameters: $151643 \times 4096 = 621,129,728$

    Reduced output layer parameters: $32000 \times 4096 = 131,072,000$

    Parameters saved: $621,129,728 - 131,072,000 = 490,057,728$ (approx. 490 million)

    Percentage: $490,057,728 / 621,129,728 \approx 78.9\%$

    **Analysis of negative impacts**:

    Reducing the vocabulary from 151,643 to 32,000 would most directly result in a significant decrease in Chinese encoding efficiency. An important reason Qwen2 uses a large vocabulary is that Chinese requires more subword entries to cover common phrases and expressions. After vocabulary reduction, many Chinese phrases that previously had dedicated subword entries would be split into finer-grained individual characters or even bytes, leading to:

    1. **Increased sequence length**: The same text would produce more tokens after tokenization, lengthening the sequence. Since Transformer attention computation has $O(n^2)$ complexity, doubling the sequence length means a 4x increase in attention computation, significantly raising training and inference costs.
    2. **Semantic fragmentation**: Phrases that were previously encoded as a single unit (such as "machine learning") would be split into individual characters. The model would need to relearn the combinatorial relationships between characters over a longer context, increasing learning difficulty.
    3. **Insufficient representation of low-frequency subwords**: When the vocabulary shrinks, the retained subwords are predominantly high-frequency, general-purpose ones. Domain-specific vocabulary and low-frequency subwords would have insufficient training samples, degrading representation quality.

    This is the trade-off in vocabulary design: a larger vocabulary trades more parameters for shorter sequences and more complete semantic units, while a smaller vocabulary saves parameters but increases sequence length and the risk of semantic fragmentation. Qwen2's choice of a vocabulary of 150,000 is precisely aimed at gaining an advantage in Chinese encoding efficiency, even at the cost of nearly 500 million additional output layer parameters.
    </details>
