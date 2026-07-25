# Pretraining Data Engineering

In 2020, when OpenAI released GPT-3, the world was amazed by its 175 billion parameters. But another number in the GPT paper was largely overlooked by the public — GPT-3's training data contained approximately 500 billion (500B) tokens. Three years later, Meta's LLaMA pushed this to 1.4T, and by the end of 2024, DeepSeek-V3 used 14.8T tokens, nearly 30 times the amount of GPT-3's training data. Behind these astronomical figures lies a meticulously designed data pipeline: collecting raw text from multiple sources, passing it through rigorous quality filters, and then mixing it in specific proportions.

Data defines the upper bound of model capability. No matter how sophisticated the model architecture or how advanced the training algorithm, they can only approach this upper bound. If data quality is poor or coverage is insufficient, the model's capability will inevitably be limited, regardless of algorithmic excellence. This chapter discusses how to obtain high-quality data for the pretraining process, systematically covering the data engineering workflow: from data sources to quality filtering, from mixing strategies to contamination detection, and finally exploring the potential of synthetic data.

## Data Sources

Before building a pretraining dataset, we must first answer where training text comes from. The internet contains endless amounts of text, but not all of it is suitable for training models. Texts from different sources vary dramatically in quality, length, and domain coverage. This section introduces several data sources widely used by modern LLMs.

- **CommonCrawl**: In 2007, a non-profit project called **CommonCrawl** began regularly crawling publicly available web pages, making the raw HTML data freely available to everyone. By 2023, a single crawl contained approximately 2.5 billion web pages, with raw HTML exceeding 250 TB. After cleaning and deduplication, the usable training text can still reach trillions of tokens. Models such as GPT-3, LLaMA, and DeepSeek all use CommonCrawl as their primary data source.

    The CommonCrawl dataset contains virtually everything — news, blogs, forums, encyclopedias, e-commerce pages — covering over 100 languages. Since crawling is continuous, the data is timely, with the latest information and events readily available. However, because the crawler indiscriminately collects information, the quality of CommonCrawl data is highly uneven. Alongside news articles and Wikipedia entries, there is a large amount of SEO spam, advertising copy, and machine-generated content. HTML tags, navigation bars, footers, and other irrelevant information also take up a considerable proportion. Duplication caused by the same article being republished across multiple websites is a chronic problem for web data. Therefore, transforming CommonCrawl's raw HTML into usable training data requires a complete processing pipeline:

    ```mermaid compact
    graph LR
            A["<b>Raw HTML</b><br/>250+ TB"] --> B["<b>Text Extraction</b><br/>Remove HTML tags"]
            B --> C["<b>Language Detection</b><br/>Filter non-target languages"]
            C --> D["<b>Quality Filtering</b><br/>Denoising, deduplication"]
            D --> E["<b>Toxicity Filtering</b><br/>Remove harmful content"]
            E --> F["<b>Training Data</b><br/>Trillions of tokens"]
    ```
    *Figure: CommonCrawl data processing pipeline*

- **Books3**: While web-crawled data is abundant, individual documents are typically short and cannot help the model learn long-distance semantic dependencies. A blog post may only be a few hundred words, and a forum post might be just a sentence or two. For the model to understand long-text structures — where earlier passages set up foreshadowing and later passages provide resolution — the best approach is to use complete books as training material.

    The Books3 dataset was created precisely for this purpose. It contains approximately 200,000 books sourced from the Bibliotik private tracker. Compared to fragmented web pages, books are complete long-form narratives or systematic expositions with higher information density, helping the model learn discourse-level coherence. For the same 10,000 words, a textbook carries far more knowledge than dozens of blog posts. Books have been edited and proofread, making the language more standardized, with content spanning fiction, non-fiction, textbooks, and professional works across multiple domains.

    The biggest problem with book data is copyright. Books3 was taken down due to copyright issues, sparking widespread discussion about intellectual property rights in training data. Copyright concerns have forced researchers to seek alternative sources. The most well-known is Project Gutenberg, which contains approximately 70,000 books in the public domain. Although far fewer in number than Books3, it has the advantage of being free from legal controversy.

- **GitHub**: Web text teaches models how to speak, books teach models how to write articles, and code data teaches models how to program. Code data is crucial for a model's code generation ability and logical reasoning capability. GPT-4, DeepSeek-Coder, and other models have all been trained on large amounts of code data.

    The primary source of code data is public repositories on GitHub, along with problems and solutions from competitive programming platforms like Codeforces and LeetCode, as well as official documentation and Q&A from Stack Overflow. Unlike natural language text, code is a formal language with a clear syntactic structure — every line must conform to grammar rules. The control flow in code, such as conditionals, loops, and recursion, naturally serves as chain-of-thought training material, while comments act as a bridge between natural language and formal language, helping the model learn to switch between the two modes of expression.

- **arXiv**: In 1991, American physicist Paul Ginsparg created the arXiv website, initially to allow physicists to quickly share preprint papers before formal publication. More than three decades later, arXiv has become the most important preprint platform in physics, mathematics, computer science, and related fields, and it serves as a vital data source for language models to acquire specialized knowledge and academic writing style.

    The value of academic literature lies in its specialization. Technical details in deep learning papers, derivations in quantum physics papers, and rigorous logic in mathematical proofs — these are virtually absent from general web text. Papers also follow a standard structural format, from abstract and introduction to methods and conclusions. This structure itself is a valuable training signal. LaTeX formulas are a distinctive feature of arXiv; mathematical expression is a special language that language models need to learn, and LaTeX source code provides an abundance of training material. Additionally, citation relationships between papers can be used to build knowledge graphs.

    However, processing academic literature also presents unique challenges. LaTeX source code must be correctly parsed into trainable text. How to tokenize mathematical formulas is a separate technical problem, and the figures and tables in papers remain difficult to utilize directly.

- **Wikipedia**: Unlike the needle-in-a-haystack approach of CommonCrawl, Wikipedia provides high-quality encyclopedia data that has undergone human review. Nearly all LLM training datasets include Wikipedia and give it a sampling weight far exceeding its natural proportion.

    Wikipedia's core advantage is factual accuracy. Unlike the rumors and misinformation that are everywhere on the internet, Wikipedia entries undergo community review, making the information relatively reliable and providing the model with trustworthy factual knowledge. Its structure is also very clear: well-defined heading hierarchies and category systems help the model learn how knowledge is organized, and internal links within entries help the model understand relationships between entities. In terms of multilingual coverage, each language version of Wikipedia is an independent knowledge base, and different language versions of the same entry can be used for alignment learning, enabling the model to learn that "apple" and "pomme" refer to the same thing.

## Data Quality

Among the petabytes of raw HTML in CommonCrawl, the proportion of text truly suitable for training may be less than 10%. Feeding such noisy data directly to a model is like asking a student to read a book full of typos and formatting errors — not only will they fail to learn correct knowledge, but they may also develop bad habits. Data quality filtering is the foremost step in the data pipeline.

### Denoising

Most posts on internet forums contain meaningful discussion, but they are mixed with garbled text, awkward machine translations, spam ads saying "click here for a free download," and filler content that repeats the same sentence across an entire page.

Humans can identify such garbage text by intuition, but with billions of documents to process, manual inspection is impossible — automated methods are essential. Heuristic rules are a commonly used denoising technique. A normal English text will have a reasonable number of punctuation marks, a certain proportion of stopwords (the, is, are, etc.), and will not have highly repetitive vocabulary. If a text violates these basic patterns, it is likely low quality. The following code demonstrates how to use a set of heuristic rules to detect low-quality text:

```python runnable
# Low-quality text detection example
import re

def detect_low_quality(text):
    """Detect low-quality text"""
    issues = []
    
    # 1. Length check
    if len(text) < 50:
        issues.append("Text too short")
    elif len(text) > 100000:
        issues.append("Text too long")
    
    # 2. Character ratio check
    # Garbled text detection: abnormal proportion of non-ASCII characters
    non_ascii_ratio = len(re.findall(r'[^\x00-\x7F]', text)) / max(len(text), 1)
    if non_ascii_ratio > 0.5 and not any(c in text for c in '中文日本語한국어'):
        issues.append(f"Suspected garbled text (non-ASCII ratio: {non_ascii_ratio:.1%})")
    
    # 3. Punctuation check
    # Texts lacking punctuation may be low quality
    punct_ratio = len(re.findall(r'[.!?]', text)) / max(len(text.split()), 1)
    if punct_ratio < 0.01:
        issues.append("Too few punctuation marks")
    
    # 4. Repeated word check
    words = text.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append(f"High lexical repetition (unique word ratio: {unique_ratio:.1%})")
    
    # 5. Stopword check
    # High-quality text usually contains a certain proportion of stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    stopword_count = sum(1 for w in words if w in stopwords)
    stopword_ratio = stopword_count / max(len(words), 1)
    if stopword_ratio < 0.05 and len(words) > 50:
        issues.append(f"Abnormally low stopword ratio ({stopword_ratio:.1%})")
    
    # 6. Special pattern detection
    # Common patterns in SEO spam text
    seo_patterns = [
        r'click here',
        r'buy now',
        r'free download',
    ]
    for pattern in seo_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(f"SEO spam pattern: {pattern}")
            break
    
    return issues

# Test examples
test_cases = [
    ("This is a normal English sentence with proper punctuation and structure.", "Normal English"),
    ("click here to buy now free download limited offer", "SEO spam"),
    ("asdfghjkl qwertyuiop zxcvbnm", "Garbled text"),
    ("the the the the the the the the the the", "Repeated words"),
]

print("Low-quality text detection example:\n")
for text, label in test_cases:
    issues = detect_low_quality(text)
    status = "Issues: " + ", ".join(issues) if issues else "Passed"
    print(f"[{label}]")
    print(f"  Text: {text[:50]}...")
    print(f"  {status}\n")
```

Heuristic rules are simple and efficient, but they can only capture known low-quality patterns. They are powerless against more subtle noise, such as the grammatically correct but semantically vacuous AI-generated text that has begun to appear in large quantities in recent years. This reality dictates that modern data pipelines can only use heuristic rules as a front-end coarse filter, and must incorporate model-based filtering at the back end — training a lightweight classifier to distinguish between high-quality and low-quality text.

### Deduplication

A single press release on the internet may be republished by dozens of websites, a piece of code may be forked by countless projects, and a piece of knowledge may be repeatedly answered across Q&A platforms. Duplicate data not only wastes training resources but, more dangerously, can lead the model to memorize rather than learn — it recites familiar content flawlessly but is helpless when faced with unseen material.

The most basic form of deduplication is exact deduplication, such as computing a hash (e.g., MD5, SHA256) for each document and deleting documents with identical hashes. However, exact deduplication cannot identify near-duplicates. Two documents may differ by only a few words (e.g., a republished article with a modified title or an added source credit), and exact hashing would treat them as completely different documents.

In 1997, Israeli computer scientist Andrei Broder, while researching document similarity detection for search engines, proposed the MinHash method. Instead of directly comparing the content of two documents, MinHash maps them to a set of short signatures and estimates the similarity of the original documents by comparing the similarity of these signatures. Specifically, MinHash represents each document as a set of words, and document similarity is measured using Jaccard similarity:

$$Jaccard(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

The numerator $|A \cap B|$ is the number of words common to both documents, and the denominator $|A \cup B|$ is the total number of unique words across both documents. The overall ratio measures how much the two documents overlap. If two documents are identical, the Jaccard similarity is 1; if they share no words, it is 0. Directly computing Jaccard similarity requires a full comparison of the two sets, which is expensive. The cleverness of MinHash lies in using a set of random permutations to map sets to signatures, such that the similarity of two signatures approximates their Jaccard similarity. Using multiple independent random permutations yields a MinHash signature vector; comparing the proportion of matching positions in two signature vectors provides an unbiased estimate of the Jaccard similarity.

```python runnable
# MinHash deduplication demonstration
import hashlib
from collections import defaultdict

class SimpleMinHash:
    """Simplified MinHash implementation"""
    
    def __init__(self, num_hashes=128):
        self.num_hashes = num_hashes
        # Use different hash seeds to simulate different permutations
        self.seeds = [i * 1000003 for i in range(num_hashes)]
    
    def _hash(self, token, seed):
        """Compute hash value for a single token"""
        h = hashlib.md5(f"{seed}{token}".encode()).hexdigest()
        return int(h, 16)
    
    def get_signature(self, tokens):
        """Compute MinHash signature for text"""
        signature = []
        for seed in self.seeds:
            min_hash = float('inf')
            for token in tokens:
                h = self._hash(token, seed)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature
    
    def jaccard_similarity(self, sig1, sig2):
        """Estimate Jaccard similarity from signatures"""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

# Demonstration of deduplication
minhash = SimpleMinHash(num_hashes=64)

# Simulated documents
docs = {
    'doc1': "The quick brown fox jumps over the lazy dog",
    'doc2': "The quick brown fox jumps over the lazy dog",  # Exact duplicate
    'doc3': "A quick brown fox jumped over a lazy dog",    # Near duplicate
    'doc4': "Machine learning is a subset of artificial intelligence",  # Completely different
}

# Compute signatures
signatures = {}
for doc_id, text in docs.items():
    tokens = text.lower().split()
    signatures[doc_id] = minhash.get_signature(tokens)

# Compute similarity matrix
print("Document similarity matrix (MinHash estimate):\n")
doc_ids = list(docs.keys())
print("           ", "  ".join(f"{d:6}" for d in doc_ids))
for i, id1 in enumerate(doc_ids):
    row = [id1.ljust(10)]
    for j, id2 in enumerate(doc_ids):
        sim = minhash.jaccard_similarity(signatures[id1], signatures[id2])
        row.append(f"{sim:.2f}  ")
    print(" ".join(row))
```

Even with MinHash signatures, the complexity of pairwise comparison remains $O(n^2)$, which becomes computationally prohibitive when the number of documents reaches billions. **Locality Sensitive Hashing** (LSH) divides the signatures into buckets, such that only documents falling into the same bucket require detailed comparison, reducing the complexity to $O(n)$. In practice, deduplication is typically performed at three granularities using different strategies:

| Dedup Level | Method | Threshold | Effect |
|:------------|:-------|:----------|:-------|
| Document-level | MinHash + LSH | Jaccard > 0.8 | Removes ~30-50% duplicate documents |
| Sentence-level | Exact hash | Exact match | Removes ~10-20% duplicate sentences |
| Token-level | N-Gram dedup | n=13 | Prevents model from memorizing long sequences |

Document-level deduplication removes documents that are highly similar overall. Sentence-level deduplication handles duplicate sentences within or across documents. Token-level [N-Gram](../architecture-basics/language-model-tokenization.md#n-gram-language-models) deduplication is the most fine-grained, ensuring that no consecutive 13-token sequence appears identically in the training data, thereby preventing the model from cheating by memorizing long sequences.

### Toxicity Filtering

Beyond lexical and grammatical quality issues, data also contains semantic safety concerns, including hate speech, violent descriptions, pornographic content, and discriminatory language. The model will absorb these as knowledge during training and reproduce them in its output. The goal of toxicity filtering is to remove as much of this content as possible before training.

Filtering methods progress in complexity. The simplest is keyword filtering, maintaining a list of sensitive words and directly filtering out documents containing them. However, this method is prone to false positives — medical literature frequently contains sexuality-related terms, and legal texts may describe violence, none of which are toxic. A more refined approach is classifier-based filtering: training a toxicity detection model to score documents and only filtering those that exceed a threshold. Commonly used models include Google's Perspective API, HateBERT fine-tuned on hate speech data, and Meta's LLM-based safety guard model, Llama Guard. The most refined approach directly uses a large model (such as GPT-4) to judge whether content is harmful, then uses its labeled data to train a small model for large-scale filtering, balancing precision and efficiency.

The key to toxicity filtering is threshold selection. A threshold that is too low will over-filter, losing large amounts of useful data, while a threshold that is too high will allow harmful content to slip through. In practice, thresholds are typically set separately for different dimensions (violence, hate, pornography, self-harm, etc.), with special handling for professional domains such as medicine and law.

### Language Detection

Before feeding text to a language model for training, it is also necessary to identify the language of each document and decide whether to keep it and how to sample it. Facebook's FastText language identification model is the most commonly used tool, supporting 176 languages with extremely fast speed. Python's langdetect library and Google's CLD3 model are also common choices. For monolingual models, only text in the target language is retained. For multilingual models, text needs to be grouped by language, and the retention ratio for each language is determined by the mixing strategy.

Language detection may seem simple — just an API call or model inference — but there are several noteworthy caveats. Detection accuracy tends to be lower for short texts, as a single sentence may be grammatically compatible with multiple languages. A single document may also mix multiple languages (such as a technical blog post written in both Chinese and English). For low-resource languages, the detection model's training data is itself insufficient, so accuracy is naturally limited.

## Data Mixing Strategies

After source collection and quality filtering, we have high-quality text from different domains: the web, books, code, academic papers, and more. Using these texts is like cooking — good ingredients still need the right proportions; they cannot simply be mixed together indiscriminately. For example, if 99% of the data is web text, the model might develop a distinctly "internet vernacular" tone; if code data is too scarce, the model's programming ability will be weak. Data mixing strategies address the allocation of sampling proportions across different sources, ensuring the model is adequately trained in all areas.

### Fixed Proportioning

Looking at the data proportions of mainstream LLMs in recent years, some converging empirical patterns emerge. Taking LLaMA as an example, CommonCrawl and its cleaned version C4 together accounted for approximately 82% of its training data, GitHub code about 5%, Wikipedia and books about 4.5% each, with arXiv and StackExchange supplementing about 4.5% of specialized knowledge:

| Data Source | Proportion | Token Count |
|:------------|:-----------|:------------|
| CommonCrawl | 67% | ~940B |
| C4 | 15% | ~210B |
| GitHub | 4.5% | ~63B |
| Wikipedia | 4.5% | ~63B |
| Books3 | 4.5% | ~63B |
| arXiv | 2.5% | ~35B |
| StackExchange | 2% | ~28B |

The logic behind this proportioning is worth considering: CommonCrawl occupies an absolutely dominant position because it provides the broadest coverage of general knowledge and language ability. Although CommonCrawl's share in the raw data is far higher than 82%, a large portion is eliminated after quality filtering, drastically reducing its usable proportion, and it ultimately needs supplementation from other high-quality sources. Code data, while only accounting for 4.5%, significantly improves the model's logical reasoning capability. DeepSeek reported in its technical paper that when the proportion of code data was increased from 7% to 14%, the model's pass rate on HumanEval improved by nearly 10 percentage points. Although Wikipedia and books account for a small proportion, they provide a high-quality knowledge baseline, ensuring the model does not stray from factual accuracy. arXiv and StackExchange supplement the depth of knowledge in specialized domains.

### Dynamic Domain Weighting

While manually set weights are effective, they are ultimately empirical, and fixed source proportions implicitly assume that all documents within the same source are equally valuable. This is not the case. Among web texts, a Wikipedia-quality article and a forum spam post are worlds apart in quality. Since the information density of Wikipedia is far higher than that of an average web page, even if it accounts for only 4.5% of the raw proportion, it can be given a higher sampling weight, allowing the model to encounter Wikipedia texts more frequently during training.

Domain weighting aims to break away from manually set fixed weights by assigning different sampling weights to data from different domains. In 2023, Google Research proposed the DoReMi (Domain Reweighting with Minimax Optimization) algorithm, which lets the model itself decide what data it needs. DoReMi first trains a small proxy model, then allows the proxy model to dynamically adjust the sampling weights of different domain data during training. If the loss in a particular domain decreases slowly, it indicates that the model still has room to learn in that domain, so the sampling weight for that domain is increased, and vice versa. This approach makes the data proportioning no longer fixed during training, but dynamically adjusted according to the model's learning state, without requiring manual tuning. Moreover, the weights themselves are interpretable, intuitively reflecting the learning difficulty of each domain.

## Data Contamination

In 2023, an embarrassing discovery shook the NLP community: several open-source large language models achieved abnormally high scores on benchmarks such as MMLU and GSM8K, and the reason was that test set data had been mixed into the training set. The models had not learned how to solve the problems — they had seen the answers beforehand. This phenomenon is called **data contamination**, and it is one of the most easily overlooked issues in pretraining data engineering.

This contamination was not the result of someone deliberately inserting test questions into the training data. The way it occurred was much more subtle: someone on Wikipedia added the complete questions and answers of MMLU items; someone on Stack Overflow discussed solutions to LeetCode problems, and LeetCode happens to be a source of code evaluation data; an arXiv paper included GSM8K math problems and their standard solutions in its appendix. Once such content is crawled into the training data, the model will cheat during evaluation, producing inflated scores that do not accurately reflect the model's true capability.

Contamination can be categorized into three levels of severity. The most severe is exact input + output matching, where both the question and the standard answer appear verbatim in the training data — the model only needs to recall them. Next is input matching, where only the question appears — the model has seen the question but not the standard answer, yet it still has an advantage over models that have never seen it. The mildest is approximate matching, where content similar to the test question appears in the training data, potentially providing the model with some problem-solving clues.

The real challenge is how to efficiently find overlapping segments with the test set from trillions of tokens of training data. The most direct method is N-Gram matching: splitting each sample in the test set into consecutive n-grams and searching the training data for identical sequences. The GPT-2 technical report used both 8-Gram and 10-Gram granularities for detection. N-Gram matching is simple and reliable, but it cannot detect approximate contamination where the wording has been altered. Only semantic-based detection methods can effectively identify this. Word embedding models can map text into a vector space, and vector similarity can be computed to identify texts that are semantically highly similar but worded differently — however, this comes with high computational cost.

The approach to handling data contamination depends on when it is discovered. If it is still in the data preparation stage, the simplest method is to directly delete the contaminated documents or paragraphs. If the model has already been trained, the only recourse is to exclude contaminated samples during evaluation and report scores only on uncontaminated samples. This does not fix the model itself, but at least it makes the evaluation results more meaningful.

Detecting data contamination may seem like a technical issue, but at its core, it is about research integrity. A student who has peeked at the exam in advance cannot claim their high score reflects their true ability. As the community's awareness of data contamination grows, mainstream evaluation benchmarks now require reporting decontaminated results, and new training datasets have universally incorporated decontamination as a standard procedure.

## Synthetic Data

All the data sources discussed so far — whether web crawling or book scanning — essentially consume text that humans have already produced. As model sizes grow, high-quality human-generated data is being rapidly consumed. Some studies estimate that, at current usage rates, high-quality text on the internet could be exhausted for training purposes by around 2026. Facing the threat of data depletion, researchers have begun exploring a new idea: can we use model-generated text to train models?

In 2023, Microsoft Research's work on the Phi series proposed the following approach: rather than having the model read massive amounts of low-quality text from the internet, use strong models like GPT-3.5 or GPT-4 to carefully generate high-quality training data according to a specific curriculum. This textbook-level synthetic data is far superior in quality to ordinary web text. Phi-1, with only 1.3B parameters and far less training data than LLaMA, achieved impressive results on programming benchmarks, powerfully demonstrating the decisive impact of data quality on model performance.

The logic behind this approach is that quality trumps quantity. In traditional pretraining data, large amounts of low-quality text occupy training resources without contributing effective learning signals, and may even have negative effects. Synthetic data, on the other hand, can be generated on demand, targeting the model's weak areas with supplementary training material. Specifically, there are three main methods for generating synthetic data. The first is strong model generation: using large models like GPT-4 with carefully designed prompts to generate text of a specific domain, difficulty, and style. The Phi series was trained on programming textbooks and exercises generated by GPT-3.5. The second is self-distillation: using the model's own output to train itself — typically, the model generates multiple candidate responses, and a quality signal (such as a reward model score) is used to filter out high-quality responses for inclusion in the training data. The third is data augmentation: transforming existing data through rewriting, expansion, translation, and other operations to generate new training samples — this has the lowest cost but also relatively limited diversity.

Phi-2, with only 2.7B parameters, outperformed LLaMA-2 7B and Mistral 7B (which have 2-3 times more parameters) on multiple benchmarks, proving that high-quality synthetic data can indeed break the simple linear assumption that bigger models are always better. While the prospects of synthetic data are enticing, its limitations are equally real.

The most fundamental concern is **model collapse**. In 2023, Ilia Shumailov from the University of Oxford and colleagues theoretically demonstrated that if a model is trained entirely on the output of another model, the generated distribution will gradually degenerate over successive generations. The model forgets the long-tail distribution of the original data, increasingly tending to generate high-frequency common content, ultimately leading to a loss of diversity. This is like repeatedly photocopying a document — each generation loses some detail, eventually becoming a blurry shadow.

Synthetic data also suffers from **bias amplification**: model-generated content inherits and amplifies biases present in the training data. If this content is then used to train the next generation of models, the bias snowballs. In terms of factual accuracy, models may generate plausible-sounding but factually incorrect information. If such errors seep into the training data, they can severely damage the reliability of the next-generation model. Furthermore, synthetic data generated by current models remains limited to content within the training distribution and cannot create truly novel knowledge the way humans can.

Therefore, the prevailing view is that synthetic data can serve as a supplement to human data, but cannot completely replace it. The most pragmatic strategy is human-machine collaboration: let the model generate a draft, then have humans review and correct it. Alternatively, use synthetic data for data augmentation — expanding diversity on top of human data — rather than starting from scratch with purely synthetic training.

## Chapter Summary

Pretraining data engineering is a精密 pipeline that transforms "raw materials" into "training fuel": collecting text from multiple sources — CommonCrawl, books, code, academic papers, and Wikipedia — then passing it through quality filtering and decontamination, and finally mixing it according to a strategic proportioning plan. Through this chapter, we have learned where data comes from, how it is cleaned, and how it is mixed. But when faced with trillions of tokens of data, how much computational resource is actually needed to train a powerful model? Is there a predictable relationship between model size, data size, and computational cost? The next chapter will explore scaling laws and the mathematical answers to these questions.

## Exercises

1. Implement a simple data quality filtering pipeline that sequentially applies length filtering, language detection, repetition detection, and SEO spam detection to input text. Each filtering step should output the specific reason for filtering.
   <details>
   <summary>Reference Answer</summary>
   
   ```python runnable
   import re
   
   def quality_pipeline(text):
       """Multi-step data quality filtering pipeline"""
       reasons = []
       
       # Step 1: Length filtering
       if len(text) < 30:
           reasons.append("Too short")
       elif len(text) > 100000:
           reasons.append("Too long")
       
       # Step 2: Language detection
       chinese_ratio = len(re.findall(r'[一-鿿]', text)) / max(len(text), 1)
       if chinese_ratio > 0.3:
           lang = "Chinese"
       else:
           lang = "English/Other"
       
       # Step 3: Repetition detection
       words = text.lower().split()
       if len(words) > 10:
           unique_ratio = len(set(words)) / len(words)
           if unique_ratio < 0.3:
               reasons.append(f"Vocabulary repetition too high ({unique_ratio:.1%})")
       
       # Step 4: SEO spam detection
       seo_patterns = [r'click here', r'buy now']
       for p in seo_patterns:
           if re.search(p, text, re.IGNORECASE):
               reasons.append(f"SEO spam pattern: {p}")
               break
       
       passed = len(reasons) == 0
       return passed, reasons, lang
   
   # Test
   samples = [
       "This is a normal academic English text discussing the application of deep learning in natural language processing.",
       "click here to buy now free download limited offer",
       "the the the the the the the the the the",
   ]
   
   for text in samples:
       passed, reasons, lang = quality_pipeline(text)
       status = "Passed" if passed else f"Filtered ({', '.join(reasons)})"
       print(f"Language: {lang}, Status: {status}")
   ```
   
   </details>

2. The expected similarity of MinHash signatures equals the Jaccard similarity. Please explain the intuitive meaning of this conclusion and the mathematical proof approach.
   <details>
   <summary>Reference Answer</summary>

    **Intuitive meaning**: MinHash applies a random permutation to a set and takes the minimum element after permutation as the signature value. For two sets $A$ and $B$, the probability that they happen to pick the same minimum element under this permutation is equivalent to the probability that the first element in the permutation falls within $A \cap B$ — and this probability is exactly $|A \cap B| / |A \cup B|$, i.e., the Jaccard similarity. Therefore, after multiple independent permutations, the proportion of equal signatures is an unbiased estimate of the Jaccard similarity.

    **Mathematical proof approach**: Fix a random permutation $\pi$, and let $h_\pi(S) = \min_{x \in S} \pi(x)$ be the MinHash value of set $S$ under this permutation. We need to prove:

    $$\Pr[h_\pi(A) = h_\pi(B)] = J(A, B)$$

    1. Arrange the elements of the universe according to $\pi$, scanning from front to back. The first element encountered must belong to $A \cup B$.
    2. If this element belongs to $A \cap B$, then $h_\pi(A) = h_\pi(B)$ (both have the same minimum value); otherwise, it belongs to $A \setminus B$ or $B \setminus A$, and the minimum values differ.
    3. Therefore, $h_\pi(A) = h_\pi(B)$ if and only if the first element of $\pi$ that lies in $A \cup B$ falls in $A \cap B$.
    4. Since the permutation is random, the probability that the first element falls in $A \cap B$ is $|A \cap B| / |A \cup B| = J(A, B)$.

    Using $k$ independent permutations yields a $k$-bit signature vector. The proportion of matching signatures is an unbiased estimator of $J(A, B)$, with variance decreasing as $k$ increases.

   </details>
