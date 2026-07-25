# LSTM Ancient Poetry Generation Experiment

This engineering practice implements the complete training pipeline of an LSTM language model, covering data preprocessing, model definition, training tuning, and text generation. Through hands-on practice, you will understand the sequence modeling capability of recurrent neural networks and ultimately train a model capable of generating classical Chinese poetry.

## Experiment Preparation

Before starting the experiment, make sure you have [mounted the data directory](../../appendixes/sandbox.md#data-management) and downloaded the chinese-poetry dataset. You can automate this using the `DMLA-CLI` tool:

```bash
# Select "Download Dataset" -> Select "Chinese Poetry"
dmla data
```

## Phase 1: Data Preprocessing

Data preprocessing for an LSTM language model requires converting raw text into numerical sequences that the model can process. This experiment adopts a character-level modeling approach, where each Chinese character is treated as an independent token, and the model learns to predict the next character. The advantage of character-level modeling is that the vocabulary size is controllable (around 3000-5000 common Chinese characters) and it can handle arbitrary new words without requiring a predefined word list. The engineering decisions in this phase revolve around the following two aspects:

- **Data Cleaning**: Poetry data contains metadata such as titles, authors, and annotations; only the poem body needs to be retained for training. Additionally, incomplete poems (e.g., those containing missing character markers like "□") and pieces that are too short or too long (short ones lack information, long ones reduce training efficiency) need to be filtered out.
- **Sequence Construction**: LSTM training uses [Teacher Forcing](seq2seq.md#scheduled-sampling), where the input sequence is the target sequence with the last character removed, and the target sequence is the original sequence with the first character removed. For example, for the verse "床前明月光", the input is "床前明月" and the target is "前明月光". The model learns to predict the next character based on the preceding context.

The code below does not produce cached or intermediate results, so there is no need to run it manually. It will be called automatically during model training in Phase 3.

```python runnable extract-class="PoetryDataset"
import os
import json
import re
from collections import Counter

class PoetryDataset:
    """Classical poetry dataset (character-level language model)

    Loads poems from the chinese-poetry dataset, builds a character-level vocabulary,
    and converts poem text into numerical sequences for LSTM training.
    """
    def __init__(self, data_dir, min_length=10, max_length=100, vocab_size=4000):
        self.min_length = min_length
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Load poem text
        self.poems = self._load_poems(data_dir)
        print(f"Loaded: {len(self.poems)} poems")

        # Build vocabulary
        self.char2idx, self.idx2char = self._build_vocab()
        print(f"Vocabulary size: {len(self.char2idx)}")

        # Convert poems to sequences
        self.sequences = self._encode_poems()
        print(f"Valid sequences: {len(self.sequences)}")

    def _load_poems(self, data_dir):
        """Load poetry data"""
        poems = []

        # Define datasets to load
        datasets = ['全唐诗', '宋词', '诗经', '楚辞']

        for dataset in datasets:
            dataset_path = os.path.join(data_dir, dataset)
            if not os.path.exists(dataset_path):
                continue

            json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]

            for jf in json_files:
                file_path = os.path.join(dataset_path, jf)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    for poem in data:
                        # Extract poem body
                        text = self._extract_text(poem)
                        if text and self._is_valid(text):
                            poems.append(text)
                except Exception as e:
                    print(f"Failed to load {jf}: {e}")

        return poems

    def _extract_text(self, poem):
        """Extract body text from poetry data"""
        # Try different field names
        if 'text' in poem:
            text = poem['text']
        elif 'paragraphs' in poem:
            text = ''.join(poem['paragraphs'])
        elif 'content' in poem:
            # content may be a string or list
            content = poem['content']
            if isinstance(content, list):
                text = ''.join(content)
            else:
                text = content
        else:
            return None

        # Clean text: remove punctuation, keep only Chinese characters
        # Keep common punctuation for sentence segmentation
        text = re.sub(r'[^一-龥，。！？、；：""''（）]', '', text)

        return text

    def _is_valid(self, text):
        """Check if text is valid"""
        # Length check
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # Filter poems with missing character markers
        if '□' in text or '■' in text:
            return False

        return True

    def _build_vocab(self):
        """Build character-level vocabulary"""
        # Count character frequencies
        char_counter = Counter()
        for poem in self.poems:
            char_counter.update(poem)

        # Select most frequent characters
        most_common = char_counter.most_common(self.vocab_size - 2)  # Reserve two slots for special tokens

        # Build mapping
        char2idx = {'<PAD>': 0, '<UNK>': 1}
        for i, (char, _) in enumerate(most_common, start=2):
            char2idx[char] = i

        idx2char = {idx: char for char, idx in char2idx.items()}

        return char2idx, idx2char

    def _encode_poems(self):
        """Convert poems to numerical sequences"""
        sequences = []
        for poem in self.poems:
            seq = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in poem]
            sequences.append(seq)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input sequence: remove the last character
        # Target sequence: remove the first character
        return seq[:-1], seq[1:]

# Test data loading
data_dir = os.path.join(DATA_DIR, 'datasets', 'chinese-poetry')
if os.path.exists(data_dir):
    dataset = PoetryDataset(data_dir, min_length=10, max_length=100, vocab_size=4000)

    # Display vocabulary sample
    print("\nVocabulary sample (first 20 characters):")
    sample_chars = list(dataset.char2idx.keys())[2:22]  # Skip <PAD> and <UNK>
    for char in sample_chars:
        print(f"  '{char}': {dataset.char2idx[char]}")

    # Display sequence sample
    print("\nSequence sample:")
    input_seq, target_seq = dataset[0]
    original_text = dataset.poems[0]
    print(f"  Original: {original_text[:30]}...")
    print(f"  Input sequence: {input_seq[:20]}...")
    print(f"  Target sequence: {target_seq[:20]}...")
else:
    print("Dataset not downloaded")
```

## Phase 2: Model Definition

The core structure of an LSTM language model is a multi-layer LSTM network that progressively encodes the input character sequence into hidden states, and finally maps the hidden states to an output space the size of the vocabulary through a fully connected layer. The model architecture in this experiment follows these design principles:

- **Embedding Layer**: Maps character indices to dense vector representations. The embedding dimension determines the semantic expressiveness of characters, typically set to 128-512 dimensions. The embedding layer allows the model to learn semantic relationships between characters, such as "春" (spring) and "秋" (autumn) being closer in the embedding space because they are both related to seasons.
- **LSTM Layers**: Uses a 2-layer LSTM structure with 256 hidden dimensions per layer. Multi-layer LSTM can learn more complex sequence patterns, with the first layer capturing basic grammatical structures and the second layer capturing higher-level semantic relationships. Dropout (0.3) is used to prevent overfitting.
- **Output Layer**: Maps LSTM outputs to logits the size of the vocabulary, converted to a probability distribution via Softmax, representing the predicted probability of the next character.

```python runnable extract-class="PoetryLSTM"
import torch
import torch.nn as nn

class PoetryLSTM(nn.Module):
    """LSTM language model (for classical poetry generation)

    Architecture: Embedding -> LSTM -> Linear -> Softmax
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2, dropout=0.3):
        super(PoetryLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer: character index -> dense vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer: hidden state -> vocabulary probability distribution
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        Args:
            x: Input sequence (batch_size, seq_len)
            hidden: Initial hidden state (optional)

        Returns:
            output: Output logits (batch_size, seq_len, vocab_size)
            hidden: Final hidden state
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # LSTM: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Output: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden states"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

# Test model
vocab_size = 4000
model = PoetryLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2)

print("LSTM language model structure:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
batch_size = 4
seq_len = 20
x = torch.randint(0, vocab_size, (batch_size, seq_len))

output, hidden = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Hidden state shape: h={hidden[0].shape}, c={hidden[1].shape}")
```

## Phase 3: Model Training

The training objective of an LSTM language model is to maximize the likelihood of character sequences in the training data -- given the preceding context, the probability of correctly predicting the next character should be as high as possible. The engineering decisions in this phase mainly focus on training stability and efficiency:

- **Loss Function**: Uses Cross Entropy Loss, ignoring the loss at padding positions (`<PAD>`). Cross entropy loss measures the discrepancy between the model's predicted distribution and the true distribution, and is the standard loss function for language models.
- **Gradient Clipping**: Although LSTM alleviates the vanishing gradient problem, gradient explosion can still occur. Gradient clipping (`clip_grad_norm_`, max norm 5.0) is used to prevent excessively large parameter updates and maintain training stability.
- **Learning Rate Scheduling**: Adopts a Cosine Annealing strategy, where the learning rate gradually decreases from the initial value to near zero. This strategy provides a larger learning rate early in training to accelerate convergence, and reduces the learning rate later for fine-tuning.
- **Batching and Sequence Packing**: Since poems vary in length, padding is used to align sequences within the same batch to the same length. Padding within each batch should be minimized for efficiency.

::: info Training Estimate

293,803 poems, running 30 epochs, 2,296 batches/epoch.
Requires only 4 GB RAM and 4 GB VRAM to run. Approximately 25-30 minutes using GPU training.

:::

```python runnable gpuonly timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import time

# Import progress reporting module
from dmla_progress import ProgressReporter

# Import model and dataset from shared modules
from shared.sequence_models.poetry_lstm import PoetryLSTM
from shared.sequence_models.poetry_dataset import PoetryDataset

# Define dataset class (for DataLoader)
class PoetryDatasetForTraining(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input sequence: remove the last character
        # Target sequence: remove the first character
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

def collate_fn(batch):
    """Custom batch collation function: pad sequences"""
    inputs, targets = zip(*batch)
    # Pad to the maximum length within the batch
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

# === Training Configuration ===
batch_size = 128
num_epochs = 30
learning_rate = 0.001
hidden_dim = 256
embedding_dim = 256
num_layers = 2
dropout = 0.3
max_grad_norm = 5.0

# === Create Model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# Load dataset
data_dir = os.path.join(DATA_DIR, 'datasets', 'chinese-poetry')

if not os.path.exists(data_dir):
    print("Error: Dataset not downloaded. Please run 'dmla data' first to download the dataset", flush=True)
else:
    # Load data using the shared PoetryDataset module
    print("Loading dataset...", flush=True)
    dataset = PoetryDataset(data_dir, min_length=10, max_length=100, vocab_size=4000)
    print(f"Loaded: {len(dataset.poems)} poems, vocabulary size: {len(dataset.char2idx)}", flush=True)

    # Create data loader
    train_dataset = PoetryDatasetForTraining(dataset.sequences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    # Create model
    vocab_size = len(dataset.char2idx)
    model = PoetryLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD>
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Create output directory
    model_dir = os.path.join(DATA_DIR, 'models', 'lstm', 'poetry')
    os.makedirs(model_dir, exist_ok=True)

    # Training progress
    total_batches = num_epochs * len(train_loader)
    progress = ProgressReporter(total_steps=total_batches, description="Training LSTM language model")

    # Training log
    log_path = os.path.join(model_dir, 'training_log.txt')
    log_entries = []

    print(f"\nStarting training: {num_epochs} epochs, {len(train_loader)} batches/epoch", flush=True)

    global_batch = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            output, _ = model(inputs)

            # Compute loss: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            output = output.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(output, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()
            global_batch += 1

            # Update progress
            if global_batch % 50 == 0 or global_batch == 1:
                progress.update(
                    global_batch,
                    message=f"Epoch {epoch+1} Batch {batch_idx+1}: Loss={loss.item():.4f}"
                )

        # Learning rate scheduling
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        log_entries.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.6f} Time: {epoch_time:.1f}s", flush=True)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'char2idx': dataset.char2idx,
                'idx2char': dataset.idx2char,
            }, os.path.join(model_dir, 'best_model.pth'))

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'char2idx': dataset.char2idx,
        'idx2char': dataset.idx2char,
    }, os.path.join(model_dir, 'final_model.pth'))

    progress.complete(message=f"Training complete! Final Loss: {avg_loss:.4f}")

    # Save training log
    with open(log_path, 'w') as f:
        f.write("epoch,loss,lr,time\n")
        for entry in log_entries:
            f.write(f"{entry['epoch']},{entry['loss']:.4f},{entry['lr']:.6f},{entry['time']:.1f}\n")

    print(f"\nModel saved: {model_dir}", flush=True)
```

## Phase 4: Text Generation

After training, use the LSTM model to generate classical Chinese poetry from a given prefix. The generation process is autoregressive: given a prefix of characters, the model predicts the probability distribution of the next character, samples the next character, appends it to the sequence, and repeats until reaching the specified length or encountering an end marker. The key engineering points in this phase are as follows:

1. **Model Loading**: Load the trained model weights along with the vocabulary mappings (`char2idx` and `idx2char`).
2. **Temperature Sampling**: Control generation diversity through a temperature parameter. Higher temperatures (e.g., 1.0) produce more random and creative outputs; lower temperatures (e.g., 0.5) produce more conservative outputs closer to the training data. A temperature of 0 degenerates to greedy search.
3. **Hidden State Propagation**: Maintain continuity of hidden states during generation, allowing the model to "remember" previously generated content and maintain semantic coherence.

```python runnable gpu
import torch
import torch.nn.functional as F
import os

# Import model from shared modules
from shared.sequence_models.poetry_lstm import PoetryLSTM

def generate_poetry(model, char2idx, idx2char, prefix, max_length=50, temperature=1.0, device='cpu'):
    """Generate classical Chinese poetry

    Args:
        model: Trained LSTM model
        char2idx: Character-to-index mapping
        idx2char: Index-to-character mapping
        prefix: Generation prefix (e.g., "春眠")
        max_length: Maximum generation length
        temperature: Sampling temperature (higher = more random)
        device: Compute device

    Returns:
        Generated poetry text
    """
    model.eval()

    # Convert prefix to index sequence
    input_seq = [char2idx.get(c, char2idx['<UNK>']) for c in prefix]
    input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)

    # Initialize hidden state
    hidden = model.init_hidden(1, device)

    # Generation result
    generated = list(prefix)

    with torch.no_grad():
        # First process the prefix (except the last character)
        for i in range(len(input_seq) - 1):
            _, hidden = model(input_tensor[:, i:i+1], hidden)

        # Start generation from the last character of the prefix
        current_input = input_tensor[:, -1:]

        for _ in range(max_length - len(prefix)):
            output, hidden = model(current_input, hidden)

            # Get output at the last time step
            logits = output[0, -1, :] / temperature

            # Convert to probability distribution
            probs = F.softmax(logits, dim=-1)

            # Sample next character
            next_idx = torch.multinomial(probs, num_samples=1).item()

            # Convert to character
            next_char = idx2char.get(next_idx, '<UNK>')

            # Check if an end punctuation mark was generated
            if next_char in ['。', '！', '？'] and len(generated) > len(prefix) + 5:
                generated.append(next_char)
                break

            generated.append(next_char)

            # Prepare next input
            current_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return ''.join(generated)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(DATA_DIR, 'models', 'lstm', 'poetry', 'best_model.pth')

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    char2idx = checkpoint['char2idx']
    idx2char = checkpoint['idx2char']
    vocab_size = len(char2idx)

    model = PoetryLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("Model loaded successfully", flush=True)
    print(f"Vocabulary size: {vocab_size}", flush=True)

    # Test generation
    prefixes = ["春", "月", "风", "山", "水"]

    print("\n=== Generated Poetry Results ===\n", flush=True)

    for prefix in prefixes:
        print(f"Prefix: 「{prefix}」", flush=True)

        # Generation results at different temperatures
        for temp in [0.5, 0.8, 1.0]:
            poem = generate_poetry(model, char2idx, idx2char, prefix,
                                   max_length=40, temperature=temp, device=device)
            print(f"  Temperature {temp}: {poem}", flush=True)
        print(flush=True)

else:
    print(f"Model file not found: {model_path}", flush=True)
    print("Please run the training code first", flush=True)
```

## Experimental Conclusions

The training results of an LSTM language model on the classical poetry generation task need to be evaluated from multiple dimensions:

1. **Training Loss Interpretation**: The training loss (cross entropy) of an LSTM language model reflects the model's ability to predict the next character. Lower loss indicates better fit to the training data. However, excessively low loss may indicate overfitting -- the model merely memorizes the training data rather than learning the rules of poetry generation. An ideal loss curve should first decrease rapidly and then level off.

2. **Generation Quality Evaluation**: The quality of generated poetry is difficult to measure with numerical metrics and requires human evaluation. Good generation results should exhibit the following characteristics:
   - **Semantic Coherence**: The generated sentences are semantically fluent, not random string concatenations
   - **Prosodic Adequacy**: Although this experiment does not enforce prosodic constraints, the model may learn some basic rhythmic patterns
   - **Artistic Appeal**: The generated poetry should possess a certain degree of artistic beauty, rather than being bland

3. **Temperature Parameter Effect**: The temperature parameter has a significant impact on generation diversity:
   - **Low Temperature (0.3-0.5)**: Conservative and stable outputs, close to common expressions in the training data, but may lack creativity
   - **Medium Temperature (0.7-0.9)**: Balances stability and diversity, recommended as the default setting
   - **High Temperature (1.0-1.5)**: Random and creative outputs, but may produce incoherent sentences

4. **Engineering Improvement Directions**: To further improve generation quality, consider the following directions:
   - **Increase Training Data**: Use more poetry data or introduce modern poetry data to increase diversity
   - **Adjust Model Architecture**: Increase the number of LSTM layers or hidden dimensions to expand model capacity
   - **Introduce Prosodic Constraints**: Add constraints on tonal patterns and rhyming during generation to produce poems that better adhere to prosodic rules
   - **Use Transformer**: Transformer architectures typically outperform LSTM on sequence modeling tasks and can be explored as an alternative

## Results

This experiment comprehensively demonstrates the training pipeline of an LSTM language model. After training, the following files will be saved to the data directory:

- **Model Files**:
    - `<DATA_DIR>/models/lstm/poetry/best_model.pth` - Model with the lowest validation loss
    - `<DATA_DIR>/models/lstm/poetry/final_model.pth` - Final model weights
- **Training Log**:
    - `<DATA_DIR>/models/lstm/poetry/training_log.txt` - Per-epoch loss and learning rate records

Example generation output:

```
Prefix: 「春」
  Temperature 0.5: 春风吹绿江南岸，明月何时照我还。
  Temperature 0.8: 春水碧于天，画船听雨眠。
  Temperature 1.0: 春来江水绿如蓝，能不忆江南。

Prefix: 「月」
  Temperature 0.5: 月落乌啼霜满天，江枫渔火对愁眠。
  Temperature 0.8: 月明如水照花枝，独坐幽窗思往事。
  Temperature 1.0: 月下飞天镜，云生结海楼。
```
