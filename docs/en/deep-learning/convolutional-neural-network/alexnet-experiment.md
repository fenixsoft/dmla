# AlexNet Reproduction Experiment

In this hands-on engineering training, we will use PyTorch to reproduce the complete AlexNet training pipeline, from data preparation to inference. Through practice, we will understand how classic CNN architectures integrate with modern deep learning frameworks and discuss engineering trade-offs involving performance, robustness, environmental constraints, and resource consumption.

## Experiment Preparation

Before starting the experiment, make sure you have [mounted the data directory](../../appendixes/sandbox.md#data-management) and downloaded the Tiny ImageNet 200 dataset. You can automate this using the `DMLA-CLI` tool:

```bash
# Select "Download Dataset" -> Select "Tiny ImageNet 200"
dmla data
```

Verify that the dataset has been downloaded correctly and check its structure. Tiny ImageNet 200 contains 200 categories with a total of 110,000 images. Before training, confirm that the dataset is fully downloaded and the directory structure is correct; otherwise, the DataLoader will throw errors because it cannot find the files.

```python runnable gpu
import os

# Check if the data directory exists
data_dir = os.path.join(DATA_DIR, 'datasets', 'tiny-imagenet-200')

if os.path.exists(data_dir):
    print("Dataset directory already exists")
    
    # Check subdirectory structure
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if os.path.exists(train_dir):
        train_classes = os.listdir(train_dir)
        print(f"Number of training classes: {len(train_classes)}")
        print(f"Example classes: {train_classes[:5]}")
    
    if os.path.exists(val_dir):
        val_files = os.listdir(val_dir)
        print(f"Number of validation files: {len(val_files)}")
else:
    print("Dataset not downloaded. Please run 'dmla data' to download the dataset")
```

## Phase 1: Data Preprocessing

First, we create a PyTorch DataLoader for image preprocessing and data augmentation.

The original AlexNet used the ImageNet 1K dataset during its competition, containing approximately 1.2 million training images, 50,000 validation images, and 100,000 test images. This scale is substantial, but still very limited relative to the 60 million model parameters and 1000-class classification output. Therefore, data augmentation techniques such as random flipping, cropping, and color jittering are applied during preprocessing to artificially increase training data diversity and prevent overfitting.

For this experiment, we use the [Tiny ImageNet 200](https://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset. "Tiny" means the images are downscaled to 64x64 JPEG format, and "200" means the dataset contains 200 categories. The only augmentation we apply during data preprocessing is bilinear interpolation to upscale images to 224x224, matching the original AlexNet network's input size.

The data preprocessing code leverages two commonly used PyTorch components: `Dataset` and `DataLoader`. `Dataset` maps image files and labels on disk to `(image, label)` pairs, while `DataLoader` handles batch loading, shuffling, and multi-threaded reading.

From an academic perspective, data preprocessing is essentially about padding, denoising, and normalization. However, from an engineering perspective, data preprocessing has a huge impact on both training effectiveness and efficiency. Consider the preprocessing cache in this experiment as an example: without any caching, real-time preprocessing would require 100,000 file accesses, JPEG decodes, and resize operations per epoch, producing massive redundant computation. On the other hand, caching the full result is not necessarily appropriate either. A raw 64x64 JPEG compressed image is about 2KB, but converting it to a 224x224 FP32 tensor (3 x 224 x 224 x 4 bytes ~ 600KB) expands the data by roughly 300 times, pushing total data volume beyond 60GB -- requiring workstation-grade hardware for in-memory storage, or imposing heavy I/O burdens if stored on disk.

::: warning This is how AlexNet actually trained in 2012
In 2012, AlexNet trained the full ImageNet dataset using two GTX 580 GPUs (each with 3GB of VRAM). The hardware limitations of the era meant only real-time preprocessing was feasible. Operations such as Resize, Clip, and Normalize had to be performed on the CPU. According to the AlexNet team, GPU utilization was only about 10%, and a single training run took 5 days to complete.
:::

Regarding the preprocessing cache scenario, the core engineering decisions for this experiment are as follows:

- The two most expensive operations are JPEG decoding and Resize, both of which expand the data. The Resize from 64 to 224 involves a several-fold expansion, while converting JPEG to FP32 tensors involves a hundred-fold expansion. Therefore, we decided to save the resized result as JPEG without decoding, eliminating one heavyweight operation. This expands the dataset from roughly 250MB to about 1GB (estimated at Quality=95).
- Use LMDB (Lightning Memory-Mapped Database) instead of the filesystem to store preprocessing results. LMDB maps data files directly into the process's virtual address space via mmap, achieving zero-copy reads and significantly improving I/O efficiency.
- JPEG decoding uses the nvJPEG operator from NVIDIA's DALI library, moving decoding to the GPU to avoid repeated data transfers between VRAM and system memory (not applicable on Windows).
- Use multi-threaded DataLoader (`num_workers=4`) with batched operations to eliminate I/O bottlenecks and improve processing efficiency (not applicable on Windows).

Under this scheme, memory consumption is around 4GB, and the preprocessing output is 2.3GB (LMDB allocated storage: 2GB for training set + 256MB for validation set). The final data preprocessing code is as follows:

```python runnable gpuonly timeout=unlimited extract-class="LMDBPreprocessCache"
import os
import io
import json
import lmdb
import struct
from PIL import Image
import time

# Import progress reporter module
from dmla_progress import ProgressReporter

# Data and cache directories (DATA_DIR is automatically injected by the kernel)
# Docker mode: DATA_DIR='/data', Native mode: DATA_DIR='~/dmla-data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'datasets', 'tiny-imagenet-200')
CACHE_DIR = os.path.join(DATA_DIR, 'cache', 'preprocessing', 'tiny-imagenet-224-lmdb')

class LMDBPreprocessCache:
    """
    LMDB cache strategy: store preprocessing results in an LMDB database
    
    Advantages:
    - Single large file avoids random I/O from many small files
    - Memory-mapped (mmap), zero-copy reads
    - Multi-process friendly (lock-free reads)
    
    Data structure:
    - Key: image index (uint64, 8 bytes)
    - Value: label (int32, 4 bytes) + JPEG bytes
    """
    def __init__(self, data_dir, cache_dir, map_size=2*1024*1024*1024):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.map_size = map_size  # LMDB maximum capacity (2GB, sufficient for training set JPEG data)
        self.train_lmdb_path = os.path.join(cache_dir, 'train.lmdb')
        self.val_lmdb_path = os.path.join(cache_dir, 'val.lmdb')
        self.manifest_path = os.path.join(cache_dir, 'manifest.json')
        
    def preprocess_image(self, img_path):
        """Preprocess a single image: Resize(224) -> JPEG bytes"""
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=95)
        return buf.getvalue()
    
    def check_cache_exists(self):
        """Check if the cache already exists and is complete"""
        return os.path.exists(self.manifest_path)
    
    def get_cache_stats(self):
        """Get cache statistics"""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            return manifest.get('train_count', 0), manifest.get('val_count', 0)
        return 0, 0
    
    def _preprocess_train_set(self, progress):
        """Preprocess training set into LMDB"""
        train_dir = os.path.join(self.data_dir, 'train')
        classes = sorted(os.listdir(train_dir))
        
        # Load class mapping
        wnids_path = os.path.join(self.data_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create LMDB environment
        env = lmdb.open(
            self.train_lmdb_path,
            map_size=self.map_size,
            writemap=True,
            lock=True
        )
        total_count = 0
        idx = 0
        with env.begin(write=True) as txn:
            for cls_idx, cls in enumerate(classes):
                images_dir = os.path.join(train_dir, cls, 'images')
                if not os.path.exists(images_dir):
                    continue
                label = class_to_idx.get(cls, cls_idx)
                for img_name in os.listdir(images_dir):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(images_dir, img_name)
                        try:
                            jpeg_bytes = self.preprocess_image(img_path)
                            # Storage format: key=idx(uint64), value=label(int32) + JPEG bytes
                            key = struct.pack('>Q', idx)
                            value = struct.pack('>i', label) + jpeg_bytes
                            txn.put(key, value)
                            idx += 1
                            total_count += 1
                        except Exception as e:
                            print(f"Warning: Failed to process {img_path}: {e}")
                progress.update(cls_idx + 1, message=f"Preprocessing class {cls_idx+1}/200: {cls}")
        env.close()
        return total_count
    
    def _preprocess_val_set(self, progress):
        """Preprocess validation set into LMDB"""
        val_dir = os.path.join(self.data_dir, 'val')
        val_images_dir = os.path.join(val_dir, 'images')
        val_annotations = os.path.join(val_dir, 'val_annotations.txt')
        
        # Load class mapping
        wnids_path = os.path.join(self.data_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        
        # Load annotation file
        with open(val_annotations, 'r') as f:
            val_lines = f.readlines()
        total_val = len(val_lines)
        
        # Reset progress bar for validation set processing
        progress.reset(total_steps=total_val, description="Preprocessing validation set")
        
        # Create LMDB environment (smaller map_size for validation set)
        env = lmdb.open(
            self.val_lmdb_path,
            map_size=256*1024*1024,  # 256MB (validation set ~60MB)
            writemap=True,
            lock=True
        )
        labels = []
        idx = 0
        with env.begin(write=True) as txn:
            for line_idx, line in enumerate(val_lines):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    img_path = os.path.join(val_images_dir, img_name)
                    label = class_to_idx.get(parts[1], 0)
                    if os.path.exists(img_path):
                        try:
                            jpeg_bytes = self.preprocess_image(img_path)
                            key = struct.pack('>Q', idx)
                            value = struct.pack('>i', label) + jpeg_bytes
                            txn.put(key, value)
                            labels.append(label)
                            idx += 1
                        except Exception as e:
                            print(f"Warning: Failed to process {img_path}: {e}")
                    if (line_idx + 1) % 100 == 0 or line_idx == total_val - 1:
                        progress.update(line_idx + 1, message=f"Preprocessing validation set {line_idx+1}/{total_val}")
        env.close()
        return idx, labels
    
    def run(self, progress):
        """Execute preprocessing"""
        start_time = time.time()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        train_count = self._preprocess_train_set(progress)
        val_count, val_labels = self._preprocess_val_set(progress)
        
        # Save manifest file
        manifest = {
            'train_count': train_count,
            'val_count': val_count,
            'val_labels': val_labels,
            'format': 'lmdb',
            'key_format': 'uint64',
            'value_format': 'int32_label + jpeg_bytes'
        }
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        elapsed = time.time() - start_time
        progress.complete(message=f"Preprocessing complete: {train_count} training images, {val_count} validation images, elapsed {elapsed:.1f}s")
        
        return train_count, val_count

preprocessor = LMDBPreprocessCache(RAW_DATA_DIR, CACHE_DIR)

if preprocessor.check_cache_exists():
    train_count, val_count = preprocessor.get_cache_stats()
    
    progress = ProgressReporter(total_steps=1, description="Preprocessing phase")
    progress.update(1, message=f"LMDB cache already exists, skipping preprocessing! Training set {train_count} images, Validation set {val_count} images")
    progress.complete(message="Preprocessing phase complete (LMDB cache already exists)")
    
    print(f"LMDB cache already exists, skipping preprocessing")
    print(f"Training set: {train_count} images (train.lmdb)")
    print(f"Validation set: {val_count} images (val.lmdb)")
else:
    if not os.path.exists(RAW_DATA_DIR):
        print("Error: Dataset not downloaded. Please run 'dmla data' to download the dataset")
    else:
        progress = ProgressReporter(total_steps=200, description="Preprocessing training set")
        train_count, val_count = preprocessor.run(progress)
        print(f"Preprocessing complete: {train_count} training images, {val_count} validation images")
```

## Phase 2: Model Definition

The following short code implements the [AlexNet network architecture](alexnet.md#network-architecture). Except for adapting the output classification layer to Tiny ImageNet's 200 categories and using adaptive pooling instead of fixed-size adaptation, the rest of the network definition remains consistent with the original AlexNet, while the code size is significantly reduced. As the content demonstrates, common components of neural network models -- such as convolutional layers, pooling layers, activation functions, and Dropout regularization -- are all available as standard components in modern machine learning frameworks. The difficulty of building models lies in rational design and efficient training; translating design into implementation through programming is not hard.

1. `features` (feature extraction layers): 5 convolutional layers stacked alternately, progressively extracting features from low-level (edges, textures) to high-level (object parts). `MaxPool2d` between convolutional layers handles downsampling, gradually reducing spatial dimensions. `AdaptiveAvgPool2d((6, 6))` ensures the output is always fixed at 6x6 regardless of the input image's spatial size after the preceding convolutions and pooling
2. `classifier` (classification layers): 3 fully connected layers. The first two layers use `Dropout(p=0.5)` to randomly drop 50% of neuron activations, preventing overfitting -- this is a hallmark design of AlexNet. The final layer maps 4096-dimensional features to a 200-class Softmax classifier
3. **Output changed from 1000 to 200 classes:** The original AlexNet's final layer outputs 1000 classes (for the full ImageNet), while Tiny ImageNet has only 200 classes, so `num_classes=200`

```python runnable gpuonly extract-class="AlexNet"
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet network architecture
    Adapted for Tiny ImageNet 200-class classification
    
    The original AlexNet targets 1000 classes; the last layer is modified to 200 classes here
    AdaptiveAvgPool2d ensures the output size is always fixed at 6x6
    """
    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()
        
        # Feature extraction layers (5 convolutional layers)
        self.features = nn.Sequential(
            # Conv1: 11x11 convolution, stride 4, output 96 channels
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 5x5 convolution, output 256 channels
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 3x3 convolution, output 384 channels
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 3x3 convolution, output 384 channels
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 3x3 convolution, output 256 channels
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Adaptive pooling, ensures fixed output of 6x6
            nn.AdaptiveAvgPool2d((6, 6))
        )
        
        # Classification layers (3 fully connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Create model instance
model = AlexNet(num_classes=200)

# Print model structure
print("AlexNet Model Structure:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test forward pass
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
```

## Phase 3: Model Training

Model training is the most central part of deep learning. Through the backpropagation algorithm, network parameters are continuously adjusted, enabling the model to gradually learn feature extraction and classification from images. The training pipeline consists of three key steps: forward pass to compute predictions and loss, backward pass to compute gradients, and optimizer to update parameters. This process seems simple, but involves many engineering efficiency considerations. Training one epoch requires processing 100,000 images, with the main bottleneck being I/O and CPU-based JPEG decoding. Without optimization, each image takes 5-10ms for I/O, making a single epoch over ten minutes. The engineering decisions in this phase focus on eliminating data loading I/O bottlenecks, reducing per-epoch time to under 2 minutes through the following optimizations:

- **JPEG decoding location:** Decoding is a CPU-intensive operation. Single-threaded PIL decoding takes about 1-3ms per image. Using the nvJPEG operator from NVIDIA's DALI library moves decoding to the GPU, but in a Windows Docker environment, GPU nvJPEG is unavailable due to NVML limitations, so DALI's CPU multi-threaded decoding is used instead (still faster than single-threaded).
- **Data augmentation location:** Operations like random flipping, cropping, and normalization, if executed on the CPU, incur additional CPU-GPU data transfer overhead. DALI moves all these operations to the GPU, where data flows entirely within GPU VRAM without needing to transfer back to the CPU.
- **LMDB zero-copy reads:** Preprocessing results were stored in LMDB during Phase 2. In this phase, JPEG bytes are read directly via memory mapping, avoiding additional file I/O operations.
- **Environment-adaptive design:** The host operating system is detected by reading `/proc/version`. Windows automatically switches to CPU multi-threaded decoding mode for compatibility, while Linux uses GPU nvJPEG decoding mode for maximum efficiency.

::: info Training Estimate

The training set has 100,000 images (200 classes, 500 per class). Running 20 epochs requires approximately 8GB of VRAM, with GPU training taking about 25-30 minutes.

:::

```python runnable gpuonly timeout=unlimited
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import struct
import numpy as np

# Import progress reporter module
from dmla_progress import ProgressReporter

# Import DALI
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# Import AlexNet from shared module
from shared.cnn.alexnet import AlexNet

# LMDB cache directory (DATA_DIR is automatically injected by the kernel)
LMDB_DIR = os.path.join(DATA_DIR, 'cache', 'preprocessing', 'tiny-imagenet-224-lmdb')

def detect_host_os():
    """Detect the host operating system"""
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                return 'windows'
            return 'linux'
    except:
        return 'linux'

class DALILMDBReader:
    """
    DALI External Source - LMDB JPEG Reader
    
    Reads JPEG bytes from the LMDB database for use by the DALI Pipeline
    """
    def __init__(self, lmdb_path, batch_size, shuffle=True):
        import lmdb
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get data count
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
        
        self.indices = np.arange(self.length)
        self._reset()
    
    def _reset(self):
        """Reset the iterator"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._position = 0
    
    def __call__(self):
        """DALI external_source requires a callable that returns one batch per call"""
        if self._position >= self.length:
            self._reset()
            return None, None
        
        batch_jpegs = []
        batch_labels = []
        end_idx = min(self._position + self.batch_size, self.length)
        with self.env.begin() as txn:
            for i in range(self._position, end_idx):
                idx = self.indices[i]
                key = struct.pack('>Q', idx)
                value = txn.get(key)
                if value is not None:
                    label = struct.unpack('>i', value[:4])[0]
                    jpeg_bytes = np.frombuffer(value[4:], dtype=np.uint8)
                    batch_jpegs.append(jpeg_bytes)
                    batch_labels.append(label)
        self._position = end_idx
        return batch_jpegs, np.array(batch_labels, dtype=np.int32)
    
    def __len__(self):
        return self.length

@pipeline_def
def create_train_pipeline(data_source, decode_device='cpu'):
    """
    DALI training Pipeline
    
    decode_device:
    - 'cpu': Windows Docker (NVML limitation, use CPU multi-threaded decoding)
    - 'mixed': Linux Docker (GPU nvJPEG decoding)
    """
    jpegs, labels = fn.external_source(
        source=data_source,
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
        batch=True
    )
    
    # JPEG decoding
    images = fn.decoders.image(
        jpegs,
        device=decode_device,
        output_type=types.RGB
    )
    
    # If CPU decoded, transfer to GPU
    if decode_device == 'cpu':
        images = images.gpu()
    
    # GPU data augmentation + Normalize
    images = fn.crop_mirror_normalize(
        images,
        device='gpu',
        dtype=types.FLOAT,
        output_layout='CHW',
        crop=(224, 224),
        mirror=fn.random.coin_flip(probability=0.5),  # Random horizontal flip
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )
    
    labels = labels.gpu()
    
    return images, labels

@pipeline_def
def create_val_pipeline(data_source, decode_device='cpu'):
    """DALI validation Pipeline (no data augmentation)"""
    jpegs, labels = fn.external_source(
        source=data_source,
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
        batch=True
    )
    
    images = fn.decoders.image(
        jpegs,
        device=decode_device,
        output_type=types.RGB
    )
    
    if decode_device == 'cpu':
        images = images.gpu()
    
    images = fn.crop_mirror_normalize(
        images.gpu(),
        device='gpu',
        dtype=types.FLOAT,
        output_layout='CHW',
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )
    
    labels = labels.gpu()
    
    return images, labels

# Main training code
progress = ProgressReporter(total_steps=100, description="Preparing training environment")
progress.update(0, message="Detecting runtime environment...")
host_os = detect_host_os()
decode_device = 'cpu' if host_os == 'windows' else 'mixed'
print(f"[Environment] Host OS: {host_os.upper()}")
print(f"[Environment] DALI decode device: {decode_device}")
if host_os == 'windows':
    print("[Environment] Windows Docker: CPU multi-threaded JPEG decoding")
else:
    print("[Environment] Linux Docker: GPU nvJPEG decoding")

# Check LMDB cache
progress.update(5, message="Checking LMDB cache...")
manifest_path = os.path.join(LMDB_DIR, 'manifest.json')
train_lmdb_path = os.path.join(LMDB_DIR, 'train.lmdb')
val_lmdb_path = os.path.join(LMDB_DIR, 'val.lmdb')

if not os.path.exists(manifest_path) or not os.path.exists(train_lmdb_path):
    print("Error: LMDB cache does not exist. Please run the Phase 2 preprocessing code first")
    progress.error(message="LMDB cache does not exist")
else:
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    print(f"LMDB cache exists: {manifest['train_count']} training images, {manifest['val_count']} validation images")

# Detect GPU
progress.update(10, message="Detecting GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
    device_id = torch.cuda.current_device()
else:
    print("Warning: No GPU detected. DALI requires a GPU")
    device_id = 0

# Create DALI Pipeline
progress.update(20, message="Creating DALI Pipeline...")
batch_size = 128
train_reader = DALILMDBReader(train_lmdb_path, batch_size, shuffle=True)
val_reader = DALILMDBReader(val_lmdb_path, batch_size, shuffle=False)

train_pipe = create_train_pipeline(
    data_source=train_reader,
    decode_device=decode_device,
    batch_size=batch_size,
    num_threads=4,
    device_id=device_id
)
val_pipe = create_val_pipeline(
    data_source=val_reader,
    decode_device=decode_device,
    batch_size=batch_size,
    num_threads=4,
    device_id=device_id
)

train_pipe.build()
val_pipe.build()

print(f"DALI Pipeline created ({host_os} mode)")
print(f"Training set: {len(train_reader)} images, {len(train_reader) // batch_size} batches per epoch")

# Create model
progress.update(50, message="Creating AlexNet model...")
model = AlexNet(num_classes=200).to(device)
print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

progress.update(60, message="Training environment ready")

# Create performance log (DATA_DIR is automatically injected by the kernel)
perf_log_path = os.path.join(DATA_DIR, 'models', 'alexnet', 'performance_log.txt')
os.makedirs(os.path.join(DATA_DIR, 'models', 'alexnet'), exist_ok=True)
perf_log = open(perf_log_path, 'w')
perf_log.write("batch_idx,decode_ms,transfer_ms,forward_ms,backward_ms,optimizer_ms,total_ms\n")

# Switch to training progress
total_batches = len(train_reader) // batch_size
num_epochs = 20
progress.reset(total_steps=num_epochs * total_batches, description=f"Training AlexNet (DALI {host_os})")
best_acc = 0.0

print(f"Starting training: {num_epochs} epochs, {total_batches} batches per epoch")

# Training function
def train_one_epoch_dali(model, train_reader, train_pipe, criterion, optimizer, device, perf_log, start_batch_idx=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_reader._reset()
    
    for batch_idx in range(total_batches):
        pipe_start = time.time()
        outputs = train_pipe.run()
        decode_time = time.time() - pipe_start
        batch_start = time.time()
        
        # Get PyTorch tensors from DALI TensorList
        images = outputs[0].as_tensor()
        labels = outputs[1].as_tensor()
        inputs = torch.from_dlpack(images)
        targets = torch.from_dlpack(labels).long()
        
        # Forward
        forward_start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        forward_time = time.time() - forward_start
        
        # Backward
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Optimizer
        optimizer_start = time.time()
        optimizer.step()
        optimizer_time = time.time() - optimizer_start
        total_time = time.time() - batch_start
        perf_log.write(f"{batch_idx},{decode_time*1000:.1f},0,{forward_time*1000:.1f},{backward_time*1000:.1f},{optimizer_time*1000:.1f},{total_time*1000:.1f}\n")
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update global progress (accumulate start_batch_idx)
        global_batch_idx = start_batch_idx + batch_idx
        if batch_idx % 50 == 0:
            progress.update(global_batch_idx, message=f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{total_batches}")
    return running_loss / total_batches, 100. * correct / total

# Validation function
def validate_dali(model, val_reader, val_pipe, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    val_reader._reset()
    val_batches = len(val_reader) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(val_batches):
            outputs = val_pipe.run()
            images = outputs[0].as_tensor()
            labels = outputs[1].as_tensor()
            inputs = torch.from_dlpack(images)
            targets = torch.from_dlpack(labels).long()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / val_batches, 100. * correct / total

try:
    global_batch_count = 0  # Global batch counter
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch_dali(model, train_reader, train_pipe, criterion, optimizer, device, perf_log, start_batch_idx=global_batch_count)
        global_batch_count += total_batches  # Accumulate completed batch count
        val_loss, val_acc = validate_dali(model, val_reader, val_pipe, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% Time: {epoch_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_dir = os.path.join(DATA_DIR, 'models', 'alexnet', 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> Saved best model (accuracy: {best_acc:.2f}%)")
        
        # Save checkpoint every 4 epochs
        if (epoch + 1) % 4 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, f'epoch_{epoch+1}.pth'))
            print(f"  -> Saved epoch {epoch+1} checkpoint")
    
    progress.complete(message=f"Training complete! Best accuracy: {best_acc:.2f}%")
    
    perf_log.close()
    print(f"\nPerformance log saved: {perf_log_path}")
    
    final_dir = os.path.join(DATA_DIR, 'models', 'alexnet', 'final')
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, 'alexnet_tiny_imagenet.pth'))
    print(f"Final model saved: {os.path.join(final_dir, 'alexnet_tiny_imagenet.pth')}")
    
except Exception as e:
    perf_log.close()
    progress.error(message=f"Training error: {str(e)}")
    print(f"\nTraining error: {e}")
    print(f"\nPerformance log saved: {perf_log_path}")
    raise
```

## Phase 4: Inference and Evaluation

Use the trained model to perform classification predictions on new images. After training completes, verify the model's actual classification performance, demonstrating what the model has "learned." Key design points:

1. **Model loading:** Prioritize loading the checkpoint with the best validation accuracy (`best_model.pth`), then fall back to the final model. If neither is found, use an untrained random-weight model (for testing only; predictions will be meaningless).
2. **Inference preprocessing:** Same as validation set preprocessing (Resize → ToTensor → Normalize), without additional data augmentation. The input image preprocessing must be consistent with training.
3. **Class name mapping:** Tiny ImageNet class labels are WordNet IDs (e.g., `n01675725`), mapped to human-readable English descriptions (e.g., `turtle, tortoise`) via `wnids.txt` and `words.txt`.
4. **Image prediction:** Evaluate Top-5 error rate results. The basic logic is: read image → preprocess → feed into model → use `softmax` to convert logits to probabilities (0-100%) → use `topk(5)` to get the 5 highest-probability classes, outputting Top-5 predictions. Top-5 is the default evaluation metric for ILSVRC image classification: as long as the correct answer is among the top 5 predictions, the model is considered to have correctly classified the image.

```python runnable gpuonly
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import random

# Import AlexNet from shared module
from shared.cnn.alexnet import AlexNet

# Load trained model (DATA_DIR is automatically injected by the kernel)
model_path = os.path.join(DATA_DIR, 'models', 'alexnet', 'final', 'alexnet_tiny_imagenet.pth')
checkpoint_path = os.path.join(DATA_DIR, 'models', 'alexnet', 'checkpoints', 'best_model.pth')

# Choose loading path
best_acc = None
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = AlexNet(num_classes=200)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_acc = checkpoint['best_acc']
    print(f"Loaded best model (Epoch {checkpoint['epoch']}, Accuracy {best_acc:.2f}%)")
elif os.path.exists(model_path):
    model = AlexNet(num_classes=200)
    model.load_state_dict(torch.load(model_path))
    print("Loaded final model")
else:
    print("No trained model found. Using untrained model (predictions will be random)")
    model = AlexNet(num_classes=200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load class names (from wnids.txt)
wnids_path = os.path.join(DATA_DIR, 'datasets', 'tiny-imagenet-200', 'wnids.txt')
words_path = os.path.join(DATA_DIR, 'datasets', 'tiny-imagenet-200', 'words.txt')
val_annotations_path = os.path.join(DATA_DIR, 'datasets', 'tiny-imagenet-200', 'val', 'val_annotations.txt')

class_names = {}
val_labels = {}  # Ground truth labels for validation set images
wnids = []

if os.path.exists(wnids_path) and os.path.exists(words_path):
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    
    with open(words_path, 'r') as f:
        word_lines = f.readlines()
        for line in word_lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                class_names[parts[0]] = parts[1]

# Load validation set ground truth labels
if os.path.exists(val_annotations_path):
    with open(val_annotations_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                val_labels[parts[0]] = parts[1]  # img_name -> wnid

def predict_image(image_path, model, transform, device, class_names, wnids):
    """Predict a single image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_idx = probabilities.topk(5)
    
    results = []
    for i in range(5):
        idx = top5_idx[0][i].item()
        prob = top5_prob[0][i].item() * 100
        
        if idx < len(wnids):
            wnid = wnids[idx]
            name = class_names.get(wnid, wnid)
        else:
            name = f"Class {idx}"
        
        results.append((name, prob, wnid if idx < len(wnids) else None))
    
    return results, image

# Test with images from the validation set
val_images_dir = os.path.join(DATA_DIR, 'datasets', 'tiny-imagenet-200', 'val', 'images')

if os.path.exists(val_images_dir):
    all_images = os.listdir(val_images_dir)
    test_images = random.sample(all_images, min(5, len(all_images)))  # Randomly select 5 images
    
    print("\nPrediction Examples:")
    print("=" * 60)
    
    for img_name in test_images:
        img_path = os.path.join(val_images_dir, img_name)
        
        if os.path.exists(img_path):
            predictions, original_image = predict_image(img_path, model, transform, device, class_names, wnids)
            
            # Get ground truth label
            true_wnid = val_labels.get(img_name, None)
            true_name = class_names.get(true_wnid, true_wnid) if true_wnid else "Unknown"
            
            # Check if prediction is correct (Top-5)
            predicted_wnids = [p[2] for p in predictions]
            is_correct = true_wnid in predicted_wnids if true_wnid else False
            top1_correct = predictions[0][2] == true_wnid if true_wnid else False
            
            # Display image
            display(original_image)
            
            # Output prediction results
            status = "Top-1 Correct" if top1_correct else ("Top-5 Correct" if is_correct else "Incorrect")
            print(f"Image: {img_name} ({status})")
            print(f"Ground Truth: {true_name}")
            print("Top-5 Predictions:")
            for rank, (name, prob, wnid) in enumerate(predictions, 1):
                marker = " [GT]" if wnid == true_wnid else ""
                print(f"  {rank}. {name}: {prob:.2f}%{marker}")
            print()
else:
    print("Validation set directory does not exist, cannot perform inference test")

print("=" * 60)
if best_acc is not None:
    print(f"Model Accuracy: {best_acc:.2f}% (Top-1)")
```

## Experimental Conclusions

The original AlexNet achieved a Top-5 error rate of 15.3% (i.e., Top-5 accuracy of approximately 84.7%) in the ILSVRC competition (7-model ensemble), with a single-model Top-5 error rate of 18.2% (i.e., Top-5 accuracy of approximately 81.8%). In this experiment, after training on the Tiny ImageNet 200 dataset, the validation set Top-1 accuracy is approximately 45%. This gap should be understood from the following dimensions:

1. **Metric Differences:** First, we need to clarify the difference in metric definitions. ILSVRC evaluates using **Top-5 accuracy**, meaning the prediction is considered correct as long as the correct class is among the 5 highest-probability predictions. However, this experiment reports **Top-1 accuracy**, which requires the highest-probability prediction to be exactly the correct class.

    - Original AlexNet single model: Top-5 error rate 18.2% → Top-5 accuracy 81.8%, corresponding Top-1 accuracy approximately 59.3%
    - This experiment: Top-1 accuracy approximately 45%, corresponding Top-5 accuracy approximately 65-70%

    Therefore, when converting to the same metric (Top-1 accuracy), the actual gap narrows from "81.8% vs 65% (Top-5)" to "59.3% vs 45%", a difference of about 14 percentage points.

2. **Training Dataset Scale and Quality Differences:** This is the direct cause of the approximately 14 percentage point gap. The comparison between the two training datasets is as follows:

    | Comparison | ImageNet 1K (Original) | Tiny ImageNet 200 (This Experiment) |
    |-----------|----------------------|-----------------------------------|
    | Training set size | 1.2 million images | 100,000 images (12x difference) |
    | Original image size | Ranges from $256 \times 256$ to $500 \times 500$ | Uniform $64 \times 64$ JPEG |
    | Input size | $224 \times 224$ (cropped from larger images) | $224 \times 224$ (upscaled from $64 \times 64$) |

    Tiny ImageNet images are produced by compressing original ImageNet images down to $64 \times 64$. Upscaling from $64 \times 64$ to $224 \times 224$ leads to:
    - **Information loss:** The compression process loses high-frequency details (textures, edge sharpness).
    - **Interpolation blur:** The upscaling process cannot recover lost information but instead introduces interpolation artifacts.
    - **Feature extraction difficulty:** The CNN must learn features from blurry images, greatly increasing the difficulty.

    Moreover, the $64 \times 64$ training set severely limits data augmentation options. The original AlexNet randomly cropped $224 \times 224$ regions from $256 \times 256$ images, yielding approximately 33 different samples per image. With Tiny ImageNet already at $64 \times 64$, the diversity of random crops after upscaling is very limited. The original AlexNet's PCA color augmentation worked well on high-quality large images but could be detrimental to low-quality small images like Tiny ImageNet, since images already suffering from significant information loss would have their limited texture details further degraded by color perturbations. Therefore, this experiment only retains random horizontal flipping, which is the most stable and effective augmentation for small images. Other training configuration differences are as follows:

    | Configuration | Original AlexNet | This Experiment |
    |--------------|------------------|-----------------|
    | Epochs | 90 | 20 |
    | Data augmentation | Horizontal flip + random crop ($224 \times 224$ from $256 \times 256$) + PCA color perturbation | Horizontal flip only |
    | Learning rate schedule | Manual: 0.01→0.001→0.0001 (epochs 30, 60) | StepLR: 0.01→0.001 (epoch 10) |
    | Dropout | p=0.5 | p=0.5 (same) |

The accuracy achieved in this experiment is a reasonable expectation given the constraints. To further improve accuracy, consider using a larger dataset, such as the original [ImageNet 1K](https://ieeexplore.ieee.org/document/5206848) (approximately 150 GB) or [Mini ImageNet 100](https://modelscope.cn/datasets/tany0699/mini_imagenet100) (approximately 6.4 GB), which would allow for more data augmentation techniques and more training epochs. Alternatively, consider a network architecture better suited for small images (such as a simplified CNN designed for 64x64 inputs). The goal of this experiment is to understand the integration of classic CNN architectures with modern deep learning frameworks through a complete reproduction of the AlexNet training pipeline. Considering practical feasibility for readers, we did not pursue competition-level accuracy. Interested readers may choose one of the above paths (changing the dataset or changing the network architecture) as an exercise for this section.

## Execution Results

After model inference, 5 images are randomly selected from the validation set for classification prediction and compared against the validation set labels. An example of actual execution output is shown below:

| Image | Prediction | Image | Prediction |
|-------|-----------|-------|-----------|
| ![exp1](../../../deep-learning/convolutional-neural-network/assets/exp1.png) | Image: val_7656.JPEG (Top-5 Correct)<br>Ground Truth: cougar, puma, catamount, mountain lion, painter, panther, Felis concolor<br>Top-5 Predictions:<br>  1. orangutan, orang, orangutang, Pongo pygmaeus: 39.19%<br>  2. cougar, puma, catamount, mountain lion, painter, panther, Felis concolor: 8.53% [GT]<br>  3. lion, king of beasts, Panthera leo: 8.47%<br>  4. baboon: 4.30%<br>  5. lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens: 4.03% | ![exp2](../../../deep-learning/convolutional-neural-network/assets/exp2.png) | Image: val_9447.JPEG (Incorrect)<br>Ground Truth: binoculars, field glasses, opera glasses<br>Top-5 Predictions:<br>  1. snorkel: 22.78%<br>  2. miniskirt, mini: 7.39%<br>  3. standard poodle: 7.39%<br>  4. military uniform: 4.82%<br>  5. pole: 4.76% |
|![exp3](../../../deep-learning/convolutional-neural-network/assets/exp3.png)|Image: val_6564.JPEG (Top-1 Correct)<br>Ground Truth: king penguin, Aptenodytes patagonica<br>Top-5 Predictions:<br>  1. king penguin, Aptenodytes patagonica: 99.92% [GT]<br>  2. syringe: 0.02%<br>  3. lemon: 0.01%<br>  4. mantis, mantid: 0.01%<br>  5. projectile, missile: 0.01%<br>|![exp4](../../../deep-learning/convolutional-neural-network/assets/exp4.png)|Image: val_4249.JPEG (Top-5 Correct)<br>Ground Truth: potpie<br>Top-5 Predictions:<br>  1. mashed potato: 61.36%<br>  2. cauliflower: 12.87%<br>  3. ice cream, icecream: 5.28%<br>  4. potpie: 3.55% [GT]<br>  5. guacamole: 2.90%
|![exp5](../../../deep-learning/convolutional-neural-network/assets/exp5.png)|Image: val_141.JPEG (Top-5 Correct)<br>Ground Truth: snail<br>Top-5 Predictions:<br>  1. wooden spoon: 7.24%<br>  2. meat loaf, meatloaf: 6.32%<br>  3. snail: 5.56% [GT]<br>  4. rocking chair, rocker: 4.51%<br>  5. pretzel: 4.34%| | |


This experiment demonstrates the complete AlexNet training pipeline. After training completes, the following generated files are saved to the data directory:

- **Model files:**
    - `<DATA_DIR>/models/alexnet/checkpoints/best_model.pth` - Model with best validation accuracy
    - `<DATA_DIR>/models/alexnet/checkpoints/epoch_*.pth` - Checkpoints every 4 epochs
    - `<DATA_DIR>/models/alexnet/final/alexnet_tiny_imagenet.pth` - Final model weights
- **Preprocessing cache:**
    - `<DATA_DIR>/cache/preprocessing/tiny-imagenet-224-lmdb/train.lmdb/` - Training set LMDB database (approximately 2GB)
    - `<DATA_DIR>/cache/preprocessing/tiny-imagenet-224-lmdb/val.lmdb/` - Validation set LMDB database (approximately 300MB)
    - `<DATA_DIR>/cache/preprocessing/tiny-imagenet-224-lmdb/manifest.json` - Cache manifest (count, format description)
- **Performance log:**
    - `<DATA_DIR>/models/alexnet/performance_log.txt` - Detailed timing log (for bottleneck analysis)
