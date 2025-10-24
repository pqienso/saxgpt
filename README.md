# SaxGPT: Audio Encoder-Decoder Transformer

A multi-stream encoder-decoder transformer trained to generate saxophone solos on jazz backing tracks. SaxGPT uses neural audio tokenization and a custom architecture to learn the relationship between rhythm sections and saxophone improvisations.

## Overview

SaxGPT takes a jazz backing track as input and generates an accompanying saxophone solo. The model uses:
- **Encodec** for audio tokenization (4 parallel token streams at 50Hz)
- **Encoder-decoder transformer** with KV-caching for efficient generation
- **Multi-codebook embeddings** with delay pattern interleaving for better convergence

### Key Features
- Custom dataset creation from YouTube jazz recordings
- Source separation using Demucs to isolate saxophone and rhythm tracks
- Multi-stream token generation with delayed codebook patterns
- Mixed precision training with gradient accumulation
- Full KV-caching support for fast inference

## Architecture

The model consists of:
- **Encoder**: Processes tokenized backing track audio (rhythm section)
- **Decoder**: Generates tokenized saxophone audio autoregressively
- **4 parallel token streams** per audio (from Encodec), each with vocab size 2048
- **Delay pattern interleaving**: Each codebook starts at a different position for better convergence

```
Input (Backing Track) → Encoder → Memory
                                    ↓
Start Tokens → Decoder (with KV cache) → 4 Logit Heads → Saxophone Tokens → Audio
```

## Installation

### Requirements
- Python 3.9+ (for Demucs)
- Python 3.x (for training)
- CUDA-compatible GPU recommended
- FFmpeg (for audio processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pqienso/SaxGPT
cd SaxGPT
```

2. Create virtual environments for data processing:
```bash
bash scripts/build_venvs.sh
```

This creates two separate environments:
- `.venv/demucs` - For stem separation with Demucs
- `.venv/encodec` - For audio tokenization with Encodec


## Dataset Creation

### 1. Configure Your Data

Create or modify a config file in `config/data/`:

```yaml
url: 'https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID'

data_paths:
  dl_dest: 'data/your_dataset/download/'
  stem_dest: 'data/your_dataset/stems/'
  metadata_path: 'data/your_dataset/metadata.csv'
  codes_dest: 'data/your_dataset/codes.pt'
  datasets_dest: 'data/your_dataset/datasets/'

intermediates: # keep / discard intermediate outputs
  keep_dl: true
  keep_stems: true
  keep_clips: true
  keep_aug_clips: true
  keep_codes: true

demucs:
  n_splits: 2          # Higher = higher quality target, slower
  n_shifts: 2          # Averaging shifts for equivariance
  n_jobs: 16           # Parallel jobs
  normalize_before: true
  normalize_after: true

rms_window:
  rms_threshold: 0.05  # Minimum RMS energy for valid sections
  min_window_size_seconds: 30.0
  rms_frame_length_seconds: 10.0
  rms_stride_seconds: 0.2

augmentation: null
# augmentation:
#   sample_rate: 32000
#   semitone_steps: [-2, 2] # Shift audio by these semitones
#   tempo_ratios: [0.9, 1.1] # Speed up / slow down audio
#   n_fft: 4096
#   hop_length: 512

encodec:
  chunk_len_s: 25      # Process audio in chunks to manage memory

train_test_split:
  seed: 67
  test: 0.1            # 10% test set
  val: 0.1             # 10% validation set

dataset:
  seq_len: 1500        # 30 seconds at 50Hz
  stride: 150          # 10x overlap for augmentation
  padding_idx: 2048
```

### 2. Build the Dataset

```bash
bash scripts/build_dataset.sh --config config/data/your_config.yaml [--cuda]
```

This pipeline:
1. Downloads audio from YouTube
2. Separates stems using Demucs (sax vs rhythm section)
3. Creates metadata with valid audio windows
4. Clips audio to obtain relevant parts
5. Augments audio (if applicable)
6. Tokenizes audio using Encodec
7. Splits into train/val/test sets

**Note**: The `--cuda` flag enables GPU acceleration for stem separation and tokenization.

## Training

### 1. Configure Training

Create or modify a config file in `config/model/`:

```yaml
data:
  train_path: "data/your_dataset/datasets/train.pt"
  val_path: "data/your_dataset/datasets/val.pt"

model:
  vocab_size: 2049
  num_codebooks: 4
  d_model: 256
  nhead: 16
  num_encoder_layers: 16
  num_decoder_layers: 16
  dim_feedforward: 1024
  dropout: 0.1
  activation: "relu"
  norm_first: false
  max_seq_len: 1505
  padding_idx: 2048
  scale_embeddings: true

training:
  num_epochs: 1000
  batch_size: 2
  gradient_accumulation_steps: 16  # Effective batch size = 32
  max_grad_norm: 1.0
  
  scheduler:
    type: "cosine"
    warmup_steps: 1000
    min_lr: 1.0e-6
  
  optimizer:
    type: "adamw"
    lr: 0.0001
    betas: [0.9, 0.98]
    eps: 1.0e-9
    weight_decay: 0.01
  
  output_dir: "models/your_model"
  log_interval: 2
  save_interval: 50
  num_workers: 8
  
  resume_from_checkpoint: null  # Or path to checkpoint
```

### 2. Start Training

```bash
python -m src.training.train --config config/model/your_config.yaml
```

Training features:
- Mixed precision (FP16) with gradient scaling
- Gradient accumulation for large effective batch sizes
- Automatic checkpointing (best model + periodic saves)
- Interrupt recovery (Ctrl+C saves checkpoint)
- JSONL metrics logging for real-time monitoring

### 3. Monitor Training

**Text monitoring:**
```bash
python -m src.training.monitor_metrics --config config/model/your_config.yaml --watch
```

**Live plotting:**
```bash
python -m src.training.monitor_metrics --config config/model/your_config.yaml --live
```

**Static plot:**
```bash
python -m src.training.monitor_metrics \
    --config config/model/your_config.yaml \
    --plot \
    --output training_progress.png
```

### 4. Resume Training

If training is interrupted:
```bash
# The checkpoint path is automatically saved in config
python -m src.training.train --config config/model/your_config.yaml
```

## Project Structure

```
SaxGPT/
├── config/
│   ├── data/              # Data pipeline configurations
│   └── model/             # Model training configurations
├── src/
│   ├── data/
│   │   ├── pipeline/      # 7-step data processing pipeline
│   │   └── util/          # Audio processing utilities
│   ├── model/             # Transformer architecture
│   └── training/          # Training and evaluation
├── requirements/
│   ├── demucs.txt        # For stem splitting (Python 3.9)
│   └── encodec.txt       # For training / tokenization (Python 3.10+)
└── scripts/
    ├── build_venvs.sh
    └── run_data_pipeline.sh
```

## Model Sizes

Pre-configured model sizes:

| Model | Parameters | d_model | Layers | Config |
|-------|-----------|---------|--------|--------|
| Small | 4.8M | 128 | 6/6 | `config/model/small.yaml` |
| Medium | ~30M | 256 | 16/16 | `config/model/medium.yaml` |


## Technical Details

### Audio Tokenization
- **Sample rate**: 32kHz
- **Token rate**: 50Hz (20ms frames)
- **Codebooks**: 4 parallel streams
- **Vocabulary**: 2048 tokens per codebook (+ 1 padding token)
- **Delay pattern**: CB0 starts at t=1, CB1 at t=2, CB2 at t=3, CB3 at t=4

### Attention Mechanism
- Flash Attention 2 for memory efficiency
- KV-caching for fast autoregressive generation
- Causal masking for decoder self-attention
- Proper handling of padding masks

### Training Optimizations
- Mixed precision (FP16) training
- Gradient accumulation for larger batch sizes
- Gradient clipping (default: 1.0)
- Learning Rate scheduling:
    - Cosine learning rate schedule with warmup
    - Plateau learning rate schedule
    - Linear learning rate schedule
- Early stopping on validation loss

## Known Issues & Limitations

1. **Data requirements**: Model requires substantial training data (100+ hours recommended)
2. **Overfitting**: Small datasets lead to overfitting; regularization techniques help but are not a complete solution
3. **Audio quality**: Generated audio quality depends on Encodec compression artifacts.
4. **Data Leakage**: Some bleeding of source audio into target audio due to poor separation with 6-stem Demucs model
4. **Computational cost**: Training requires GPU with ≥16GB VRAM for larger models

## References

- **MusicGen**: [Copet et al., 2023](https://arxiv.org/pdf/2306.05284) - Delay pattern and multi-codebook inspiration
- **Encodec**: [Défossez et al., 2022](https://arxiv.org/pdf/2210.13438) - Neural audio codec
- **Demucs**: [Rouard et al., 2022](https://arxiv.org/abs/2211.08553)
