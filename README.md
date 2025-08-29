# ECG Contrastive + Linear Probe Classifier

## 1) Overview

This project takes **12-lead ECG** signals as input and follows a two-stage pipeline:

1. **Contrastive Pretraining** (self-supervised learning)
2. **Linear Probe Supervised Fine-tuning**

to perform binary classification.

* Input length: variable length → unified by padding/clipping (4096)
* Sampling rate: resampled to **400 Hz**
* Core model: `ResNet1D + SE Block + Transformer Block` encoder, trained contrastively and then fine-tuned with a linear probe classifier

All hyperparameters are decided empirically
---

## 2) Preprocessing

### `SignalProcessor`

* Resample (→ 400 Hz)
* Replace NaNs with finite values
* **Bandpass filter**: 0.5–45 Hz (4th-order Butterworth)
* **Notch filter**: 60 Hz powerline noise removal

### Normalization

```python
def normalize_leads(arr):
    (arr - mean) / std  # per-lead normalization
```

### Padding

* Pad all samples to shape `(max_len, num_leads)`

---

## 3) Data Augmentation (`augment_signal_v2`)

* Add Gaussian noise
* Random scaling (0.9 \~ 1.1)
* Time masking (mask 10% of the signal with zeros)
* Random temporal shift (±5 samples)

Used to generate two augmented views (v1, v2) for contrastive learning.

---

## 4) Dataset Classes

* **ContrastiveECGDataset**:
  Takes one ECG signal and applies `augment_signal_v2` twice → returns (v1, v2).

* **SupervisedECGDataset**:
  Returns (signal, label) pairs → used for linear probe training.

---

## 5) Model Architecture

### (1) SEBlock1D

* **Squeeze-and-Excitation** module (channel-wise attention)
* Global Average Pooling → Conv1d(reduction) → ReLU → Conv1d → Sigmoid

### (2) TransformerBlock1D

* Multihead Self-Attention (`nn.MultiheadAttention`)
* LayerNorm + residual connections
* Position-wise MLP (Linear → GELU → Linear)

### (3) BasicBlock1D

* ResNet-style 1D convolutional block
* Conv-BN-ReLU → Conv-BN → SEBlock → Residual connection

### (4) ResNet1DEncoder

* Stem: Conv(7x1, stride=2) + BN + ReLU + MaxPool
* Layer1–4: ResNet blocks (64 → 128 → 256 → 512)
* Global Avg Pooling → TransformerBlock → Projection Head

Outputs:

* `x`: encoder feature (512-d)
* `z`: projection head embedding (contrastive learning, normalized)

---

## 6) Contrastive Loss: NT-Xent

* Input: `z1`, `z2` (augmentation pair)
* Positive pairs: different views of the same sample
* Negative pairs: all other samples in the batch
* Softmax + CrossEntropy-based loss

---

## 7) Linear Probe Classifier

### LinearProbeHead

* 2-layer MLP:

  * Input: encoder feature (512-d)
  * Hidden: 128-d
  * Output: `num_classes` (default=2)

### LinearProbeTrainer

* Encoder is **frozen**; only the linear head is trained
* Optimizer: Adam (`lr=1e-4`)
* Loss: Weighted CrossEntropy

  * `class_weights = [0.5, 1.0]` (to address class imbalance)
* Model saving: `torch.save(self.head.state_dict(), save_path)`

---

## 8) Training Flow

### Phase 1: Contrastive Pretraining

1. Apply augmentations to ECG signal → create (v1, v2) pairs
2. Pass through encoder + projection head
3. Train using NT-Xent loss

### Phase 2: Linear Probe Fine-tuning

1. Freeze encoder
2. Load labeled ECG data via `SupervisedECGDataset`
3. Train only the linear probe head with CrossEntropyLoss

---

## 9) Key Design Choices

* **ResNet + SEBlock + Transformer**: combines local pattern extraction, channel attention, and global dependencies
* Contrastive pretraining improves generalization on limited/noisy labels
* Linear probe allows efficient use of labels
* Design considers ECG-specific challenges (noise, variability, varying sequence lengths)

---

## 10) Minimal Example

```python
# Pretrain contrastive
dataset = ContrastiveECGDataset(signals)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
encoder = ResNet1DEncoder(in_ch=12)

# Fine-tune with Linear Probe
trainer = LinearProbeTrainer(
    encoder, signals_tensor, labels_array, save_path="linear_probe.pt",
    batch_size=64, lr=1e-4, device="cuda"
)
trainer.train(epochs=5)
```