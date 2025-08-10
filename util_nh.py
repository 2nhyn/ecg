#!/usr/bin/env python

import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# preprocessor
class SignalProcessor:
    def __init__(self, target_fs=400, bp_low=0.5, bp_high=45.0, notch_freq=60.0):
        self.target_fs = target_fs
        self.bp_lowcut = bp_low
        self.bp_highcut = bp_high
        self.notch_freq = notch_freq
        self.notch_q = 30.0

    def preprocess(self, signal: np.ndarray, fs: int) -> np.ndarray:
        if fs != self.target_fs:
            signal = resample_poly(signal, self.target_fs, fs)
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)
        
        # Bandpass
        nyq = 0.5 * self.target_fs
        b, a = butter(4, [self.bp_lowcut/nyq, self.bp_highcut/nyq], btype='band')
        signal = filtfilt(b, a, signal)
        
        # Notch
        b, a = iirnotch(self.notch_freq, self.notch_q, self.target_fs)
        signal = filtfilt(b, a, signal)
        return signal

def normalize_leads(arr: np.ndarray) -> np.ndarray:
    arr_norm = arr.copy()
    for i in range(arr.shape[1]):
        mean = np.mean(arr[:, i])
        std = np.std(arr[:, i])
        if std < 1e-6:
            std = 1.0
        arr_norm[:, i] = (arr[:, i] - mean) / std
    return arr_norm


# padding
def pad_signals(signals, max_len):
    num_samples = len(signals)
    if num_samples == 0:
        return np.array([], dtype=np.float32)
    num_leads = signals[0].shape[1]
    padded_signals = np.zeros((num_samples, max_len, num_leads), dtype=np.float32)
    for i, s in enumerate(signals):
        seq_len = s.shape[0]
        padded_signals[i, :seq_len, :] = s[:min(seq_len, max_len), :]
    return padded_signals


# Data Augmentation
def augment_signal_v2(x):
    # Gaussian noise
    x = x + torch.randn_like(x) * 0.025
    # Random scaling
    scale = torch.empty(1).uniform_(0.9, 1.1).to(x.device)
    x = x * scale
    # Time masking
    t = x.size(1)
    mask_len = int(t * 0.1)
    start = torch.randint(0, t - mask_len, (1,)).item()
    x[:, start:start + mask_len] = 0
    # Random shift
    shift = torch.randint(-5, 6, (1,)).item()
    if shift > 0:
        pad = torch.zeros(x.size(0), shift, device=x.device)
        x = torch.cat([x[:, shift:], pad], dim=1)
    elif shift < 0:
        pad = torch.zeros(x.size(0), -shift, device=x.device)
        x = torch.cat([pad, x[:, :shift]], dim=1)
    return x

# dataset class
class ContrastiveECGDataset(Dataset):
    def __init__(self, signals):
        self.signals = signals
    def __len__(self): return len(self.signals)
    def __getitem__(self, idx):
        sig = self.signals[idx]
        v1 = augment_signal_v2(sig)
        v2 = augment_signal_v2(sig)
        return v1, v2

class SupervisedECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = torch.from_numpy(labels).long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.signals[idx], self.labels[idx]


# ===========================
# --- 모델 아키텍처 정의 ---
# ===========================

# 1. SE Block
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), 
                                nn.Conv1d(channels, channels // reduction, 1), 
                                nn.ReLU(inplace=True), 
                                nn.Conv1d(channels // reduction, channels, 1), 
                                nn.Sigmoid())
        
    def forward(self, x): 
        return x * self.fc(x)


# 2. Transformer Block
class TransformerBlock1D(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    def forward(self, x):
        res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = res + attn_out
        res2 = x
        x = self.mlp(self.norm2(res2))
        return res2 + x

# 3. Basic Block
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 7, stride, 3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 7, 1, 3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.se    = SEBlock1D(out_c)
        self.down  = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)   
        if self.down:
            identity = self.down(x)
        out += identity
        return self.relu(out)


# 4. ResNet block -> simple ver
class ResNet1DEncoder(nn.Module):
    def __init__(self, in_ch, emb_dim=128, layers=[2,2,2,2], dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.layer1 = self._make_layer(64, 64, layers[0])
        self.layer2 = self._make_layer(64, 128, layers[1], 2)
        self.layer3 = self._make_layer(128, 256, layers[2], 2)
        self.layer4 = self._make_layer(256, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.transformer = TransformerBlock1D(512)
        
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim)
        )
        
    def _make_layer(self, in_c, out_c, blocks, stride=1):
        down = None
        if stride!=1 or in_c!=out_c:
            down = nn.Sequential(
                nn.Conv1d(in_c, out_c,1,stride,bias=False),
                nn.BatchNorm1d(out_c)
            )
        layers = [BasicBlock1D(in_c, out_c, stride, down)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_c, out_c))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        x = x.unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        z = self.proj(x)
        return x, F.normalize(z, dim=1)


# =================================
# Loss  :  NTXent Loss
# =================================
class NTXentLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], 0)
        sim = torch.matmul(z, z.T) / self.temp  # similarity matrix
        mask = ~torch.eye(2*B, device=sim.device).bool()
        pos = torch.cat([sim.diag(B), sim.diag(-B)], 0).unsqueeze(1)
        neg = sim[mask].view(2*B, -1)
        logits = torch.cat([pos, neg], 1)
        labels = torch.zeros(2*B, dtype=torch.long, device=sim.device)
        return self.ce(logits, labels)
    


# ========================
# Classifier
# =======================

class LinearProbeHead(nn.Module):
    """ 2-Layer MLP 분류기 헤드 """
    def __init__(self, in_dim=512, num_classes=2, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.mlp(x)

class LinearProbeTrainer:
    def __init__(self, encoder, signals_tensor, labels_array, save_path, batch_size=64, lr=1e-4, device='cpu', verbose=False):
        self.device = device
        self.encoder = encoder.to(self.device)
        self.encoder.eval() 
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = LinearProbeHead(in_dim=512, num_classes=len(np.unique(labels_array))).to(self.device)
        self.save_path = save_path
        self.verbose = verbose

        # data loader
        dataset = SupervisedECGDataset(signals_tensor, labels_array)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # class weight
        class_weights = torch.tensor([0.5, 1.0], dtype=torch.float32).to(self.device)
        
        self.opt = optim.Adam(self.head.parameters(), lr=lr)
        self.crit = nn.CrossEntropyLoss(weight=class_weights)

    def train(self, epochs=5):
        if self.verbose:
            try:
                from tqdm import tqdm
                epoch_iterator = tqdm(range(epochs), desc="Fine-tuning Head")
            except ImportError:
                epoch_iterator = range(epochs)
        else:
            epoch_iterator = range(epochs)

        for e in epoch_iterator:
            self.head.train()
            total_loss = 0
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                with torch.no_grad():
                    h, _ = self.encoder(xb)
                logits = self.head(h)
                loss = self.crit(logits, yb)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.loader)
            
            if self.verbose:
                print(f'[Fine-tune] Epoch {e+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        # final model
        torch.save(self.head.state_dict(), self.save_path)
        if self.verbose:
            print(f"   -> Model saved after {epochs} epochs to {self.save_path}")