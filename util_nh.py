#!/usr/bin/env python

import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader,  WeightedRandomSampler
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Tuple
from scipy.signal import butter, filtfilt, resample_poly, iirnotch
import os, sys, math, json, glob
from torch.cuda.amp import GradScaler, autocast
from helper_code import *
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
   
# =========================
# Full ECG Training Module
# =========================
# - Robust preprocessing (resample, notch/bandpass, median/MAD normalize)
# - ECG-aware augmentations (two views for contrastive)
# - Datasets (contrastive / supervised / multi-source with confidence)
# - Encoders:
#     * ResNet1DEncoder (원본 호환 유지)
#     * EnhancedECGEncoder (LeadGate + True Temporal Attention + Multi-Scale + Residual LSTM)
# - Heads (classification + contrastive projection)
# - Losses: NT-Xent (원본), SupCon (confidence-weighted)
# - Trainers: LinearProbeTrainer (원본 보강), JointContrastiveClassifierTrainer (멀티태스크)
# ===============================================================

# ---------------------------------
# 1) Preprocessor & Normalization
# ---------------------------------
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
        signal = filtfilt(b, a, signal, axis=0)
        
        # Notch
        b, a = iirnotch(self.notch_freq, self.notch_q, self.target_fs)
        signal = filtfilt(b, a, signal, axis=0)
        return signal

def normalize_leads(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Median/MAD normalization per lead.
    Input/Output: (T, C)
    """
    if arr.ndim != 2:
        raise ValueError("arr must be (T, C)")
    med = np.median(arr, axis=0, keepdims=True)
    mad = np.median(np.abs(arr - med), axis=0, keepdims=True)
    scale = 1.4826 * mad
    scale[scale < eps] = 1.0
    return (arr - med) / scale


def pad_signals(signals, max_len):
    """
    signals: list of (T, C) → returns (N, max_len, C) float32
    """
    num_samples = len(signals)
    if num_samples == 0:
        return np.array([], dtype=np.float32)
    num_leads = signals[0].shape[1]
    padded = np.zeros((num_samples, max_len, num_leads), dtype=np.float32)
    for i, s in enumerate(signals):
        L = min(s.shape[0], max_len)
        padded[i, :L, :] = s[:L, :]
    return padded

# ============================
# dataloader
# ============================

def load_all_sources(data_folder: str, verbose: bool = False):
    ref_channels = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
    proc = SignalProcessor(target_fs=400, bp_low=0.5, bp_high=45.0)

    ptb_sig, ptb_lab = [], []
    sami_sig, sami_lab = [], []
    code_sig, code_lab, code_conf = [], [], []

    records = find_records(data_folder, file_extension='.hea')
    for rel in tqdm(records, desc="Preprocessing ECG signals"):
        rec_path = os.path.join(data_folder, rel)
        base = os.path.basename(rel)
        try:
            header = load_header(rec_path)
            n_sig = get_num_signals(header)
            if n_sig is None or n_sig < 2:
                if verbose: print(f"[skip] {base}: not a multi-lead ECG")
                continue

            # WFDB 신호 로드
            sig, fields = load_signals(rec_path)             # (T, C_raw)
            fs = float(fields.get('fs', get_sampling_frequency(header) or 400.0))
            ch_names = fields.get('sig_name', get_signal_names(header) or [])
            sig = reorder_signal(sig, ch_names, ref_channels)  # (T, C≤12)
            if sig.shape[1] != 12:
                if verbose: print(f"[skip] {base}: not 12-lead ({sig.shape[1]})")
                continue

            # 전처리 + 정규화
            sig = proc.preprocess(sig, fs=int(fs))
            sig = normalize_leads(sig).astype(np.float32, copy=False)

            # 소스/라벨
            src = (get_source(header) or "").casefold()
            try:
                lab = int(get_label(header, allow_missing=False))
            except Exception:
                if verbose: print(f"[skip] {base}: missing label")
                continue

            if "ptb" in src:
                ptb_sig.append(sig);  ptb_lab.append(lab)
            elif ("sami" in src) or ("trop" in src):
                sami_sig.append(sig); sami_lab.append(lab)
            elif ("code" in src):
                code_sig.append(sig); code_lab.append(lab); code_conf.append(1.0)
            else:
                low = rel.casefold()
                if "ptb" in low:
                    ptb_sig.append(sig);  ptb_lab.append(lab)
                elif ("sami" in low) or ("trop" in low):
                    sami_sig.append(sig); sami_lab.append(lab)
                else:
                    code_sig.append(sig); code_lab.append(lab); code_conf.append(1.0)

        except Exception as e:
            if verbose: print(f"[skip] {base} error: {e}")
            continue

    return (
        (ptb_sig, np.asarray(ptb_lab, dtype=int)),
        (sami_sig, np.asarray(sami_lab, dtype=int)),
        (code_sig, np.asarray(code_lab, dtype=int), np.asarray(code_conf, dtype=float))
    )

from torch.utils.data._utils.collate import default_collate

def _to_hard_tensor(x):
    # 항상 새/연속 스토리지로 강제
    if torch.is_tensor(x):
        return x.detach().cpu().contiguous().clone()
    elif isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x)      # 연속화
        return torch.tensor(arr)           # 이 버전에선 기본이 'copy' 동작
    else:
        return torch.tensor(x)  

def collate_hard(batch):
    b0 = batch[0]
    if isinstance(b0, (list, tuple)) and len(b0) == 3:
        xs, ys, metas = zip(*batch)
        xs = torch.stack([_to_hard_tensor(x) for x in xs], dim=0)  # 직접 stack
        ys = default_collate(ys)
        return xs, ys, list(metas)
    fixed = [tuple(_to_hard_tensor(e) for e in ex) for ex in batch]
    return default_collate(fixed)


def collate_safe(batch):
    def _fix(x):
        if torch.is_tensor(x):
            return x.contiguous().clone()
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, copy=True)
        return x
    b0 = batch[0]
    if isinstance(b0, (list, tuple)) and len(b0) == 3:
        xs, ys, metas = zip(*batch)
        xs = default_collate([_fix(x) for x in xs])
        ys = default_collate(ys)
        return xs, ys, list(metas)
    fixed = [tuple(_fix(e) for e in ex) for ex in batch]
    return default_collate(fixed)

# ---------------------------------
# 2) ECG-aware Augmentations
# ---------------------------------

def _time_shift(x: torch.Tensor, max_shift: int = 8) -> torch.Tensor:
    # x: (C, T)
    T = x.size(1)
    shift = int(torch.randint(low=-max_shift, high=max_shift + 1, size=(1,)))
    if shift == 0:
        return x
    if shift > 0:
        pad = torch.zeros(x.size(0), shift, device=x.device, dtype=x.dtype)
        return torch.cat([x[:, shift:], pad], dim=1)
    else:
        pad = torch.zeros(x.size(0), -shift, device=x.device, dtype=x.dtype)
        return torch.cat([pad, x[:, :shift]], dim=1)


def _baseline_wander(x: torch.Tensor, fs: int = 400) -> torch.Tensor:
    # Add/remove low-freq sinusoidal baseline
    C, T = x.shape
    t = torch.arange(T, device=x.device, dtype=x.dtype) / fs
    freq = torch.empty(1, device=x.device).uniform_(0.2, 0.5).item()
    amp  = torch.empty(1, device=x.device).uniform_(0.02, 0.06).item()
    wander = amp * torch.sin(2 * math.pi * freq * t)  # (T,)
    return x + wander.unsqueeze(0)


def _bandstop_hum(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    # Simulate mild 50/60Hz interference by additive narrowband sine (random phase).
    if torch.rand(1).item() > p:
        return x
    C, T = x.shape
    fs = 400.0
    f = 50.0 if torch.rand(1).item() < 0.5 else 60.0
    t = torch.arange(T, device=x.device, dtype=x.dtype) / fs
    phase = torch.rand(1, device=x.device) * 2 * math.pi
    hum = 0.01 * torch.sin(2 * math.pi * f * t + phase)  # small amplitude
    return x + hum.unsqueeze(0)


def _lead_dropout(x: torch.Tensor, drop_prob: float = 0.15) -> torch.Tensor:
    # Randomly attenuate or drop 1~2 leads
    C, T = x.shape
    if torch.rand(1).item() > drop_prob:
        return x
    n_drop = 1 if torch.rand(1).item() < 0.7 else 2
    idx = torch.randperm(C)[:n_drop]
    scale = 0.0 if torch.rand(1).item() < 0.5 else float(torch.empty(1).uniform_(0.2, 0.5))
    x = x.clone()
    x[idx, :] = x[idx, :] * scale
    return x


def _gaussian_noise(x: torch.Tensor, std_range: Tuple[float, float] = (0.01, 0.03)) -> torch.Tensor:
    std = float(torch.empty(1).uniform_(*std_range))
    return x + torch.randn_like(x) * std


def augment_signal(x: torch.Tensor) -> torch.Tensor:
    """
    ECG-safe view. x: (C, T). Keep morphology but add realistic artifacts.
    """
    x = _gaussian_noise(x)
    x = _time_shift(x, max_shift=4)
    x = _baseline_wander(x, fs=400)
    x = _bandstop_hum(x, p=0.5)
    x = _lead_dropout(x, drop_prob=0.15)
    # mild amplitude jitter
    scale = float(torch.empty(1).uniform_(0.9, 1.1))
    x = x * scale
    return x

# score_code15_batch 등에서 사용하는 약~중간 강도 증강
def augment_signal_v2(x: torch.Tensor) -> torch.Tensor:
    # x: (C, T)
    x = x + torch.randn_like(x) * 0.025
    scale = torch.empty(1).uniform_(0.9, 1.1).to(x.device)
    x = x * scale
    t = x.size(1)
    mask_len = int(t * 0.1)
    if mask_len > 0 and t - mask_len > 0:
        start = torch.randint(0, t - mask_len, (1,)).item()
        x[:, start:start + mask_len] = 0
    return _time_shift(x, max_shift=5)


# ---------------------------------
# 3) Datasets
# ---------------------------------

class ContrastiveECGDataset(Dataset):
    """
    Unlabeled or mixed data; returns two augmented views for contrastive.
    signals: torch.Tensor of shape (N, C, T) or list of (C, T)
    """
    def __init__(self, signals):
        self.signals = signals

    def __len__(self): 
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx]  # (C, T)
        if isinstance(sig, np.ndarray):
            sig = torch.from_numpy(np.ascontiguousarray(sig.T)).float()
        assert sig.dim() == 2, "signal must be (C, T)"
        aug = augment_signal
        v1 = aug(sig)
        v2 = aug(sig)
        return v1, v2


class SupervisedECGDataset(Dataset):
    """
    Labeled data for classification training/eval.
    signals: torch.Tensor (N, C, T) or list of (C, T)
    labels : np.ndarray or torch.LongTensor of shape (N,)
    """
    def __init__(self, signals, labels):
        self.signals = signals
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.signals[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x.T)).float()
        return x, self.labels[idx]


class MultiSourceECGDataset(Dataset):
    """
    Labeled data with per-sample confidence (for weak labels like CODE-15).
    signals: (N, C, T)
    labels : (N,)
    conf   : (N,) in [0,1]
    aug_for_train: whether to create two views for joint contrastive loss
    """
    def __init__(self, signals, labels, confidences=None, aug_for_train: bool = True):
        self.signals = signals
        self.labels  = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()
        if confidences is None:
            self.confidences = torch.ones_like(self.labels, dtype=torch.float32)
        else:
            self.confidences = torch.from_numpy(confidences).float() if isinstance(confidences, np.ndarray) else confidences.float()
        self.aug = aug_for_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.signals[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x.T)).float()
        y = self.labels[idx]
        c = self.confidences[idx]
        if self.aug:
            v1 = augment_signal(x)
            v2 = augment_signal(x)
            return v1, v2, y, c
        else:
            return x, y, c


# ---------------------------------
# 4) Building Blocks & Encoders
# ---------------------------------
class MultiScaleConv1D(nn.Module):
    """Multi-scale convolution to capture different ECG wave components."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Ensure divisibility by 3
        assert out_channels % 3 == 0, "out_channels must be divisible by 3"
        k_small, k_medium, k_large = 3, 7, 15
        self.conv_small  = nn.Conv1d(in_channels, out_channels // 3, kernel_size=k_small, padding=k_small // 2)
        self.conv_medium = nn.Conv1d(in_channels, out_channels // 3, kernel_size=k_medium, padding=k_medium // 2)
        self.conv_large  = nn.Conv1d(in_channels, out_channels // 3, kernel_size=k_large, padding=k_large // 2)
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        x1 = self.conv_small(x)
        x2 = self.conv_medium(x)
        x3 = self.conv_large(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return F.relu(self.bn(x))


class LeadGate(nn.Module):
    """Learnable per-lead gating at the raw input stage."""
    def __init__(self, num_leads=12):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_leads, 1))
    def forward(self, x):
        # x: (B, C=12, T)
        return x * self.alpha


class TrueTemporalAttention(nn.Module):
    """Time-wise attention mask via 1D conv over channel-averaged trace."""
    def __init__(self, kernel_size=9):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: (B, C, T)
        t_score = x.mean(dim=1, keepdim=True)   # (B, 1, T)
        mask = self.sigmoid(self.conv(t_score)) # (B, 1, T)
        return x * mask


class ResidualLSTMBlock(nn.Module):
    """BiLSTM with residual connection (keeps (B, C, T) interface)."""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, input_dim)
        self.norm = nn.LayerNorm(input_dim)
    def forward(self, x):
        # x: (B, C, T)
        res = x
        x = x.transpose(1, 2)        # (B, T, C)
        lstm_out, _ = self.lstm(x)   # (B, T, 2H)
        x = self.proj(lstm_out)      # (B, T, C)
        x = self.norm(x + res.transpose(1, 2))
        return x.transpose(1, 2)     # (B, C, T)


class EnhancedECGEncoder(nn.Module):
    """
    Enhanced backbone for ECG:
      - LeadGate on raw leads
      - Multi-scale conv -> deep conv blocks
      - True time-wise attention after each stage
      - Residual BiLSTM
      - Dual pooling (avg+max) -> feature vector
    """
    def __init__(self, in_channels=12, dropout=0.3):
        super().__init__()
        self.lead_gate = LeadGate(in_channels)

        self.stem = nn.Sequential(
            MultiScaleConv1D(in_channels, 66),
            nn.MaxPool1d(2)
        )

        self.layer1 = self._make_layer(66, 128, blocks=2, stride=1)
        self.tatt1  = TrueTemporalAttention()

        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2)
        self.tatt2  = TrueTemporalAttention()

        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2)
        self.tatt3  = TrueTemporalAttention()

        self.lstm_block = ResidualLSTMBlock(512, 256, num_layers=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.in_dim = 512 * 2  # avg+max concat -> 1024

        self.dropout = nn.Dropout(dropout)

    def _make_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, 7, stride, 3, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 7, 1, 3, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = [self._make_block(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(self._make_block(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        # x: (B, C, T)
        x = self.lead_gate(x)
        x = self.stem(x)

        x = self.layer1(x); x = self.tatt1(x)
        x = self.layer2(x); x = self.tatt2(x)
        x = self.layer3(x); x = self.tatt3(x)

        x = self.lstm_block(x)
        avg_feat = self.avg_pool(x).squeeze(-1)
        max_feat = self.max_pool(x).squeeze(-1)
        feat = torch.cat([avg_feat, max_feat], dim=1)  # (B, 1024)
        feat = self.dropout(feat)
        return feat

    def forward(self, x):
        return self.forward_features(x)


# ---------------------------------
# 5) Heads (classification + contrastive)
# ---------------------------------

class Heads(nn.Module):
    """
    Two heads:
      - classification logits
      - contrastive projection (normalized)
    """
    def __init__(self, in_dim=1024, n_classes=2, proj_dim=128, dropout=0.3):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, feat):
        logits = self.cls(feat)
        z = F.normalize(self.proj(feat), dim=1)
        return logits, z


# ---------------------------------
# 6) Losses (NT-Xent original + SupCon)
# ---------------------------------

class NTXentLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], 0)                 # (2B, D)
        sim = torch.matmul(z, z.T) / self.temp     # (2B, 2B)
        mask = ~torch.eye(2*B, device=sim.device).bool()
        pos = torch.cat([sim.diag(B), sim.diag(-B)], 0).unsqueeze(1)  # (2B, 1)
        neg = sim[mask].view(2*B, -1)                                 # (2B, 2B-1)
        logits = torch.cat([pos, neg], 1)                              # (2B, 2B)
        labels = torch.zeros(2*B, dtype=torch.long, device=sim.device)
        return self.ce(logits, labels)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with optional confidence weights per pair.
    Ref: Khosla et al., 2020
    """
    def __init__(self, temperature: float = 0.1, eps: float = 1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor, y: torch.Tensor, conf: Optional[torch.Tensor] = None):
        """
        z: (B, D) normalized
        y: (B,)
        conf: (B,) in [0,1], per-sample confidence
        """
        device = z.device
        B = z.size(0)
        sim = torch.matmul(z, z.t()) / self.temperature   # (B, B)

        # Mask out self-comparisons
        logits_mask = torch.ones((B, B), device=device) - torch.eye(B, device=device)
        sim = sim - 1e9 * (1 - logits_mask)

        # Positive mask: same labels (excluding self)
        labels = y.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.t()).float() * logits_mask  # (B, B)

        # Weights (pairwise) from confidences
        if conf is None:
            w = pos_mask
        else:
            conf = conf.view(-1, 1)  # (B,1)
            w = pos_mask * (conf @ conf.t())  # (B,B)

        # Log-softmax over columns
        log_prob = F.log_softmax(sim, dim=1)  # (B, B)

        # For each anchor i, average log_prob over positives j with weights w_ij
        numerator = (w * log_prob).sum(dim=1)
        denom = w.sum(dim=1) + self.eps
        loss = -(numerator / denom)
        # Only anchors with at least one positive contribute
        valid = (denom > self.eps).float()
        return (loss * valid).sum() / valid.sum().clamp(min=1.0)


# ---------------------------------
# 7) Classifiers / Trainers
# ---------------------------------

class LinearProbeHead(nn.Module):
    """ 2-Layer MLP classifier head """
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
    """
    Original linear probe flow, but in_dim is read from encoder if available.
    """
    def __init__(self, encoder, signals_tensor, labels_array, save_path, batch_size=64, lr=1e-4, device='cpu', verbose=False):
        self.device = device
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # auto in_dim
        in_dim = getattr(self.encoder, "in_dim", 512)
        n_classes = int(len(np.unique(labels_array)))
        self.head = LinearProbeHead(in_dim=in_dim, num_classes=n_classes).to(self.device)
        self.save_path = save_path
        self.verbose = verbose

        dataset = SupervisedECGDataset(signals_tensor, labels_array)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_safe)

        # class weight (placeholder; adjust as needed)
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
            total_loss = 0.0
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)   # (B, C, T)
                with torch.no_grad():
                    if hasattr(self.encoder, "forward_features"):
                        h = self.encoder.forward_features(xb)
                    else:
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

        torch.save(self.head.state_dict(), self.save_path)
        if self.verbose:
            print(f"   -> Head saved after {epochs} epochs to {self.save_path}")


# ---------------------------
# 파이프라인 러너
# ---------------------------
import wandb

def run_phase1_contrastive(all_signals, save_dir, epochs=100, 
                           batch_size=128, lr=1e-3, temp=0.2, 
                           wd=1e-4, amp=True, num_workers=2):
    wandb.init(
    project="ecg-contrastive-final",  # 원하는 프로젝트명
    name=f"phase1_contrastive", # 실험 이름
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "temp": temp,
        "weight_decay": wd,
        "amp": amp
    })
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ContrastiveECGDataset(all_signals)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_safe, pin_memory=True, drop_last=True)

    encoder = EnhancedECGEncoder(in_channels=12).to(device)
    heads   = Heads(in_dim=encoder.in_dim, n_classes=2, proj_dim=128).to(device)
    opt = optim.AdamW(list(encoder.parameters()) + list(heads.parameters()), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=amp)
    ntx = NTXentLoss(temp=temp)

    encoder.train(); heads.train()
    for ep in range(1, epochs+1):
        encoder.train(); heads.train() 
        run_loss = 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            with autocast(enabled=amp):
                f1 = encoder.forward_features(v1); _, z1 = heads(f1)
                f2 = encoder.forward_features(v2); _, z2 = heads(f2)
                loss = ntx(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run_loss += float(loss.item())
        avg_train_loss = run_loss/len(loader)
        # === REMOVE BELOW FOR CHALLENGE SUBMISSION ===
        # Validation loss calculation (phase 1)
        encoder.eval(); heads.eval()
        val_loss = 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            with torch.no_grad(), autocast(enabled=amp):
                f1 = encoder.forward_features(v1); _, z1 = heads(f1)
                f2 = encoder.forward_features(v2); _, z2 = heads(f2)
                loss = ntx(z1, z2)
            val_loss += float(loss.item())
        avg_val_loss = val_loss/len(loader)
        wandb.log({"phase1_loss": avg_train_loss, "phase1_val_loss": avg_val_loss, "epoch": ep})
        # === END REMOVE ===
        print(f"[P1] epoch {ep}/{epochs} loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")
        # === REMOVE BELOW FOR CHALLENGE SUBMISSION ===
        torch.save({"encoder": encoder.state_dict()}, os.path.join(save_dir, f"phase1_encoder_epoch{ep}.pt"))
        # === END REMOVE ===

    os.makedirs(save_dir, exist_ok=True)
    torch.save({"encoder": encoder.state_dict()}, os.path.join(save_dir, "phase1_encoder.pt"))
    return os.path.join(save_dir, "phase1_encoder.pt")

# =============================================
# ============================
# Phase 2 with pseudo-labeling
# ============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os

# ---- Temperature scaling (확률 보정) ----
class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor(float(np.log(init_T))))

    def forward(self, logits):
        T = self.logT.exp()
        return logits / T

@torch.no_grad()
def evaluate_acc(encoder, heads, loader, device):
    encoder.eval(); heads.eval()
    total, correct = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        feat = encoder.forward_features(xb)
        logits, _ = heads(feat)
        pred = logits.argmax(1)
        total += yb.size(0); correct += (pred==yb).sum().item()
    return correct / max(1, total)

def calibrate_temperature(encoder, heads, val_loader, device, max_steps=200, lr=1e-2):
    """
    PTB-XL + SaMi-Trop 검증셋으로 T-scaling 파라미터 한 개 학습.
    """
    encoder.eval(); heads.eval()
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_steps, line_search_fn='strong_wolfe')

    xs, ys = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            feat = encoder.forward_features(xb)
            logits, _ = heads(feat)
            xs.append(logits); ys.append(yb)
    X = torch.cat(xs, 0); Y = torch.cat(ys, 0)

    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = ce(scaler(X), Y)
        loss.backward()
        return loss
    opt.step(closure)
    return scaler  # forward(logits)로 보정된 로짓 얻기

@torch.no_grad()
def build_class_prototypes(encoder, heads, pos_loader, neg_loader, device, n_aug=1):
    """
    SaMi(양성), PTB-XL(음성)에서 임베딩(z: normalized) 평균 → 프로토타입 생성.
    """
    encoder.eval(); heads.eval()
    zs_pos, zs_neg = [], []
    for _ in range(n_aug):
        for xb, yb in pos_loader:
            xb = xb.to(device)
            feat = encoder.forward_features(xb)
            _, z = heads(feat)
            zs_pos.append(z)
        for xb, yb in neg_loader:
            xb = xb.to(device)
            feat = encoder.forward_features(xb)
            _, z = heads(feat)
            zs_neg.append(z)
    proto_pos = torch.cat(zs_pos, 0).mean(0)   # (D,)
    proto_neg = torch.cat(zs_neg, 0).mean(0)   # (D,)
    proto_pos = F.normalize(proto_pos, dim=0)
    proto_neg = F.normalize(proto_neg, dim=0)
    return proto_pos.detach(), proto_neg.detach()

@torch.no_grad()
def score_code15_batch(encoder, heads, temp_scaler, xb, proto_pos, proto_neg, device, n_tta=8):
    """
    CODE-15 배치 x에 대해:
      - TTA로 N회 추론 → 평균확률, 표준편차, 마진
      - 임베딩 평균과 프로토타입 유사도 마진
      - 종합 신뢰도 s 반환
    """
    encoder.eval(); heads.eval()
    probs, embeds = [], []
    for _ in range(n_tta):
        # 1. 배치(xb)에 있는 신호를 하나씩 꺼내(x_sample) 증강하고 리스트에 담습니다.
        x_aug_list = [augment_signal_v2(x_sample) for x_sample in xb]
        # 2. 증강된 신호 리스트를 다시 하나의 배치 텐서로 합칩니다.
        x_aug = torch.stack(x_aug_list, dim=0)

        feat = encoder.forward_features(x_aug.to(device))
        logits, z = heads(feat)
        logits = temp_scaler(logits) if temp_scaler is not None else logits
        p = torch.softmax(logits, dim=1)  # (B,2)
        probs.append(p)
        embeds.append(z)
    P = torch.stack(probs, 0)             # (N_aug,B,2)
    p_mean = P.mean(0)                     # (B,2)
    p_std  = P.std(0).amax(dim=1)  # (B,)
    margin = (p_mean[:,1] - p_mean[:,0]).abs()  # (B,)
    z_mean = torch.stack(embeds, 0).mean(0)     # (B,D) normalized (already)

    # 프로토타입 유사도 (cosine)
    # z_mean, proto_* 는 모두 정규화되었다고 가정 → dot == cosine
    sim_pos = (z_mean @ proto_pos)  # (B,)
    sim_neg = (z_mean @ proto_neg)  # (B,)
    proto_margin = sim_pos - sim_neg

    # 종합 스코어 (가중치는 경험적 기본값)
    s = (0.40 * p_mean.amax(1)
         + 0.20 * margin
         - 0.20 * p_std
         + 0.20 * torch.tanh(proto_margin))
    # [0,1] 구간에 클램프
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    pred = p_mean.argmax(1)  # (B,)
    return s.cpu(), pred.cpu(), p_mean.cpu()

def run_phase2(
    ptb, sami, code15, save_dir, encoder_ckpt=None,
    # linear probe + warmup finetune
    lin_epochs=6, ft_epochs_base=18,
    # pseudo-labeling 추가 파인튜닝
    ft_epochs_pl=12, n_tta=8, tau_pos=0.85, tau_neg=0.85,
    # loaders/opt
    batch_size=64, lr=1e-3, wd=1e-4, focal_gamma=None, amp=True, num_workers=2
):
    """
    Phase 2:
      1) Linear probe → 2) Warm-up fine-tune(PTB+SaMi 중심, CODE15 소량 랜덤) →
      3) CODE-15 신뢰도 산출 & 라벨 일치 필터 → 4) 추가 fine-tune
    """
    wandb.init(
    project="ecg-contrastive-final",
    name="phase2_finetune",
    config={
        "lin_epochs": lin_epochs,
        "ft_epochs_base": ft_epochs_base,
        "ft_epochs_pl": ft_epochs_pl,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": wd
    })
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ptb_sig, ptb_lab), (sami_sig, sami_lab) = ptb, sami
    if isinstance(code15, (list, tuple)) and len(code15) == 3:
        code_sig, code_lab, _ = code15
    else:
        code_sig, code_lab = code15

    # ---------- 데이터셋 ----------
    MAX_LEN = 4096

    p = 0.2
    N = len(code_sig); k = max(1, int(N*p))
    sel = np.random.choice(N, size=k, replace=False) if k < N else np.arange(N)
    
    tr_sig_unpadded = list(ptb_sig) + list(sami_sig) + [code_sig[i] for i in sel]
    tr_sig = pad_signals(tr_sig_unpadded, MAX_LEN) # 패딩 추가
    tr_lab = np.concatenate([ptb_lab, sami_lab, np.asarray([code_lab[i] for i in sel])])

    val_sig_unpadded = list(ptb_sig) + list(sami_sig)
    val_sig = pad_signals(val_sig_unpadded, MAX_LEN) # 패딩 추가
    val_lab = np.concatenate([ptb_lab, sami_lab])

    train_ds = MultiSourceECGDataset(tr_sig, tr_lab, confidences=None, aug_for_train=False)
    val_ds   = SupervisedECGDataset(val_sig, val_lab)

    neg_ds = SupervisedECGDataset(pad_signals(list(ptb_sig), MAX_LEN), ptb_lab)   # 음성
    pos_ds = SupervisedECGDataset(pad_signals(list(sami_sig), MAX_LEN), sami_lab) # 양성

    neg_loader = DataLoader(neg_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)
    pos_loader = DataLoader(pos_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)

    # ---------- 모델 ----------
    encoder = EnhancedECGEncoder(in_channels=12)
    if encoder_ckpt and os.path.exists(encoder_ckpt):
        state = torch.load(encoder_ckpt, map_location="cpu")
        encoder.load_state_dict(state["encoder"], strict=False)
        print(f"[Phase2] loaded encoder: {encoder_ckpt}")
    heads = Heads(in_dim=encoder.in_dim, n_classes=2, proj_dim=128)

    encoder, heads = encoder.to(device), heads.to(device)

    # ---------- Linear probe ----------
    lp = LinearProbeTrainer(
        encoder=encoder,
        signals_tensor=train_ds.signals,
        labels_array=train_ds.labels.cpu().numpy(),
        save_path=os.path.join(save_dir, "phase2_linear_head.pt"),
        batch_size=batch_size, lr=lr, device=device, verbose=True
    )
    lp.train(epochs=lin_epochs)

    # ---------- Warm-up fine-tune ----------
    y = train_ds.labels.cpu().numpy()
    classes, counts = np.unique(y, return_counts=True)
    inv = {c: 1.0 / max(1, cnt) for c, cnt in zip(classes, counts)}
    w = np.array([inv[int(lbl)] for lbl in y], dtype=np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate_safe)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(heads.parameters()), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=amp)

    def ce_or_focal(logits, targets):
        if focal_gamma is None:
            return F.cross_entropy(logits, targets, reduction='none')
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1-pt)**focal_gamma) * ce

    best_acc, best_path = 0.0, os.path.join(save_dir, "phase2_finetuned.pt")
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, ft_epochs_base+1):
        encoder.train(); heads.train()
        tr_loss = 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(enabled=amp):
                feat = encoder.forward_features(xb)
                logits, _ = heads(feat)
                loss = ce_or_focal(logits, yb).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())

        avg_train_loss = tr_loss/len(train_loader)
        # === REMOVE BELOW FOR CHALLENGE SUBMISSION ===
        # Validation loss calculation (phase 2 warmup)
        encoder.eval(); heads.eval()
        val_loss = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad(), autocast(enabled=amp):
                feat = encoder.forward_features(xb)
                logits, _ = heads(feat)
                loss = ce_or_focal(logits, yb).mean()
            val_loss += float(loss.item())
        avg_val_loss = val_loss/len(val_loader)
        acc = evaluate_acc(encoder, heads, val_loader, device)
        wandb.log({
            "warmup_loss": avg_train_loss,
            "warmup_val_loss": avg_val_loss,
            "val_acc": acc,
            "epoch": ep
        })
        # === END REMOVE ===
        print(f"[Phase2][warmup {ep}/{ft_epochs_base}] loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | val_acc={acc:.4f}")
        # === REMOVE BELOW FOR CHALLENGE SUBMISSION ===
        torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, os.path.join(save_dir, f"phase2_finetuned_epoch{ep}.pt"))
        # === END REMOVE ===
        if acc > best_acc:
            best_acc = acc
            torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, best_path)
            print(f"  ↳ saved: {best_path} (acc={best_acc:.4f})")

    # ---------- CODE-15 신뢰도 산출 + 서브셋 선택 ----------
    # 1) 캘리브레이션(T-scaling): 검증 로더로 온도 보정
    temp_scaler = calibrate_temperature(encoder, heads, val_loader, device)

    # 2) 클래스 프로토타입(임베딩 평균) 구성
    proto_pos, proto_neg = build_class_prototypes(encoder, heads, pos_loader, neg_loader, device, n_aug=1)

    # 3) CODE-15 전체를 배치로 훑으며 스코어링
    code_ds = SupervisedECGDataset(pad_signals(list(code_sig), MAX_LEN), np.asarray(code_lab))

    code_loader = DataLoader(code_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)
    kept_signals, kept_labels, kept_weights = [], [], []

    with torch.no_grad():
        for xb, yb in code_loader:
            xb = xb.to(device)
            s, pred, p_mean = score_code15_batch(encoder, heads, temp_scaler, xb, proto_pos, proto_neg, device, n_tta=n_tta)
            yb_cpu = yb.cpu()
            # 라벨 일치 + 클래스별 임계값
            match = (pred == yb_cpu)

            # 클래스별 임계값 적용
            keep_mask = torch.zeros_like(match, dtype=torch.bool)
            keep_mask[(match) & (pred==1) & (s>=tau_pos)] = True
            keep_mask[(match) & (pred==0) & (s>=tau_neg)] = True

            # 선택된 샘플만 수집
            for i, keep in enumerate(keep_mask.tolist()):
                if keep:
                    # (C,T) → (T,C)로 바꿔 "unpadded"로 보관
                    x_i = (
                        xb[i]
                        .detach().cpu().contiguous().clone()
                        .permute(1, 0)      # to (T,C)
                        .numpy()
                    )
                    kept_signals.append(x_i)
                    kept_labels.append(int(yb_cpu[i].item()))
                    kept_weights.append(float(s[i].item()))

    print(f"[Phase2][PL] selected {len(kept_signals)} / {len(code_sig)} CODE-15 samples")

    if len(kept_signals) == 0:
        print("[Phase2][PL] No samples passed the threshold; skipping PL fine-tune.")
        return best_path

    # 4) 추가 파인튜닝(선택 서브셋 포함, 가중치 사용)
    #    - 학습세트 = 기존 PTB+SaMi + 선택된 CODE-15
    # (1) 먼저 (T,C) "unpadded" 리스트 만들기
    ext_signals_unpadded = list(ptb_sig) + list(sami_sig) + kept_signals
    ext_labels  = np.concatenate([ptb_lab, sami_lab, np.asarray(kept_labels, dtype=int)])

    # (2) 공통 길이 패딩 (T,C) -> (N, MAX_LEN, C)
    MAX_LEN = 4096
    ext_signals = pad_signals(ext_signals_unpadded, MAX_LEN)   # (T,C) 가정에 맞게 패딩
    # 참고: pad_signals의 사양은 파일 상단에 명시됨.  (T,C) -> (N,max_len,C)  :contentReference[oaicite:6]{index=6}

    # (3) 모델 입력 (C,T) 텐서로 변환 + 새/연속 스토리지 보장
    def _to_ct_tensor(x):
        t = torch.tensor(np.ascontiguousarray(x))  # x는 (T,C) numpy
        t = t.transpose(0, 1).contiguous()         # (C,T)
        return t.clone()

    ext_signals = [_to_ct_tensor(x) for x in ext_signals]

    # (4) sanity check (12채널 보장)
    assert all(isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] == 12 for x in ext_signals)

    # sampler 가중치 = 클래스 역빈도 × 선택 샘플의 s (PTB/SaMi는 s=1.0로 취급)
    y_all = ext_labels
    classes, counts = np.unique(y_all, return_counts=True)
    inv = {c: 1.0 / max(1, cnt) for c, cnt in zip(classes, counts)}
    base_w = np.array([inv[int(lbl)] for lbl in y_all], dtype=np.float32)
    s_w = np.concatenate([np.ones(len(ptb_lab) + len(sami_lab), dtype=np.float32),
                          np.asarray(kept_weights, dtype=np.float32)])
    w_all = base_w * s_w.clip(0.5, 1.0)  # s를 [0.5,1.0]로 클램프해서 너무 큰 편차 방지

    ext_train = MultiSourceECGDataset(ext_signals, ext_labels, confidences=None, aug_for_train=False)
    sampler2 = WeightedRandomSampler(torch.from_numpy(w_all), num_samples=len(w_all), replacement=True)
    ext_loader = DataLoader(
        ext_train, batch_size=batch_size, sampler=sampler2,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_hard  # <-- 여기 포인트
    )

    for ep in range(1, ft_epochs_pl+1):
        encoder.train(); heads.train()
        tr_loss = 0.0
        for xb, yb, _ in ext_loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(enabled=amp):
                feat = encoder.forward_features(xb)
                logits, _ = heads(feat)
                loss = ce_or_focal(logits, yb).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())

        avg_train_loss = tr_loss/len(ext_loader)
        # === REMOVE BELOW FOR CHALLENGE SUBMISSION ===
        # Validation loss calculation (phase 2 PL)
        encoder.eval(); heads.eval()
        val_loss = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad(), autocast(enabled=amp):
                feat = encoder.forward_features(xb)
                logits, _ = heads(feat)
                loss = ce_or_focal(logits, yb).mean()
            val_loss += float(loss.item())
        avg_val_loss = val_loss/len(val_loader)
        acc = evaluate_acc(encoder, heads, val_loader, device)
        wandb.log({
            "pl_loss": avg_train_loss,
            "pl_val_loss": avg_val_loss,
            "val_acc": acc,
            "epoch": ep
        })
        # === END REMOVE ===
        print(f"[Phase2][PL {ep}/{ft_epochs_pl}] loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | val_acc={acc:.4f}")
        # === REMOVE BELOW FOR CHALLENGE SUBMISSION ===
        torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, os.path.join(save_dir, f"phase2_finetuned_epoch{ep}.pt"))
        # === END REMOVE ===
        if acc > best_acc:
            best_acc = acc
            torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, best_path)
            print(f"  ↳ saved: {best_path} (acc={best_acc:.4f})")

    return best_path


# ===========================코드 터졌을 때 ,,

def resume_phase2_pl(
    ptb, sami, code15, save_dir,
    best_ckpt_path=None,              # warmup 베스트(.pt) 또는 None
    phase1_ckpt=None,                 # redo_warmup=True일 때 사용할 phase1_encoder.pt
    redo_warmup=True,               
    lin_epochs=2, ft_epochs_base=4,   # warm-up용
    ft_epochs_pl=5, n_tta=8, tau_pos=0.85, tau_neg=0.85,
    batch_size=64, lr=1e-3, wd=3e-4, focal_gamma=None, amp=True, num_workers=2
):
    import os, numpy as np, torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torch import amp as _amp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ptb_sig, ptb_lab), (sami_sig, sami_lab) = ptb, sami
    if isinstance(code15, (list, tuple)) and len(code15) == 3:
        code_sig, code_lab, _ = code15
    else:
        code_sig, code_lab = code15

    MAX_LEN = 4096
    # --- 공통 로더들 ---
    neg_ds = SupervisedECGDataset(pad_signals(list(ptb_sig),  MAX_LEN), ptb_lab)
    pos_ds = SupervisedECGDataset(pad_signals(list(sami_sig), MAX_LEN), sami_lab)
    val_ds = SupervisedECGDataset(pad_signals(list(ptb_sig)+list(sami_sig), MAX_LEN),
                                  np.concatenate([ptb_lab, sami_lab]))
    neg_loader = DataLoader(neg_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)
    pos_loader = DataLoader(pos_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)

    # --- 모델 준비 ---
    encoder = EnhancedECGEncoder(in_channels=12).to(device)
    heads   = Heads(in_dim=encoder.in_dim, n_classes=2, proj_dim=128).to(device)

    # --- (옵션) Warm-up 다시 수행 ---
    if redo_warmup:
        assert phase1_ckpt and os.path.exists(phase1_ckpt), f"phase1 ckpt not found: {phase1_ckpt}"
        st = torch.load(phase1_ckpt, map_location="cpu")
        encoder.load_state_dict(st["encoder"], strict=False)
        # Linear probe
        train_ds_for_lp = MultiSourceECGDataset(
            pad_signals(list(ptb_sig)+list(sami_sig), MAX_LEN),
            np.concatenate([ptb_lab, sami_lab]),
            confidences=None, aug_for_train=False
        )
        lp = LinearProbeTrainer(
            encoder=encoder,
            signals_tensor=train_ds_for_lp.signals,
            labels_array=train_ds_for_lp.labels.cpu().numpy(),
            save_path=os.path.join(save_dir, "phase2_linear_head.pt"),
            batch_size=batch_size, lr=lr, device=device, verbose=True
        )
        lp.train(epochs=lin_epochs)

        # Warm-up fine-tune
        # PTB+SaMi+CODE-15 일부(비율 0.1) 혼입
        p = 0.1
        N = len(code_sig); k = max(1, int(N*p))
        sel = np.random.choice(N, size=k, replace=False) if k < N else np.arange(N)
        tr_sig_unpadded = list(ptb_sig) + list(sami_sig) + [code_sig[i] for i in sel]
        tr_lab = np.concatenate([ptb_lab, sami_lab, np.asarray([code_lab[i] for i in sel])])
        tr_sig = pad_signals(tr_sig_unpadded, MAX_LEN)
        train_ds = MultiSourceECGDataset(tr_sig, tr_lab, confidences=None, aug_for_train=True)

        y = train_ds.labels.cpu().numpy()
        classes, counts = np.unique(y, return_counts=True)
        inv = {c: 1.0/max(1,cnt) for c,cnt in zip(classes, counts)}
        w = np.array([inv[int(lbl)] for lbl in y], dtype=np.float32)
        sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True,
                                  collate_fn=collate_safe)

        opt = torch.optim.AdamW(list(encoder.parameters())+list(heads.parameters()), lr=lr, weight_decay=wd)
        scaler = _amp.GradScaler("cuda", enabled=amp)
        best_acc, best_path = 0.0, os.path.join(save_dir, "phase2_finetuned.pt")

        for ep in range(1, ft_epochs_base+1):
            encoder.train(); heads.train()
            tr_loss = 0.0
            for xb, yb, _ in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                with _amp.autocast("cuda", enabled=amp):
                    feat = encoder.forward_features(xb)
                    logits, _ = heads(feat)
                    loss = F.cross_entropy(logits, yb).mean()
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                tr_loss += float(loss.item())
            acc = evaluate_acc(encoder, heads, val_loader, device)
            print(f"[Warmup {ep}/{ft_epochs_base}] loss={tr_loss/len(train_loader):.4f} | val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, best_path)
                print(f"  ↳ saved: {best_path} (acc={best_acc:.4f})")
        best_ckpt_path = os.path.join(save_dir, "phase2_finetuned.pt")

    # --- 여기서부터는 기존 PL 파이프라인 그대로 ---
    assert best_ckpt_path and os.path.exists(best_ckpt_path), f"ckpt not found: {best_ckpt_path}"
    state = torch.load(best_ckpt_path, map_location="cpu")
    encoder.load_state_dict(state["encoder"], strict=False)
    heads.load_state_dict(state["heads"], strict=False)
    print(f"[Resume] loaded warmup-best: {best_ckpt_path}")

    temp_scaler = calibrate_temperature(encoder, heads, val_loader, device)
    proto_pos, proto_neg = build_class_prototypes(encoder, heads, pos_loader, neg_loader, device, n_aug=1)

    code_ds = SupervisedECGDataset(pad_signals(list(code_sig), MAX_LEN), np.asarray(code_lab))
    code_loader = DataLoader(code_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)

    kept_signals, kept_labels, kept_weights = [], [], []
    with torch.no_grad():
        for xb, yb in code_loader:
            xb = xb.to(device)
            s, pred, _ = score_code15_batch(encoder, heads, temp_scaler, xb, proto_pos, proto_neg, device, n_tta=n_tta)
            yb_cpu = yb.cpu()
            match = (pred == yb_cpu)
            keep_mask = torch.zeros_like(match, dtype=torch.bool)
            keep_mask[(match) & (pred==1) & (s>=tau_pos)] = True
            keep_mask[(match) & (pred==0) & (s>=tau_neg)] = True
            for i, keep in enumerate(keep_mask.tolist()):
                if keep:
                    x_i = xb[i].detach().cpu().contiguous().clone().permute(1,0).numpy()  # (T,C)
                    kept_signals.append(x_i)
                    kept_labels.append(int(yb_cpu[i].item()))
                    kept_weights.append(float(s[i].item()))
    print(f"[PL] selected {len(kept_signals)} samples")

    if len(kept_signals) == 0:
        print("[PL] nothing selected; stop.")
        return best_ckpt_path

    # --- PL fine-tune (패딩→(C,T) 텐서화→collate_hard) ---
    ext_signals_unpadded = list(ptb_sig) + list(sami_sig) + kept_signals
    ext_labels = np.concatenate([ptb_lab, sami_lab, np.asarray(kept_labels, dtype=int)])
    ext_signals = pad_signals(ext_signals_unpadded, MAX_LEN)  # (T,C)

    def _to_ct_tensor(x):
        t = torch.tensor(np.ascontiguousarray(x))  # (T,C)
        return t.transpose(0,1).contiguous().clone()  # (C,T)
    ext_signals = [_to_ct_tensor(x) for x in ext_signals]
    assert all(x.ndim==2 and x.shape[0]==12 for x in ext_signals)

    y_all = ext_labels
    classes, counts = np.unique(y_all, return_counts=True)
    inv = {c: 1.0/max(1,cnt) for c,cnt in zip(classes, counts)}
    base_w = np.array([inv[int(lbl)] for lbl in y_all], dtype=np.float32)
    s_w = np.concatenate([np.ones(len(ptb_lab)+len(sami_lab), dtype=np.float32),
                          np.asarray(kept_weights, dtype=np.float32)])
    w_all = base_w * s_w.clip(0.5, 1.0)

    ext_train = MultiSourceECGDataset(ext_signals, ext_labels, confidences=None, aug_for_train=False)
    sampler2 = WeightedRandomSampler(torch.from_numpy(w_all), num_samples=len(w_all), replacement=True)
    ext_loader = DataLoader(ext_train, batch_size=batch_size, sampler=sampler2,
                            num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate_hard)

    opt = torch.optim.AdamW(list(encoder.parameters())+list(heads.parameters()), lr=lr, weight_decay=wd)
    scaler = _amp.GradScaler("cuda", enabled=amp)
    def ce_or_focal(logits, targets):
        if focal_gamma is None:
            return F.cross_entropy(logits, targets, reduction='none')
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1-pt)**focal_gamma) * ce

    best_acc, best_path = 0.0, os.path.join(save_dir, "phase2_finetuned.pt")
    if os.path.exists(best_path):
        best_acc = evaluate_acc(encoder, heads, val_loader, device)
        print(f"[Resume] starting acc={best_acc:.4f}")

    for ep in range(1, ft_epochs_pl+1):
        encoder.train(); heads.train()
        tr_loss = 0.0
        for xb, yb, _ in ext_loader:
            xb, yb = xb.to(device), yb.to(device)
            with _amp.autocast("cuda", enabled=amp):
                feat = encoder.forward_features(xb)
                logits, _ = heads(feat)
                loss = ce_or_focal(logits, yb).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())
        acc = evaluate_acc(encoder, heads, val_loader, device)
        print(f"[PL {ep}/{ft_epochs_pl}] loss={tr_loss/len(ext_loader):.4f} | val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, best_path)
            print(f"  ↳ saved: {best_path} (acc={best_acc:.4f})")
    return best_path
