#!/usr/bin/env python
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Tuple
from scipy.signal import butter, filtfilt, resample_poly, iirnotch
import os, sys, math, json, glob
from torch.cuda.amp import GradScaler, autocast
from helper_code import *
from model import *

from torch.utils.data._utils.collate import default_collate

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

### 전처리된 npy 파일에서 데이터 로드
def load_all_sources_from_npy(data_folder: str, verbose: bool = False):
    """
    전처리된 NPY 파일에서 데이터 로드
    data_folder: processed .npy 파일들이 있는 폴더
    """
    import os
    import numpy as np
    
    # NPY 파일 경로
    signals_path = os.path.join(data_folder, "signals.npy")
    labels_path = os.path.join(data_folder, "labels.npy") 
    sources_path = os.path.join(data_folder, "sources.npy")
    exam_ids_path = os.path.join(data_folder, "exam_ids.npy")
    
    # 파일 존재 확인
    for path in [signals_path, labels_path, sources_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    if verbose:
        print("Loading preprocessed NPY files...")
    
    # 데이터 로드
    signals = np.load(signals_path)  # (N, T, C) 형태 예상
    labels = np.load(labels_path)    # (N,) 형태
    sources = np.load(sources_path, allow_pickle=True)  # (N,) 문자열 배열
    
    if verbose:
        print(f"Loaded signals shape: {signals.shape}")
        print(f"Loaded labels shape: {labels.shape}")
        print(f"Loaded sources shape: {sources.shape}")
        print(f"Unique sources: {np.unique(sources)}")
        print(f"Label distribution: {np.bincount(labels)}")
    
    # 소스별로 데이터 분리
    ptb_sig, ptb_lab = [], []
    sami_sig, sami_lab = [], []
    code_sig, code_lab, code_conf = [], [], []
    
    for i in range(len(signals)):
        sig = signals[i]  # (T, C) 또는 (C, T)
        lab = labels[i]
        src = str(sources[i]).lower()
        
        # 신호가 (C, T) 형태가 아니라면 transpose 필요할 수 있음
        # 일반적으로 ECG는 (T, C) 형태로 저장되므로 확인 필요
        if sig.shape[1] == 12:  # (T, 12) 형태
            pass  # 이미 올바른 형태
        elif sig.shape[0] == 12:  # (12, T) 형태
            sig = sig.T  # (T, 12)로 변환
        else:
            if verbose:
                print(f"Warning: unexpected signal shape {sig.shape} for sample {i}")
            continue
            
        # 소스별 분류
        if "ptb" in src:
            ptb_sig.append(sig.astype(np.float32))
            ptb_lab.append(lab)
        elif "sami" in src or "trop" in src:
            sami_sig.append(sig.astype(np.float32))
            sami_lab.append(lab)
        elif "code" in src:
            code_sig.append(sig.astype(np.float32))
            code_lab.append(lab)
            code_conf.append(1.0)  # 기본 신뢰도
        else:
            # 소스가 명확하지 않은 경우 CODE로 분류
            code_sig.append(sig.astype(np.float32))
            code_lab.append(lab)
            code_conf.append(1.0)
    
    if verbose:
        print(f"PTB samples: {len(ptb_sig)}")
        print(f"SaMi samples: {len(sami_sig)}")
        print(f"CODE samples: {len(code_sig)}")
    
    return (
        (ptb_sig, np.asarray(ptb_lab, dtype=int)),
        (sami_sig, np.asarray(sami_lab, dtype=int)),
        (code_sig, np.asarray(code_lab, dtype=int), np.asarray(code_conf, dtype=float))
    )
# =============================제출전에 삭제


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

# 평범한 normalization
def normalize_leads(arr: np.ndarray) -> np.ndarray:
    arr_norm = arr.copy()
    for i in range(arr.shape[1]):
        mean = np.mean(arr[:, i])
        std = np.std(arr[:, i])
        if std < 1e-6:
            std = 1.0
        arr_norm[:, i] = (arr[:, i] - mean) / std
    return arr_norm


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


# =========================
# Collate Functions
# =========================


def _to_hard_tensor(x):
    if torch.is_tensor(x):
        return x.detach().cpu().contiguous().clone()
    elif isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x)
        return torch.tensor(arr)
    else:
        return torch.tensor(x)  

def collate_hard(batch):
    b0 = batch[0]
    if isinstance(b0, (list, tuple)) and len(b0) == 3:
        xs, ys, metas = zip(*batch)
        xs = torch.stack([_to_hard_tensor(x) for x in xs], dim=0)
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


# ---------------------------------
# 3) Datasets
# ---------------------------------

class ContrastiveECGDataset(Dataset):
    """Contrastive learning용 데이터셋"""
    def __init__(self, signals):
        self.signals = signals

    def __len__(self): 
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx]
        if isinstance(sig, np.ndarray):
            sig = torch.from_numpy(np.ascontiguousarray(sig.T)).float()
        v1 = augment_signal_v2(sig)
        v2 = augment_signal_v2(sig)
        return v1, v2


class SupervisedECGDataset(Dataset):
    """분류 학습용 데이터셋"""
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
    Labeled data with per-sample confidence (for weak labels  CODE-15).
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
            v1 = augment_signal_v2(x) # edit
            v2 = augment_signal_v2(x) # edit
            return v1, v2, y, c
        else:
            return x, y, c

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



