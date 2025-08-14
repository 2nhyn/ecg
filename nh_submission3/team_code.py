#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import sys
import random
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
import wfdb

from helper_code import *
from models import EnhancedECGEncoder, Heads
from util_nh import *

# ===================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================================
# TRAIN FUNCTION
# ==================================
def train_model(data_folder, model_folder, verbose=False):
    os.makedirs(model_folder, exist_ok=True)
    if verbose:
        print("Setting up hyperparameters and device...")
        print("Finding and preprocessing data...")

    # 1) 데이터 로딩
#    (ptb_sig, ptb_lab), (sami_sig, sami_lab), (code_sig, code_lab, code_conf) = load_all_sources(data_folder, verbose=verbose)
    # new!@
    (ptb_sig, ptb_lab), (sami_sig, sami_lab), (code_sig, code_lab, code_conf) = load_all_sources_from_npy(data_folder, verbose=verbose)




    if len(ptb_sig)+len(sami_sig)+len(code_sig) == 0:
        raise ValueError("No data could be processed. Please check the data folder and format.")

    # 2) Phase 1 (contrastive learning)
    all_signals = list(ptb_sig) + list(sami_sig) + list(code_sig)
    MAX_LEN = 4096

    padded_signals_array = pad_signals(all_signals, MAX_LEN)
    all_signals_padded = [padded_signals_array[i] for i in range(len(padded_signals_array))]
    p1_ckpt = run_phase1_contrastive(all_signals_padded, 
                                     save_dir=model_folder, 
                                     epochs=20, 
                                     batch_size=256, 
                                     lr=1e-3, 
                                     temp=0.2)

    # 3) Phase 2 (분류 파인튜닝: confidence-weighted)
    best_ckpt = run_phase2(
        (ptb_sig, ptb_lab), (sami_sig, sami_lab), (code_sig, code_lab, code_conf),
        save_dir=model_folder, encoder_ckpt=p1_ckpt,
        lin_epochs=5, ft_epochs_base=3, batch_size=128, 
        lr=1e-3, wd=1e-4,
        focal_gamma=None, amp=True,
        num_workers=2
    )
    # 5) 제출 포맷에 필요한 메타 저장(클래스 맵 등)
    meta = {
        "classes": ["negative", "positive"],
        "preprocess": {"fs": 400, "bp_low": 0.5, "bp_high": 45.0, "norm": "medianMAD"},
        "checkpoints": {"phase1": p1_ckpt, "phase2_best": best_ckpt}
    }
    with open(os.path.join(model_folder, "meta.json"), "w") as f:
        json.dump(meta, f)
    if verbose:
        print("Training finished. Artifacts saved to:", model_folder)


# ==================================
# load model
# ==================================
def _pad_or_truncate(tc: np.ndarray, max_len: int) -> np.ndarray:
    # tc: (T, C=12)
    T, C = tc.shape
    if T == max_len: 
        return tc
    out = np.zeros((max_len, C), dtype=np.float32)
    if T >= max_len:
        out[:] = tc[:max_len]
    else:
        out[:T] = tc
    return out

def load_model(model_folder: str, verbose: bool = False):
    """
    - encoder/head 가중치 로드
    - 전처리기/입력 길이 준비
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 메타(있으면) 읽기
    meta_path = os.path.join(model_folder, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        max_len = int(meta.get("preprocess", {}).get("max_len", 4096))
    else:
        meta = {}
        max_len = 4096

    # 모델 구성 (네 클래스 수 2로 고정)
    encoder = EnhancedECGEncoder(in_channels=12).to(device).eval()
    heads   = Heads(in_dim=encoder.in_dim, n_classes=2, proj_dim=128).to(device).eval()

    # 체크포인트 로드 (phase2가 최우선, 없으면 phase1)
    ckpt_p2 = os.path.join(model_folder, "phase2_finetuned.pt")
    ckpt_p1 = os.path.join(model_folder, "phase1_encoder_epoch10.pt")
    if os.path.exists(ckpt_p2):
        state = torch.load(ckpt_p2, map_location="cpu")
        encoder.load_state_dict(state["encoder"], strict=False)
        if "heads" in state:
            heads.load_state_dict(state["heads"], strict=False)
        if verbose: print(f"[load_model] loaded {ckpt_p2}")
    elif os.path.exists(ckpt_p1):
        state = torch.load(ckpt_p1, map_location="cpu")
        encoder.load_state_dict(state["encoder"], strict=False)
        if verbose: print(f"[load_model] loaded {ckpt_p1}")
    else:
        raise FileNotFoundError("No checkpoint (.pth) found in model folder")

    # 전처리기 (학습 때와 동일)
    proc = SignalProcessor(target_fs=400, bp_low=0.5, bp_high=45.0)

    return {
        "device": device,
        "encoder": encoder,
        "heads": heads,
        "proc": proc,
        "max_len": max_len
    }



# =============================
# run_model
# =============================
def run_model(record: str, model: dict, verbose: bool = False):
    """
    - record: helper_code.find_records로 받은 경로(확장자 제외/포함 모두 load_*가 처리)
    - 반환: (binary_output, probability_output)
    """
    device   = model["device"]
    encoder  = model["encoder"]
    heads    = model["heads"]
    proc     = model["proc"]
    max_len  = model["max_len"]
    REF_12_LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

    # 1) 읽기
    header = load_header(record)
    sig, fields = load_signals(record)  # (T, C_raw)
    names = fields.get('sig_name', get_signal_names(header) or [])
    sig = reorder_signal(sig, names, REF_12_LEADS)  # (T, 12)
    fs = int(get_sampling_frequency(header) or 400)

    # 2) 전처리/정규화/길이 맞추기
    x = proc.preprocess(sig, fs=fs)             # (T, 12)
    x = normalize_leads(x).astype(np.float32)   # (T, 12)
    x = _pad_or_truncate(x, max_len)            # (T*, 12)

    xt = torch.from_numpy(x.T).unsqueeze(0).to(device)  # (1, 12, T*)

    # 3) 추론
    with torch.no_grad():
        with autocast(enabled=False):  # 필요 시 True로 바꿔도 됨
            feat = encoder.forward_features(xt)
            logits, _ = heads(feat)           # (1, 2)
            prob_pos = F.softmax(logits, dim=1)[0, 1].item()

    pred = int(prob_pos >= 0.5)  # 필요하면 임계값 조정
    return pred, float(prob_pos)


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)

    # Extract the age from the record.
    age = get_age(header)
    age = np.array([age])

    # Extract the sex from the record and represent it as a one-hot encoded vector.
    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex.casefold().startswith('f'):
        sex_one_hot_encoding[0] = 1
    elif sex.casefold().startswith('m'):
        sex_one_hot_encoding[1] = 1
    else:
        sex_one_hot_encoding[2] = 1

    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)

    # Load the signal data and fields. Try fields.keys() to see the fields, e.g., fields['fs'] is the sampling frequency.
    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels in case they are in a different order in the signal data.
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)

    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal)
        else:
            signal_std = 0.0

    # Return the features.

    return age, sex_one_hot_encoding, source, signal_mean, signal_std

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)