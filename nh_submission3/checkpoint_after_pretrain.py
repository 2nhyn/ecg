#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from util_nh import *

# ---------- 데이터 로더: .npy 우선, 없으면 .npz ----------
def load_cached_revised(cache_dir, max_len=4096, assume_padded=True):
    p = lambda n: os.path.join(cache_dir, n)
    labels  = np.load(p("labels.npy"),  allow_pickle=True).astype(int)
    sources = np.load(p("sources.npy"), allow_pickle=True)

    signals = None
    # .npy 우선 시도
    for cand in ("processed_signals.npy", "signals.npy"):
        f = p(cand)
        if os.path.isfile(f):
            signals = np.load(f, allow_pickle=True, mmap_mode='r')
            break
    # .npz 폴백
    if signals is None:
        f = p("processed_signals.npz")
        if os.path.isfile(f):
            with np.load(f, allow_pickle=True) as z:
                key = "signals" if "signals" in z.files else z.files[0]
                signals = z[key]
        else:
            raise FileNotFoundError("processed_signals(.npy|.npz) 또는 signals.npy가 필요합니다.")

    # (T,12) 리스트로 맞추기
    sig_list = []
    if isinstance(signals, np.ndarray) and signals.dtype == object:
        for x in signals:
            x = np.asarray(x)
            if x.ndim == 1:
                x = x[:, None]
            if x.shape[1] != 12 and x.shape[0] == 12:
                x = x.T
            sig_list.append(x.astype(np.float32))
        assume_padded = False
    else:
        X = np.asarray(signals)
        if X.ndim != 3:
            raise ValueError(f"signals ndim={X.ndim}, expected 3")
        if X.shape[-1] == 12:      # (N,T,12)
            sig_list = [x.astype(np.float32) for x in X]
        elif X.shape[1] == 12:     # (N,12,T) -> (T,12)
            sig_list = [x.transpose(1, 0).astype(np.float32) for x in X]
        else:
            raise ValueError(f"expect 12 leads, got shape {X.shape}")

    # 필요시 패딩/컷
    if not assume_padded:
        def _pad_or_cut(tc, L=max_len):
            out = np.zeros((L, tc.shape[1]), dtype=np.float32)
            T = min(L, tc.shape[0])
            out[:T] = tc[:T]
            return out
        sig_list = [_pad_or_cut(x, max_len) for x in sig_list]

    src = np.asarray(sources).astype(str)
    y   = np.asarray(labels).astype(int)

    def _sel(mask):
        idx = np.where(mask)[0]
        return [sig_list[i] for i in idx], y[idx]

    m_ptb  = np.char.find(src, "PTB")  >= 0
    m_sami = (np.char.find(src, "SaMi") >= 0) | (np.char.find(src, "SAMI") >= 0)
    m_code = np.char.find(src, "CODE") >= 0

    ptb_sig,  ptb_lab  = _sel(m_ptb)
    sami_sig, sami_lab = _sel(m_sami)
    code_sig, code_lab = _sel(m_code)
    return (ptb_sig, ptb_lab), (sami_sig, sami_lab), (code_sig, code_lab)

# ---------- OrderedDict(encoder-only) → {"encoder": state_dict} 래핑 ----------
def ensure_phase1_ckpt_format(src_ckpt_path, out_ckpt_path=None):
    sd = torch.load(src_ckpt_path, map_location="cpu")
    # 이미 {"encoder": ...} 형태면 그대로 반환
    if isinstance(sd, dict) and "encoder" in sd and isinstance(sd["encoder"], dict):
        return src_ckpt_path
    # module. 접두사 제거
    if isinstance(sd, dict):
        sd = { (k.replace("module.", "", 1) if k.startswith("module.") else k): v for k,v in sd.items() }
    else:
        raise ValueError("지원하지 않는 체크포인트 형식입니다.")

    if out_ckpt_path is None:
        base, ext = os.path.splitext(src_ckpt_path)
        out_ckpt_path = base + "_wrapped.pt"
    torch.save({"encoder": sd}, out_ckpt_path)
    return out_ckpt_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="labels.npy, sources.npy, processed_signals(.npy|.npz) 위치")
    ap.add_argument("--save_dir", required=True, help="체크포인트 저장 폴더")
    ap.add_argument("--phase1_ckpt", required=True, help="encoder state_dict(.pth) 또는 {'encoder':...} 형식")
    ap.add_argument("--redo_warmup", action="store_true", help="LP+warm-up부터 다시")
    ap.add_argument("--lin_epochs", type=int, default=2)
    ap.add_argument("--ft_epochs_base", type=int, default=2)
    ap.add_argument("--ft_epochs_pl", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--tau_pos", type=float, default=0.90)
    ap.add_argument("--tau_neg", type=float, default=0.97)
    ap.add_argument("--n_tta", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--best_ckpt_path", default=None, help="redo_warmup=False일 때 warm-up 베스트(.pt)")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) 데이터 로드
    (ptb, sami, code15) = load_cached_revised(args.cache_dir, max_len=4096, assume_padded=True)

    # 2) phase1_ckpt 형식 보정
    phase1_path = ensure_phase1_ckpt_format(args.phase1_ckpt)

    # 3) 실행
    best = resume_phase2_pl_simple(
        ptb=ptb,
        sami=sami,
        code15=(code15[0], code15[1]),
        save_dir=args.save_dir,
        best_ckpt_path=(None if args.redo_warmup else args.best_ckpt_path),
        phase1_ckpt=phase1_path,
        redo_warmup=args.redo_warmup,
        lin_epochs=args.lin_epochs,
        ft_epochs_base=args.ft_epochs_base,
        ft_epochs_pl=args.ft_epochs_pl,
        n_tta=args.n_tta,
        tau_pos=args.tau_pos,
        tau_neg=args.tau_neg,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        focal_gamma=None,
        amp=args.amp,
        num_workers=args.num_workers
    )
    print("Best checkpoint:", best)

if __name__ == "__main__":
    main()


# python /Data/nh25/ECG/nh_submission3/checkpoint_after_pretrain.py \
#   --cache_dir /Data/nh25/ECG/0_Data/processed_revised \
#   --save_dir  /Data/nh25/ECG/nh_submission3/model_from_encoder_epoch10 \
#   --phase1_ckpt /Data/nh25/ECG/nh_submission3/model/phase1_encoder_epoch10.pt \
#   --redo_warmup


# python /Data/nh25/ECG/nh_submission3/checkpoint_after_pretrain.py \
#   --cache_dir /Data/nh25/ECG/0_Data/processed_revised \
#   --save_dir  /Data/nh25/ECG/nh_submission3/model_from_encoder_epoch10_simpler \
#   --phase1_ckpt /Data/nh25/ECG/nh_submission3/model/phase1_encoder_epoch10.pt \
#   --redo_warmup