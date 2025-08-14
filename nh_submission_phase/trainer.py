import sys
import os
sys.path.append(os.getcwd())
from model import ShallowECGEncoder, DualBranchECGEncoder, ImprovedHeads, TemperatureScaler
from util_nh import *
from helper_code import *
from team_code import *

# ---------------------------
# 개선된 파이프라인 러너
# ---------------------------
import wandb

def run_phase1_contrastive_improved(all_signals, save_dir, encoder_type='shallow',
                                   epochs=25, batch_size=256, lr=1e-3, temp=0.1, 
                                   wd=1e-4, amp=True, num_workers=0, verbose=False,
                                   wandb_proj_name="improved-ecg-contrastive"):
    """
    개선된 Phase 1 Contrastive Learning
    - 정보 소실 방지를 위한 개선된 인코더 사용
    - 안정적인 학습 파라미터
    """
    if verbose:
        print(f"🚀 Phase 1: Improved Contrastive Learning")
        print(f"   📊 Encoder: {encoder_type}")
        print(f"   🔢 Epochs: {epochs}, Batch size: {batch_size}")
    
    wandb.init(
        project=wandb_proj_name,
        name=f"phase1_contrastive_{encoder_type}", 
        config={
            "encoder_type": encoder_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "temp": temp,
            "weight_decay": wd,
            "amp": amp
        })
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ContrastiveECGDataset(all_signals)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, 
                       num_workers=num_workers, collate_fn=collate_safe, 
                       pin_memory=True, drop_last=True)

    # 개선된 인코더 선택
    if encoder_type == 'shallow':
        encoder = ShallowECGEncoder(in_channels=12, dropout=0.1).to(device)
    elif encoder_type == 'dual':
        encoder = DualBranchECGEncoder(in_channels=12, dropout=0.1).to(device)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    # 개선된 헤드 (contrastive projection만 사용)
    proj_head = nn.Sequential(
        nn.Linear(encoder.in_dim, 512),
        nn.LeakyReLU(0.1),
        nn.Linear(512, 128)
    ).to(device)
    
    # Optimizer 및 Loss
    opt = optim.AdamW(list(encoder.parameters()) + list(proj_head.parameters()), 
                     lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=amp)
    ntx = NTXentLoss(temp=temp)

    if verbose:
        print(f"   🎯 Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"   💾 Embedding dimension: {encoder.in_dim}")

    # Learning rate scheduler 추가
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    encoder.train(); proj_head.train()
    best_loss = float('inf')
    
    for ep in range(1, epochs+1):
        encoder.train(); proj_head.train() 
        run_loss = 0.0
        
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            with autocast(enabled=amp):
                # 개선된 인코더 사용
                f1 = encoder.forward_features(v1)
                f2 = encoder.forward_features(v2)
                
                # Projection
                z1 = F.normalize(proj_head(f1), dim=1)
                z2 = F.normalize(proj_head(f2), dim=1)
                
                loss = ntx(z1, z2)
            
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run_loss += float(loss.item())
        
        scheduler.step()
        avg_train_loss = run_loss/len(loader)
        
        # Validation (same data, no augmentation)
        encoder.eval(); proj_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_count = 0
            for v1, v2 in loader:
                if val_count >= 10:  # 빠른 validation을 위해 제한
                    break
                v1, v2 = v1.to(device), v2.to(device)
                with autocast(enabled=amp):
                    f1 = encoder.forward_features(v1)
                    f2 = encoder.forward_features(v2)
                    z1 = F.normalize(proj_head(f1), dim=1)
                    z2 = F.normalize(proj_head(f2), dim=1)
                    loss = ntx(z1, z2)
                val_loss += float(loss.item())
                val_count += 1
        
        avg_val_loss = val_loss/max(1, val_count)
        
        # Wandb logging
        wandb.log({
            "phase1_loss": avg_train_loss, 
            "phase1_val_loss": avg_val_loss, 
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": ep
        })
        
        if verbose:
            print(f"[P1] epoch {ep:2d}/{epochs} loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}")
        
        # 체크포인트 저장 (best model만)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"phase1_{encoder_type}_encoder_best.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "proj_head": proj_head.state_dict(),
                "epoch": ep,
                "loss": best_loss
            }, ckpt_path)
            
            if verbose and ep % 5 == 0:
                print(f"   💾 Saved best model: {ckpt_path}")

    # 최종 모델 저장
    final_path = os.path.join(save_dir, f"phase1_{encoder_type}_encoder_final.pt")
    torch.save({
        "encoder": encoder.state_dict(),
        "proj_head": proj_head.state_dict(),
        "epoch": epochs,
        "loss": avg_train_loss
    }, final_path)
    
    if verbose:
        print(f"✅ Phase 1 completed. Best validation loss: {best_loss:.4f}")
    
    wandb.finish()
    return ckpt_path  # best model 경로 반환


@torch.no_grad()
def evaluate_acc_improved(encoder, heads, loader, device):
    """개선된 정확도 평가"""
    encoder.eval(); heads.eval()
    total, correct = 0, 0
    all_probs = []
    all_labels = []
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        feat = encoder.forward_features(xb)
        logits, _ = heads(feat)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(1)
        
        total += yb.size(0)
        correct += (pred == yb).sum().item()
        
        all_probs.append(probs.cpu())
        all_labels.append(yb.cpu())
    
    accuracy = correct / max(1, total)
    
    # 추가 메트릭 계산
    all_probs = torch.cat(all_probs, 0)
    all_labels = torch.cat(all_labels, 0)
    
    # 클래스별 정확도
    class_accs = {}
    for cls in torch.unique(all_labels):
        mask = (all_labels == cls)
        if mask.sum() > 0:
            cls_preds = all_probs[mask].argmax(1)
            cls_acc = (cls_preds == cls).float().mean().item()
            class_accs[f"class_{cls.item()}_acc"] = cls_acc
    
    return accuracy, class_accs


def run_phase2_improved(
    ptb, sami, code15, save_dir, encoder_ckpt=None, encoder_type='shallow',
    # 개선된 학습 파라미터
    lin_epochs=8, ft_epochs_base=5, ft_epochs_pl=5, 
    # pseudo-labeling 파라미터
    n_tta=3, tau_pos=0.90, tau_neg=0.95,
    # 옵티마이저 설정
    batch_size=128, lr=5e-4, wd=1e-4, focal_gamma=None, amp=True, num_workers=0,
    verbose=False
):
    """
    개선된 Phase 2: Classification Fine-tuning
    - 개선된 인코더 및 헤드 사용
    - 더 안정적인 학습 설정
    - 향상된 pseudo-labeling
    """
    if verbose:
        print(f"🚀 Phase 2: Improved Classification Fine-tuning")
        print(f"   📊 Encoder: {encoder_type}")
        print(f"   🔢 Linear epochs: {lin_epochs}, FT epochs: {ft_epochs_base}, PL epochs: {ft_epochs_pl}")
    
    wandb.init(
        project="improved-ecg-classification",
        name=f"phase2_finetune_{encoder_type}",
        config={
            "encoder_type": encoder_type,
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

    # 데이터셋 준비
    MAX_LEN = 4096

    # 초기 학습용 데이터 (CODE15 일부만 포함)
    p = 0.1  # CODE15의 10%만 초기에 사용
    N = len(code_sig); k = max(1, int(N*p))
    sel = np.random.choice(N, size=k, replace=False) if k < N else np.arange(N)
    
    tr_sig_unpadded = list(ptb_sig) + list(sami_sig) + [code_sig[i] for i in sel]
    tr_sig = pad_signals(tr_sig_unpadded, MAX_LEN)
    tr_lab = np.concatenate([ptb_lab, sami_lab, np.asarray([code_lab[i] for i in sel])])

    val_sig_unpadded = list(ptb_sig) + list(sami_sig)
    val_sig = pad_signals(val_sig_unpadded, MAX_LEN)
    val_lab = np.concatenate([ptb_lab, sami_lab])

    train_ds = MultiSourceECGDataset(tr_sig, tr_lab, confidences=None, aug_for_train=False)
    val_ds = SupervisedECGDataset(val_sig, val_lab)

    # 프로토타입용 데이터셋
    neg_ds = SupervisedECGDataset(pad_signals(list(ptb_sig), MAX_LEN), ptb_lab)
    pos_ds = SupervisedECGDataset(pad_signals(list(sami_sig), MAX_LEN), sami_lab)

    neg_loader = DataLoader(neg_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)
    pos_loader = DataLoader(pos_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)

    # 개선된 모델 로드
    if encoder_type == 'shallow':
        encoder = ShallowECGEncoder(in_channels=12, dropout=0.1)
    elif encoder_type == 'dual':
        encoder = DualBranchECGEncoder(in_channels=12, dropout=0.1)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    # Phase 1 가중치 로드
    if encoder_ckpt and os.path.exists(encoder_ckpt):
        state = torch.load(encoder_ckpt, map_location="cpu")
        encoder.load_state_dict(state["encoder"], strict=False)
        if verbose:
            print(f"   📥 Loaded encoder from: {encoder_ckpt}")
    
    # 개선된 헤드
    heads = ImprovedHeads(in_dim=encoder.in_dim, n_classes=2, proj_dim=128, dropout=0.2)
    
    encoder, heads = encoder.to(device), heads.to(device)
    
    if verbose:
        print(f"   🧠 Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"   💾 Embedding dimension: {encoder.in_dim}")

    # Linear probe 단계
    if verbose:
        print(f"   🔧 Step 1: Linear Probe ({lin_epochs} epochs)")
    
    lp = LinearProbeTrainer(
        encoder=encoder,
        signals_tensor=train_ds.signals,
        labels_array=train_ds.labels.cpu().numpy(),
        save_path=os.path.join(save_dir, f"phase2_{encoder_type}_linear_head.pt"),
        batch_size=batch_size, lr=lr*2, device=device, verbose=verbose  # Linear probe는 더 큰 LR
    )
    lp.train(epochs=lin_epochs)

    # Warm-up fine-tune 단계
    if verbose:
        print(f"   🔥 Step 2: Warm-up Fine-tune ({ft_epochs_base} epochs)")
    
    # 클래스 균형 샘플링
    y = train_ds.labels.cpu().numpy()
    classes, counts = np.unique(y, return_counts=True)
    inv = {c: 1.0 / max(1, cnt) for c, cnt in zip(classes, counts)}
    w = np.array([inv[int(lbl)] for lbl in y], dtype=np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate_safe)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # 더 작은 learning rate로 안정적인 학습
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(heads.parameters()), 
                           lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs_base)
    scaler = GradScaler(enabled=amp)

    def ce_or_focal(logits, targets):
        if focal_gamma is None:
            return F.cross_entropy(logits, targets, reduction='none')
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1-pt)**focal_gamma) * ce

    best_acc, best_path = 0.0, os.path.join(save_dir, f"phase2_{encoder_type}_finetuned.pt")
    os.makedirs(save_dir, exist_ok=True)

    # Warm-up fine-tuning
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

        scheduler.step()
        avg_train_loss = tr_loss/len(train_loader)
        
        # Validation
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
        acc, class_accs = evaluate_acc_improved(encoder, heads, val_loader, device)
        
        # Wandb logging
        log_dict = {
            "warmup_loss": avg_train_loss,
            "warmup_val_loss": avg_val_loss,
            "val_acc": acc,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": ep
        }
        log_dict.update(class_accs)
        wandb.log(log_dict)
        
        if verbose:
            print(f"[Phase2][warmup {ep:2d}/{ft_epochs_base}] loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | val_acc={acc:.4f}")
            for cls_name, cls_acc in class_accs.items():
                print(f"    {cls_name}: {cls_acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, best_path)
            if verbose:
                print(f"  ↳ 💾 saved best: {best_path} (acc={best_acc:.4f})")

    # Pseudo-labeling 단계
    if verbose:
        print(f"   🎯 Step 3: Pseudo-labeling ({ft_epochs_pl} epochs)")
    
    # Temperature scaling
    temp_scaler = calibrate_temperature(encoder, heads, val_loader, device)
    
    # Class prototypes
    proto_pos, proto_neg = build_class_prototypes(encoder, heads, pos_loader, neg_loader, device, n_aug=1)
    
    # CODE-15 전체 스코어링
    code_ds = SupervisedECGDataset(pad_signals(list(code_sig), MAX_LEN), np.asarray(code_lab))
    code_loader = DataLoader(code_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=collate_safe)
    
    kept_signals, kept_labels, kept_weights = [], [], []

    with torch.no_grad():
        for xb, yb in code_loader:
            xb = xb.to(device)
            s, pred, p_mean = score_code15_batch_improved(encoder, heads, temp_scaler, xb, 
                                                        proto_pos, proto_neg, device, n_tta=n_tta)
            yb_cpu = yb.cpu()
            match = (pred == yb_cpu)

            # 더 보수적인 임계값 적용
            keep_mask = torch.zeros_like(match, dtype=torch.bool)
            keep_mask[(match) & (pred==1) & (s>=tau_pos)] = True
            keep_mask[(match) & (pred==0) & (s>=tau_neg)] = True

            for i, keep in enumerate(keep_mask.tolist()):
                if keep:
                    x_i = xb[i].detach().cpu().contiguous().clone().permute(1, 0).numpy()
                    kept_signals.append(x_i)
                    kept_labels.append(int(yb_cpu[i].item()))
                    kept_weights.append(float(s[i].item()))

    if verbose:
        print(f"   📊 Selected {len(kept_signals)} / {len(code_sig)} CODE-15 samples for pseudo-labeling")

    if len(kept_signals) == 0:
        if verbose:
            print("   ⚠️ No samples passed the threshold; skipping PL fine-tune.")
        wandb.finish()
        return best_path

    # 확장된 데이터셋으로 추가 fine-tuning
    ext_signals_unpadded = list(ptb_sig) + list(sami_sig) + kept_signals
    ext_labels = np.concatenate([ptb_lab, sami_lab, np.asarray(kept_labels, dtype=int)])
    ext_signals = pad_signals(ext_signals_unpadded, MAX_LEN)

    def _to_ct_tensor(x):
        t = torch.tensor(np.ascontiguousarray(x))
        t = t.transpose(0, 1).contiguous()
        return t.clone()

    ext_signals = [_to_ct_tensor(x) for x in ext_signals]

    y_all = ext_labels
    classes, counts = np.unique(y_all, return_counts=True)
    
    if verbose:
        print(f"   📈 Extended dataset: {dict(zip(classes, counts))}")

    # 더 균형잡힌 가중치
    weight_dict = {0: 0.6, 1: 0.7}  # 약간 더 보수적
    w_all = np.array([weight_dict.get(int(lbl), 0.5) for lbl in y_all], dtype=np.float32)

    ext_train = MultiSourceECGDataset(ext_signals, ext_labels, confidences=None, aug_for_train=False)
    sampler2 = WeightedRandomSampler(torch.from_numpy(w_all), num_samples=len(w_all), replacement=True)
    ext_loader = DataLoader(ext_train, batch_size=batch_size, sampler=sampler2,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            collate_fn=collate_hard)

    # Pseudo-labeling fine-tuning (더 작은 learning rate)
    opt_pl = torch.optim.AdamW(list(encoder.parameters()) + list(heads.parameters()), 
                              lr=lr*0.5, weight_decay=wd)  # 절반 학습률
    scheduler_pl = optim.lr_scheduler.CosineAnnealingLR(opt_pl, T_max=ft_epochs_pl)

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
            scaler.step(opt_pl); scaler.update(); opt_pl.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())

        scheduler_pl.step()
        avg_train_loss = tr_loss/len(ext_loader)
        
        # Validation on original validation set
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
        acc, class_accs = evaluate_acc_improved(encoder, heads, val_loader, device)
        
        # Wandb logging
        log_dict = {
            "pl_loss": avg_train_loss,
            "pl_val_loss": avg_val_loss,
            "val_acc": acc,
            "pl_learning_rate": scheduler_pl.get_last_lr()[0],
            "epoch": ep
        }
        log_dict.update({f"pl_{k}": v for k, v in class_accs.items()})
        wandb.log(log_dict)
        
        if verbose:
            print(f"[Phase2][PL {ep:2d}/{ft_epochs_pl}] loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | val_acc={acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, best_path)
            if verbose:
                print(f"  ↳ 💾 saved best PL: {best_path} (acc={best_acc:.4f})")

    if verbose:
        print(f"✅ Phase 2 completed. Best accuracy: {best_acc:.4f}")
    
    wandb.finish()
    return best_path


@torch.no_grad()
def score_code15_batch_improved(encoder, heads, temp_scaler, xb, proto_pos, proto_neg, device, n_tta=3):
    """
    개선된 CODE15 배치 스코어링
    - 더 안정적인 TTA
    - 개선된 신뢰도 계산
    """
    encoder.eval(); heads.eval()
    B = xb.size(0)
    
    # 효율적인 TTA
    xb_repeated = xb.repeat(n_tta, 1, 1)
    
    # 더 안정적인 augmentation
    x_aug_batch = torch.stack([
        augment_signal_v2(xb_repeated[i]) if i % 2 == 1 else xb_repeated[i]  # 50%만 augment
        for i in range(len(xb_repeated))
    ], dim=0)
    
    feat_batch = encoder.forward_features(x_aug_batch.to(device))
    logits_batch, z_batch = heads(feat_batch)
    
    if temp_scaler is not None:
        logits_batch = temp_scaler(logits_batch)
    
    probs_batch = torch.softmax(logits_batch, dim=1)
    
    # Reshape
    probs = probs_batch.view(n_tta, B, 2)
    embeds = z_batch.view(n_tta, B, -1)
    
    # 통계 계산
    p_mean = probs.mean(0)
    p_std = probs.std(0, correction=0).amax(dim=1)
    z_mean = embeds.mean(0)
    
    # 개선된 신뢰도 계산
    max_prob = p_mean.max(1)[0]
    margin = (p_mean[:, 1] - p_mean[:, 0]).abs()
    consistency = torch.exp(-p_std * 2)  # 더 민감하게
    
    # 프로토타입 일치도
    if proto_pos is not None and proto_neg is not None:
        sim_pos = F.cosine_similarity(z_mean, proto_pos.unsqueeze(0), dim=1)
        sim_neg = F.cosine_similarity(z_mean, proto_neg.unsqueeze(0), dim=1)
        proto_confidence = torch.sigmoid((sim_pos - sim_neg) * 2)  # 더 강한 차이 강조
    else:
        proto_confidence = torch.ones(B, device=device)
    
    # 엔트로피 기반
    entropy = -(p_mean * torch.log(p_mean + 1e-8)).sum(dim=1)
    uncertainty = 1.0 - entropy / math.log(2)
    
    # 더 보수적인 가중치 조합
    confidence = (0.3 * max_prob + 
                 0.3 * margin + 
                 0.2 * consistency + 
                 0.1 * proto_confidence + 
                 0.1 * uncertainty)
    
    pred = p_mean.argmax(1)
    
    return confidence.cpu(), pred.cpu(), p_mean.cpu()


def calibrate_temperature(encoder, heads, val_loader, device, max_steps=200, lr=1e-2):
    """
    개선된 Temperature Scaling
    PTB-XL + SaMi-Trop 검증셋으로 T-scaling 파라미터 학습
    """
    encoder.eval(); heads.eval()
    scaler = TemperatureScaler().to(device) 
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_steps, line_search_fn='strong_wolfe')

    # 모든 검증 데이터 수집
    xs, ys = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            feat = encoder.forward_features(xb)
            logits, _ = heads(feat)
            xs.append(logits); ys.append(yb)
    
    if len(xs) == 0:
        return scaler
    
    X = torch.cat(xs, 0); Y = torch.cat(ys, 0)

    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = ce(scaler(X), Y)
        loss.backward()
        return loss
    
    try:
        opt.step(closure)
    except:
        # LBFGS 실패 시 Adam으로 fallback
        opt_adam = torch.optim.Adam(scaler.parameters(), lr=lr)
        for _ in range(50):
            opt_adam.zero_grad()
            loss = ce(scaler(X), Y)
            loss.backward()
            opt_adam.step()
    
    return scaler


@torch.no_grad()
def build_class_prototypes(encoder, heads, pos_loader, neg_loader, device, n_aug=1):
    """
    개선된 클래스 프로토타입 생성
    SaMi(양성), PTB-XL(음성)에서 임베딩 평균으로 프로토타입 생성
    """
    encoder.eval(); heads.eval()
    zs_pos, zs_neg = [], []
    
    # 여러 번 augmentation하여 더 robust한 프로토타입 생성
    for aug_round in range(n_aug):
        # 양성 프로토타입 (SaMi)
        for xb, yb in pos_loader:
            xb = xb.to(device)
            if aug_round > 0:  # 첫 번째는 원본, 나머지는 augmented
                xb = augment_signal_v2(xb)
            feat = encoder.forward_features(xb)
            _, z = heads(feat)
            zs_pos.append(z)
        
        # 음성 프로토타입 (PTB)
        for xb, yb in neg_loader:
            xb = xb.to(device)
            if aug_round > 0:
                xb = augment_signal_v2(xb)
            feat = encoder.forward_features(xb)
            _, z = heads(feat)
            zs_neg.append(z)
    
    if len(zs_pos) == 0 or len(zs_neg) == 0:
        return None, None
    
    # 프로토타입 계산 및 정규화
    proto_pos = torch.cat(zs_pos, 0).mean(0)   # (D,)
    proto_neg = torch.cat(zs_neg, 0).mean(0)   # (D,)
    proto_pos = F.normalize(proto_pos, dim=0)
    proto_neg = F.normalize(proto_neg, dim=0)
    
    return proto_pos.detach(), proto_neg.detach()


class LinearProbeTrainer(nn.Module):
    """
    개선된 Linear Probe 학습기
    """
    def __init__(self, encoder, signals_tensor, labels_array, save_path, 
                 batch_size=64, lr=1e-3, device='cpu', verbose=False):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        
        # Encoder 파라미터 고정
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Linear head 생성
        in_dim = getattr(self.encoder, "in_dim", 512)
        n_classes = int(len(np.unique(labels_array)))
        
        # 개선된 Linear head (Dropout 추가)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        ).to(self.device)
        
        self.save_path = save_path
        self.verbose = verbose

        # 데이터셋 및 로더
        dataset = SupervisedECGDataset(signals_tensor, labels_array)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=collate_safe, pin_memory=True)

        # 클래스 가중치 (불균형 데이터 대응)
        classes, counts = np.unique(labels_array, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(weights)  # 정규화
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # 옵티마이저 및 손실함수
        self.opt = optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-4)
        self.crit = nn.CrossEntropyLoss(weight=class_weights)
        
        if self.verbose:
            print(f"   🔧 Linear probe setup: {in_dim} -> {n_classes}")
            print(f"   ⚖️ Class weights: {class_weights.cpu().numpy()}")

    def train(self, epochs=5):
        if self.verbose:
            print(f"   🏃 Training linear probe for {epochs} epochs...")
        
        self.head.train()
        best_loss = float('inf')
        
        for e in range(epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                # Encoder features (frozen)
                with torch.no_grad():
                    if hasattr(self.encoder, "forward_features"):
                        h = self.encoder.forward_features(xb)
                    else:
                        h, _ = self.encoder(xb)
                
                # Linear head 학습
                logits = self.head(h)
                loss = self.crit(logits, yb)
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                total_loss += loss.item()
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_samples += yb.size(0)
            
            avg_loss = total_loss / len(self.loader)
            accuracy = total_correct / max(1, total_samples)
            
            if self.verbose:
                print(f"     Epoch {e+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.4f}")
            
            # Best model 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.head.state_dict(), self.save_path)
        
        if self.verbose:
            print(f"   ✅ Linear probe completed. Best loss: {best_loss:.4f}")
            print(f"   💾 Saved to: {self.save_path}")


# 호환성을 위한 alias 함수들
def run_phase1_contrastive(all_signals, save_dir, epochs=100, 
                           batch_size=128, lr=1e-3, temp=0.2, 
                           wd=1e-4, amp=True, num_workers=0,
                           wandb_proj_name="ecg-contrastive-pseudo-labeling"):
    """
    기존 호환성을 위한 wrapper 함수
    기본적으로 개선된 shallow encoder 사용
    """
    return run_phase1_contrastive_improved(
        all_signals=all_signals,
        save_dir=save_dir,
        encoder_type='shallow',  # 기본값을 shallow로
        epochs=min(epochs, 30),  # 너무 긴 학습 방지
        batch_size=batch_size,
        lr=lr,
        temp=temp,
        wd=wd,
        amp=amp,
        num_workers=num_workers,
        verbose=True,
        wandb_proj_name=wandb_proj_name
    )


def run_phase2(ptb, sami, code15, save_dir, encoder_ckpt=None,
               lin_epochs=3, ft_epochs_base=5, ft_epochs_pl=5, 
               n_tta=3, tau_pos=0.90, tau_neg=0.95,
               batch_size=128, lr=1e-3, wd=1e-4, focal_gamma=None, 
               amp=True, num_workers=0):
    """
    기존 호환성을 위한 wrapper 함수
    기본적으로 개선된 shallow encoder 사용
    """
    return run_phase2_improved(
        ptb=ptb, sami=sami, code15=code15,
        save_dir=save_dir,
        encoder_ckpt=encoder_ckpt,
        encoder_type='shallow',  # 기본값을 shallow로
        lin_epochs=max(lin_epochs, 8),  # 최소 8 에포크
        ft_epochs_base=max(ft_epochs_base, 5),
        ft_epochs_pl=ft_epochs_pl,
        n_tta=n_tta,
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        batch_size=batch_size,
        lr=lr*0.5,  # 더 안정적인 학습률
        wd=wd,
        focal_gamma=focal_gamma,
        amp=amp,
        num_workers=num_workers,
        verbose=True
    )