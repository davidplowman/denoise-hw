"""BPTT training script for the recurrent (temporal) DenoiseNet on short clip sequences."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_temporal import load_datasets, worker_init_fn
from denoisenet_temporal import DenoiseNet, depth_to_space, space_to_depth


def _receptive_field(model: DenoiseNet, h: int = 128) -> int:
    """Estimate receptive field in full-resolution Bayer pixels via gradient backpropagation.

    Same idea as train.py's version, but the model takes 32 input channels
    (current frame + previous output) instead of 16 -- doesn't change the
    spatial receptive-field calculation itself.
    """
    device = next(model.parameters()).device
    x = torch.rand(1, 32, h, h, device=device, requires_grad=True)
    was_training = model.training
    model.eval()
    with torch.enable_grad():
        out = model(x)
    if was_training:
        model.train()

    cy, cx = h // 2, h // 2
    out[0, :, cy, cx].sum().backward()

    grad = x.grad[0].abs().sum(dim=0)  # (h, h)
    nz = grad.nonzero()
    rf_h = nz[:, 0].max().item() - nz[:, 0].min().item() + 1
    rf_w = nz[:, 1].max().item() - nz[:, 1].min().item() + 1
    return max(rf_h, rf_w) * 4  # packed → full-resolution Bayer pixels


def _macs_per_pixel(model: DenoiseNet, patch_size: int = 256) -> int:
    """Count Conv2d MACs for a single frame and return the per-full-resolution-pixel figure."""
    total = 0

    def _hook(module, inp, out):
        nonlocal total
        _, C_out, H_out, W_out = out.shape
        C_in_per_group = inp[0].shape[1] // module.groups
        kH, kW = module.kernel_size
        total += C_out * H_out * W_out * C_in_per_group * kH * kW

    handles = [m.register_forward_hook(_hook)
               for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    device = next(model.parameters()).device
    h = patch_size // 4
    with torch.no_grad():
        model(torch.zeros(1, 32, h, h, device=device))
    for hnd in handles:
        hnd.remove()
    return total // (patch_size * patch_size)


def _split(t: torch.Tensor) -> torch.Tensor:
    """(B, H, W) → (B, 4, H/2, W/2) Bayer channel split."""
    return torch.stack([
        t[:, 0::2, 0::2],
        t[:, 0::2, 1::2],
        t[:, 1::2, 0::2],
        t[:, 1::2, 1::2],
    ], dim=1)


def _pack_frame(frame: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
    """(B, H, W) raw Bayer frame, wl (B,1,1,1) -> (B, 16, H/4, W/4) packed, normalised."""
    return space_to_depth(_split(frame) / wl, 2)


def _to_wb(x: torch.Tensor, channel_gains: torch.Tensor) -> torch.Tensor:
    """(B, 16, H/4, W/4) normalised, packed Bayer -> white-balanced, gamma-corrected
    space (the space the loss works in). channel_gains: (B, 16) per-channel WB gains."""
    gains = channel_gains.view(-1, 16, 1, 1)
    eps = 1e-6
    return (torch.clamp(x * gains, 0.0, 1.0) + eps) ** (1 / 2.2)


def _smooth(x: torch.Tensor, k: int) -> torch.Tensor:
    """Spatially average a (B, 16, H/4, W/4) packed tensor over k x k raw pixels.

    The packing is undone first, so each average is over same-colour Bayer
    samples: returns (B, 4, H/k, W/k) colour planes (non-overlapping average).
    k must be 1 (no smoothing; x is returned as is) or even.
    """
    if k == 1:
        return x
    return F.avg_pool2d(depth_to_space(x, 2), k // 2)


def _ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float, step: int) -> None:
    """Move ema_model's weights towards model's: ema = d*ema + (1-d)*model.

    The decay is ramped up from 0.1 at the first step towards `decay`, so
    the EMA isn't dominated by the initial weights early in training.
    """
    d = min(decay, (1 + step) / (10 + step))
    with torch.no_grad():
        for e, p in zip(ema_model.parameters(), model.parameters()):
            e.lerp_(p, 1.0 - d)
        for e, b in zip(ema_model.buffers(), model.buffers()):
            e.copy_(b)


def _loss(
    pred_wb: torch.Tensor,
    gt_wb: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Loss in white-balanced, gamma-corrected space. Same as train.py's _loss.

    Args:
        pred_wb, gt_wb: (B, 16, H/4, W/4) packed tensors, already in WB/gamma
                        space (see _to_wb).
        weight:         optional (B, 16, H/4, W/4) per-pixel weight on the L1
                        term (not normalised, so weight 1 everywhere is the
                        plain loss). The SSIM term is unweighted.
    """
    l1 = F.l1_loss(pred_wb, gt_wb) if weight is None else (weight * (pred_wb - gt_wb).abs()).mean()
    return 100.0 * ((1.0 - ssim(pred_wb, gt_wb, data_range=1.0)) + 6 * l1)


def _run_sequence(
    model: DenoiseNet, batch: dict, device: torch.device, diff_weight: float = 0.0,
    temporal_weight: float = 0.0, temporal_start: int = 1, flicker_scale: int = 4,
) -> torch.Tensor:
    """Run BPTT over one batch of T-frame sequences, returning the mean per-frame loss.

    batch["noisy"]/batch["gt"]: (B, T, H, W) raw Bayer sequences.
    The previous frame's *output* is concatenated onto the next frame's
    packed input (zeros for the first frame); no detaching between frames,
    so gradients flow through the whole sequence (true BPTT -- T is only
    10-15, so this is the whole window, not a truncation).

    With diff_weight > 0, each pixel's L1 loss (from the second frame on) is
    weighted by 1 + diff_weight * |gt_t - gt_{t-1}| (in the loss's WB/gamma
    space), to emphasise regions where the scene changed and the previous
    output must not be trusted (e.g. detail being occluded).

    With temporal_weight > 0, frames from temporal_start on also get a
    frame-to-frame consistency term,
    |(pred_t - pred_{t-1}) - (gt_t - gt_{t-1})| = |err_t - err_{t-1}|, which
    penalises changes in the error (scintillation) but not genuine scene
    changes. Skipping the first few frames avoids penalising the large,
    wanted drop in error while the recurrence converges. Both differences are
    first averaged over flicker_scale x flicker_scale raw pixels (see _smooth),
    which suppresses the GT's own pixel-level noise and targets the blobby
    shimmer that is most visible.
    """
    wl = batch["white_level"].float().to(device).view(-1, 1, 1, 1)
    noisy_seq = batch["noisy"].to(device)  # (B, T, H, W)
    gt_seq    = batch["gt"].to(device)
    cg = batch["channel_gains"].to(device).repeat_interleave(4, dim=1)  # (B, 16)

    T = noisy_seq.shape[1]
    prev_output = None
    pred_wb_prev = gt_wb_prev = None
    total_loss = 0.0
    for t in range(T):
        x_t  = _pack_frame(noisy_seq[:, t], wl)
        gt_t = _pack_frame(gt_seq[:, t], wl)
        if prev_output is None:
            prev_output = torch.zeros_like(x_t)
        pred = model(torch.cat([x_t, prev_output], dim=1))
        pred_wb = _to_wb(pred, cg)
        gt_wb   = _to_wb(gt_t, cg)
        weight = None
        if diff_weight > 0 and gt_wb_prev is not None:
            weight = 1.0 + diff_weight * (gt_wb - gt_wb_prev).abs()
        total_loss = total_loss + _loss(pred_wb, gt_wb, weight)
        if temporal_weight > 0 and t >= temporal_start:
            # pred_wb_prev is not detached: gradients reach both frames.
            tc = (_smooth(pred_wb - pred_wb_prev, flicker_scale)
                  - _smooth(gt_wb - gt_wb_prev, flicker_scale)).abs().mean()
            total_loss = total_loss + 100.0 * temporal_weight * tc
        prev_output = pred
        pred_wb_prev, gt_wb_prev = pred_wb, gt_wb

    return total_loss / T


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the recurrent (temporal) DenoiseNet.")
    parser.add_argument("--dataset",      default="/opt/datasets/IMX662_temporal")
    parser.add_argument("--filter",       nargs="*",  default=None)
    parser.add_argument("--val-frac",     type=float, default=0.1)
    parser.add_argument("--seq-len",      type=int,   default=10)
    parser.add_argument("--patch-size",   type=int,   default=256)
    parser.add_argument("--gain-aug-prob", type=float, default=0.5,
                         help="probability of applying brightness augmentation "
                              "to a training sequence (0 disables it)")
    parser.add_argument("--gain-aug-max", type=float, default=2.0,
                         help="upper bound of the [1.0, gain_aug_max] brightness gain range")
    parser.add_argument("--gain-min",     type=float, default=4.0,
                         help="lower bound of the synthetic noise gain range")
    parser.add_argument("--gain-max",     type=float, default=400.0,
                         help="upper bound of the synthetic noise gain range "
                              "(sampled log-uniformly; set equal to --gain-min "
                              "for a fixed gain)")
    parser.add_argument("--window-stride", type=int, default=0,
                         help="use every --seq-len window of each clip starting "
                              "at a multiple of this stride, per epoch (e.g. "
                              "--seq-len 10 --window-stride 5 starts at 0, 5, "
                              "10, ...); 0 = one randomly placed window per tile")
    parser.add_argument("--diff-weight",  type=float, default=0.0,
                         help="weight each pixel's L1 loss by 1 + K * |gt_t - gt_{t-1}| "
                              "(WB/gamma space), emphasising regions where the scene "
                              "changed; 0 disables it")
    parser.add_argument("--temporal-weight", type=float, default=0.0,
                         help="weight of the frame-to-frame consistency term "
                              "|(pred_t - pred_{t-1}) - (gt_t - gt_{t-1})| (WB/gamma "
                              "space; the L1 term's weight is 6); 0 disables it")
    parser.add_argument("--temporal-start", type=int, default=1,
                         help="first frame (0-based) the consistency term applies to; "
                              "skipping a few frames avoids penalising convergence")
    parser.add_argument("--flicker-scale", type=int, default=4,
                         help="average the consistency term's frame differences over "
                              "K x K raw pixels first (1 or even; 1 = no smoothing)")
    parser.add_argument("--ema",          type=float, default=0.95,
                         help="decay of an exponential moving average of the weights; "
                              "the EMA model is the one validated and saved (0 disables it)")
    parser.add_argument("--val-seed",     type=int,   default=0,
                         help="seed of the fixed, gain-stratified validation set "
                              "(-1: random validation draws every epoch instead)")
    parser.add_argument("--resume",       default=None, metavar="WEIGHTS")
    parser.add_argument("--lr",           type=float, default=2e-3)
    parser.add_argument("--lr-min",       type=float, default=1e-5)
    parser.add_argument("--lr-patience",  type=int,   default=10)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=800)
    parser.add_argument("--no-pbar",      action="store_true")
    parser.add_argument("--ch",      type=int,       default=None)
    parser.add_argument("--scaling", type=str, default=None,
                         help="space-separated scaling factors, e.g. \"2 2\"; "
                              "\"\" means no downscaling at all")
    parser.add_argument("--encoders", nargs="+", type=int, default=None)
    args = parser.parse_args()
    if not 0 <= args.ema < 1:
        parser.error("--ema must be in [0, 1)")
    if not 0 < args.gain_min <= args.gain_max:
        parser.error("need 0 < --gain-min <= --gain-max")
    if args.temporal_start < 1:
        parser.error("--temporal-start must be at least 1 (frame 0 has no previous frame)")
    if args.flicker_scale != 1 and (args.flicker_scale < 2 or args.flicker_scale % 2):
        parser.error("--flicker-scale must be 1 or an even number")
    if args.scaling is not None:
        args.scaling = [int(v) for v in args.scaling.split()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets...")
    train_ds, val_ds = load_datasets(
        dataset_root=args.dataset,
        filters=args.filter,
        val_frac=args.val_frac,
        patch_size=args.patch_size,
        seq_len=args.seq_len,
        gain_aug_prob=args.gain_aug_prob,
        gain_aug_max=args.gain_aug_max,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        window_stride=args.window_stride,
        val_seed=None if args.val_seed < 0 else args.val_seed,
    )
    print(f"  train: {len(train_ds)} sequences,  val: {len(val_ds)} sequences  "
          f"(seq_len={args.seq_len}, patch_size={args.patch_size})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    state_dict = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            if args.ch is None:
                args.ch = ckpt.get("ch")
            if args.scaling is None:
                args.scaling = ckpt.get("scaling_factors")
            if args.encoders is None:
                args.encoders = ckpt.get("num_encoders")
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

    model = DenoiseNet(ch=args.ch, scaling_factors=args.scaling, num_encoders=args.encoders).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(f"Resumed from {args.resume}")
    n_params = sum(p.numel() for p in model.parameters())
    macs = _macs_per_pixel(model, patch_size=args.patch_size)
    rf = _receptive_field(model)
    print(f"Model parameters: {n_params:,}  MACs/pixel: {macs:,}  RF: {rf}×{rf}px")
    print(f"Gain min: {args.gain_min} max: {args.gain_max}")
    print(f"GT-difference loss weight: {args.diff_weight}")
    print(f"Temporal consistency weight: {args.temporal_weight}  (from frame {args.temporal_start}, scale {args.flicker_scale})")
    print(f"Weight EMA decay: {args.ema or 'off'}  Validation: "
          + ("random draws" if args.val_seed < 0 else f"fixed, gain-stratified (seed {args.val_seed})"))

    def make_ema(m: DenoiseNet) -> DenoiseNet | None:
        if not args.ema:
            return None
        e = copy.deepcopy(m)
        e.requires_grad_(False)
        return e

    ema_model = make_ema(model)
    step = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=args.lr_patience,
    )
    Path("checkpoints").mkdir(exist_ok=True)
    best_val = float("inf")
    consecutive_nans = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, disable=args.no_pbar):
            optimizer.zero_grad()
            loss = _run_sequence(model, batch, device, args.diff_weight,
                                 args.temporal_weight, args.temporal_start, args.flicker_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_model is not None:
                _ema_update(ema_model, model, args.ema, step)
            step += 1
            train_loss += loss.item()
        train_loss /= len(train_loader)

        if math.isnan(train_loss):
            consecutive_nans += 1
            if consecutive_nans >= 10:
                print(f"Epoch {epoch}  NaN loss — {consecutive_nans} in a row, quitting.")
                break
            model = DenoiseNet(ch=args.ch, scaling_factors=args.scaling, num_encoders=args.encoders).to(device)
            if best_val < float("inf"):
                ckpt = torch.load("checkpoints/best_temporal.pth", map_location=device)
                model.load_state_dict(ckpt["state_dict"])
                print(f"Epoch {epoch}  NaN loss ({consecutive_nans}/10) — "
                      f"reloaded checkpoints/best_temporal.pth (val {best_val:.5f}) and recreated optimizer.")
            else:
                print(f"Epoch {epoch}  NaN loss ({consecutive_nans}/10) — "
                      f"no checkpoint yet, recreating model from scratch.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)
            ema_model = make_ema(model)
            step = 0
            continue

        consecutive_nans = 0

        # validate (and save) the EMA weights when enabled
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False, disable=args.no_pbar):
                val_loss += _run_sequence(eval_model, batch, device, args.diff_weight,
                                            args.temporal_weight, args.temporal_start, args.flicker_scale).item()
        val_loss /= len(val_loader)

        elapsed = round(time.time() - t0)
        lr = optimizer.param_groups[0]["lr"]
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(
                {"state_dict": eval_model.state_dict(), "ch": model.ch, "scaling_factors": model.scaling_factors, "num_encoders": model.num_encoders,
                 "diff_weight": args.diff_weight,
                 "temporal_weight": args.temporal_weight, "temporal_start": args.temporal_start,
                 "flicker_scale": args.flicker_scale},
                "checkpoints/best_temporal.pth",
            )
        tag = f"  -> improved: {val_loss:.5f}" if improved else ""
        print(f"Epoch {epoch}  train {train_loss:.5f}  val {val_loss:.5f}  lr {lr:.2e}  {elapsed}s{tag}")

        scheduler.step(val_loss)
        if optimizer.param_groups[0]["lr"] <= args.lr_min:
            print("LR below minimum — stopping.")
            break


if __name__ == "__main__":
    main()
