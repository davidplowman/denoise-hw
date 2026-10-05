"""BPTT training script for the recurrent (temporal) DenoiseNet on short clip sequences."""

from __future__ import annotations

import argparse
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
from denoisenet_temporal import DenoiseNet, space_to_depth


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


def _loss(
    pred: torch.Tensor,
    gt_ch: torch.Tensor,
    channel_gains: torch.Tensor,
) -> torch.Tensor:
    """MSE loss in white-balanced, gamma-corrected space. Same as train.py's _loss.

    Args:
        pred, gt_ch:    (B, 16, H/4, W/4) normalised, packed Bayer tensors.
        channel_gains:  (B, 16) per-channel WB gains.
    """
    gains   = channel_gains.view(-1, 16, 1, 1)
    eps = 1e-6
    pred_wb = (torch.clamp(pred  * gains, 0.0, 1.0) + eps) ** (1 / 2.2)
    gt_wb   = (torch.clamp(gt_ch * gains, 0.0, 1.0) + eps) ** (1 / 2.2)
    return 100.0 * ((1.0 - ssim(pred_wb, gt_wb, data_range=1.0)) + 6 * F.l1_loss(pred_wb, gt_wb))


def _run_sequence(
    model: DenoiseNet, batch: dict, device: torch.device,
) -> torch.Tensor:
    """Run BPTT over one batch of T-frame sequences, returning the mean per-frame loss.

    batch["noisy"]/batch["gt"]: (B, T, H, W) raw Bayer sequences.
    The previous frame's *output* is concatenated onto the next frame's
    packed input (zeros for the first frame); no detaching between frames,
    so gradients flow through the whole sequence (true BPTT -- T is only
    10-15, so this is the whole window, not a truncation).
    """
    wl = batch["white_level"].float().to(device).view(-1, 1, 1, 1)
    noisy_seq = batch["noisy"].to(device)  # (B, T, H, W)
    gt_seq    = batch["gt"].to(device)
    cg = batch["channel_gains"].to(device).repeat_interleave(4, dim=1)  # (B, 16)

    T = noisy_seq.shape[1]
    prev_output = None
    total_loss = 0.0
    for t in range(T):
        x_t  = _pack_frame(noisy_seq[:, t], wl)
        gt_t = _pack_frame(gt_seq[:, t], wl)
        if prev_output is None:
            prev_output = torch.zeros_like(x_t)
        pred = model(torch.cat([x_t, prev_output], dim=1))
        total_loss = total_loss + _loss(pred, gt_t, cg)
        prev_output = pred

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
    parser.add_argument("--resume",       default=None, metavar="WEIGHTS")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--lr-min",       type=float, default=1e-5)
    parser.add_argument("--lr-patience",  type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=800)
    parser.add_argument("--no-pbar",      action="store_true")
    parser.add_argument("--ch",      type=int,       default=None)
    parser.add_argument("--scaling", type=str, default=None,
                         help="space-separated scaling factors, e.g. \"2 2\"; "
                              "\"\" means no downscaling at all")
    parser.add_argument("--encoders", nargs="+", type=int, default=None)
    args = parser.parse_args()
    if not 0 < args.gain_min <= args.gain_max:
        parser.error("need 0 < --gain-min <= --gain-max")
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
            loss = _run_sequence(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
            continue

        consecutive_nans = 0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False, disable=args.no_pbar):
                val_loss += _run_sequence(model, batch, device).item()
        val_loss /= len(val_loader)

        elapsed = round(time.time() - t0)
        lr = optimizer.param_groups[0]["lr"]
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(
                {"state_dict": model.state_dict(), "ch": model.ch, "scaling_factors": model.scaling_factors, "num_encoders": model.num_encoders},
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
