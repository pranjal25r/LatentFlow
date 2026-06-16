"""Train the LatentFlow Diffusion Transformer on cached VAE latents.

Pipeline (Stage 2 of latent diffusion):
  cached latents --> add noise (q_sample) --> DiT predicts noise --> MSE loss.

Features:
  - Config-driven (configs/default.yaml is the source of truth).
  - Mixed precision (fp16 / bf16) via torch.amp.
  - EMA of weights (sampling uses the EMA model).
  - Warmup + cosine learning-rate schedule.
  - Resumable checkpoints (model + EMA + optimizer + scheduler + step).
  - TensorBoard loss logging + periodic decoded-sample previews.

Run:
  python scripts/train.py --config configs/default.yaml --preset s2
  python scripts/train.py --config configs/default.yaml --resume checkpoints/ckpt_step5000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make the project root importable when run as `python scripts/train.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from latentflow.models.dit import DiT, DiT_S_2, DiT_B_2  # noqa: E402
from latentflow.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from latentflow.data.dataset import CachedLatentDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# Small self-contained helpers (no coupling to utils/ or vae.py)
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    """Load the YAML config as a plain nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any], preset: str = "config") -> DiT:
    """Construct a DiT according to the chosen preset.

    Args:
        config: Parsed config dict.
        preset: One of {"config", "s2", "b2"}. "config" uses the `dit` block.

    Returns:
        An (un-trained) DiT instance.
    """
    pred = config["diffusion"]["prediction_type"]
    lc = config["latent_channels"]
    if preset == "s2":
        return DiT_S_2(latent_channels=lc, prediction_type=pred)
    if preset == "b2":
        return DiT_B_2(latent_channels=lc, prediction_type=pred)
    return DiT.from_pretrained_config(config)


class EMA:
    """Exponential moving average of model parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the shadow weights toward the current model weights."""
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        """Load the shadow weights into `model` (in place)."""
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}


def build_vae(config: Dict[str, Any], device: torch.device):
    """Load the frozen pretrained VAE (diffusers) for decoding previews.

    Kept independent of the repo's own vae.py so this script has no coupling.
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(config["vae_model"]).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, scale: float) -> torch.Tensor:
    """Decode scaled latents back to images in [-1, 1]."""
    imgs = vae.decode(latents / scale).sample
    return imgs.clamp(-1.0, 1.0)


def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay to ~0."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device}")

    # Precision setup
    prec = config["training"]["mixed_precision"]
    use_amp = prec in ("fp16", "bf16") and device_type == "cuda"
    amp_dtype = torch.float16 if prec == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(device_type, enabled=(use_amp and prec == "fp16"))

    # Data (CLI overrides config for quick local runs)
    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else config["training"]["num_workers"]
    latent_dir = config["paths"]["latent_cache_dir"]
    dataset = CachedLatentDataset(latent_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
    )
    print(f"Cached latents: {len(dataset)} | batches/epoch: {len(loader)}")

    # Model + diffusion
    model = build_model(config, args.preset).to(device)
    print(f"Model preset: {args.preset} | parameters: {model.num_parameters():,}")
    diffusion = GaussianDiffusion(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        prediction_type=config["diffusion"]["prediction_type"],
    ).to(device)

    ema = EMA(model, config["training"]["ema_decay"])

    # Optimizer + schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    steps_per_epoch = len(loader)
    total_steps = args.max_steps or config["training"]["num_epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(config["training"]["warmup_steps"], total_steps)
    )

    # Output dirs + logging
    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    log_dir = Path(config["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer: Optional[SummaryWriter] = SummaryWriter(str(log_dir))
    except Exception as e:  # pragma: no cover
        print(f"[warn] TensorBoard unavailable ({e}); console logging only.")
        writer = None

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at step {global_step}")

    vae = None  # built lazily for previews
    model.train()
    t0 = time.time()
    running = 0.0

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        for latents in loader:
            latents = latents.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                loss = diffusion.training_loss(model, latents)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            global_step += 1
            running += loss.item()

            # ---- logging ----
            if global_step % args.log_every == 0:
                avg = running / args.log_every
                running = 0.0
                its = args.log_every / (time.time() - t0)
                t0 = time.time()
                lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch} step {global_step}/{total_steps} "
                    f"loss {avg:.4f} lr {lr:.2e} {its:.2f} it/s"
                )
                if writer is not None:
                    writer.add_scalar("train/loss", avg, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/it_per_s", its, global_step)

            # ---- decoded preview from EMA model ----
            if args.preview_every and global_step % args.preview_every == 0:
                try:
                    if vae is None:
                        vae = build_vae(config, device)
                    _save_preview(
                        config, model, ema, diffusion, vae, device,
                        log_dir / f"preview_step{global_step}.png", args.sample_preview_n,
                    )
                    model.train()
                except Exception as e:
                    print(f"[warn] preview failed ({e}); continuing training.")

            # ---- checkpoint ----
            if global_step % args.save_every == 0:
                _save_ckpt(ckpt_dir / f"ckpt_step{global_step}.pt", model, ema,
                           optimizer, scheduler, global_step, epoch, config, args.preset)

            if args.max_steps and global_step >= args.max_steps:
                _save_ckpt(ckpt_dir / "ckpt_final.pt", model, ema, optimizer,
                           scheduler, global_step, epoch, config, args.preset)
                print("Reached max_steps; stopping.")
                return

    _save_ckpt(ckpt_dir / "ckpt_final.pt", model, ema, optimizer, scheduler,
               global_step, epoch, config, args.preset)
    print(f"Training complete. Final step {global_step}. "
          f"Approx GPU-time: {(time.time() - t0):.0f}s for last window "
          f"(use TensorBoard it/s + total steps for a full estimate).")


def _save_ckpt(path: Path, model, ema, optimizer, scheduler, step, epoch, config, preset) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": config,
            "preset": preset,
        },
        path,
    )
    print(f"  saved checkpoint: {path}")


@torch.no_grad()
def _save_preview(config, model, ema, diffusion, vae, device, out_path: Path, n: int) -> None:
    """Sample n images from the EMA model and save a grid (visual sanity only)."""
    from torchvision.utils import make_grid, save_image

    # Snapshot training weights, load EMA weights, sample, then restore.
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    model.eval()

    shape = (n, config["latent_channels"], config["latent_height"], config["latent_width"])
    steps = config["sampling"]["ddim_steps"]
    latents = diffusion.ddim_sample(model, shape=shape, num_steps=steps, progress_bar=False)
    imgs = decode_latents(vae, latents, config["latent_scale_factor"])
    grid = make_grid((imgs + 1) / 2, nrow=int(math.sqrt(n)) or 1)
    save_image(grid, str(out_path))
    print(f"  saved preview: {out_path}")

    model.load_state_dict(backup)  # restore training weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LatentFlow DiT.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--preset", type=str, default="s2", choices=["config", "s2", "b2"],
                   help="Model size. 's2' (~33M) is recommended for a single GPU.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override config batch size (use a small value on CPU).")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Override dataloader workers (use 0 on CPU to silence worker spam).")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--preview-every", type=int, default=2000,
                   help="0 disables decoded previews during training.")
    p.add_argument("--sample-preview-n", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override total steps (otherwise num_epochs * batches).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
