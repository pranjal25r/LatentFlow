"""Generate face images from a trained LatentFlow checkpoint.

Loads the EMA weights, samples latents with DDIM (or DDPM), decodes them with
the frozen VAE, and saves a grid plus individual PNGs.

Run:
  python scripts/sample.py --ckpt checkpoints/ckpt_final.pt --n 16 --out samples/
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from latentflow.models.dit import DiT, DiT_S_2, DiT_B_2  # noqa: E402
from latentflow.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402


def build_model(config: Dict[str, Any], preset: str) -> DiT:
    pred = config["diffusion"]["prediction_type"]
    lc = config["latent_channels"]
    if preset == "s2":
        return DiT_S_2(latent_channels=lc, prediction_type=pred)
    if preset == "b2":
        return DiT_B_2(latent_channels=lc, prediction_type=pred)
    return DiT.from_pretrained_config(config)


def build_vae(config: Dict[str, Any], device: torch.device):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(config["vae_model"]).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, scale: float) -> torch.Tensor:
    return vae.decode(latents / scale).sample.clamp(-1.0, 1.0)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    config: Dict[str, Any] = ckpt["config"]
    preset: str = ckpt.get("preset", "config")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Model with EMA weights (falls back to raw weights if requested/missing).
    model = build_model(config, preset).to(device)
    weights = ckpt["ema"] if (args.use_ema and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(weights)
    model.eval()
    print(f"Loaded {'EMA' if args.use_ema else 'raw'} weights | preset {preset} "
          f"| params {model.num_parameters():,}")

    diffusion = GaussianDiffusion(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        prediction_type=config["diffusion"]["prediction_type"],
    ).to(device)

    vae = build_vae(config, device)

    shape = (args.n, config["latent_channels"], config["latent_height"], config["latent_width"])
    sampler = args.sampler or config["sampling"]["sampler"]
    steps = args.steps or config["sampling"]["ddim_steps"]

    print(f"Sampling {args.n} images via {sampler} ({steps} steps)...")
    if sampler == "ddim":
        latents = diffusion.ddim_sample(model, shape=shape, num_steps=steps,
                                        progress_bar=True, eta=args.eta)
    else:
        latents = diffusion.p_sample_loop(model, shape=shape, progress_bar=True)

    imgs = decode_latents(vae, latents, config["latent_scale_factor"])
    imgs01 = (imgs + 1) / 2  # [0, 1] for saving

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from torchvision.utils import make_grid, save_image

    grid = make_grid(imgs01, nrow=int(math.sqrt(args.n)) or 1)
    grid_path = Path(args.grid_path)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(grid_path))
    for i, img in enumerate(imgs01):
        save_image(img, str(out_dir / f"sample_{i:04d}.png"))

    print(f"Saved grid -> {grid_path}")
    print(f"Saved {args.n} images -> {out_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a LatentFlow checkpoint.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n", type=int, default=16, help="Number of images to generate.")
    p.add_argument("--out", type=str, default="samples/")
    p.add_argument("--grid-path", type=str, default="assets/samples.png")
    p.add_argument("--sampler", type=str, default=None, choices=[None, "ddim", "ddpm"])
    p.add_argument("--steps", type=int, default=None, help="Sampler steps (DDIM).")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
