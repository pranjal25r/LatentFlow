"""Evaluate a LatentFlow checkpoint with FID against real CelebA-HQ images.

Generates N samples (in batches), decodes them to PNGs, then computes FID with
clean-fid. Prints the FID alongside N and the sampler settings so the number is
reproducible and honest to report.

Run:
  python scripts/eval_fid.py --ckpt checkpoints/ckpt_final.pt \
      --real-dir data/celebahq --num-samples 5000 --batch-size 64

Requires: pip install clean-fid
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

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
def generate_to_dir(args, config, model, diffusion, vae, device, out_dir: Path) -> int:
    """Generate args.num_samples images into out_dir as PNGs. Returns count."""
    from torchvision.utils import save_image

    sampler = args.sampler or config["sampling"]["sampler"]
    steps = args.steps or config["sampling"]["ddim_steps"]
    lc, lh, lw = config["latent_channels"], config["latent_height"], config["latent_width"]
    scale = config["latent_scale_factor"]

    saved = 0
    while saved < args.num_samples:
        bs = min(args.batch_size, args.num_samples - saved)
        shape = (bs, lc, lh, lw)
        if sampler == "ddim":
            latents = diffusion.ddim_sample(model, shape=shape, num_steps=steps,
                                            progress_bar=False, eta=args.eta)
        else:
            latents = diffusion.p_sample_loop(model, shape=shape, progress_bar=False)
        imgs01 = (decode_latents(vae, latents, scale) + 1) / 2
        for img in imgs01:
            save_image(img, str(out_dir / f"gen_{saved:05d}.png"))
            saved += 1
        print(f"  generated {saved}/{args.num_samples}")
    return saved


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    config: Dict[str, Any] = ckpt["config"]
    preset: str = ckpt.get("preset", "config")

    model = build_model(config, preset).to(device)
    weights = ckpt["ema"] if (args.use_ema and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(weights)
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        prediction_type=config["diffusion"]["prediction_type"],
    ).to(device)
    vae = build_vae(config, device)

    sampler = args.sampler or config["sampling"]["sampler"]
    steps = args.steps or config["sampling"]["ddim_steps"]

    gen_dir = Path(args.gen_dir) if args.gen_dir else Path(tempfile.mkdtemp(prefix="latentflow_fid_"))
    gen_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.num_samples} samples ({sampler}, {steps} steps) -> {gen_dir}")
    generate_to_dir(args, config, model, diffusion, vae, device, gen_dir)

    try:
        from cleanfid import fid
    except ImportError:
        print("\n[error] clean-fid not installed. Run: pip install clean-fid")
        print(f"Generated images are in {gen_dir} if you want to compute FID separately.")
        return

    print(f"Computing FID: real={args.real_dir}  vs  generated={gen_dir}")
    score = fid.compute_fid(str(gen_dir), str(args.real_dir))

    print("\n" + "=" * 60)
    print(f"FID: {score:.2f}")
    print(f"  samples generated : {args.num_samples}")
    print(f"  sampler           : {sampler} ({steps} steps, eta={args.eta})")
    print(f"  model preset      : {preset}")
    print(f"  weights           : {'EMA' if args.use_ema else 'raw'}")
    print("=" * 60)
    print("Report this number with N and sampler settings — that's what makes it honest.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FID for a LatentFlow checkpoint.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--real-dir", type=str, required=True,
                   help="Directory of real CelebA-HQ images to compare against.")
    p.add_argument("--gen-dir", type=str, default=None,
                   help="Where to write generated PNGs (default: temp dir).")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sampler", type=str, default=None, choices=[None, "ddim", "ddpm"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
