"""Stage 1: encode CelebA-HQ images with the frozen VAE and cache the latents.

Runs the pretrained VAE encoder over the dataset ONCE and writes one .pt file per
image (a plain [4, H/8, W/8] tensor, scaled by latent_scale_factor). Training then
reads these cached latents and never touches the VAE — the main compute saving.

The latents are multiplied by `latent_scale_factor` here so they match the
decode step (which divides by the same factor) in sample.py / eval_fid.py.

Run (full dataset):
  python scripts/precompute_latents.py --config configs/default.yaml

Run (quick CPU smoke test — cache only 200 images):
  python scripts/precompute_latents.py --config configs/default.yaml --limit 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from latentflow.data.dataset import CelebAHQDataset  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(requested: str) -> torch.device:
    """Resolve device: explicit request, else cuda -> mps -> cpu."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vae(config: Dict[str, Any], device: torch.device):
    """Load the frozen pretrained VAE (diffusers), independent of the repo's vae.py."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(config["vae_model"]).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    set_seed(args.seed)

    device = pick_device(args.device)
    print(f"Device: {device}")

    # Resolve the image directory: --data-dir, else <data_root>/<dataset_name>.
    data_dir = args.data_dir or os.path.join(
        config["paths"]["data_root"], config["paths"]["dataset_name"]
    )
    out_dir = Path(args.out_dir or config["paths"]["latent_cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = config["latent_scale_factor"]
    dataset = CelebAHQDataset(
        data_dir=data_dir,
        resolution=config["image_size"],
        vae=None,  # we encode here in batches; do NOT encode inside the dataset
    )
    n_total = len(dataset)
    n_target = min(args.limit, n_total) if args.limit else n_total
    print(f"Found {n_total} images in {data_dir} | caching {n_target} -> {out_dir}/")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    vae = build_vae(config, device)

    try:
        from tqdm import tqdm
        bar = tqdm(total=n_target, desc="Encoding")
    except ImportError:
        bar = None

    saved = 0
    latent_shape = None
    for images in loader:
        images = images.to(device, non_blocking=True)  # [B, 3, H, W] in [-1, 1]
        # Encode -> sample from the latent distribution -> scale (SD convention).
        posterior = vae.encode(images).latent_dist
        latents = posterior.sample() * scale  # [B, 4, H/8, W/8]
        latents = latents.cpu()

        for i in range(latents.shape[0]):
            if saved >= n_target:
                break
            z = latents[i].contiguous()  # [4, H/8, W/8]
            torch.save(z, out_dir / f"{saved:06d}.pt")  # plain tensor (weights_only-safe)
            latent_shape = tuple(z.shape)
            saved += 1
            if bar is not None:
                bar.update(1)

        if saved >= n_target:
            break

    if bar is not None:
        bar.close()

    print(f"\nDone. Cached {saved} latents of shape {latent_shape} in {out_dir}/")
    print("You can now run: python scripts/train.py --config configs/default.yaml --preset s2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute & cache VAE latents for LatentFlow.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Image folder (default: <data_root>/<dataset_name>).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write .pt latents (default: paths.latent_cache_dir).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Cache only the first N images (use for a quick CPU test).")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' (cuda>mps>cpu), or force 'cpu' / 'mps' / 'cuda'.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
