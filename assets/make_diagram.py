"""Render the LatentFlow architecture diagram to assets/architecture.png.

A simple, dependency-light pipeline figure for the README:

  Images -> [Frozen VAE Encoder] -> Latents (cached)
         -> [DiT: patchify -> N x Transformer blocks (adaLN-Zero, timestep cond.)
            -> unpatchify -> predict noise]
         -> [DDIM Sampler] -> [Frozen VAE Decoder] -> Faces

Run:
  python assets/make_diagram.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Palette
FROZEN = "#cbd5e1"   # gray-blue: frozen / pretrained
FROZEN_E = "#64748b"
TRAINED = "#6366f1"  # indigo: the part we train (DiT)
TRAINED_E = "#4338ca"
DATA = "#f1f5f9"     # near-white: data/tensors
DATA_E = "#94a3b8"
TEXT = "#0f172a"


def box(ax, x, y, w, h, label, face, edge, text_color=TEXT, fontsize=10, weight="bold"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, weight=weight, wrap=True)
    return (x + w, y + h / 2), (x, y + h / 2)  # right-mid, left-mid


def arrow(ax, p_from, p_to, color="#475569"):
    ax.add_patch(FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.8, color=color, shrinkA=2, shrinkB=2,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    y = 2.6
    h = 1.0

    # Stage 1 — frozen VAE encode
    img_r, _ = box(ax, 0.2, y, 1.5, h, "Face\nImages\n256x256", DATA, DATA_E, fontsize=9)
    enc_r, enc_l = box(ax, 2.1, y, 1.9, h, "Frozen VAE\nEncoder", FROZEN, FROZEN_E)
    lat_r, lat_l = box(ax, 4.4, y, 1.7, h, "Latents\n4x32x32\n(cached)", DATA, DATA_E, fontsize=9)

    # Stage 2 — trained DiT
    dit_r, dit_l = box(
        ax, 6.5, y - 0.35, 3.2, h + 0.7,
        "Diffusion Transformer (DiT)\npatchify -> N x blocks\n(adaLN-Zero, timestep)\n-> unpatchify -> noise",
        TRAINED, TRAINED_E, text_color="white", fontsize=9,
    )

    # Inference tail
    samp_r, samp_l = box(ax, 10.1, y, 1.4, h, "DDIM\nSampler", DATA, DATA_E, fontsize=9)
    # Decoder sits to the right; faces below it to keep things on canvas
    dec_top, dec_l = box(ax, 11.7, y, 1.1, h, "Frozen\nVAE\nDecoder", FROZEN, FROZEN_E, fontsize=8)

    arrow(ax, img_r, enc_l)
    arrow(ax, enc_r, lat_l)
    arrow(ax, lat_r, dit_l)
    arrow(ax, dit_r, samp_l)
    arrow(ax, samp_r, dec_l)

    # "Faces" output below the decoder
    faces_pt, _ = box(ax, 11.45, 0.5, 1.55, 0.9, "Generated\nFaces", "#ecfdf5", "#10b981",
                      text_color="#065f46", fontsize=9)
    arrow(ax, (12.25, y), (12.25, 1.4))

    # Stage brackets / labels
    ax.text(3.05, y + h + 0.55, "Stage 1 — Perceptual compression (frozen)",
            ha="center", color=FROZEN_E, fontsize=10, weight="bold")
    ax.plot([2.1, 6.1], [y + h + 0.35, y + h + 0.35], color=FROZEN_E, lw=1.2)

    ax.text(8.1, y + h + 0.55, "Stage 2 — Latent diffusion (trained)",
            ha="center", color=TRAINED_E, fontsize=10, weight="bold")
    ax.plot([6.5, 9.7], [y + h + 0.35, y + h + 0.35], color=TRAINED_E, lw=1.2)

    ax.text(11.4, y + h + 0.55, "Inference",
            ha="center", color="#475569", fontsize=10, weight="bold")
    ax.plot([10.1, 12.8], [y + h + 0.35, y + h + 0.35], color="#475569", lw=1.2)

    ax.text(6.5, 0.15, "LatentFlow: a Diffusion Transformer trained on cached VAE latents",
            ha="center", color=TEXT, fontsize=11, weight="bold")

    fig.tight_layout()
    out = "assets/architecture.png"
    import os
    os.makedirs("assets", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
