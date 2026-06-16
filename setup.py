#!/usr/bin/env python3
"""Setup file for LatentFlow."""

from setuptools import setup, find_packages

setup(
    name="latentflow",
    version="0.1.0",
    description="Diffusion Transformer (DiT) image generator built from scratch",
    author="Pranjal",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "numpy>=1.24.3",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "diffusers>=0.28.0",
        "huggingface_hub>=0.22.0",
        "transformers>=4.38.0",
        "einops>=0.7.0",
        "torchmetrics>=1.0.0",
        "clean-fid>=0.1.35",
        "tensorboard>=2.15.0",
        "Pillow>=10.1.0",
    ],
)
