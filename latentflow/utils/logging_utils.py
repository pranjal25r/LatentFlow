"""Logging and monitoring utilities.

Provides console logging helpers and TensorBoard integration.
"""

from typing import Dict, Optional, Any
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger.
    
    Args:
        name: Logger name (typically __name__).
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    
    # Formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


class TensorBoardLogger:
    """Simple TensorBoard wrapper for scalars and histograms.
    
    Args:
        log_dir: Directory for TensorBoard logs.
    """

    def __init__(self, log_dir: str) -> None:
        """Initialize TensorBoard logger."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value.
        
        Args:
            tag: Metric name (e.g., "loss/train").
            value: Scalar value.
            step: Global step / iteration.
        """
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log histogram.
        
        Args:
            tag: Histogram name.
            values: Values to histogram.
            step: Global step.
        """
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor, step: int) -> None:
        """Log image.
        
        Args:
            tag: Image name.
            img_tensor: Image tensor (C, H, W) or (B, C, H, W).
            step: Global step.
        """
        self.writer.add_image(tag, img_tensor, step)

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()


def log_metrics(
    logger: Optional[logging.Logger],
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Log metrics to console.
    
    Args:
        logger: Logger instance (if None, uses print).
        metrics: Dictionary of metric names -> values.
        step: Optional step/epoch number.
        prefix: Prefix for all metric names.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Format metrics
    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_strs.append(f"{prefix}{key}={value:.4f}")
        else:
            metric_strs.append(f"{prefix}{key}={value}")
    
    # Log with step if provided
    if step is not None:
        logger.info(f"[Step {step}] {', '.join(metric_strs)}")
    else:
        logger.info(", ".join(metric_strs))
