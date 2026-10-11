"""Device resolution for inference (MPS / CPU)."""

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Return the best available device string.

    Args:
        requested: One of 'auto', 'mps', or 'cpu'.

    Returns:
        'mps' if MPS is available and requested in {'auto', 'mps'}, otherwise 'cpu'.
    """
    if requested not in ("auto", "mps", "cpu"):
        logger.warning("Unknown device '%s', falling back to 'auto'", requested)
        requested = "auto"

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    if requested in ("auto", "mps") and mps_available:
        logger.info("Using device: mps")
        return "mps"

    if requested == "mps" and not mps_available:
        logger.warning("MPS requested but unavailable, falling back to cpu")

    logger.info("Using device: cpu")
    return "cpu"
