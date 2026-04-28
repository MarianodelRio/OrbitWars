"""Utility for resolving device strings including 'auto'."""

import torch


def resolve_device(device_str: str) -> str:
    """Resolve a device string to a concrete device.

    Args:
        device_str: One of 'auto', 'cpu', 'cuda', or any valid torch device string.

    Returns:
        'cuda' if device_str is 'auto' and CUDA is available, 'cpu' if device_str
        is 'auto' and CUDA is not available, or device_str unchanged otherwise.
    """
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str
