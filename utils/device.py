# utils/device.py
"""
Hardware-agnostic device management.
Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
"""
import torch
import platform
import os


def get_device() -> torch.device:
    """
    Auto-detect the best available device.

    Returns:
        torch.device configured for the best available backend.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    Get the appropriate dtype for the device.
    MPS has limited fp16 support during training, so we use float32.
    CUDA can use fp16 for inference and mixed precision for training.
    """
    if device.type == "mps":
        return torch.float32
    elif device.type == "cuda":
        return torch.float16
    else:
        return torch.float32


def get_device_info() -> dict:
    """Return a summary of the current hardware environment."""
    device = get_device()
    info = {
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
    }

    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 1
        )

    if device.type == "mps":
        # Apple Silicon unified memory — report system RAM as proxy
        try:
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            info["system_memory_gb"] = round(mem_bytes / 1e9, 1)
        except (ValueError, OSError):
            pass

    return info


def print_device_info():
    """Pretty-print device information."""
    info = get_device_info()
    print("=" * 50)
    print("  Device Configuration")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k:.<30} {v}")
    print("=" * 50)
