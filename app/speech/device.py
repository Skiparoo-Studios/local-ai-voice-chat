"""Compute device selection.

The brief is explicit that CUDA must not be assumed. ``auto`` therefore probes
and falls back to CPU, while an explicit ``cuda`` request fails loudly rather
than silently running somewhere the user did not intend.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

DeviceProbe = Callable[[], bool]


class DeviceUnavailableError(Exception):
    """Raised when a requested compute device cannot be used."""


def cudaAvailable() -> bool:
    """Whether torch reports a usable CUDA device."""
    try:
        import torch
    except ImportError:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception as error:  # driver faults vary  # noqa: BLE001
        logger.debug("CUDA probe failed: %s", error)
        return False


def describeCudaDevice() -> str | None:
    """Name of the active CUDA device, if there is one."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        totalBytes = torch.cuda.get_device_properties(0).total_memory
        return f"{name} ({totalBytes / 1024**3:.1f} GiB)"
    except Exception:  # diagnostics must never break startup  # noqa: BLE001
        return None


def resolveDevice(preference: str, probe: DeviceProbe = cudaAvailable) -> str:
    """Turn a configured preference into a concrete device string."""
    if preference == "cpu":
        return "cpu"

    available = probe()

    if preference == "cuda":
        if not available:
            raise DeviceUnavailableError(
                "device is set to 'cuda' but no CUDA device is available. "
                "Check that an NVIDIA driver and a CUDA build of torch are installed, "
                "or set the device to 'auto' to fall back to CPU."
            )
        return "cuda"

    if preference != "auto":
        raise DeviceUnavailableError(
            f"Unknown device {preference!r}; expected 'auto', 'cpu' or 'cuda'."
        )

    if available:
        return "cuda"

    logger.info("No CUDA device detected, using CPU")
    return "cpu"
