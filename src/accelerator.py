"""Optional accelerator detection for CPU and Hygon DCU environments."""
import os


def resolve_backend(requested="auto"):
    """Return ``cpu`` or ``dcu`` without importing optional PyTorch on CPU."""
    requested = requested.lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "dcu":
            raise RuntimeError("--backend dcu requires a DTK-compatible PyTorch installation")
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "dcu" and not available:
        raise RuntimeError("--backend dcu requested, but no accelerator is visible")
    return "dcu" if available and requested in ("auto", "dcu") else "cpu"


def describe_backend(backend):
    if backend == "cpu":
        return "cpu (NumPy/SciPy)"
    import torch
    return "dcu (PyTorch device: {})".format(torch.cuda.get_device_name(0))
