"""Lazy device selection so the backbone works without PyTorch installed."""


def resolve_device(requested="auto"):
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device: {requested}")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        if requested == "auto":
            return "cpu"
        raise RuntimeError(f"Device {requested} requires PyTorch") from exc
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if requested == "auto":
        return next((name for name, usable in available.items() if usable), "cpu")
    if not available[requested]:
        raise RuntimeError(f"Requested device {requested} is unavailable")
    return requested
