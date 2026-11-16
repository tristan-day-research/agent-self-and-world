"""Additive residual-stream steering, shared by real and toy-model runs."""

from contextlib import contextmanager
import math


@contextmanager
def steer_layer(layer, direction, strength, scope="all"):
    """Apply h <- h + strength * direction; preserve decoder row magnitude.

    Hooks run during both prefill and cached decoding. `last` affects only the
    last position of each forward call; `all` matches the paper's token scope.
    Zero strength is an exact no-op. This is steering, not SAE feature ablation.
    """
    if not math.isfinite(strength) or scope not in {"all", "last"}:
        raise ValueError("Strength must be finite and scope must be all or last")
    stats = {"calls": 0, "sequence_lengths": [], "positions_steered": 0}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or direction.shape != (hidden.shape[-1],):
            raise ValueError("Steering direction does not match the layer's hidden size")
        stats["calls"] += 1
        stats["sequence_lengths"].append(hidden.shape[1])
        if strength == 0:
            return output
        delta = strength * direction.to(device=hidden.device, dtype=hidden.dtype)
        changed = hidden.clone()
        if scope == "all":
            changed += delta
            stats["positions_steered"] += hidden.shape[0] * hidden.shape[1]
        else:
            changed[:, -1, :] += delta
            stats["positions_steered"] += hidden.shape[0]
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = layer.register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()
