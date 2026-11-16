"""Small model-agnostic residual capture, patching, and held-out linear probes."""

from contextlib import contextmanager


def write_torch(path, value):
    """Atomically checkpoint tensor dictionaries inside a run directory."""
    import os
    import torch
    from pathlib import Path
    import tempfile
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def capture_layers(layers, indices, position=-1):
    """Capture one residual vector per selected layer in a single forward."""
    records, handles = {}, []
    try:
        for index in indices:
            def hook(_module, _inputs, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                records[index] = hidden[0, position].detach().float().cpu().clone()
            handles.append(layers[index].register_forward_hook(hook))
        yield records
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_layer(layer, replacement, position=-1):
    """Replace one residual at a layer output; K/V in earlier layers remain target-run values."""
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != 1 or replacement.numel() != hidden.shape[-1]:
            raise ValueError('Residual patch requires batch one and matching width')
        changed = hidden.clone()
        changed[0, position] = replacement.to(device=hidden.device, dtype=hidden.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def fit_ridge_probe(vectors, labels, ridge=1.0):
    """Binary ridge classifier, fit only on training families; returns source threshold."""
    import torch
    x = torch.stack([v.float().flatten() for v in vectors]).double()
    y = torch.tensor([1.0 if label else -1.0 for label in labels], dtype=torch.float64)
    if x.shape[0] < 4 or len(set(labels)) != 2 or ridge <= 0:
        raise ValueError('Probe requires both labels, at least four samples, and positive ridge')
    center = x.mean(0)
    x = x - center
    scale = x.norm(dim=1).mean().clamp_min(1e-12)
    x = x / scale
    gram = x @ x.T
    coeff = torch.linalg.solve(gram + ridge * torch.eye(len(x), dtype=x.dtype), y)
    weight = x.T @ coeff / scale
    bias = -(center @ weight)
    return weight, bias


def score_probe(weight, bias, vectors, labels):
    import torch
    scores = [float(v.float().flatten().double() @ weight + bias) for v in vectors]
    positive = [s for s, label in zip(scores, labels) if label]
    negative = [s for s, label in zip(scores, labels) if not label]
    auc = None
    if positive and negative:
        auc = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive)*len(negative))
    return {'n': len(scores), 'accuracy': sum((s > 0) == bool(y) for s, y in zip(scores, labels))/len(scores),
            'auroc': auc, 'mean_positive_score': sum(positive)/len(positive) if positive else None,
            'mean_negative_score': sum(negative)/len(negative) if negative else None}
