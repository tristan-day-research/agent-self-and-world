"""Small, inference-only pieces of a JumpReLU SAE.

Only selected columns/rows are loaded. This is sufficient for tokenwise feature
recording and decoder-direction steering without keeping the full 65k feature
activation tensor in memory.
"""

from contextlib import contextmanager


class SelectedJumpReLU:
    def __init__(self, feature_ids, w_enc, b_enc, threshold, w_dec):
        import torch

        self.feature_ids = tuple(feature_ids)
        self.w_enc = w_enc.float()       # [d_model, n_features]
        self.b_enc = b_enc.float()       # [n_features]
        self.threshold = threshold.float()
        self.w_dec = w_dec.float()       # [n_features, d_model]
        expected = (self.w_dec.shape[1], len(self.feature_ids))
        if self.w_enc.shape != expected:
            raise ValueError(f"Encoder shape {tuple(self.w_enc.shape)} != {expected}")
        if self.w_dec.shape[0] != len(self.feature_ids):
            raise ValueError("Decoder row count does not match selected features")
        if self.b_enc.shape != (len(self.feature_ids),) or self.threshold.shape != self.b_enc.shape:
            raise ValueError("Bias/threshold shape does not match selected features")
        tensors = (self.w_enc, self.b_enc, self.threshold, self.w_dec)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("SAE parameters contain non-finite values")

    def encode(self, hidden):
        """Match SAELens JumpReLU: ReLU(pre) when pre > threshold, else zero."""
        import torch

        pre = hidden.float() @ self.w_enc.to(hidden.device) + self.b_enc.to(hidden.device)
        return torch.relu(pre) * (pre > self.threshold.to(hidden.device)).to(pre.dtype)

    def to(self, device):
        self.w_enc = self.w_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.threshold = self.threshold.to(device)
        self.w_dec = self.w_dec.to(device)
        return self


@contextmanager
def record_sae_features(layer, sae, capture_residuals=False):
    """Record the layer output before any subsequently registered steering hook."""
    record = {
        "calls": 0,
        "sequence_lengths": [],
        "feature_ids": list(sae.feature_ids),
        "activation_chunks": [],
        "residual_chunks": [],
    }

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[-1] != sae.w_enc.shape[0]:
            raise ValueError("Recorded residual does not match SAE input width")
        record["calls"] += 1
        record["sequence_lengths"].append(hidden.shape[1])
        record["activation_chunks"].append(sae.encode(hidden).detach().cpu())
        if capture_residuals:
            record["residual_chunks"].append(hidden.detach().float().cpu())

    handle = layer.register_forward_hook(hook)
    try:
        yield record
    finally:
        handle.remove()


def flatten_record(record):
    """Flatten batch-size-one prefill/decode chunks into token-major tensors."""
    import torch

    if not record["activation_chunks"]:
        raise RuntimeError("SAE recording hook never ran")
    if any(chunk.shape[0] != 1 for chunk in record["activation_chunks"]):
        raise ValueError("Token trace currently supports batch size one only")
    activations = torch.cat([chunk[0] for chunk in record["activation_chunks"]], dim=0)
    residuals = None
    if record["residual_chunks"]:
        residuals = torch.cat([chunk[0] for chunk in record["residual_chunks"]], dim=0)
    return activations, residuals


@contextmanager
def record_residuals(layer):
    """Read-only residual capture independent of an SAE or feature dictionary."""
    record = {"residual_chunks": [], "sequence_lengths": []}
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("Residual capture requires batch size one")
        record["residual_chunks"].append(hidden.detach().float().cpu())
        record["sequence_lengths"].append(hidden.shape[1])
    handle = layer.register_forward_hook(hook)
    try:
        yield record
    finally:
        handle.remove()


def flatten_residuals(record):
    import torch
    if not record["residual_chunks"]:
        raise RuntimeError("Residual hook never ran")
    return torch.cat([chunk[0] for chunk in record["residual_chunks"]], dim=0)
