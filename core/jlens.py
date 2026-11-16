"""Load and apply the public Gemma-3-4B-IT averaged Jacobian lens."""

JLENS_ID = "neuronpedia/jacobian-lens"
JLENS_FILE = "gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt"


def transported_vectors(jacobian, residuals):
    """Unnormalized final-basis vectors, not vocabulary logits or probabilities."""
    import torch
    with torch.inference_mode():
        if jacobian is None:
            return residuals.detach().float().cpu().clone()
        return (residuals.to(jacobian.device).float() @ jacobian.float().T).detach().cpu()


def load_jacobians(layers, expected_width, revision=None):
    """Load one checkpoint once; validate every requested source layer."""
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    revision = revision or HfApi().model_info(JLENS_ID).sha
    path = hf_hub_download(JLENS_ID, JLENS_FILE, revision=revision)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("d_model") != expected_width or "J" not in checkpoint:
        raise ValueError("J-Lens checkpoint is incompatible with the model width")
    matrices = {}
    for layer in layers:
        if layer not in checkpoint["J"]:
            raise ValueError(f"J-Lens has no layer {layer}; available: {sorted(checkpoint['J'])}")
        matrix = checkpoint["J"][layer].float()
        if matrix.shape != (expected_width, expected_width) or not torch.isfinite(matrix).all():
            raise ValueError(f"Invalid layer-{layer} Jacobian shape or values")
        matrices[layer] = matrix
    metadata = {
        "repo": JLENS_ID,
        "revision": revision,
        "filename": JLENS_FILE,
        "n_prompts": checkpoint.get("n_prompts"),
        "d_model": checkpoint.get("d_model"),
        "source_layers": checkpoint.get("source_layers", sorted(checkpoint["J"])),
        "selected_layers": list(layers),
    }
    return matrices, metadata


def load_jacobian(layer, expected_width):
    matrices, metadata = load_jacobians([layer], expected_width)
    return matrices[layer], metadata


def residual_logits(jacobian, residuals, model):
    """Transport residuals, then apply final norm/unembedding (full vocabulary)."""
    import torch

    if residuals.ndim != 2 or (jacobian is not None and residuals.shape[1] != jacobian.shape[1]):
        raise ValueError("Residuals and Jacobian have incompatible shapes")
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm, lm_head = model.model.norm, model.lm_head
    elif (hasattr(model, "lm_head") and hasattr(model, "model")
          and hasattr(model.model, "language_model")):
        final_norm, lm_head = model.model.language_model.norm, model.lm_head
    else:
        raise ValueError("Unsupported model wrapper for final norm/unembedding")
    with torch.inference_mode():
        transported = (residuals.to(device) if jacobian is None else
                       residuals.to(device=device, dtype=jacobian.dtype) @ jacobian.to(device).T)
        normalized = final_norm(transported.to(model_dtype))
        logits = lm_head(normalized).float()
    return logits


def jspace_top_tokens(jacobian, residuals, model, tokenizer, top_k=10):
    """Vocabulary rankings from the same full readout used by decision metrics."""
    import torch
    logits = residual_logits(jacobian, residuals, model)
    values, ids = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
    return [
        [
            {"token_id": int(token_id), "token": tokenizer.decode([int(token_id)]),
             "logit": float(value)}
            for token_id, value in zip(row_ids, row_values)
        ]
        for row_ids, row_values in zip(ids.cpu(), values.cpu())
    ]
