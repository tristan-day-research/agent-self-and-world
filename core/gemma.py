"""The paper's fixed Gemma/SAE pair."""

import json
from pathlib import Path

MODEL_ID = "google/gemma-3-4b-it"
MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
SAE_ID = "google/gemma-scope-2-4b-it"
SAE_REVISION = "3e94b68be95290aada5b7525cf431d3040f81bb1"
SAE_PATH = "resid_post/layer_9_width_65k_l0_medium"
LAYER = 9  # Zero-based decoder block index, post-block residual stream.


def load_selected_sae(features, layer=LAYER):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import torch
    from core.sae import SelectedJumpReLU

    sae_path = f"resid_post/layer_{layer}_width_65k_l0_medium"
    def download(name):
        return hf_hub_download(SAE_ID, f"{sae_path}/{name}", revision=SAE_REVISION)

    config = json.loads(Path(download("config.json")).read_text())
    expected = {"model_name": MODEL_ID, "architecture": "jump_relu", "width": 65536,
                "hf_hook_point_out": f"model.layers.{layer}.output", "affine_connection": False}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected SAE config: {config}")
    w_enc_columns, w_dec_rows, b_enc_values, threshold_values = [], [], [], []
    with safe_open(download("params.safetensors"), framework="pt", device="cpu") as weights:
        required = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
        if not required.issubset(weights.keys()):
            raise ValueError(f"Missing SAE parameters: {required - set(weights.keys())}")
        encoder = weights.get_slice("w_enc")
        decoder = weights.get_slice("w_dec")
        b_enc = weights.get_slice("b_enc")
        threshold = weights.get_slice("threshold")
        if encoder.get_shape() != [2560, 65536]:
            raise ValueError(f"Unexpected encoder shape: {encoder.get_shape()}")
        if decoder.get_shape() != [65536, 2560]:
            raise ValueError(f"Unexpected decoder shape: {decoder.get_shape()}")
        for feature in features:
            if feature < 0 or feature >= config["width"]:
                raise ValueError(f"Feature out of range: {feature}")
            w_enc_columns.append(encoder[:, feature:feature + 1].clone().float())
            w_dec_rows.append(decoder[feature:feature + 1, :].clone().float())
            b_enc_values.append(b_enc[feature:feature + 1].clone().float())
            threshold_values.append(threshold[feature:feature + 1].clone().float())
    sae = SelectedJumpReLU(features, torch.cat(w_enc_columns, dim=1),
                           torch.cat(b_enc_values), torch.cat(threshold_values),
                           torch.cat(w_dec_rows, dim=0))
    if (sae.w_dec.norm(dim=1) == 0).any():
        raise ValueError("Invalid SAE decoder directions")
    return sae, config


def load_decoder_rows(features):
    """Compatibility helper for callers needing only decoder directions."""
    sae, config = load_selected_sae(features)
    return sae.w_dec, config


def load_gemma(device, dtype):
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=getattr(torch, dtype),
        attn_implementation="eager",
    ).to(device).eval()
    layer = model.get_submodule(f"model.language_model.layers.{LAYER}")
    return model, tokenizer, layer
