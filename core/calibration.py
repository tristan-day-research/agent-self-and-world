"""Reference-set feature peaks, with provenance distinct from the paper corpus."""
import hashlib
import json
from pathlib import Path

from core.runs import write_json
from core.sae import flatten_record, record_sae_features


def read_reference_prompts(path):
    raw = Path(path).read_bytes()
    samples = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Calibration needs nonempty samples with unique IDs")
    for sample in samples:
        if not sample.get("messages") or any(not isinstance(m.get("content"), str)
                                              for m in sample["messages"]):
            raise ValueError("Each reference sample needs text messages")
    return samples, hashlib.sha256(raw).hexdigest()


def collect_reference_peaks(model, tokenizer, layer, sae, samples, digest, device,
                            max_new_tokens, seed, provenance, output_path):
    import torch
    peaks = {str(feature): 0.0 for feature in sae.feature_ids}
    records = []
    artifact = {"kind": "reference_set_peaks", "exact_paper_corpus": False,
                "source_sha256": digest, "max_new_tokens": max_new_tokens,
                "seed": seed, "do_sample": False, **provenance}
    for sample in samples:
        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
        torch.manual_seed(seed)
        with record_sae_features(layer, sae) as record, torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    use_cache=True, pad_token_id=0)
        activations, _ = flatten_record(record)
        maxima = dict(zip(map(str, sae.feature_ids), activations.max(dim=0).values.tolist()))
        for feature, value in maxima.items():
            peaks[feature] = max(peaks[feature], value)
        records.append({"id": sample["id"], "condition": sample.get("condition", "reference"),
                        "messages": sample["messages"], "formatted_prompt": text,
                        "input_ids": inputs["input_ids"][0].tolist(), "sample_maxima": maxima,
                        "processed_tokens": len(activations),
                        "text": tokenizer.decode(output[0, inputs["input_ids"].shape[1]:].tolist(),
                                                 skip_special_tokens=True)})
        artifact.update(feature_peaks=peaks, samples=records, completed_samples=len(records),
                        planned_samples=len(samples), complete=len(records) == len(samples))
        write_json(output_path, artifact)
    return artifact


def validated_peaks(artifact, features, provenance):
    import math
    if artifact.get("kind") != "reference_set_peaks" or not artifact.get("complete"):
        raise ValueError("Reference calibration is missing or incomplete")
    for key, value in provenance.items():
        if artifact.get(key) != value:
            raise ValueError(f"Calibration provenance differs at {key}")
    peaks = {int(key): float(value) for key, value in artifact["feature_peaks"].items()}
    for feature in features:
        if feature not in peaks or not math.isfinite(peaks[feature]) or peaks[feature] <= 0:
            raise ValueError(f"Reference peak for {feature} is absent, nonfinite, or zero")
    return peaks
