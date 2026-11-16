"""Binary J-Space capture and within-layer comparisons on matching text prefixes."""
import os
from pathlib import Path
import tempfile


def save_vector_artifact(result, directory, index):
    """Remove private tensors from JSON readouts and atomically save them once."""
    import torch
    spaces = {}
    for name, source in (("free", result), ("fixed", result.get("fixed_continuation", {}))):
        records = []
        for point in source.get("jspace", []):
            for phase in ("pre", "post", "observed"):
                vector = point.pop(f"_vector_{phase}", None)
                if vector is not None:
                    records.append({"layer": point["layer"], "position": point["position"],
                                    "phase": phase, "lens_kind": point.get("lens_kind", "jacobian"),
                                    "vector": vector.detach().float().cpu()})
        if records:
            spaces[name] = records
    if not spaces:
        return
    relative = Path("jspace") / f"comparison_{index:04d}.pt"
    target = Path(directory) / relative
    target.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save({"schema": 1, "basis": "unnormalized J @ residual; direct residual at final block",
                    "spaces": spaces}, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    result["jspace_vectors_file"] = str(relative)


def vector_effects(results, directory):
    """Never compare free-generation vectors after token-prefix divergence."""
    import torch
    directory = Path(directory).resolve()
    def load(result):
        path = (directory / result["jspace_vectors_file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Vector artifact path escapes the run directory")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != 1:
            raise ValueError("Unsupported vector artifact schema")
        return {(name, point["layer"], point["position"], point["phase"]): point["vector"]
                for name, points in payload["spaces"].items() for point in points}
    baselines = {r.get("prompt_id", "synthetic"): r for r in results
                 if r["strength"] is None and not r.get("query_intervention") and r.get("jspace_vectors_file")}
    baseline_vectors = {pid: load(result) for pid, result in baselines.items()}
    comparisons, skipped = [], 0
    for index, result in enumerate(results):
        pid = result.get("prompt_id", "synthetic")
        if not result.get("jspace_vectors_file") or pid not in baselines:
            continue
        base = baselines[pid]
        for key, current in load(result).items():
            if key not in baseline_vectors[pid]:
                continue
            space, layer, position, phase = key
            source = result if space == "free" else result.get("fixed_continuation", {})
            base_source = base if space == "free" else base.get("fixed_continuation", {})
            ids = result.get("input_ids", []) + source.get("token_ids", [])
            base_ids = base.get("input_ids", []) + base_source.get("token_ids", [])
            if space == "fixed" and ids != base_ids:
                raise ValueError("Fixed-continuation token IDs differ across conditions")
            if len(ids) <= position or len(base_ids) <= position or ids[:position+1] != base_ids[:position+1]:
                skipped += 1
                continue
            reference = baseline_vectors[pid][key]
            if current.shape != reference.shape or not torch.isfinite(current).all() or not torch.isfinite(reference).all():
                raise ValueError("Invalid or mismatched J-Space vectors")
            norm, base_norm = float(current.norm()), float(reference.norm())
            delta = float((current-reference).norm())
            cosine = float(torch.dot(current, reference)/(current.norm()*reference.norm())) if norm and base_norm else None
            comparisons.append({"result_index": index, "prompt_id": pid, "space": space,
                                "layer": layer, "position": position, "phase": phase,
                                "source": "prompt" if position < len(result.get("input_ids", [])) else "generated",
                                "baseline_norm": base_norm, "norm": norm, "delta_l2": delta,
                                "relative_delta_l2": delta/base_norm if base_norm else None,
                                "cosine": max(-1., min(1., cosine)) if cosine is not None else None})
    return {"metric_scope": "within-layer, same text prefix, unnormalized transported vectors",
            "skipped_nonmatching_prefixes": skipped, "comparisons": comparisons}
