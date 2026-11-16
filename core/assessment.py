"""Reusable token-level SAE assessment and controlled generation."""
from contextlib import ExitStack, nullcontext
from core.sae import flatten_record, record_sae_features, record_residuals, flatten_residuals
from core.steering import steer_layer
from core.attention import record_attention


def matched_random_direction(direction, seed):
    """An isotropic control with the same norm, reproducible across devices."""
    import torch
    random = torch.randn(direction.shape, generator=torch.Generator().manual_seed(seed))
    return (random / random.norm() * direction.detach().float().cpu().norm()).to(direction.device)


def token_trace(record, inputs, generated_ids, tokenizer, jacobian=None, text_model=None, top_k=10,
                direction=None, strength=None, scope="all", post_record=None, stride=0,
                save_vectors=False):
    activations, residuals = flatten_record(record)
    input_ids = inputs["input_ids"][0].tolist()
    available_ids = input_ids + generated_ids
    if activations.shape[0] > len(available_ids):
        raise RuntimeError("More recorded positions than available token IDs")
    traced_ids = available_ids[:activations.shape[0]]
    rows = []
    for index, (token_id, values) in enumerate(zip(traced_ids, activations.tolist())):
        rows.append({
            "position": index,
            "source": "prompt" if index < len(input_ids) else "generated",
            "generated_index": None if index < len(input_ids) else index - len(input_ids),
            "token_id": token_id,
            "token": tokenizer.decode([token_id]) if tokenizer is not None else str(token_id),
            "features": {str(feature): value for feature, value in zip(record["feature_ids"], values)},
        })
    if post_record is not None:
        post_activations, _ = flatten_record(post_record)
        if post_activations.shape != activations.shape:
            raise RuntimeError("Before/after SAE trace shapes differ")
        for row, values in zip(rows, post_activations.tolist()):
            row["features_after_steering"] = dict(zip(map(str, record["feature_ids"]), values))
    jspace = []
    if jacobian is not None:
        from core.jlens import jspace_top_tokens

        positions = sorted({len(input_ids) - 1, min(len(input_ids), len(rows) - 1), len(rows) - 1})
        if stride:
            positions = sorted(set(positions) | set(range(len(input_ids), len(rows), stride)))
        positions = [position for position in positions if 0 <= position < len(rows)]
        pre_readouts = jspace_top_tokens(jacobian, residuals[positions], text_model, tokenizer, top_k)
        # Capture actual model-dtype addition, rather than reconstructing it in float32.
        post_residuals = flatten_record(post_record)[1] if post_record is not None else residuals
        post_readouts = jspace_top_tokens(
            jacobian, post_residuals[positions], text_model, tokenizer, top_k)
        jspace = [{"position": position, "source": rows[position]["source"],
                   "token": rows[position]["token"],
                   "pre_current_steering_top_tokens": pre_readout,
                   "post_current_steering_top_tokens": post_readout}
                  for position, pre_readout, post_readout
                  in zip(positions, pre_readouts, post_readouts)]
        if save_vectors:
            from core.jlens import transported_vectors
            before = transported_vectors(jacobian, residuals[positions])
            after = transported_vectors(jacobian, post_residuals[positions])
            for index, point in enumerate(jspace):
                point.update(_vector_pre=before[index], _vector_post=after[index])
    return rows, jspace


def multi_layer_readouts(records, lenses, inputs, tokens, tokenizer, model, top_k, stride,
                         save_vectors=False):
    """Sparse position readouts; final layer uses a direct vocabulary readout."""
    from core.jlens import jspace_top_tokens
    result = []
    prompt_length = inputs["input_ids"].shape[1]
    ids = inputs["input_ids"][0].tolist() + tokens
    for layer_id, record in records.items():
        hidden = flatten_residuals(record)
        if len(hidden) > len(ids):
            raise RuntimeError("Lens trace exceeds available token IDs")
        positions = {prompt_length - 1, min(prompt_length, len(hidden)-1), len(hidden)-1}
        if stride:
            positions.update(range(prompt_length, len(hidden), stride))
        positions = sorted(p for p in positions if 0 <= p < len(hidden))
        readouts = jspace_top_tokens(lenses[layer_id], hidden[positions], model, tokenizer, top_k)
        if save_vectors:
            from core.jlens import transported_vectors
            vectors = transported_vectors(lenses[layer_id], hidden[positions])
        for index, (position, readout) in enumerate(zip(positions, readouts)):
            point = {"layer": layer_id, "position": position,
                           "source": "prompt" if position < prompt_length else "generated",
                           "token": tokenizer.decode([ids[position]]),
                           "measurement": "actual layer output; includes upstream intervention effects",
                           "lens_kind": "direct_final" if lenses[layer_id] is None else "jacobian",
                           "top_tokens": readout}
            if save_vectors:
                point["_vector_observed"] = vectors[index]
            result.append(point)
    return result


def generate_comparison(model, layer, inputs, direction, strengths, scope, max_new_tokens, seed,
                        sae=None, tokenizer=None, jacobian=None, text_model=None, jlens_top_k=10,
                        monitors=None, include_controls=True, lens_layers=None,
                        steering_layer_index=9, jlens_stride=0, temperature=0.0,
                        save_vectors=False, attention=None, query_patch=None):
    """Yield baseline, exact zero-hook control, then each nonzero strength."""
    import torch

    baseline = None
    for strength in ([None, 0.0] if include_controls else []) + list(dict.fromkeys(s for s in strengths if s != 0)):
        torch.manual_seed(seed)
        with ExitStack() as stack:
            from core.query_interventions import replace_prefill_queries
            stack.enter_context(replace_prefill_queries(model, query_patch))
            attention_record = stack.enter_context(record_attention(model, inputs["input_ids"], inputs["input_ids"].shape[1], attention))
            record = stack.enter_context(record_sae_features(
                layer, sae, capture_residuals=jacobian is not None)) if sae is not None else None
            extra_records = {key: stack.enter_context(record_sae_features(module, encoder))
                             for key, (module, encoder) in (monitors or {}).items()}
            lens_records = {key: stack.enter_context(record_residuals(module))
                            for key, (module, _) in (lens_layers or {}).items()
                            if key != steering_layer_index}
            context = nullcontext(None) if strength is None else steer_layer(layer, direction, strength, scope)
            stats = stack.enter_context(context)
            post_record = stack.enter_context(record_sae_features(
                layer, sae, capture_residuals=jacobian is not None)) if sae is not None else None
            generation_options = {"do_sample": temperature > 0}
            if temperature > 0:
                generation_options.update(temperature=temperature, top_p=1.0, top_k=0)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        use_cache=True, pad_token_id=0, **generation_options)
        tokens = output[0, inputs["input_ids"].shape[1]:].tolist()
        if strength is None:
            baseline = tokens
        elif strength == 0 and tokens != baseline:
            raise RuntimeError("Zero-strength control differs from baseline; stop before interpreting steering")
        if stats is not None and not stats["calls"]:
            raise RuntimeError("The steering hook never ran")
        trace, jspace = (token_trace(record, inputs, tokens, tokenizer, jacobian, text_model,
                                     jlens_top_k, direction, strength, scope, post_record, jlens_stride,
                                     save_vectors)
                         if record is not None else ([], []))
        for layer_id, extra in extra_records.items():
            extra_trace, _ = token_trace(extra, inputs, tokens, tokenizer)
            if len(extra_trace) != len(trace):
                raise RuntimeError("Layer traces have different token counts")
            for row, other in zip(trace, extra_trace):
                row["features"].update({f"L{layer_id}:{key}": value
                                        for key, value in other["features"].items()})
        for point in jspace:
            point.update(layer=steering_layer_index, lens_kind="jacobian")
        if lens_records:
            jspace += multi_layer_readouts(
                lens_records, {key: matrix for key, (_, matrix) in lens_layers.items()},
                inputs, tokens, tokenizer, text_model or model, jlens_top_k, jlens_stride, save_vectors)
        record_summary = None if record is None else {
            "calls": record["calls"], "sequence_lengths": record["sequence_lengths"],
            "feature_ids": record["feature_ids"], "token_count": len(trace),
            "activation_location": "layer output before the current-token steering addition",
        }
        yield {**({"attention": attention_record} if attention else {}), "condition": "baseline" if strength is None else "steered",
               "strength": strength, "token_ids": tokens, "hook": stats,
               "sae_record": record_summary, "token_trace": trace, "jspace": jspace}


def assess_fixed_continuation(model, layer, inputs, tokens, direction, strength, scope, sae,
                              tokenizer, monitors=None, lens_layers=None, jacobian=None,
                              steering_layer_index=9, jlens_top_k=10, jlens_stride=16,
                              save_vectors=False, attention=None, query_patch=None):
    """Teacher-force exactly the baseline continuation; no sampled-text confound."""
    import torch
    if not tokens:
        raise ValueError("A fixed continuation must contain tokens")
    prefix = torch.tensor([tokens[:-1]], dtype=inputs["input_ids"].dtype,
                          device=inputs["input_ids"].device)
    fixed_ids = torch.cat([inputs["input_ids"], prefix], dim=1)
    fixed_inputs = {"input_ids": fixed_ids, "attention_mask": torch.ones_like(fixed_ids)}
    with ExitStack() as stack:
        from core.query_interventions import replace_prefill_queries
        stack.enter_context(replace_prefill_queries(model, query_patch))
        attention_record = stack.enter_context(record_attention(model, fixed_ids, inputs["input_ids"].shape[1], attention))
        pre = stack.enter_context(record_sae_features(layer, sae, capture_residuals=jacobian is not None))
        extra = {key: stack.enter_context(record_sae_features(module, encoder))
                 for key, (module, encoder) in (monitors or {}).items()}
        lens_records = {key: stack.enter_context(record_residuals(module))
                        for key, (module, _) in (lens_layers or {}).items()
                        if key != steering_layer_index}
        if strength is not None:
            stack.enter_context(steer_layer(layer, direction, strength, scope))
        post = stack.enter_context(record_sae_features(layer, sae, capture_residuals=jacobian is not None))
        with torch.inference_mode():
            output = model(**fixed_inputs, use_cache=False)
            # Score the same continuation in bounded chunks, using actual model logits.
            log_probabilities = []
            start = inputs["input_ids"].shape[1]-1
            labels = torch.tensor(tokens,device=output.logits.device)
            for offset in range(0,len(tokens),16):
                logits = output.logits[0,start+offset:start+offset+16].float()
                target = labels[offset:offset+len(logits)]
                losses = torch.nn.functional.cross_entropy(logits,target,reduction="none")
                log_probabilities.extend((-losses).cpu().tolist())
    trace, jspace = token_trace(pre, inputs, tokens, tokenizer, jacobian, model, jlens_top_k,
                               post_record=post, stride=jlens_stride, save_vectors=save_vectors)
    for layer_id, record in extra.items():
        other, _ = token_trace(record, inputs, tokens, tokenizer)
        for row, values in zip(trace, other):
            row["features"].update({f"L{layer_id}:{key}": value
                                    for key, value in values["features"].items()})
    for point in jspace:
        point.update(layer=steering_layer_index, lens_kind="jacobian")
    if lens_records:
        jspace += multi_layer_readouts(lens_records, {k: matrix for k, (_, matrix) in lens_layers.items()},
                                      inputs, tokens, tokenizer, model, jlens_top_k, jlens_stride,
                                      save_vectors)
    return {**({"attention": attention_record} if attention else {}), "reference": "same prompt and unmodified baseline continuation for every condition",
            "token_ids": tokens, "token_trace": trace, "jspace": jspace,
            "continuation_likelihood": {"token_log_probabilities": log_probabilities,
                "mean_nll": -sum(log_probabilities)/len(log_probabilities),
                "first_token_log_probability": log_probabilities[0]},
            "scope_note": "last means last position of this full teacher-forced forward; use all for comparison to cached generation"}


