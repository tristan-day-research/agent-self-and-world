"""Bounded Gemma-4 prompt-boundary attention and Query recording.

Use eager attention so selected self-attention hooks expose actual attention
probabilities. This does not capture cached generation or infer causal use.
"""

from contextlib import contextmanager


def locate_spans(tokenizer, formatted, input_ids, excerpts):
    """Map exact, unique prompt excerpts to token indices; fail on ambiguity."""
    encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != list(input_ids):
        raise ValueError('Offset-tokenized text differs from Gemma-4 processor input')
    offsets = encoded['offset_mapping']
    spans = {}
    for name, excerpt in excerpts.items():
        if formatted.count(excerpt) != 1:
            raise ValueError(f'Span {name} must occur exactly once')
        start = formatted.index(excerpt)
        stop = start + len(excerpt)
        positions = [i for i, (left, right) in enumerate(offsets)
                     if right > left and left < stop and right > start]
        if not positions:
            raise ValueError(f'Span {name} aligned to no tokens')
        spans[name] = positions
    return spans


@contextmanager
def record_boundary_attention(layers, indices, *, prompt_length):
    """Capture final-position Q before/after norm and actual attention rows."""
    if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= len(layers) for i in indices):
        raise ValueError('Attention layer indices must be unique and valid')
    if prompt_length < 1:
        raise ValueError('Prompt must contain tokens')
    records, handles = {}, []
    try:
        for index in indices:
            attn = layers[index].self_attn
            if getattr(attn.config, '_attn_implementation', None) != 'eager':
                raise ValueError('Gemma-4 attention recording requires eager backend')
            record = records[index] = {}

            def raw_hook(_module, _inputs, output, record=record):
                # q_proj is [batch, tokens, heads * head_dim].
                value = output[0] if isinstance(output, tuple) else output
                if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != prompt_length:
                    raise ValueError('Unexpected raw Query shape')
                source = _inputs[0]
                if source.ndim != 3 or source.shape[:2] != (1, prompt_length):
                    raise ValueError('Unexpected Query source shape')
                record['q_input'] = source[0, -1].detach().float().cpu().clone()
                record['q_raw'] = value[0, -1].detach().float().cpu().clone()

            def norm_hook(_module, _inputs, output, record=record):
                value = output[0] if isinstance(output, tuple) else output
                if value.ndim != 4 or value.shape[:2] != (1, prompt_length):
                    raise ValueError('Unexpected normalized Query shape')
                record['q_normalized'] = value[0, -1].detach().float().cpu().clone()

            def attention_hook(_module, _inputs, output, record=record):
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise ValueError('Eager Gemma-4 attention weights were not returned')
                weights = output[1]
                if weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[2:] != (prompt_length, prompt_length):
                    raise ValueError('Unexpected attention weight shape')
                row = weights[0, :, -1].detach().float().cpu().clone()
                if not (row.isfinite().all() and (row.sum(-1) - 1).abs().max() < .02):
                    raise ValueError('Attention rows are not finite normalized probabilities')
                record['weights'] = row

            handles.append(attn.q_proj.register_forward_hook(raw_hook))
            handles.append(attn.q_norm.register_forward_hook(norm_hook))
            handles.append(attn.register_forward_hook(attention_hook))
        yield records
        for index, record in records.items():
            if set(record) != {'q_input', 'q_raw', 'q_normalized', 'weights'}:
                raise RuntimeError(f'Incomplete attention capture at layer {index}')
    finally:
        for handle in handles:
            handle.remove()


def summarize_attention(records, spans, token_ids, tokenizer):
    """Per-head routing, Query norm, entropy, and top keys, with raw vectors kept separate."""
    import torch
    summary = {}
    for layer, record in records.items():
        weights = record['weights']
        q = record['q_normalized']
        if q.shape[0] != weights.shape[0] or len(token_ids) != weights.shape[1]:
            raise ValueError('Query/head/token dimensions disagree')
        entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
        top_values, top_positions = weights.topk(min(5, weights.shape[1]), dim=-1)
        heads = []
        for head in range(weights.shape[0]):
            heads.append({'head': head, 'entropy_nats': float(entropy[head]),
                          'q_raw_norm': float(record['q_raw'].reshape(weights.shape[0], -1)[head].norm()),
                          'q_normalized_norm': float(q[head].norm()),
                          'span_mass': {name: float(weights[head, positions].sum())
                                        for name, positions in spans.items()},
                          'top_keys': [{'position': int(pos), 'weight': float(weight),
                                        'token': tokenizer.decode([token_ids[int(pos)]])}
                                       for weight, pos in zip(top_values[head], top_positions[head])]})
        summary[str(layer)] = {'q_input_norm': float(record['q_input'].norm()),
                               'heads': heads}
    return summary
