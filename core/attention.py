"""Bounded, observational Gemma-3 attention capture. No backend replacement.

Capture prefill/teacher-forced rows only; cached decode is deliberately skipped.
All heads in selected blocks, selected query positions, all available key positions.
"""
from contextlib import contextmanager, ExitStack
from pathlib import Path


def configure_attention(parser):
    parser.add_argument('--attention', action='store_true', help='Save bounded attention/Q diagnostics on prompt and fixed-text passes')
    parser.add_argument('--attention-layers', type=int, nargs='+', default=[8,9,10,17,22,29,32,33],
                        help='Zero-based Gemma blocks; use -1 alone for all blocks')
    parser.add_argument('--attention-max-positions', type=int, default=4)
    parser.add_argument('--attention-stride', type=int, default=16)


def attention_options(args):
    if not args.attention:
        return None
    if args.attention_max_positions < 1 or args.attention_stride < 1:
        raise ValueError('Attention position limit and stride must be positive')
    return {'layers':args.attention_layers, 'max_positions':args.attention_max_positions,
            'stride':args.attention_stride}


def decoder_layers(model):
    for name in ('model.language_model.layers', 'model.layers'):
        try:
            return model.get_submodule(name)
        except AttributeError:
            pass
    raise ValueError('Attention recorder currently supports Gemma-3 text decoders only')


@contextmanager
def record_attention(model, input_ids, prompt_length, options=None):
    """Observe actual eager weights plus projected Q and per-head context vectors.

    Tensors are detached CPU float32. K/V summaries use KV-head indices; query
    heads map to KV heads by integer division by num_key_value_groups. Q vectors
    include both sides of q_norm and RoPE. No gradients or extra model forwards.
    """
    import torch
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, apply_rotary_pos_emb
    record = {'rows': [], '_tensors': {}, 'skipped_cached_calls': 0}
    if options is None:
        yield record
        return
    layers = decoder_layers(model)
    requested = options['layers']
    selected = list(range(len(layers))) if requested == [-1] else requested
    if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= len(layers) for i in selected):
        raise ValueError(f'Attention layers must be unique and in 0..{len(layers)-1}, or -1 alone')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 1 <= prompt_length <= input_ids.shape[1]:
        raise ValueError('Attention capture requires one unpadded text sequence and a valid prompt boundary')
    ids = input_ids[0].tolist()
    positions = [prompt_length-1] + list(range(prompt_length, len(ids), options['stride']))
    positions = positions[:options['max_positions']]
    record.update(schema_version=1, input_ids=ids, query_positions=positions,
                  scope='prefill or teacher-forced full text only; cached decoding skipped',
                  layers=selected, tensor_dtype='float32',
                  head_output_location='context before o_proj; projected attention output before post_attention_layernorm')
    with ExitStack() as stack:
        for index in selected:
            module = layers[index].self_attn
            if not isinstance(module, Gemma3Attention) or module.config._attn_implementation != 'eager' or module.training:
                raise ValueError('Attention capture requires eval-mode Gemma3Attention with eager backend')
            state = {}

            def before(_m, args, kwargs, state=state):
                hidden = kwargs.get('hidden_states', args[0] if args else None)
                cache_pos = kwargs.get('cache_position')
                state.clear()
                # First full-context call only. Later cached calls have nonzero origin.
                state['active'] = hidden.shape[1] == len(ids) and (cache_pos is None or int(cache_pos[0]) == 0)
                if not state['active']:
                    record['skipped_cached_calls'] += 1
                    return
                state['rope'] = kwargs['position_embeddings']

            def query(_m, args, output, state=state):
                if state.get('active'):
                    state['q_raw'] = args[0].detach()[:, :, positions, :]
                    state['q_norm'] = output.detach()[:, :, positions, :]

            def key(_m, args, output, state=state):
                if state.get('active'):
                    state['k_norms'] = output.detach().float().norm(dim=-1)[0]
                    state['k_raw_norms'] = args[0].detach().float().norm(dim=-1)[0]

            def value(m, args, output, state=state, module=module):
                if state.get('active'):
                    state['v_norms'] = output.detach().view(1,len(ids),-1,module.head_dim).float().norm(dim=-1)[0].T

            def context(_m, args, state=state, module=module):
                if state.get('active'):
                    state['context'] = args[0].detach()[0,positions].view(len(positions),-1,module.head_dim).transpose(0,1)

            def after(m, args, output, state=state, index=index):
                if not state.get('active'):
                    return
                weights = output[1]
                if weights is None or weights.shape[-1] != len(ids):
                    raise RuntimeError('Expected eager weights over the entire unpadded text prefix')
                probs = weights.detach()[0,:,positions,:].float()
                cos, sin = state['rope']
                q = state['q_norm']
                rotated, _ = apply_rotary_pos_emb(q,q,cos[:,positions],sin[:,positions])
                tensors = {'q_raw':state['q_raw'][0], 'q_normalized':q[0], 'q_rotary':rotated[0],
                           'pattern':probs, 'k_raw_norm':state['k_raw_norms'], 'k_normalized_norm':state['k_norms'], 'v_norm':state['v_norms'],
                           'head_context':state['context'], 'attention_output':output[0].detach()[0,positions]}
                for key, tensor in tensors.items():
                    record['_tensors'][f'L{index}.{key}'] = tensor.float().cpu().clone()
                probs = record['_tensors'][f'L{index}.pattern']
                raw_q = record['_tensors'][f'L{index}.q_raw']
                normalized_q = record['_tensors'][f'L{index}.q_normalized']
                head_context = record['_tensors'][f'L{index}.head_context']
                for h in range(probs.shape[0]):
                    for p, position in enumerate(positions):
                        a = probs[h,p]
                        vals, inds = a.topk(min(8,len(a)))
                        allowed = int((a>0).sum())  # nonzero support, not architectural window size
                        record['rows'].append({'layer':index,'head':h,'kv_head':h//m.num_key_value_groups,
                            'position':position,'sliding_window':m.sliding_window,
                            'entropy_nats':float(-(a*a.clamp_min(1e-30).log()).sum()),
                            'nonzero_support':allowed,'row_sum':float(a.sum()),
                            'self_mass':float(a[position]),'bos_mass':float(a[0]),
                            'recent_16_mass':float(a[max(0,position-15):position+1].sum()),
                            'prompt_mass':float(a[:prompt_length].sum()),
                            'mean_distance':float((a*(position-torch.arange(len(a),device=a.device))).sum()),
                            'q_raw_norm':float(raw_q[h,p].norm()),
                            'q_normalized_norm':float(normalized_q[h,p].norm()),
                            'context_norm':float(head_context[h,p].norm()),
                            'top_keys':[{'position':int(k),'token_id':ids[int(k)],'weight':float(v)}
                                        for k,v in zip(inds,vals)]})
                state.clear()  # release GPU references immediately

            write_seen = [False]
            def writeback(_m, _args, output, index=index, seen=write_seen):
                if not seen[0] and f'L{index}.pattern' in record['_tensors']:
                    record['_tensors'][f'L{index}.residual_write'] = output.detach()[0,positions].float().cpu().clone()
                    seen[0] = True

            stack.callback(layers[index].post_attention_layernorm.register_forward_hook(writeback).remove)
            stack.callback(module.register_forward_pre_hook(before, with_kwargs=True).remove)
            stack.callback(module.q_norm.register_forward_hook(query).remove)
            stack.callback(module.k_norm.register_forward_hook(key).remove)
            stack.callback(module.v_proj.register_forward_hook(value).remove)
            stack.callback(module.o_proj.register_forward_pre_hook(context).remove)
            stack.callback(module.register_forward_hook(after).remove)
        yield record


def save_attention_artifact(result, directory, index, tokenizer=None):
    """Move binary arrays out of JSON, including a fixed-continuation observation."""
    import torch
    for phase, owner in [('prompt', result), ('fixed', result.get('fixed_continuation',{}))]:
        record = owner.get('attention')
        if record is None:
            continue
        tensors = record.pop('_tensors', None)
        if tensors is None:
            continue
        if not tensors:
            raise RuntimeError('Enabled attention recorder captured no full-context rows')
        relative = Path('attention')/f'{index:06d}_{phase}.pt'
        target = Path(directory)/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        torch.save(tensors,target)
        if tokenizer is not None:
            record['tokens'] = [tokenizer.decode([token]) for token in record['input_ids']]
        record['tensor_file'] = str(relative)
        record['tensor_bytes'] = sum(t.numel()*t.element_size() for t in tensors.values())


def compare_attention(reference, changed):
    """Compare tensors on identical text and query positions, per head/position.

    Caller must validate metadata/prefix identity; never align two different prompts
    by raw token index. Tensor comparisons alone cannot establish causal mediation.
    """
    import torch
    rows=[]
    for name, p in reference.items():
        if not name.endswith('.pattern'):
            continue
        prefix=name.removesuffix('.pattern')
        q=changed[name]
        if p.shape != q.shape:
            raise ValueError('Mismatched attention shapes')
        # BF16 softmax rows may have small rounding errors in their total mass.
        p=p/p.sum(-1,keepdim=True); q=q/q.sum(-1,keepdim=True)
        mid=(p+q)/2
        def kl(x,y): return (x*(x.clamp_min(1e-30).log()-y.clamp_min(1e-30).log())).sum(-1)
        js=(kl(p,mid)+kl(q,mid))/2
        for h in range(p.shape[0]):
            for pos in range(p.shape[1]):
                a=reference[prefix+'.q_rotary'][h,pos]; b=changed[prefix+'.q_rotary'][h,pos]
                rows.append({'layer':int(prefix[1:]),'head':h,'query_index':pos,
                    'attention_js_nats':float(js[h,pos]),
                    'attention_l1':float((p[h,pos]-q[h,pos]).abs().sum()),
                    'q_l2':float((b-a).norm()),
                    'q_cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                    'context_l2':float((changed[prefix+'.head_context'][h,pos]-reference[prefix+'.head_context'][h,pos]).norm()),
                    'block_attention_write_l2':float((changed[prefix+'.residual_write'][pos]-reference[prefix+'.residual_write'][pos]).norm())})
    return rows


@contextmanager
def patch_queries(attention_module, normalized_queries, positions, heads=None):
    """One full-context forward: replace selected Q after q_norm, before RoPE.

    This changes only Q locally, leaving that block's K/V untouched. Source shape
    [all_query_heads, selected_positions, head_dim] matches saved q_normalized.
    Caller must use the identical token prefix and positional setup. Not a
    generation/cache API and not itself evidence of mediation.
    """
    if not positions or len(set(positions)) != len(positions) or min(positions)<0:
        raise ValueError('Query patch positions must be unique nonnegative indices')
    if normalized_queries.ndim != 3 or normalized_queries.shape[1] != len(positions):
        raise ValueError('Query source shape must be [heads, positions, head_dim]')
    selected=list(range(normalized_queries.shape[0])) if heads is None else heads
    if not selected or len(set(selected))!=len(selected) or min(selected)<0 or max(selected)>=normalized_queries.shape[0]:
        raise ValueError('Invalid query patch heads')
    calls=0
    def hook(_module, _args, output):
        nonlocal calls
        calls+=1
        if calls!=1 or output.shape[0]!=1 or max(positions)>=output.shape[2]:
            raise ValueError('Query patch requires one unpadded full-context forward')
        if output.shape[1]!=normalized_queries.shape[0] or output.shape[3]!=normalized_queries.shape[2]:
            raise ValueError('Query patch head dimensions differ')
        changed=output.clone()
        source=normalized_queries.to(output.device,output.dtype)
        for head in selected:
            changed[0,head,positions,:]=source[head]
        return changed
    handle=attention_module.q_norm.register_forward_hook(hook)
    try:
        yield
        if calls!=1:
            raise RuntimeError('Query patch did not run exactly once')
    finally:
        handle.remove()
