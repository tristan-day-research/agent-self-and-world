"""Query replacement on the first complete forward, including generation prefill."""
from contextlib import contextmanager
from core.attention import decoder_layers


@contextmanager
def replace_prefill_queries(model, specification):
    """Replace selected Q after q_norm, before RoPE; skip cached decode calls.

    specification has layer, heads, positions, source [all heads, rows, head_dim].
    The caller validates identical text for rescue, or explicitly aligned boundary
    positions for cross-context transfer. This primitive cannot infer semantic alignment.
    """
    if specification is None:
        yield
        return
    module=decoder_layers(model)[specification['layer']].self_attn
    positions=specification['positions']; heads=specification['heads']
    source=specification['source']
    if source.ndim!=3 or source.shape[1]!=len(positions) or not positions or len(set(positions))!=len(positions) or min(positions)<0:
        raise ValueError('Invalid query replacement positions/source')
    if not heads or len(set(heads))!=len(heads) or min(heads)<0 or max(heads)>=source.shape[0]:
        raise ValueError('Invalid query replacement heads')
    state={'active':False,'calls':0}
    def before(_m,args,kwargs):
        cache=kwargs.get('cache_position')
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        state['active']=state['calls']==0 and (cache is None or int(cache[0])==0)
        if state['active'] and max(positions)>=hidden.shape[1]:
            raise ValueError('Query patch positions exceed prefill')
    def patch(_m,args,output):
        if not state['active']:return output
        if output.shape[0]!=1 or source.shape[0]!=output.shape[1] or source.shape[2]!=output.shape[3]:
            raise ValueError('Query replacement shape differs')
        changed=output.clone(); values=source.to(output.device,output.dtype)
        for head in heads:changed[0,head,positions,:]=values[head]
        state['calls']+=1
        return changed
    pre=module.register_forward_pre_hook(before,with_kwargs=True)
    hook=module.q_norm.register_forward_hook(patch)
    try:
        yield
        if state['calls']!=1:raise RuntimeError('Query replacement did not execute once')
    finally:
        hook.remove();pre.remove()


def query_sources(result, layer):
    """Copy the small donor Q arrays before artifact serialization removes tensors."""
    out={}
    for phase,owner in [('prompt',result),('fixed',result['fixed_continuation'])]:
        record=owner['attention']
        out[phase]={'source':record['_tensors'][f'L{layer}.q_normalized'].clone(),
                    'positions':list(record['query_positions']), 'input_ids':list(record['input_ids'])}
    return out


def patch_specification(donor,layer,head,input_ids):
    if donor['input_ids']!=input_ids:raise ValueError('Query donor/recipient text differs')
    return {'layer':layer,'heads':[head],'source':donor['source'],'positions':donor['positions']}


def capture_boundary_query(model,inputs,layer):
    """Unsteered donor forward, returning all heads at its final prompt position."""
    import torch
    captured=[]
    module=decoder_layers(model)[layer].self_attn
    handle=module.q_norm.register_forward_hook(lambda _m,_args,output: captured.append(output.detach()[0,:,-1:,:].float().cpu().clone()))
    try:
        with torch.inference_mode():model(**inputs,use_cache=False)
    finally:handle.remove()
    if len(captured)!=1:raise RuntimeError('Donor query capture did not execute once')
    return captured[0]
