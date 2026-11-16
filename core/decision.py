"""One next-token decision with steering, SAE measurements and boundary J-Lens."""
from contextlib import ExitStack

from core.choice_metrics import distribution_metrics
from core.steering import steer_layer


def last_output(layer, record, key):
    """Capture only the last token, avoiding a full per-layer activation cache."""
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        record[key] = hidden[:, -1, :].detach().float().cpu()
    return layer.register_forward_hook(hook)


def measure_decision(model, inputs, layer, sae, direction, strength, choices,
                     tokenizer, correct_label=None, lenses=None, save_vectors=True, attention=None,
                     steering_scope='all', monitors=None):
    import torch
    from core.jlens import residual_logits, transported_vectors
    from core.attention import record_attention
    captured = {}
    with ExitStack() as stack:
        attention_record = stack.enter_context(record_attention(model, inputs['input_ids'], inputs['input_ids'].shape[1], attention))
        stack.callback(last_output(layer, captured, 'pre').remove)
        if strength is not None:
            stack.enter_context(steer_layer(layer, direction, strength, steering_scope))
        stack.callback(last_output(layer, captured, 'post').remove)
        for layer_id, (module, _matrix) in (lenses or {}).items():
            if module is not layer:
                stack.callback(last_output(module, captured, layer_id).remove)
        for layer_id, (module, _encoder) in (monitors or {}).items():
            stack.callback(last_output(module, captured, f'monitor_{layer_id}').remove)
        with torch.inference_mode():
            output = model(**inputs, use_cache=False)
            metrics = distribution_metrics(output.logits[0, -1], choices, correct_label)
            metrics['unconstrained_argmax_token'] = tokenizer.decode([metrics['unconstrained_argmax_token_id']])
    if 'pre' not in captured or 'post' not in captured:
        raise RuntimeError('Decision capture did not run')
    before = sae.encode(captured['pre'].to(sae.w_enc.device))[0].tolist()
    after = sae.encode(captured['post'].to(sae.w_enc.device))[0].tolist()
    position = inputs['input_ids'].shape[1]-1
    points = []
    for layer_id, (module, matrix) in (lenses or {}).items():
        phases = [('pre', captured['pre']), ('post', captured['post'])] if module is layer else [('observed', captured[layer_id])]
        for phase, residual in phases:
            logits = residual_logits(matrix, residual, model)[0]
            lens_metrics = distribution_metrics(logits, choices, correct_label)
            values, ids = logits.topk(min(10, len(logits)))
            point = {'layer': layer_id, 'position': position, 'source': 'prompt',
                     'phase': phase, 'lens_kind': 'direct_final' if matrix is None else 'jacobian',
                     'distribution': lens_metrics,
                     'top_tokens': [{'token_id': int(t), 'token': tokenizer.decode([int(t)]),
                                     'logit': float(v)} for t,v in zip(ids,values)]}
            if save_vectors:
                point[f'_vector_{phase}'] = transported_vectors(matrix, residual)[0]
            if matrix is None and phase != 'pre':
                # Direct final readout must match actual output, not merely its argmax.
                actual=output.logits[0,-1].float()
                point['direct_final_max_logit_error']=float((logits.to(actual.device)-actual).abs().max())
                # A one-row BF16 GEMM can round differently from the full prompt GEMM.
                low_precision=next(model.parameters()).dtype in (torch.bfloat16,torch.float16)
                torch.testing.assert_close(logits.to(actual.device),actual,
                    rtol=2e-2 if low_precision else 1e-4,atol=.25 if low_precision else 1e-4)
            points.append(point)
    paper_boundary = {}
    for layer_id, (_module, encoder) in (monitors or {}).items():
        values = encoder.encode(captured[f'monitor_{layer_id}'].to(encoder.w_enc.device))[0].tolist()
        paper_boundary.update({f'L{layer_id}:{feature}': value
                               for feature, value in zip(encoder.feature_ids, values)})
    return {**({'attention': attention_record} if attention else {}), 'distribution': metrics,
            'sae_boundary_pre': dict(zip(map(str,sae.feature_ids),before)),
            'sae_boundary_post': dict(zip(map(str,sae.feature_ids),after)),
            'paper_boundary': paper_boundary,
            'jspace': points, 'input_ids': inputs['input_ids'][0].tolist(),
            'token_ids': []}


def generate_diagnostic(model,inputs,layer,sae,direction,strength,tokenizer,max_new_tokens,
                        steering_scope='all',monitors=None):
    """Free-text manipulation check, independent of the scored decision context."""
    import torch
    from core.sae import record_sae_features,flatten_record
    with ExitStack() as stack:
        pre=stack.enter_context(record_sae_features(layer,sae))
        if strength is not None:
            stack.enter_context(steer_layer(layer,direction,strength,steering_scope))
        post=stack.enter_context(record_sae_features(layer,sae))
        other={layer_id:stack.enter_context(record_sae_features(module,encoder))
               for layer_id,(module,encoder) in (monitors or {}).items()}
        with torch.inference_mode():
            tokens=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=0)
    prompt_length=inputs['input_ids'].shape[1]
    summaries={}
    for phase,record in [('pre',pre),('post',post)]:
        values,_=flatten_record(record)
        values=values[prompt_length:]
        summaries[phase]={'processed_reply_positions':len(values),
                          'mean':dict(zip(map(str,sae.feature_ids),values.mean(0).tolist())) if len(values) else None,
                          'peak':dict(zip(map(str,sae.feature_ids),values.max(0).values.tolist())) if len(values) else None}
    paper_generated={}
    for layer_id, record in other.items():
        values,_=flatten_record(record)
        values=values[prompt_length:]
        if len(values):
            encoder=monitors[layer_id][1]
            for feature,mean,peak in zip(encoder.feature_ids,values.mean(0).tolist(),values.max(0).values.tolist()):
                paper_generated[f'L{layer_id}:{feature}']={'mean':mean,'peak':peak,
                    'active_positions':int((values[:,encoder.feature_ids.index(feature)]>0).sum())}
    return {'text':tokenizer.decode(tokens[0,prompt_length:].tolist(),skip_special_tokens=True),
            'sae_generated':summaries,'paper_generated':paper_generated,
            'last_emitted_token_not_processed':True}
