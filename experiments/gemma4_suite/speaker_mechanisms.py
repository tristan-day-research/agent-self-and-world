"""Discover Gemma-4 speaker/viewpoint candidates, then test frozen directions."""
import hashlib
import json
import statistics
from pathlib import Path

from core.runs import run_context, write_json
from experiments.gemma4_suite.speaker_design import prompts


def discovery_plan(args):
    rows = [r for r in prompts() if r['split'] != 'steer_test']
    steer_prompts = [r for r in prompts() if r['split'] == 'steer_test' and
                     (r['kind'] == 'viewpoint' or
                      (r['kind'] == 'speaker_setting' and r['person'] == 'third' and
                       r['setting'] in {'assistant', 'story'}))]
    return {'stage': 'speaker_discover', 'train_characters': 4, 'validation_characters': 2,
            'reserved_steer_test_characters': 2, 'prompts': len(rows),
            'forwards': len(rows), 'generations': len(rows),
            'steering_test_prompts': len(steer_prompts) if args.steer_after_discovery else 0,
            'steering_arms_per_prompt': 8 if args.steer_after_discovery else 0,
            'additional_steering_generations': len(steer_prompts)*8 if args.steer_after_discovery else 0,
            'layers': args.residual_layers, 'sae_layer': 30,
            'note': 'Train-only direction selection; separate validation and untouched steering-test characters.'}


def _pair_rows(rows, kind, split, positive, negative, person_filter=None):
    filtered = [r for r in rows if r['kind'] == kind and r['split'] == split and
                (person_filter is None or r['person'] == person_filter)]
    keys = ('character', 'prose') if kind == 'viewpoint' else ('character', 'person', 'prose')
    groups = {}
    for row in filtered:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, {})[row['viewpoint'] if kind == 'viewpoint' else row['setting']] = row
    return [(group[positive], group[negative]) for group in groups.values()]


def _direction(rows, vectors, layer, kind, split, positive, negative, person_filter=None):
    import torch
    pairs = _pair_rows(rows, kind, split, positive, negative, person_filter)
    differences = torch.stack([vectors[p['id']][layer].float() - vectors[n['id']][layer].float()
                               for p, n in pairs])
    mean = differences.mean(0)
    norm = float(mean.norm())
    if norm < 1e-6:
        raise ValueError('Speaker contrast collapsed')
    unit = mean / norm
    margins = differences @ unit
    return unit, {'n_pairs': len(pairs), 'mean_margin': float(margins.mean()),
                  'positive_pairs': int((margins > 0).sum()),
                  'median_pair_margin': float(margins.median())}


def _feature_candidates(rows, sparse, *, kind='viewpoint', positive='immersed',
                        negative='detached', person_filter=None):
    import torch
    pairs = _pair_rows(rows, kind, 'train', positive, negative, person_filter)
    deltas = []
    for positive, negative in pairs:
        a = torch.zeros(43008)
        b = torch.zeros(43008)
        for vector, row in ((a, positive), (b, negative)):
            entry = sparse[row['id']]
            vector[entry['indices'].long()] = entry['values'].float()
        deltas.append(a-b)
    matrix = torch.stack(deltas)
    means = matrix.mean(0)
    support = (matrix != 0).sum(0)
    eligible = support >= 2
    scores = means.abs() / (matrix.std(0, unbiased=False) + 1.)
    scores[~eligible] = -1
    ids = scores.topk(10).indices.tolist()
    if scores[ids[0]] <= 0:
        raise ValueError('No supported viewpoint SAE feature candidate')
    return [{'feature_id': i, 'train_mean_delta': float(means[i]),
             'train_support_pairs': int(support[i]), 'train_score': float(scores[i])} for i in ids]


def discover(args):
    import torch
    from core.gemma4 import MODEL_ID, load_model, render, text_layers
    from core.gemma4_sae import load_sae
    from core.residual_experiments import capture_layers, write_torch
    if 30 not in args.residual_layers or len(set(args.residual_layers)) != len(args.residual_layers):
        raise ValueError('Speaker discovery requires unique residual layers including 30')
    specification = discovery_plan(args)
    if args.dry_run:
        print(json.dumps(specification, indent=2)); return 0
    if args.smoke:
        from experiments.gemma4_suite.smoke import main as smoke_main
        return smoke_main(args, 'speaker-discover')
    rows = [r for r in prompts() if r['split'] != 'steer_test']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != 'resume_directory'}
    with run_context(args.experiment, config, args.resume_directory) as run:
        write_json(run.directory/'design.json', rows)
        model, processor, revision = load_model('cuda')
        layers = text_layers(model)
        sae, sae_meta = load_sae(30)
        sae.to('cuda')
        if args.resume_directory and (run.directory/'metadata.json').exists():
            previous = json.loads((run.directory/'metadata.json').read_text())
            if previous.get('revision') != revision or previous.get('sae', {}).get('revision') != sae_meta['revision']:
                raise ValueError('Model or SAE checkpoint revision changed on resume')
        root = Path(__file__).parents[2]
        sources = [Path(__file__), root/'experiments/gemma4_suite/speaker_design.py',
                   root/'core/residual_experiments.py', root/'core/gemma4.py',
                   root/'core/gemma4_sae.py', root/'core/steering.py']
        write_json(run.directory/'metadata.json', {'model': MODEL_ID, 'revision': revision,
            'sae': sae_meta, 'plan': specification,
            'measurement_code_sha256': {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources}})
        outputs_path = run.directory/'outputs.json'
        vectors_path = run.directory/'vectors.pt'
        results = json.loads(outputs_path.read_text())['results'] if args.resume_directory and outputs_path.exists() else []
        saved = torch.load(vectors_path, map_location='cpu', weights_only=True) if args.resume_directory and vectors_path.exists() else {'residuals': {}, 'sae': {}}
        vectors, sparse = saved['residuals'], saved['sae']
        completed = {r['id'] for r in results}
        if len(completed) != len(results):
            raise ValueError('Resume contains duplicate speaker IDs')
        if completed != set(vectors) or completed != set(sparse):
            valid = completed & set(vectors) & set(sparse)
            results = [r for r in results if r['id'] in valid]
            saved['residuals'] = vectors = {key: value for key, value in vectors.items() if key in valid}
            saved['sae'] = sparse = {key: value for key, value in sparse.items() if key in valid}
            completed = valid
            write_json(outputs_path, {'kind': 'gemma4_speaker_discovery', 'results': results})
            write_torch(vectors_path, saved)
        for row in rows:
            if row['id'] in completed:
                continue
            prompt, inputs = render(processor, row['messages'])
            inputs = inputs.to('cuda')
            with capture_layers(layers, args.residual_layers) as captured:
                with torch.inference_mode():
                    model(**inputs, use_cache=False)
            with torch.inference_mode():
                values, indices = sae.encode(captured[30].to('cuda'))
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
            raw = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                              skip_special_tokens=False).strip()
            response = processor.parse_response(raw, prefix=inputs['input_ids'][0])['content'].strip()
            vectors[row['id']] = {layer: captured[layer].half() for layer in args.residual_layers}
            sparse[row['id']] = {'indices': indices.cpu().int(), 'values': values.cpu().half()}
            results.append({**{k: v for k, v in row.items() if k != 'messages'},
                            'formatted_prompt': prompt, 'response': response, 'raw_response': raw})
            write_json(outputs_path, {'kind': 'gemma4_speaker_discovery', 'results': results})
            write_torch(vectors_path, saved)
            run.logger.info('Captured %d/%d %s', len(results), len(rows), row['id'])
        contrasts = {}
        frozen = {}
        for name, kind, positive, negative, person_filter in [
            ('assistant_story', 'speaker_setting', 'story', 'assistant', None),
            ('assistant_third_story', 'speaker_setting', 'story', 'assistant', 'third'),
            ('roleplay_story', 'speaker_setting', 'story', 'roleplay', None),
            ('viewpoint', 'viewpoint', 'immersed', 'detached', None)]:
            contrasts[name] = {}
            for layer in args.residual_layers:
                unit, training = _direction(rows, vectors, layer, kind, 'train', positive, negative, person_filter)
                validation_pairs = _pair_rows(rows, kind, 'validation', positive, negative, person_filter)
                margins = [float((vectors[p['id']][layer].float()-vectors[n['id']][layer].float()) @ unit)
                           for p, n in validation_pairs]
                contrasts[name][str(layer)] = {'train': training,
                    'validation': {'n_pairs': len(margins), 'mean_margin': statistics.mean(margins),
                                   'positive_pairs': sum(m > 0 for m in margins)}}
                if layer == 30 and name in ('assistant_third_story', 'viewpoint'):
                    frozen[name] = {'layer': layer, 'unit': unit.cpu().half(),
                                    'scale': max(1., training['median_pair_margin'] * .25)}
        feature_sets = {}
        for name, kind, positive, negative, person_filter, scale_source in [
            ('viewpoint_sae', 'viewpoint', 'immersed', 'detached', None, 'viewpoint'),
            ('assistant_third_sae', 'speaker_setting', 'story', 'assistant',
             'third', 'assistant_third_story')]:
            candidates = _feature_candidates(rows, sparse, kind=kind, positive=positive,
                                             negative=negative, person_filter=person_filter)
            winner = candidates[0]
            feature_id = winner['feature_id']
            orientation = 1 if winner['train_mean_delta'] >= 0 else -1
            decoder = sae.direction(feature_id).detach().float().cpu() * orientation
            frozen[name] = {'layer': 30, 'unit': (decoder / decoder.norm()).half(),
                            'scale': frozen[scale_source]['scale'],
                            'feature_id': feature_id, 'orientation': orientation}
            validation_pairs = _pair_rows(rows, kind, 'validation', positive, negative,
                                          person_filter)
            def value(row):
                data = sparse[row['id']]
                match = (data['indices'] == feature_id).nonzero(as_tuple=False)
                return float(data['values'][int(match[0, 0])]) if len(match) else 0.
            winner['validation_mean_delta'] = statistics.mean(value(p)-value(n) for p, n in validation_pairs)
            winner['validation_positive_pairs'] = sum(orientation*(value(p)-value(n)) > 0
                                                      for p, n in validation_pairs)
            feature_sets[name] = candidates
        write_json(run.directory/'discovery.json', {'contrasts': contrasts,
            'sae_candidates_train_ranked': feature_sets,
            'candidate_selection': 'Train characters only; layer 30 fixed before validation'})
        write_torch(run.directory/'directions.pt', {'model': MODEL_ID, 'model_revision': revision,
                    'sae_revision': sae_meta['revision'], 'directions': frozen,
                    'discovery_outputs_sha256': hashlib.sha256(outputs_path.read_bytes()).hexdigest()})
        if args.steer_after_discovery:
            _steer_test(run, args, model, processor, layers, frozen)
    return 0


def _steer_test(run, args, model, processor, layers, directions):
    """Precommitted, untouched characters; only train-selected directions."""
    import torch
    from core.gemma4 import render
    from core.steering import steer_layer
    candidates = [r for r in prompts() if r['split'] == 'steer_test' and
                  (r['kind'] == 'viewpoint' or
                   (r['kind'] == 'speaker_setting' and r['person'] == 'third' and
                    r['setting'] in {'assistant', 'story'}))]
    rng = torch.Generator(device='cpu').manual_seed(args.seed + 991)
    random = torch.randn(5376, generator=rng)
    random = random / random.norm()
    path = run.directory/'steering.json'
    results = json.loads(path.read_text())['results'] if path.exists() else []
    done = {(r['prompt_id'], r['arm']) for r in results}
    if len(done) != len(results):
        raise ValueError('Steering resume contains duplicate rows')
    for row in candidates:
        if row['kind'] == 'viewpoint':
            selected = ('viewpoint', 'viewpoint_sae')
            random_name = 'random_viewpoint'
        else:
            selected = ('assistant_third_story', 'assistant_third_sae')
            random_name = 'random_story'
        arms = [('baseline', None, None), ('zero', selected[0], 0.)]
        arms += [(name, name, sign) for name in selected for sign in (-1., 1.)]
        arms += [(random_name, random_name, sign) for sign in (-1., 1.)]
        prompt, inputs = render(processor, row['messages'])
        inputs = inputs.to('cuda')
        baseline_logits = None
        baseline_response = next((r['response'] for r in results
                                  if r['prompt_id'] == row['id'] and r['arm'] == 'baseline'), None)
        if baseline_response is not None:
            with torch.inference_mode():
                baseline_logits = model(**inputs, use_cache=False).logits[0, -1].detach().float().cpu()
        for arm, direction_name, sign in arms:
            if (row['id'], arm if sign is None else f'{arm}:{sign:+g}') in done:
                continue
            if direction_name in ('random_story', 'random_viewpoint'):
                match = 'assistant_third_story' if direction_name == 'random_story' else 'viewpoint'
                direction, magnitude = random, directions[match]['scale']
            elif direction_name:
                record = directions[direction_name]
                direction, magnitude = record['unit'].float(), record['scale']
            else:
                direction, magnitude = None, None
            from contextlib import nullcontext
            context = (nullcontext() if direction is None else
                       steer_layer(layers[30], direction, sign * magnitude, scope='last'))
            with context:
                with torch.inference_mode():
                    logits = model(**inputs, use_cache=False).logits[0, -1].detach().float().cpu()
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                               do_sample=False,
                                               pad_token_id=processor.tokenizer.eos_token_id)
            raw = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                              skip_special_tokens=False).strip()
            response = processor.parse_response(raw, prefix=inputs['input_ids'][0])['content'].strip()
            if arm == 'baseline':
                baseline_logits, baseline_response = logits, response
            if arm == 'zero' and baseline_logits is not None:
                if not torch.equal(logits, baseline_logits) or response != baseline_response:
                    raise ValueError('Zero hook changed baseline outputs')
            if baseline_logits is None:
                # On resume, recompute the baseline if its row was already saved.
                with torch.inference_mode():
                    baseline_logits = model(**inputs, use_cache=False).logits[0, -1].detach().float().cpu()
            base_logp = baseline_logits.log_softmax(-1)
            now_logp = logits.log_softmax(-1)
            kl = float((base_logp.exp() * (base_logp - now_logp)).sum())
            label = arm if sign is None else f'{arm}:{sign:+g}'
            results.append({'prompt_id': row['id'], 'kind': row['kind'],
                            'character': row['character'], 'prose': row['prose'],
                            'setting': row['setting'], 'viewpoint': row.get('viewpoint'),
                            'arm': label, 'direction_name': direction_name,
                            'coefficient': None if sign is None else sign*magnitude,
                            'response': response, 'raw_response': raw,
                            'boundary_kl_from_baseline_nats': kl,
                            'boundary_top_token_id': int(logits.argmax())})
            write_json(path, {'kind': 'gemma4_speaker_steering', 'results': results,
                              'random_seed': args.seed+991})
            run.logger.info('Steered %d/%d %s %s', len(results), len(candidates)*8, row['id'], label)
