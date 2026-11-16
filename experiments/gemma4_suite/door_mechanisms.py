"""Fresh-family crossed door behavior, held-out parity probes, and residual patches."""
import hashlib
import json
from pathlib import Path

from core.runs import run_context, write_json

DEFAULT_LAYERS = (10, 20, 30, 40, 50)


def door_design(families, seed):
    from types import SimpleNamespace
    from experiments.gemma4_suite.main import design
    rows = design(SimpleNamespace(sections=['door'], families=families, seed=seed))
    # Equal rule order across code pairs. This limits the number of forwards and
    # keeps the train/test split at the code-family level.
    return [r for r in rows if r['task'] == 'self_report' or
            (r['task'] == 'instrumental' and r['rule_order'] == 'even_first')]


def plan(args):
    rows = door_design(args.door_families, args.seed)
    train = args.door_families // 2
    heldout = args.door_families - train
    patches = heldout * 2 * 2 * len(args.residual_layers)
    self_controls = 2 * 2 * len(args.residual_layers)
    return {'design': 'crossed fresh-family door', 'prompts': len(rows),
            'train_families': train, 'heldout_families': heldout,
            'baseline_forwards_and_generations': len(rows),
            'opposite_parity_patches': patches, 'self_patch_controls': self_controls,
            'residual_layers': args.residual_layers, 'max_new_tokens': 12,
            'note': 'Probe train/test split by family. Patch target final-position residual only.'}


def _choice_for_row(row, processor, prompt):
    from core.choice_metrics import continuation_label_token_ids
    labels = ['Y', 'N'] if row['task'] == 'self_report' else ['L', 'R', 'A']
    return continuation_label_token_ids(processor.tokenizer, prompt, labels)


def _probe_results(rows, vectors, layer_ids, train_family_count):
    from core.residual_experiments import fit_ridge_probe, score_probe
    output = []
    present = ('present_knows', 'present_doesnt_know')
    targets = ('present_knows', 'present_doesnt_know', 'absent_knows', 'absent_doesnt_know')
    for layer in layer_ids:
        for source_task in ('self_report', 'instrumental'):
            for source in present:
                training = [r for r in rows if r['task'] == source_task and r['condition'] == source
                            and int(r['family'][3:]) < train_family_count]
                weights, bias = fit_ridge_probe([vectors[r['id']][layer] for r in training],
                                                [r['parity'] == 'even' for r in training])
                for target_task in ('self_report', 'instrumental'):
                    for target in targets:
                        evaluation = [r for r in rows if r['task'] == target_task and r['condition'] == target
                                      and int(r['family'][3:]) >= train_family_count]
                        metrics = score_probe(weights, bias, [vectors[r['id']][layer] for r in evaluation],
                                              [r['parity'] == 'even' for r in evaluation])
                        output.append({'layer': layer, 'source_task': source_task,
                                       'target_task': target_task, 'source_condition': source,
                                       'target_condition': target, **metrics})
    return output


def _factor_probes(rows, vectors, layer_ids, train_family_count):
    from core.residual_experiments import fit_ridge_probe, score_probe
    output = []
    for layer in layer_ids:
        for factor in ('code_available', 'attributed_knowledge'):
            for source_task in ('self_report', 'instrumental'):
                training = [r for r in rows if r['task'] == source_task and
                            int(r['family'][3:]) < train_family_count]
                weight, bias = fit_ridge_probe([vectors[r['id']][layer] for r in training],
                                               [r[factor] for r in training])
                for target_task in ('self_report', 'instrumental'):
                    heldout = [r for r in rows if r['task'] == target_task and
                               int(r['family'][3:]) >= train_family_count]
                    output.append({'layer': layer, 'factor': factor, 'source_task': source_task,
                                   'target_task': target_task,
                                   **score_probe(weight, bias,
                                       [vectors[r['id']][layer] for r in heldout],
                                       [r[factor] for r in heldout])})
    return output


def main(args):
    import torch
    from core.gemma4 import MODEL_ID, load_model, render, text_layers
    from core.choice_metrics import distribution_metrics
    from core.residual_experiments import capture_layers, patch_layer, write_torch
    from experiments.agent_vs_model.main import parse_answer
    if args.door_families < 4 or args.door_families % 2 or len(set(args.residual_layers)) != len(args.residual_layers):
        raise ValueError('Use an even number of at least four door families and unique layers')
    if any(not 0 <= layer < 60 for layer in args.residual_layers):
        raise ValueError('Residual layers must be 0..59')
    specification = plan(args)
    if args.dry_run:
        print(json.dumps(specification, indent=2)); return 0
    if args.smoke:
        from experiments.gemma4_suite.smoke import main as smoke_main
        return smoke_main(args, 'door-mechanisms')
    rows = door_design(args.door_families, args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != 'resume_directory'}
    with run_context(args.experiment, config, args.resume_directory) as run:
        write_json(run.directory/'design.json', rows)
        model, processor, revision = load_model('cuda')
        layers = text_layers(model)
        if args.resume_directory and (run.directory/'metadata.json').exists():
            previous = json.loads((run.directory/'metadata.json').read_text())
            if previous.get('revision') != revision:
                raise ValueError('Gemma-4 checkpoint revision changed on resume')
        root = Path(__file__).parents[2]
        sources = [Path(__file__), root/'core/residual_experiments.py',
                   root/'core/gemma4.py', root/'experiments/gemma4_suite/main.py',
                   root/'experiments/agent_vs_model/design.py']
        write_json(run.directory/'metadata.json', {'model': MODEL_ID, 'revision': revision,
            'plan': specification, 'measurement_code_sha256': {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources}})
        results_path = run.directory/'outputs.json'
        vectors_path = run.directory/'residuals.pt'
        results = json.loads(results_path.read_text())['results'] if args.resume_directory and results_path.exists() else []
        vectors = torch.load(vectors_path, map_location='cpu', weights_only=True) if args.resume_directory and vectors_path.exists() else {}
        completed = {r['id'] for r in results}
        if len(completed) != len(results):
            raise ValueError('Resume contains duplicate result IDs')
        if completed != set(vectors):
            # A process can stop between the two atomic checkpoints. Recompute
            # only the unpaired last rows, preserving the verified intersection.
            valid = completed & set(vectors)
            results = [r for r in results if r['id'] in valid]
            vectors = {key: value for key, value in vectors.items() if key in valid}
            completed = valid
            write_json(results_path, {'kind': 'gemma4_door_mechanisms', 'results': results})
            write_torch(vectors_path, vectors)
        for row in rows:
            if row['id'] in completed:
                continue
            prompt, inputs = render(processor, row['messages'])
            inputs = inputs.to('cuda')
            with capture_layers(layers, args.residual_layers) as capture:
                with torch.inference_mode():
                    logits = model(**inputs, use_cache=False).logits[0, -1].detach().float().cpu()
            token_ids = _choice_for_row(row, processor, prompt)
            distribution = distribution_metrics(logits, token_ids)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                           pad_token_id=processor.tokenizer.eos_token_id)
            raw = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                              skip_special_tokens=False).strip()
            response = processor.parse_response(raw, prefix=inputs['input_ids'][0])['content'].strip()
            result = {k: v for k, v in row.items() if k != 'messages'}
            result.update(formatted_prompt=prompt, response=response, raw_response=raw,
                          input_token_count=int(inputs['input_ids'].shape[1]),
                          parsed=parse_answer(row['task'], response, row['code']),
                          distribution=distribution)
            results.append(result)
            vectors[row['id']] = {layer: capture[layer].half() for layer in args.residual_layers}
            write_json(results_path, {'kind': 'gemma4_door_mechanisms', 'results': results})
            write_torch(vectors_path, vectors)
            run.logger.info('Captured %d/%d %s', len(results), len(rows), row['id'])
        probes = _probe_results(rows, vectors, args.residual_layers, args.door_families//2)
        write_json(run.directory/'probes.json', probes)
        write_json(run.directory/'factor_probes.json',
                   _factor_probes(rows, vectors, args.residual_layers, args.door_families//2))
        by_key = {(r['family'], r['parity'], r['condition'], r['task']): r for r in rows}
        baseline_by_id = {r['id']: r for r in results}
        patches_path = run.directory/'patches.json'
        patches = json.loads(patches_path.read_text())['results'] if args.resume_directory and patches_path.exists() else []
        patched = {(r['target_id'], r['layer'], r['control']) for r in patches}
        test_rows = [r for r in rows if r['task'] == 'instrumental' and r['code_available']
                     and int(r['family'][3:]) >= args.door_families//2]
        for target in test_rows:
            donor_parity = 'odd' if target['parity'] == 'even' else 'even'
            donor = by_key[(target['family'], donor_parity, target['condition'], 'instrumental')]
            if baseline_by_id[target['id']]['input_token_count'] != baseline_by_id[donor['id']]['input_token_count']:
                run.logger.warning('Skipping unequal-token-length patch pair %s / %s', target['id'], donor['id'])
                continue
            prompt, inputs = render(processor, target['messages'])
            inputs = inputs.to('cuda')
            token_ids = _choice_for_row(target, processor, prompt)
            for layer_id in args.residual_layers:
                for control, source in [('opposite_parity', donor),
                                        ('self', target) if target['family'] == test_rows[0]['family'] else (None, None)]:
                    if control is None or (target['id'], layer_id, control) in patched:
                        continue
                    with patch_layer(layers[layer_id], vectors[source['id']][layer_id]):
                        with torch.inference_mode():
                            logits = model(**inputs, use_cache=False).logits[0, -1].detach().float().cpu()
                    distribution = distribution_metrics(logits, token_ids)
                    if control == 'self':
                        reference = baseline_by_id[target['id']]['distribution']['choice_probabilities']
                        if max(abs(distribution['choice_probabilities'][label]-reference[label])
                               for label in reference) > 1e-5:
                            raise ValueError(f'Self patch changed choice probabilities at L{layer_id}')
                    patches.append({'target_id': target['id'], 'donor_id': source['id'],
                                    'layer': layer_id, 'control': control,
                                    'target_parity': target['parity'],
                                    'correct_procedure': target['correct_procedure'],
                                    'donor_procedure': donor['correct_procedure'],
                                    'distribution': distribution})
                    write_json(patches_path, {'kind': 'gemma4_residual_patches', 'results': patches})
            run.logger.info('Patched %s (%d records)', target['id'], len(patches))
    return 0
