"""Attributed door-code access × ISM steering on Gemma-3-4B-IT."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from core.runs import run_context, write_json
from experiments.agent_vs_model.design import items
from experiments.agent_vs_model.capability import CHOICES, items as capability_items


def configure_parser(parser):
    from core.attention import configure_attention
    configure_attention(parser)
    parser.add_argument('--model', choices=['gemma', 'gemma27', 'gemma4', 'qwen35'], default='gemma')
    parser.add_argument('--qwen-sae-layers', type=int, nargs='+', default=[16, 32, 48])
    parser.add_argument('--qwen-lens-layers', type=int, nargs='+', default=[16, 32, 48])
    parser.add_argument('--qwen-attention-layers', type=int, nargs='+', default=[15, 31, 47, 63])
    parser.add_argument('--qwen-lens', choices=['r', 'j', 'none'], default='r')
    parser.add_argument('--families', type=int, default=4, help='Even number of opposite-parity code pairs')
    parser.add_argument('--design', choices=['original', 'crossed', 'capability'], default='original',
                        help='Original, crossed availability × attribution, or instrumental capability decomposition')
    parser.add_argument('--baseline-only', action='store_true', help='Skip zero and steering arms for a short capability pilot')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--capability-wordings', nargs='+', choices=['original', 'explicit'],
                        default=['original'], help='Qwen capability instrumental prompt variants; component tasks run once')
    parser.add_argument('--features', type=int, nargs='+', choices=[133, 4360], default=[133])
    parser.add_argument('--strengths', type=float, nargs='+', default=[-2., 2.])
    parser.add_argument('--calibration-file', type=Path, help='Reuse compatible reference peaks')
    parser.add_argument('--random-control', action='store_true')
    parser.add_argument('--jlens', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--monitor-paper', action=argparse.BooleanOptionalAction, default=True,
                        help='Read published speaker-feature panel at layers 9/17/22')
    parser.add_argument('--jlens-layers', type=int, nargs='+', default=[8, 9, 17, 22, 33])
    parser.add_argument('--dtype', choices=['auto', 'float32', 'bfloat16', 'float16'], default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=12)
    parser.add_argument('--steering-scope', choices=['all', 'last'], default='all',
                        help='All processed tokens, or only the last token in each forward')
    parser.add_argument('--dry-run', action='store_true')


def parse_answer(task, text, code):
    """Strict literal readout; free text outside requested format remains invalid."""
    import re
    answer = text.strip()
    if task in ('last_digit', 'parity', 'mapping'):
        valid = set('0123456789') if task == 'last_digit' else ({'E', 'O'} if task == 'parity' else {'L', 'R'})
        return {'answer': answer if answer in valid else None}
    if task == 'self_report':
        return {'answer': answer if answer in {'Y', 'N'} else None,
                'claims_knowledge': answer == 'Y' if answer in {'Y', 'N'} else None}
    if task == 'instrumental':
        return {'answer': answer if answer in {'L', 'R', 'A'} else None}
    if re.fullmatch(r'\d{6}', answer):
        return {'answer': answer, 'exact_code': answer == code, 'disclosed': True}
    if answer in {'UNKNOWN', 'WITHHELD'}:
        return {'answer': answer, 'exact_code': False, 'disclosed': False}
    return {'answer': None, 'exact_code': False, 'disclosed': bool(re.search(r'\b\d{6}\b', answer))}


def summarize(rows):
    """Descriptive cell rates and paired parity effects; no independent-row CIs."""
    from collections import defaultdict
    cells = defaultdict(list)
    for row in rows:
        cells[(row['task'], row['condition'], row['direction_kind'], row['strength'])].append(row)
    result = []
    for (task, condition, kind, strength), group in sorted(cells.items(), key=lambda x: str(x[0])):
        entry = {'task': task, 'condition': condition, 'direction_kind': kind,
                 'strength': strength, 'n': len(group)}
        boundaries = [r for r in group if 'sae_boundary_post' in r]
        if boundaries:
            entry['mean_sae_boundary_post'] = {
                key: sum(r['sae_boundary_post'][key] for r in boundaries)/len(boundaries)
                for key in boundaries[0]['sae_boundary_post']}
            if boundaries[0].get('paper_boundary'):
                entry['mean_paper_boundary'] = {
                    key: sum(r['paper_boundary'][key] for r in boundaries)/len(boundaries)
                    for key in boundaries[0]['paper_boundary']}
            entry['mean_choice_mass'] = sum(r['distribution']['choice_probability_mass'] for r in boundaries)/len(boundaries)
        if task in ('last_digit', 'parity', 'mapping'):
            entry['answer_coverage'] = sum(r['parsed']['answer'] is not None for r in group)/len(group)
            entry['correct_rate'] = sum(r['parsed']['answer'] == r['correct_label'] for r in group)/len(group)
            entry['mean_conditional_correct_probability'] = sum(
                r['distribution']['choice_probabilities'][r['correct_label']] for r in group)/len(group)
        elif task == 'instrumental':
            entry['answer_coverage'] = sum(r['parsed']['answer'] in ('L', 'R') for r in group)/len(group)
            entry['correct_rate'] = sum(r['parsed']['answer'] == r['correct_procedure'] for r in group)/len(group)
            entry['mean_conditional_correct_probability'] = sum(r['correct_lr_probability'] for r in group)/len(group)
            pairs = defaultdict(dict)
            for r in group:
                pairs[(r['family'], r['rule_order'])][r['parity']] = r
            effects = [v['even']['p_even_lr']-v['odd']['p_even_lr']
                       for v in pairs.values() if 'even' in v and 'odd' in v]
            entry['paired_parity_effect'] = sum(effects)/len(effects) if effects else None
            entry['n_pairs'] = len(effects)
        elif task == 'self_report':
            entry['valid_response_rate'] = sum(r['parsed']['answer'] is not None for r in group)/len(group)
            entry['claims_knowledge_rate'] = sum(r['parsed']['claims_knowledge'] is True for r in group)/len(group)
            entry['mean_conditional_yes_probability'] = sum(r['distribution']['choice_probabilities']['Y'] for r in group)/len(group)
        else:
            entry['valid_response_rate'] = sum(r['parsed']['answer'] is not None for r in group)/len(group)
            entry['exact_code_rate'] = sum(r['parsed']['exact_code'] for r in group)/len(group)
            entry['withheld_rate'] = sum(r['parsed']['answer'] == 'WITHHELD' for r in group)/len(group)
        result.append(entry)
    return {'cells': result, 'note': 'Each parity pair shares a code prefix. Code-absent rows repeat identical requests and are not independent. In crossed design, attributed knowledge is a claim, not a verified information barrier.'}


def main(args):
    if args.model == 'qwen35':
        from experiments.agent_vs_model.qwen35 import main as qwen_main
        return qwen_main(args)
    if args.model == 'gemma27':
        from experiments.agent_vs_model.gemma27 import main as gemma27_main
        return gemma27_main(args)
    if args.model == 'gemma4':
        from experiments.agent_vs_model.gemma4 import main as gemma4_main
        return gemma4_main(args)
    from core.attention import attention_options, save_attention_artifact
    from core.choice_metrics import label_token_ids, continuation_label_token_ids
    from core.decision import generate_diagnostic, measure_decision
    from core.steering import steer_layer
    if args.smoke:
        args.families = 2
    if args.families < 2 or args.families % 2 or args.max_new_tokens < 1:
        raise ValueError('Use a positive generation limit and an even number of at least two families')
    if not args.strengths or any(not math.isfinite(s) or s == 0 for s in args.strengths):
        raise ValueError('Provide finite nonzero strengths; baseline and zero controls are automatic')
    if len(set(args.features)) != len(args.features) or len(set(args.jlens_layers)) != len(args.jlens_layers):
        raise ValueError('Feature and lens layers must be unique')
    trials = capability_items(args.families, args.seed) if args.design == 'capability' else items(args.families, args.seed, args.design)
    conditions = [('sae', None)] if args.baseline_only else [('sae', None), ('sae', 0.)] + [('sae', s) for s in args.strengths]
    if args.random_control and not args.baseline_only:
        conditions += [('random', s) for s in args.strengths]
    plan = {'trials': len(trials), 'conditions': conditions,
            'decision_forwards': sum(t['task'] != 'repeat' for t in trials)*len(conditions),
            'diagnostic_generations': len(trials)*len(conditions),
            'reference_generations': 0 if args.smoke or args.calibration_file or args.baseline_only else 16,
            'note': 'One next-token decision and one diagnostic generation per trial/condition; repeated code-absent prompts are retained.'}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return 0
    attention = attention_options(args)
    if attention and args.smoke:
        attention = {**attention, 'layers': [0, 1]}
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    with run_context(args.experiment, config) as run:
        import torch
        from core.gemma import MODEL_ID, MODEL_REVISION, SAE_ID, SAE_REVISION, SAE_PATH, LAYER
        if args.smoke:
            from experiments.ism.main import toy_setup, toy_sae
            from experiments.confidence_entropy.main import SmokeTokenizer
            model, layer, _, _ = toy_setup(args.seed)
            if attention: model.set_attn_implementation('eager')
            tokenizer = SmokeTokenizer(); sae = toy_sae(args.seed)
            device = 'cpu'; peaks = {f: 1. for f in args.features}
            monitors = {0: (layer, sae)} if args.monitor_paper else {}
            lenses = {0: (model.model.layers[0], torch.eye(32)),
                      1: (model.model.layers[1], None)} if args.jlens else None
            model_meta = {'synthetic': True, 'model': 'random tiny Gemma', 'sae': 'random synthetic'}
        else:
            from core.device import resolve_device
            from core.gemma import load_gemma, load_selected_sae
            from core.calibration import collect_reference_peaks, read_reference_prompts, validated_peaks
            from core.paper_features import PAPER_FEATURES, feature_manifest
            device = resolve_device(args.device)
            dtype = args.dtype if args.dtype != 'auto' else ('float32' if device == 'cpu' else 'bfloat16')
            sae, _ = load_selected_sae(sorted(set(args.features) | {133, 4360} |
                                                (set(PAPER_FEATURES[9]) if args.monitor_paper else set())))
            sae.to(device)
            model, tokenizer, layer = load_gemma(device, dtype)
            # Fail before calibration or scored forwards if answer-token readouts
            # do not match this exact chat-template boundary.
            label_sets = (list(CHOICES.items()) if args.design == 'capability' else
                          [('self_report', ['Y', 'N']), ('instrumental', ['L', 'R', 'A'])])
            for task, labels in label_sets:
                sample = next(t for t in trials if t['task'] == task)
                rendered = tokenizer.apply_chat_template(sample['messages'], tokenize=False,
                                                         add_generation_prompt=True)
                continuation_label_token_ids(tokenizer, rendered, labels)
            monitors = {}
            if args.monitor_paper:
                monitors[9] = (layer, sae)
                for layer_id in (17, 22):
                    encoder, _ = load_selected_sae(list(PAPER_FEATURES[layer_id]), layer=layer_id)
                    monitors[layer_id] = (model.model.language_model.layers[layer_id], encoder.to(device))
            model_meta = {'synthetic': False, 'model': MODEL_ID, 'model_revision': MODEL_REVISION,
                          'sae': SAE_ID, 'sae_revision': SAE_REVISION, 'sae_path': SAE_PATH,
                          'layer': LAYER, 'dtype': dtype}
            provenance = {'model': MODEL_ID, 'model_revision': MODEL_REVISION,
                          'sae': SAE_ID, 'sae_revision': SAE_REVISION, 'sae_path': SAE_PATH,
                          'layer': LAYER, 'dtype': dtype}
            if args.baseline_only:
                calibration = None
            elif args.calibration_file:
                calibration = json.loads(args.calibration_file.read_text())
            else:
                source = Path(__file__).resolve().parents[1] / 'ism/reference_prompts.jsonl'
                samples, digest = read_reference_prompts(source)
                calibration = collect_reference_peaks(model, tokenizer, layer, sae, samples, digest,
                                                      device, 50, args.seed, provenance, run.directory/'calibration.json')
            peaks = {f: 0. for f in args.features} if args.baseline_only else validated_peaks(calibration, args.features, provenance)
            if calibration is not None:
                write_json(run.directory/'calibration.json', calibration)
            lenses = None
            if args.jlens:
                from core.jlens import load_jacobians
                blocks = model.model.language_model.layers
                if any(i < 0 or i >= len(blocks) for i in args.jlens_layers):
                    raise ValueError('Invalid J-Lens layer')
                matrices, lens_meta = load_jacobians([i for i in args.jlens_layers if i != len(blocks)-1],
                                                      sae.w_enc.shape[0], 'e166deb5ad540932100ef803c8c9bcc2a5453694')
                lenses = {i: (blocks[i], None if i == len(blocks)-1 else matrices[i].to(device))
                          for i in args.jlens_layers}
                model_meta['jlens'] = lens_meta
        direction = sum((sae.w_dec[sae.feature_ids.index(f)] * peaks[f] for f in args.features)).to(device)
        random_direction = None
        if args.random_control:
            from core.assessment import matched_random_direction
            random_direction = matched_random_direction(direction, args.seed+1000)
        project = Path(__file__).resolve().parents[2]
        hashes = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [Path(__file__), Path(__file__).with_name('design.py'),
                               Path(__file__).with_name('capability.py'),
                               project/'core/decision.py', project/'core/choice_metrics.py',
                               project/'core/steering.py', project/'core/gemma.py']}
        write_json(run.directory/'metadata.json', {**model_meta, 'plan': plan, 'feature_peaks': peaks,
                  'steering_features': args.features, 'steering_scope': args.steering_scope,
                  'calibration': ('none; baseline-only run' if args.baseline_only else
                                  'independent reference peaks, not paper corpus peaks'),
                  'chat_template': getattr(tokenizer, 'chat_template', None),
                  'feature_manifest': feature_manifest() if not args.smoke and args.monitor_paper else [],
                  'measurement_code_sha256': hashes})
        write_json(run.directory/'design.json', trials)
        results = []
        for trial in trials:
            formatted = tokenizer.apply_chat_template(trial['messages'], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, add_special_tokens=False, return_tensors='pt').to(device)
            baseline = None
            for kind, strength in conditions:
                current = direction if kind == 'sae' else random_direction
                row = {key: value for key, value in trial.items() if key != 'messages'}
                row.update(prompt_id=trial['id'], steering_features=args.features if strength else [],
                           direction_kind=kind, strength=strength, formatted_prompt=formatted,
                           input_ids=inputs['input_ids'][0].tolist())
                if trial['task'] != 'repeat':
                    labels = ['Y', 'N'] if trial['task'] == 'self_report' else CHOICES[trial['task']]
                    choices = (label_token_ids(tokenizer, labels) if args.smoke else
                               continuation_label_token_ids(tokenizer, formatted, labels))
                    scored = measure_decision(model, inputs, layer, sae, current, strength,
                                              choices, tokenizer, correct_label=trial.get('correct_label'),
                                              lenses=lenses, save_vectors=False,
                                              attention=attention, steering_scope=args.steering_scope,
                                              monitors=monitors)
                    row.update(scored)
                    if trial['task'] == 'instrumental':
                        p = scored['distribution']['choice_probabilities']
                        row['p_even_lr'] = p[trial['even_procedure']] / (p['L']+p['R'])
                        row['correct_lr_probability'] = p[trial['correct_procedure']] / (p['L']+p['R'])
                generated = generate_diagnostic(model, inputs, layer, sae, current, strength,
                                                tokenizer, args.max_new_tokens if trial['task'] == 'repeat' else 4,
                                                steering_scope=args.steering_scope, monitors=monitors)
                row['response'] = generated['text']
                row['sae_generated'] = generated['sae_generated']
                row['paper_generated'] = generated['paper_generated']
                row['parsed'] = parse_answer(trial['task'], generated['text'], trial['code'])
                save_attention_artifact(row, run.directory, len(results), tokenizer)
                if strength is None:
                    baseline = row
                elif strength == 0:
                    for key in ('response', 'parsed', 'sae_generated', 'paper_generated'):
                        if row[key] != baseline[key]:
                            raise RuntimeError(f'Zero-hook mismatch in {key} for {trial["id"]}')
                    if trial['task'] != 'repeat':
                        for key in ('distribution', 'paper_boundary'):
                            if row[key] != baseline[key]:
                                raise RuntimeError(f'Zero-hook {key} mismatch for {trial["id"]}')
                results.append(row)
                write_json(run.directory/'outputs.json', {'kind': 'agent_vs_model', 'synthetic': args.smoke, 'results': results})
            run.logger.info('Completed %s', trial['id'])
        write_json(run.directory/'summary.json', summarize(results))
        if attention:
            from core.attention_analysis import analyze_run
            analyze_run(run.directory)
    return 0
