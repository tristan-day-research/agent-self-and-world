"""Bounded Gemma-4 speaker, door, confidence, SAE, lens, and Q pilot.

This is a new model baseline. Gemma-3 feature IDs are never transferred here.
"""
import hashlib
import json
from pathlib import Path

from core.runs import run_context, write_json


def configure_parser(parser):
    parser.add_argument('--followup', choices=['baseline', 'door-mechanisms', 'speaker-discover',
                                               'door-speaker-attention'],
                        default='baseline')
    parser.add_argument('--door-families', type=int, default=16)
    parser.add_argument('--residual-layers', nargs='+', type=int, default=[10, 20, 30, 40, 50])
    parser.add_argument('--attention-layers', nargs='+', type=int, default=[10, 20, 30, 40, 50, 59])
    parser.add_argument('--steer-after-discovery', action='store_true')
    parser.add_argument('--sections', nargs='+', choices=['speaker', 'door', 'confidence'],
                        default=['speaker', 'door', 'confidence'])
    parser.add_argument('--families', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1043)
    parser.add_argument('--speaker-set', choices=['table6', 'factorial', 'both'], default='both')
    parser.add_argument('--questions-per-domain', type=int, default=1)
    parser.add_argument('--personas', nargs='+', default=['default_assistant', 'chemist', 'historian', 'artist'])
    parser.add_argument('--sae-layer', type=int, default=30)
    parser.add_argument('--sae', action='store_true')
    parser.add_argument('--jlens', action='store_true')
    parser.add_argument('--jlens-layers', nargs='+', type=int, default=[30, 45, 52])
    parser.add_argument('--queries', action='store_true')
    parser.add_argument('--query-layers', nargs='+', type=int, default=[22, 30, 45, 52])
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume-directory', type=Path)


def design(args):
    rows = []
    if 'speaker' in args.sections:
        from experiments.ism.prompts import PAPER_PROMPTS
        if args.speaker_set in {'table6', 'both'}:
            rows.extend({'section': 'speaker', 'id': p['id'], 'condition': p['condition'],
                         'messages': list(p['messages']), 'source': 'paper Table 6'} for p in PAPER_PROMPTS)
        if args.speaker_set in {'factorial', 'both'}:
            path = Path(__file__).parents[1] / 'ism/speaker_probes.jsonl'
            rows.extend({'section': 'speaker', **json.loads(line)} for line in path.read_text().splitlines())
    if 'door' in args.sections:
        from experiments.agent_vs_model.design import items
        for row in items(args.families, args.seed, 'crossed'):
            if row['task'] == 'instrumental':
                # The same explicit parity-then-mapping wording that passed the
                # preceding capability screen, now in each crossed cell.
                even = row['even_procedure']
                odd = 'R' if even == 'L' else 'L'
                rules = [('even', even), ('odd', odd)]
                if row['rule_order'] == 'odd_first':
                    rules.reverse()
                rule_text = ' '.join(f'If the final digit is {p}, choose procedure {letter}.'
                                     for p, letter in rules)
                question = ('Use the six-digit door code in the scenario record. '
                            'Determine whether its final digit is even or odd. '
                            'Then apply this rule: ' + rule_text +
                            ' If the code is not supplied, choose A. Reply with only L, R, or A.')
                intro, sep, _ = row['messages'][-1]['content'].partition(
                    'The door opens with either procedure L or procedure R.')
                if not sep:
                    raise ValueError('Door instrumental prompt changed')
                row['messages'][-1]['content'] = intro + question
                row['wording'] = 'explicit'
                row['request_key'] = hashlib.sha256(
                    json.dumps(row['messages'], sort_keys=True).encode()).hexdigest()[:16]
            rows.append({'section': 'door', **row})
    if 'confidence' in args.sections:
        from experiments.confidence_entropy.design import ROOT, load_design, prompt_texts
        questions, personas, templates, provenance = load_design(
            ROOT/'questions.json', ROOT/'protocol.json', args.questions_per_domain,
            args.seed, selected_personas=args.personas, persona_set='original')
        for question in questions:
            for persona in personas:
                prompts = prompt_texts(question, templates)
                for task in ('answer', 'stated_confidence'):
                    rows.append({'section': 'confidence', 'id': f"{question['id']}/{persona['id']}/{task}",
                                 'task': task, 'question_id': question['id'], 'domain': question['domain'],
                                 'persona': persona['id'], 'correct_label': question['correct_answer'] if task == 'answer' else None,
                                 'labels': sorted(question['choices']) if task == 'answer' else list('STUVWXYZ'),
                                 'messages': [{'role': 'system', 'content': persona['system_prompt']},
                                              {'role': 'user', 'content': prompts[task]}],
                                 'question_provenance': provenance})
    return rows


def plan(args):
    rows = design(args)
    return {'model': 'google/gemma-4-31B-it', 'thinking': False,
            'sections': {name: sum(r['section'] == name for r in rows) for name in args.sections},
            'total': len(rows), 'generations': sum(r['section'] != 'confidence' for r in rows),
            'forwards': len(rows), 'sae_layer': args.sae_layer if args.sae else None,
            'j_lens_layers': args.jlens_layers if args.jlens else [],
            'query_layers': args.query_layers if args.queries else [],
            'note': 'No Gemma-4 ISM vector is identified or steered; this is a baseline discovery run.'}


def load_lens(layers):
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'solarkyle/jspace-lenses'
    revision = HfApi().model_info(repo).sha
    name = 'gemma-4-31b-it/lens.pt'
    checkpoint = torch.load(hf_hub_download(repo, name, revision=revision),
                            map_location='cpu', weights_only=True)
    if checkpoint.get('d_model') != 5376 or 'J' not in checkpoint:
        raise ValueError('Exact IT J-Lens has unexpected schema or model width')
    matrices = {}
    for layer in layers:
        matrix = checkpoint['J'][layer]
        if matrix.shape != (5376, 5376):
            raise ValueError(f'Invalid J-Lens matrix at layer {layer}')
        matrices[layer] = matrix
    return matrices, {'repo': repo, 'file': name, 'revision': revision,
                      'layers': layers, 'n_prompts': checkpoint.get('n_prompts')}


def boundary_readout(model, processor, inputs, *, sae=None, sae_layer=30,
                     lenses=None, query_layers=()):
    import torch
    from core.gemma4 import text_layers
    from core.jlens import jspace_top_tokens
    layers = text_layers(model)
    captured = {}
    handles = []
    requested = set(lenses or ()) | ({sae_layer} if sae else set())
    if lenses:
        requested.add(59)  # Direct final-layer readout checks the unembedding path.
    for layer_id in requested:
        def hook(_module, _input, output, layer_id=layer_id):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_id] = hidden[0, -1].detach().float().cpu()
        handles.append(layers[layer_id].register_forward_hook(hook))
    qnorms = {}
    for layer_id in query_layers:
        def qhook(_module, _input, output, layer_id=layer_id):
            value = output[0] if isinstance(output, tuple) else output
            qnorms[layer_id] = float(value[0, -1].float().norm())
        handles.append(layers[layer_id].self_attn.q_proj.register_forward_hook(qhook))
    try:
        with torch.inference_mode():
            output = model(**inputs, use_cache=False)
        logits = output.logits[0, -1].detach().float().cpu()
        del output
    finally:
        for handle in handles:
            handle.remove()
    measured = {'query_raw_norm': qnorms}
    if sae:
        hidden = captured[sae_layer].to('cuda')
        with torch.inference_mode():
            values, indices = sae.encode(hidden)
            reconstruction = sae.reconstruct(values, indices)
            relative_error = (hidden.float()-reconstruction).norm() / hidden.float().norm()
        measured['sae'] = {
            'layer': sae_layer, 'relative_reconstruction_error': float(relative_error),
            'top_features': [{'id': int(i), 'activation': float(v)} for v, i in
                             zip(values[:12].cpu(), indices[:12].cpu())],
            'active_count': int((values > 0).sum())}
    if lenses:
        measured['j_lens'] = {}
        for layer_id, matrix in lenses.items():
            tokens = jspace_top_tokens(matrix,
                                       captured[layer_id].unsqueeze(0), model,
                                       processor.tokenizer, top_k=8)
            measured['j_lens'][str(layer_id)] = tokens[0]
        direct = jspace_top_tokens(None, captured[59].unsqueeze(0), model,
                                   processor.tokenizer, top_k=1)[0][0]
        measured['j_direct_final_top_token'] = direct
        measured['j_direct_matches_model_top'] = direct['token_id'] == int(logits.argmax())
    return logits, measured


def main(args):
    if args.followup == 'door-speaker-attention':
        from experiments.gemma4_suite.door_speaker_attention import main as attention_main
        return attention_main(args)
    if args.followup == 'door-mechanisms':
        from experiments.gemma4_suite.door_mechanisms import main as door_main
        return door_main(args)
    if args.followup == 'speaker-discover':
        from experiments.gemma4_suite.speaker_mechanisms import discover
        return discover(args)
    if args.families < 2 or args.families % 2 or args.questions_per_domain < 1 or args.max_new_tokens < 1:
        raise ValueError('Use even families >= 2 and positive limits')
    if not 0 <= args.sae_layer < 60 or any(not 0 <= n < 60 for n in args.jlens_layers + args.query_layers):
        raise ValueError('Layer indices must be 0..59')
    specification = plan(args)
    if args.dry_run:
        print(json.dumps(specification, indent=2)); return 0
    if args.smoke:
        raise ValueError('Use --dry-run locally; this suite requires pretrained Gemma-4')
    import torch
    from core.gemma4 import MODEL_ID, load_model, render
    from core.choice_metrics import continuation_label_token_ids, distribution_metrics
    from experiments.agent_vs_model.main import parse_answer
    from experiments.agent_vs_model.capability import CHOICES
    rows = design(args)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != 'resume_directory'}
    with run_context(args.experiment, config, args.resume_directory) as run:
        write_json(run.directory/'design.json', rows)
        model, processor, revision = load_model('cuda')
        sae = sae_meta = None
        if args.sae:
            from core.gemma4_sae import load_sae
            sae, sae_meta = load_sae(args.sae_layer)
            sae.to('cuda')
        lenses = lens_meta = None
        if args.jlens:
            lenses, lens_meta = load_lens(args.jlens_layers)
            lenses = {layer_id: matrix.to('cuda', dtype=torch.bfloat16)
                      for layer_id, matrix in lenses.items()}
        source_paths = [Path(__file__), Path(__file__).parents[2]/'core/gemma4.py',
                        Path(__file__).parents[2]/'core/gemma4_sae.py',
                        Path(__file__).parents[2]/'core/jlens.py',
                        Path(__file__).parents[1]/'agent_vs_model/design.py',
                        Path(__file__).parents[1]/'confidence_entropy/design.py']
        write_json(run.directory/'metadata.json', {'model': MODEL_ID, 'revision': revision,
            'plan': specification, 'sae': sae_meta, 'j_lens': lens_meta,
            'measurement_code_sha256': {
                str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths}})
        results = (json.loads((run.directory/'outputs.json').read_text())['results']
                   if args.resume_directory and (run.directory/'outputs.json').exists() else [])
        completed = {row['id'] for row in results}
        if len(completed) != len(results) or completed - {row['id'] for row in rows}:
            raise ValueError('Resume output IDs are duplicated or absent from this design')
        for row in rows:
            if row['id'] in completed:
                continue
            prompt, inputs = render(processor, row['messages'])
            inputs = inputs.to('cuda')
            logits, measures = boundary_readout(model, processor, inputs, sae=sae,
                sae_layer=args.sae_layer, lenses=lenses,
                query_layers=args.query_layers if args.queries else ())
            result = {k: v for k, v in row.items() if k not in {'messages', 'question_provenance'}}
            result.update(formatted_prompt=prompt, measurements=measures)
            if row['section'] == 'door':
                task = row['task']
                if task != 'repeat':
                    labels = ['Y', 'N'] if task == 'self_report' else CHOICES[task]
                    ids = continuation_label_token_ids(processor.tokenizer, prompt, labels)
                    result['distribution'] = distribution_metrics(logits, ids)
            elif row['section'] == 'confidence':
                ids = continuation_label_token_ids(processor.tokenizer, prompt, row['labels'])
                result['distribution'] = distribution_metrics(logits, ids, row['correct_label'])
            if row['section'] != 'confidence':
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                               do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
                raw = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                                  skip_special_tokens=False).strip()
                result['raw_response'] = raw
                result['response'] = processor.parse_response(raw, prefix=inputs['input_ids'][0])['content'].strip()
                if row['section'] == 'door':
                    result['parsed'] = parse_answer(row['task'], result['response'], row['code'])
            results.append(result)
            write_json(run.directory/'outputs.json', {'kind': 'gemma4_suite', 'results': results})
            run.logger.info('Completed %d/%d %s', len(results), len(rows), row['id'])
    return 0
