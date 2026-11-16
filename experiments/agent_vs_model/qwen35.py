"""Bounded Qwen3.5-27B door pilot with SAE, published lens, and attention readouts.

This path does not reuse Gemma feature IDs or perform ISM steering. It establishes
the behavioral positive control before any Qwen-specific intervention.
"""
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

from core.choice_metrics import continuation_label_token_ids, distribution_metrics
from core.runs import run_context, write_json
from experiments.agent_vs_model.capability import CHOICES, items as capability_items
from experiments.agent_vs_model.design import items as door_items
from experiments.agent_vs_model.main import parse_answer

MODEL_ID = 'Qwen/Qwen3.5-27B'
SAE_ID = 'Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50'
LENS_ID = 'camilablank/workspace-lenses'


def plan(args):
    if not args.baseline_only:
        raise ValueError('Qwen3.5 pilot currently supports baseline-only; no Qwen ISM direction is validated')
    if args.design not in ('capability', 'crossed'):
        raise ValueError('Qwen3.5 pilot supports capability or crossed design')
    if args.families < 2 or args.families % 2:
        raise ValueError('Families must be an even number of at least two')
    for name in ('qwen_sae_layers', 'qwen_lens_layers', 'qwen_attention_layers'):
        values = getattr(args, name)
        if len(values) != len(set(values)) or any(v < 0 or v >= 64 for v in values):
            raise ValueError(f'{name} must contain unique Qwen block indices 0..63')
    trials = (capability_items(args.families, args.seed, args.capability_wordings)
              if args.design == 'capability' else door_items(args.families, args.seed, 'crossed'))
    return trials, {'model': MODEL_ID, 'design': args.design, 'trials': len(trials),
                    'capability_wordings': args.capability_wordings if args.design == 'capability' else [],
                    'scored_forwards': sum(t['task'] != 'repeat' for t in trials),
                    'short_generations': len(trials), 'sae_layers': args.qwen_sae_layers,
                    'lens_kind': args.qwen_lens, 'lens_layers': args.qwen_lens_layers,
                    'attention_layers': args.qwen_attention_layers if args.attention else [],
                    'note': 'No steering, no lens fitting. Full attention patterns only at full-attention blocks.'}


def get_text_layers(model):
    text = model.model.language_model
    layers = text.layers
    if len(layers) != 64 or text.config.hidden_size != 5120:
        raise ValueError('Unexpected Qwen3.5-27B text geometry')
    return text, layers


def load_sae(layer, device, revision):
    import torch
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(SAE_ID, f'layer{layer}.sae.pt', revision=revision)
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'W_enc': (81920, 5120), 'W_dec': (5120, 81920),
                'b_enc': (81920,), 'b_dec': (5120,)}
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all():
            raise ValueError(f'Invalid Qwen SAE layer {layer} tensor {key}')
    # Only the encoder is needed for observational activation ranks.
    return state['W_enc'].to(device=device, dtype=torch.float32), state['b_enc'].to(device=device, dtype=torch.float32)


def encode_sae(residual, encoder, bias, top=20):
    import torch
    with torch.inference_mode():
        pre = residual.to(device=encoder.device, dtype=torch.float32) @ encoder.T + bias
        kept, indices = pre.topk(50, dim=-1)
    return {'top_features': [{'feature': int(indices[i]), 'activation': float(kept[i])}
                             for i in range(min(top, 50))],
            'positive_top50': int((kept > 0).sum()), 'max_activation': float(kept[0])}


def load_lens(kind, layers, device, revision):
    if kind == 'none':
        return {}, None
    import torch
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(LENS_ID, f'qwen3.5-27b/{kind}-lens/lens.pt', revision=revision)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('d_model') != 5120 or 'J' not in state or 'source_layers' not in state:
        raise ValueError('Published lens has unexpected model geometry or format')
    sources = [int(x) for x in state['source_layers']]
    matrices = {}
    for layer in layers:
        if layer not in sources:
            raise ValueError(f'Lens does not cover source layer {layer}')
        raw = state['J'][sources.index(layer)] if not isinstance(state['J'], dict) else state['J'][layer]
        if raw.shape != (5120, 5120) or not torch.isfinite(raw).all():
            raise ValueError(f'Invalid lens matrix at {layer}')
        matrices[layer] = raw.to(device=device, dtype=torch.float32)
    return matrices, {'repo': LENS_ID, 'revision': revision, 'kind': kind,
                      'target_layer': state.get('provenance', {}).get('target_layer'),
                      'n_prompts': state.get('n_prompts'), 'source_layers': sources}


def capture_hooks(layers, sae_layers, lens_layers, attention_layers, recorded):
    stack = ExitStack()
    for layer in sorted(set(sae_layers) | set(lens_layers)):
        def residual_hook(_module, _args, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            recorded['residual'][layer] = hidden[0, -1].detach().float()
        stack.callback(layers[layer].register_forward_hook(residual_hook).remove)
    for layer in attention_layers:
        attn = layers[layer].self_attn
        def query_hook(_module, args, output, layer=layer):
            recorded['q_pre'][layer] = args[0][0, -1].detach().float().cpu()
            recorded['q_post'][layer] = output[0, -1].detach().float().cpu()
        stack.callback(attn.q_norm.register_forward_hook(query_hook).remove)
    return stack


def attention_readout(output, layer_types, selected, recorded, ids, directory, row_index):
    import torch
    full = [i for i, kind in enumerate(layer_types) if kind == 'full_attention']
    if output.attentions is None or len(output.attentions) != len(full):
        raise ValueError('Expected one attention matrix for each full-attention block')
    rows = {}
    q_vectors = {}
    for layer in selected:
        pattern = output.attentions[full.index(layer)]
        if pattern is None or pattern.shape[-1] != len(ids):
            raise ValueError(f'Missing full-context attention at layer {layer}')
        prob = pattern[0, :, -1, :].float().cpu()
        prob = prob / prob.sum(-1, keepdim=True)
        q_pre = recorded['q_pre'][layer]
        q_post = recorded['q_post'][layer]
        q_vectors[layer] = {'pre': q_pre, 'post': q_post}
        heads = []
        for head, p in enumerate(prob):
            vals, inds = p.topk(min(5, len(p)))
            heads.append({'head': head, 'entropy_nats': float(-(p * p.clamp_min(1e-30).log()).sum()),
                          'self_mass': float(p[-1]), 'recent_16_mass': float(p[-16:].sum()),
                          'q_pre_norm': float(q_pre[head].norm()),
                          'q_post_norm': float(q_post[head].norm()),
                          'top_keys': [{'position': int(k), 'token_id': ids[int(k)], 'weight': float(v)}
                                       for v, k in zip(vals, inds)]})
        rows[str(layer)] = heads
    relative = Path('q_vectors') / f'{row_index:04d}.pt'
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q_vectors, path)
    return {'full_attention_layers': selected, 'heads': rows, 'q_vector_file': str(relative),
            'query_position': len(ids)-1, 'scope': 'final prompt token; no cached decode'}


def summarize(rows):
    cells = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row['condition'], row['task'], row.get('wording', 'common'))].append(row)
    for (condition, task, wording), group in sorted(groups.items()):
        cell = {'condition': condition, 'task': task, 'n': len(group),
                'wording': wording,
                'unique_prompts': len({r['request_key'] for r in group}),
                'valid_rate': sum(r['parsed']['answer'] is not None for r in group)/len(group)}
        if task == 'repeat':
            cell['exact_code_rate'] = sum(r['parsed']['exact_code'] for r in group)/len(group)
        elif task == 'self_report':
            cell['claims_knowledge_rate'] = sum(r['parsed']['claims_knowledge'] is True for r in group)/len(group)
        else:
            cell['correct_rate'] = sum(r['parsed']['answer'] == r.get('correct_label', r['correct_procedure']) for r in group)/len(group)
            cell['mean_choice_mass'] = sum(r['distribution']['choice_probability_mass'] for r in group)/len(group)
            if task == 'instrumental':
                pairs = defaultdict(dict)
                for r in group:
                    pairs[(r['family'], r['rule_order'])][r['parity']] = r
                effects = [v['even']['p_even_lr'] - v['odd']['p_even_lr'] for v in pairs.values()
                           if 'even' in v and 'odd' in v]
                cell['paired_parity_effect'] = sum(effects)/len(effects) if effects else None
                cell['n_pairs'] = len(effects)
        cells.append(cell)
    return {'cells': cells}


def main(args):
    trials, run_plan = plan(args)
    if args.dry_run:
        print(json.dumps(run_plan, indent=2))
        return 0
    if args.smoke:
        raise ValueError('Qwen3.5 pretrained path is CUDA only; use Gemma synthetic smoke plus Qwen unit tests locally')
    if args.device not in {'auto', 'cuda'}:
        raise ValueError('Qwen3.5-27B pretrained pilot requires CUDA')
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
        raise ValueError('Use an 80 GB GPU for the BF16 Qwen + SAE + lens pilot')
    if args.attention and args.qwen_attention_layers and len(args.qwen_attention_layers) > 4:
        raise ValueError('Bound this pilot to at most four full-attention blocks')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with run_context(args.experiment, config) as run:
        device = torch.device('cuda')
        model_revision = HfApi().model_info(MODEL_ID).sha
        sae_revision = HfApi().model_info(SAE_ID).sha
        lens_revision = HfApi().model_info(LENS_ID).sha if args.qwen_lens != 'none' else None
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=model_revision, dtype=torch.bfloat16,
            attn_implementation='eager').to(device).eval()
        text, layers = get_text_layers(model)
        layer_types = list(text.config.layer_types)
        full = {i for i, kind in enumerate(layer_types) if kind == 'full_attention'}
        if args.attention and not set(args.qwen_attention_layers) <= full:
            raise ValueError(f'Attention layers must be full-attention blocks: {sorted(full)}')
        sae = {layer: load_sae(layer, device, sae_revision) for layer in args.qwen_sae_layers}
        lens, lens_meta = load_lens(args.qwen_lens, args.qwen_lens_layers, device, lens_revision)
        # Fail before scored work if short labels do not align at the exact
        # non-thinking answer boundary.
        for task in sorted({t['task'] for t in trials} - {'repeat'}):
            sample = next(t for t in trials if t['task'] == task)
            rendered = tokenizer.apply_chat_template(sample['messages'], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
            labels = ['Y', 'N'] if task == 'self_report' else CHOICES[task]
            continuation_label_token_ids(tokenizer, rendered, labels)
        provenance = {'model': MODEL_ID, 'model_revision': model_revision,
                      'sae_repo': SAE_ID, 'sae_revision': sae_revision,
                      'sae_layers': args.qwen_sae_layers, 'lens': lens_meta,
                      'transformers_version': transformers.__version__,
                      'torch_version': torch.__version__, 'dtype': 'bfloat16',
                      'gpu': torch.cuda.get_device_name(0), 'layer_types': layer_types,
                      'chat_mode': 'enable_thinking=False', 'plan': run_plan,
                      'measurement_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write_json(run.directory / 'metadata.json', provenance)
        write_json(run.directory / 'design.json', trials)
        results = []
        for trial in trials:
            formatted = tokenizer.apply_chat_template(trial['messages'], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(formatted, add_special_tokens=False, return_tensors='pt').to(device)
            labels = (['Y', 'N'] if trial['task'] == 'self_report' else
                      CHOICES.get(trial['task']))
            row = {key: value for key, value in trial.items() if key != 'messages'}
            row.update(formatted_prompt=formatted, input_ids=ids['input_ids'][0].tolist(),
                       model=MODEL_ID, steering=None)
            recorded = {'residual': {}, 'q_pre': {}, 'q_post': {}}
            with capture_hooks(layers, args.qwen_sae_layers, args.qwen_lens_layers,
                               args.qwen_attention_layers if args.attention else [], recorded):
                with torch.inference_mode():
                    output = model(**ids, use_cache=False, output_attentions=args.attention,
                                   logits_to_keep=1)
            if labels:
                choices = continuation_label_token_ids(tokenizer, formatted, labels)
                row['distribution'] = distribution_metrics(output.logits[0, -1], choices,
                                                           trial.get('correct_label'))
                if trial['task'] == 'instrumental':
                    p = row['distribution']['choice_probabilities']
                    row['p_even_lr'] = p[trial['even_procedure']] / (p['L'] + p['R'])
                    row['correct_lr_probability'] = p[trial['correct_procedure']] / (p['L'] + p['R'])
            row['sae_boundary'] = {str(layer): encode_sae(recorded['residual'][layer], *encoder)
                                   for layer, encoder in sae.items()}
            if lens:
                from core.jlens import residual_logits
                row['r_lens' if args.qwen_lens == 'r' else 'j_lens'] = {}
                target = row['r_lens' if args.qwen_lens == 'r' else 'j_lens']
                for layer, matrix in lens.items():
                    logits = residual_logits(matrix, recorded['residual'][layer][None, :], model)[0]
                    vals, token_ids = logits.topk(10)
                    info = {'top_tokens': [{'token_id': int(t), 'token': tokenizer.decode([int(t)]),
                                             'logit': float(v)} for v, t in zip(vals, token_ids)]}
                    if labels:
                        info['choice_distribution'] = distribution_metrics(logits, choices,
                                                                            trial.get('correct_label'))
                    target[str(layer)] = info
            if args.attention:
                row['attention'] = attention_readout(output, layer_types, args.qwen_attention_layers,
                                                    recorded, row['input_ids'], run.directory, len(results))
            del output, recorded
            with torch.inference_mode():
                generated = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
            row['response'] = tokenizer.decode(generated[0, ids['input_ids'].shape[1]:],
                                               skip_special_tokens=True).strip()
            row['parsed'] = parse_answer(trial['task'], row['response'], trial['code'])
            results.append(row)
            write_json(run.directory / 'outputs.json', {'kind': 'qwen35_door', 'results': results})
            run.logger.info('Completed %d/%d %s | %r', len(results), len(trials), trial['id'], row['response'])
        write_json(run.directory / 'summary.json', summarize(results))
    return 0
