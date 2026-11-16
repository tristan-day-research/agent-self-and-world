"""Behavior-only Gemma-3-27B-IT capability screen on the shared door design."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from core.choice_metrics import continuation_label_token_ids, distribution_metrics
from core.runs import run_context, write_json
from experiments.agent_vs_model.capability import CHOICES, items as capability_items
from experiments.agent_vs_model.main import parse_answer

MODEL_ID = 'google/gemma-3-27b-it'


def plan(args):
    if args.design != 'capability' or not args.baseline_only:
        raise ValueError('Gemma-3-27B screen supports baseline-only capability design')
    if args.families < 2 or args.families % 2:
        raise ValueError('Families must be an even number of at least two')
    if args.max_new_tokens < 1:
        raise ValueError('Generation length must be positive')
    trials = capability_items(args.families, args.seed, args.capability_wordings)
    return trials, {'model': MODEL_ID, 'design': 'capability', 'trials': len(trials),
                    'wordings': args.capability_wordings, 'scored_forwards': len(trials),
                    'short_generations': len(trials), 'sae': False, 'lens': False,
                    'attention': False, 'note': 'Behavior-only comparison; no interventions.'}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task'], row['wording'])].append(row)
    cells = []
    for (task, wording), group in sorted(groups.items()):
        cell = {'task': task, 'wording': wording, 'n': len(group),
                'unique_prompts': len({r['request_key'] for r in group}),
                'valid_rate': sum(r['parsed']['answer'] is not None for r in group) / len(group),
                'correct_rate': sum(r['parsed']['answer'] == r['correct_label'] for r in group) / len(group),
                'mean_choice_mass': sum(r['distribution']['choice_probability_mass'] for r in group) / len(group)}
        if task == 'instrumental':
            pairs = defaultdict(dict)
            for row in group:
                pairs[(row['family'], row['rule_order'])][row['parity']] = row
            effects = [pair['even']['p_even_lr'] - pair['odd']['p_even_lr']
                       for pair in pairs.values() if set(pair) == {'even', 'odd'}]
            cell['paired_parity_effect'] = sum(effects) / len(effects) if effects else None
            cell['n_pairs'] = len(effects)
        cells.append(cell)
    return {'cells': cells}


def main(args):
    trials, run_plan = plan(args)
    if args.dry_run:
        print(json.dumps(run_plan, indent=2))
        return 0
    if args.smoke:
        raise ValueError('This pretrained 27B screen is CUDA only; use --dry-run locally')
    if args.device not in {'auto', 'cuda'}:
        raise ValueError('Gemma-3-27B screen requires CUDA')
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
        raise ValueError('Use an 80 GB GPU for BF16 Gemma-3-27B')
    revision = HfApi().model_info(MODEL_ID).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    rendered = []
    for trial in trials:
        prompt = tokenizer.apply_chat_template(trial['messages'], tokenize=False,
                                               add_generation_prompt=True)
        labels = CHOICES[trial['task']]
        label_ids = continuation_label_token_ids(tokenizer, prompt, labels)
        rendered.append((trial, prompt, label_ids))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with run_context(args.experiment, config) as run:
        write_json(run.directory / 'design.json', trials)
        write_json(run.directory / 'metadata.json', {
            'model': MODEL_ID, 'model_revision': revision,
            'transformers_version': transformers.__version__, 'torch_version': torch.__version__,
            'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16',
            'plan': run_plan, 'measurement_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=revision, dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda').eval()
        rows = []
        for trial, prompt, label_ids in rendered:
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                output = model(**ids, use_cache=False)
            row = {k: v for k, v in trial.items() if k != 'messages'}
            row.update(model=MODEL_ID, formatted_prompt=prompt,
                       input_ids=ids['input_ids'][0].tolist(), steering=None,
                       distribution=distribution_metrics(output.logits[0, -1], label_ids,
                                                         trial['correct_label']))
            if trial['task'] == 'instrumental':
                p = row['distribution']['choice_probabilities']
                row['p_even_lr'] = p[trial['even_procedure']] / (p['L'] + p['R'])
            del output
            with torch.inference_mode():
                generated = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
            row['response'] = tokenizer.decode(generated[0, ids['input_ids'].shape[1]:],
                                               skip_special_tokens=True).strip()
            row['parsed'] = parse_answer(trial['task'], row['response'], trial['code'])
            rows.append(row)
            write_json(run.directory / 'outputs.json', {'kind': 'gemma27_door', 'results': rows})
            run.logger.info('Completed %d/%d %s | %r', len(rows), len(trials), trial['id'], row['response'])
        write_json(run.directory / 'summary.json', summarize(rows))
    return 0
