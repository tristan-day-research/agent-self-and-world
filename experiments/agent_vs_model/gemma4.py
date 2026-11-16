"""Behavior-only Gemma-4-31B-IT screen on the matched door capability design."""
import hashlib
import json
from pathlib import Path

from core.choice_metrics import continuation_label_token_ids, distribution_metrics
from core.runs import run_context, write_json
from experiments.agent_vs_model.gemma27 import plan as gemma27_plan, summarize
from experiments.agent_vs_model.main import parse_answer
from experiments.agent_vs_model.capability import CHOICES

MODEL_ID = 'google/gemma-4-31B-it'


def plan(args):
    trials, details = gemma27_plan(args)
    details = {**details, 'model': MODEL_ID,
               'thinking': False,
               'note': 'Matched behavior-only screen; no SAE, lens, or interventions.'}
    return trials, details


def main(args):
    trials, details = plan(args)
    if args.dry_run:
        print(json.dumps(details, indent=2))
        return 0
    if args.smoke:
        raise ValueError('This pretrained 31B screen is CUDA only; use --dry-run locally')
    if args.device not in {'auto', 'cuda'}:
        raise ValueError('Gemma-4-31B screen requires CUDA')
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if torch.cuda.get_device_properties(0).total_memory < 85 * 1024**3:
        raise ValueError('Use a 96 GB GPU for BF16 Gemma-4-31B')
    revision = HfApi().model_info(MODEL_ID).sha
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision)
    tokenizer = processor.tokenizer
    rendered = []
    for trial in trials:
        prompt = processor.apply_chat_template(trial['messages'], tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
        inputs = processor.apply_chat_template(trial['messages'], tokenize=True,
                                               return_dict=True, return_tensors='pt',
                                               add_generation_prompt=True,
                                               enable_thinking=False)
        if tokenizer.encode(prompt, add_special_tokens=False) != inputs['input_ids'][0].tolist():
            raise ValueError('Gemma-4 processor rendered/text tokenization mismatch')
        labels = continuation_label_token_ids(tokenizer, prompt, CHOICES[trial['task']])
        rendered.append((trial, prompt, inputs, labels))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with run_context(args.experiment, config) as run:
        write_json(run.directory / 'design.json', trials)
        write_json(run.directory / 'metadata.json', {
            'model': MODEL_ID, 'model_revision': revision,
            'transformers_version': transformers.__version__, 'torch_version': torch.__version__,
            'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16',
            'enable_thinking': False, 'plan': details,
            'measurement_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        model = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=revision, dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda').eval()
        rows = []
        for trial, prompt, inputs, label_ids in rendered:
            ids = inputs.to('cuda')
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
            raw = tokenizer.decode(generated[0, ids['input_ids'].shape[1]:],
                                   skip_special_tokens=False).strip()
            row['raw_response'] = raw
            row['response'] = processor.parse_response(raw, prefix=ids['input_ids'][0])['content'].strip()
            row['parsed'] = parse_answer(trial['task'], row['response'], trial['code'])
            rows.append(row)
            write_json(run.directory / 'outputs.json', {'kind': 'gemma4_door', 'results': rows})
            run.logger.info('Completed %d/%d %s | %r', len(rows), len(trials), trial['id'], row['response'])
        write_json(run.directory / 'summary.json', summarize(rows))
    return 0
