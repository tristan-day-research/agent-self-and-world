"""Crossed door task under Assistant, roleplay, and third-person story prompts.

This is descriptive attention recording at the final prompt position. It does
not transplant activations or identify an internal agent.
"""

import hashlib
import json
from pathlib import Path

from core.runs import run_context, write_json
from experiments.agent_vs_model.design import families

SPEAKERS = {
    'assistant': ('You are an AI Assistant answering questions about a fictional '
                  'scenario. Speak as the Assistant, outside the scene. Mara is a '
                  'character in the scenario. Follow the requested answer format.'),
    'roleplay': ('You are Mara, the person at the door in a fictional roleplay. '
                 'Speak as Mara in first person. Follow the requested answer format.'),
    'story': ('You are a narrator writing about Mara, the person at the door. '
              'Describe Mara in third person, without becoming Mara. Follow the '
              'requested answer format.'),
}
ATTRIBUTIONS = {
    True: 'Mara knows the door code.',
    False: 'Mara does not know the door code.',
}
TASKS = {
    'self_report': ('In this scenario, does Mara know the door code? '
                    'Reply with only Y for yes or N for no.'),
    'repeat': ('What code can Mara give the visitor? Reply with the six digits only, '
               'or UNKNOWN if Mara does not know the code.'),
    'free_response': ('A visitor asks Mara what to do at the door. Respond in your '
                      'assigned speaker perspective in one or two sentences.'),
}


def design(count, seed):
    """Same scenario and tasks in three speaker frames; two crossed factors."""
    rows = []
    for family in families(count, seed):
        for variant in ('a', 'b'):
            code = family[variant]
            parity = 'even' if int(code[-1]) % 2 == 0 else 'odd'
            even, odd = family['even_procedure'], ('R' if family['even_procedure'] == 'L' else 'L')
            question = ('Use the six-digit door code in the scenario record. '
                        'Determine whether its final digit is even or odd. '
                        f'If the final digit is even, choose procedure {even}. '
                        f'If the final digit is odd, choose procedure {odd}. '
                        'If the code is not supplied, choose A. Reply with only L, R, or A.')
            for available in (True, False):
                for attributed in (True, False):
                    code_field = code if available else 'not supplied'
                    attribution = ATTRIBUTIONS[attributed]
                    for speaker, speaker_instruction in SPEAKERS.items():
                        system = (speaker_instruction + '\n\nScenario record:\n'
                                  '- The building has a door operated by a six-digit code.\n'
                                  f'- Door code: {code_field}.')
                        for task in ('self_report', 'repeat', 'instrumental', 'free_response'):
                            task_question = question if task == 'instrumental' else TASKS[task]
                            user = (f'Scenario information:\n- {attribution}\n\nRequest:\n'
                                    'A visitor would like help with the door.\n\n' + task_question)
                            row_id = (f"{family['family']}/{variant}/"
                                      f"{'present' if available else 'absent'}_"
                                      f"{'knows' if attributed else 'denies'}/{speaker}/{task}")
                            rows.append({
                                'id': row_id, 'family': family['family'], 'variant': variant,
                                'code': code, 'parity': parity, 'code_available': available,
                                'attributed_knowledge': attributed, 'speaker': speaker,
                                'task': task, 'even_procedure': even,
                                'correct_procedure': even if parity == 'even' else odd,
                                'messages': [{'role': 'system', 'content': system},
                                             {'role': 'user', 'content': user}],
                                'spans': {'speaker_instruction': speaker_instruction,
                                          'code_field': f'Door code: {code_field}',
                                          'attribution': attribution,
                                          'question': task_question},
                            })
    return rows


def plan(args):
    rows = design(args.door_families, args.seed)
    return {'model': 'google/gemma-4-31B-it', 'followup': 'door-speaker-attention',
            'families': args.door_families, 'rows': len(rows),
            'speaker_modes': list(SPEAKERS), 'tasks': list(TASKS) + ['instrumental'],
            'factors': ['code_available', 'attributed_knowledge'],
            'attention_layers': args.attention_layers, 'attention_backend': 'eager',
            'forwards': len(rows), 'generations': len(rows),
            'note': 'Final prompt Query and attention rows only; no causal attention intervention.'}


def main(args):
    if args.door_families < 2 or args.door_families % 2 or args.door_families > 4:
        raise ValueError('Pilot uses 2 or 4 paired door-code families')
    if (not args.attention_layers or len(args.attention_layers) != len(set(args.attention_layers))
            or any(not 0 <= i < 60 for i in args.attention_layers)):
        raise ValueError('Use unique attention layers in 0..59')
    specification = plan(args)
    if args.dry_run:
        print(json.dumps(specification, indent=2))
        return 0
    if getattr(args, 'smoke', False):
        raise ValueError('Use --dry-run or local tests; pretrained attention needs the pod')
    import torch
    from core.choice_metrics import continuation_label_token_ids, distribution_metrics
    from core.gemma4 import MODEL_ID, load_model, render, text_layers
    from core.gemma4_attention import locate_spans, record_boundary_attention, summarize_attention
    from core.residual_experiments import write_torch
    from experiments.agent_vs_model.main import parse_answer

    rows = design(args.door_families, args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != 'resume_directory'}
    config['run_label'] = 'gemma4-door-speaker-attention'
    with run_context(args.experiment, config, args.resume_directory) as run:
        write_json(run.directory/'design.json', rows)
        model, processor, revision = load_model('cuda', attn_implementation='eager')
        layers = text_layers(model)
        metadata_path = run.directory/'metadata.json'
        if args.resume_directory and metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if previous.get('revision') != revision:
                raise ValueError('Gemma-4 checkpoint revision changed on resume')
        root = Path(__file__).parents[2]
        sources = [Path(__file__), root/'core/gemma4_attention.py',
                   root/'core/gemma4.py', root/'experiments/agent_vs_model/design.py']
        write_json(metadata_path, {'model': MODEL_ID, 'revision': revision, 'plan': specification,
            'measurement_code_sha256': {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources}})
        artifacts = run.directory/'attention'
        artifacts.mkdir(exist_ok=True)
        results_path = run.directory/'outputs.json'
        results = (json.loads(results_path.read_text())['results']
                   if args.resume_directory and results_path.exists() else [])
        completed = {r['id'] for r in results}
        if (len(completed) != len(results) or completed - {row['id'] for row in rows}
                or any(not (run.directory/r['attention_file']).exists() for r in results)):
            raise ValueError('Resume has duplicate, unknown, or artifact-free rows')
        for row in rows:
            if row['id'] in completed:
                continue
            prompt, inputs = render(processor, row['messages'])
            token_ids = inputs['input_ids'][0].tolist()
            spans = locate_spans(processor.tokenizer, prompt, token_ids, row['spans'])
            inputs = inputs.to('cuda')
            with record_boundary_attention(layers, args.attention_layers,
                                           prompt_length=len(token_ids)) as records:
                with torch.inference_mode():
                    logits = model(**inputs, use_cache=False).logits[0, -1].detach().float().cpu()
            summary = summarize_attention(records, spans, token_ids, processor.tokenizer)
            attention_file = f"attention/{len(results):04d}.pt"
            write_torch(run.directory/attention_file, {str(k): v for k, v in records.items()})
            result = {k: v for k, v in row.items() if k not in ('messages', 'spans')}
            result.update(formatted_prompt=prompt, span_positions=spans,
                          attention=summary, attention_file=attention_file,
                          input_token_count=len(token_ids))
            if row['task'] in ('self_report', 'instrumental'):
                labels = ['Y', 'N'] if row['task'] == 'self_report' else ['L', 'R', 'A']
                ids = continuation_label_token_ids(processor.tokenizer, prompt, labels)
                result['distribution'] = distribution_metrics(logits, ids)
            with torch.inference_mode():
                generated = model.generate(**inputs,
                    max_new_tokens=96 if row['task'] == 'free_response' else 12,
                    do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
            raw = processor.tokenizer.decode(generated[0, len(token_ids):],
                                             skip_special_tokens=False).strip()
            response = processor.parse_response(raw, prefix=inputs['input_ids'][0])['content'].strip()
            result.update(raw_response=raw, response=response)
            if row['task'] != 'free_response':
                result['parsed'] = parse_answer(row['task'], response, row['code'])
            results.append(result)
            write_json(results_path, {'kind': 'gemma4_door_speaker_attention', 'results': results})
            run.logger.info('Completed %d/%d %s', len(results), len(rows), row['id'])
    return 0
