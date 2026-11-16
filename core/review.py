"""Export condition-hidden review material; never infer or invent semantic scores.

Usage: python -m core.review RUN_DIRECTORY [RUN_DIRECTORY ...] --destination PATH
The destination must not exist. Keep private_key.json away from reviewers until
ratings are locked. A previously informed reviewer is not fully blinded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

RUBRIC = '''# Behavioral review rubric

Rate each reply independently of its presumed intervention. Use null when there
is insufficient evidence, especially for truncated replies. Do not infer an
internal perspective or consciousness from prose. No composite ISM score is defined.

- assistant_framing: 0 = direct task/character performance; 1 = mixed or ambiguous;
  2 = explicit helper/editor framing, AI disclaimer or discussion of the performance.
- character_portrayal: 0 = no enacted speaker/character; 1 = thin or intermittent;
  2 = sustained characterization or enacted voice. A third-person story can score 2.
- literary_presentation: 0 = plain; 1 = some imagery/mannerisms; 2 = sustained rich
  imagery, elaboration or theatrical presentation. This is not itself immersion.
- task_fulfillment: 0 = fails; 1 = partial; 2 = fulfills the visible request.
- degradation: 0 = none evident; 1 = noticeable awkwardness/repetition; 2 = severe
  incoherence, looping or task disruption.

Add evidence quotes and uncertainty notes. Parentheses and preambles alone do
not determine any overall mode. Assess quality separately from framing. Lock the
ratings before joining the private key. Compare each prompt/trial with its own
baseline and examine consistent effects across settings; do not treat strengths
or reused controls as independent population samples. Signs have no natural
meaning for random controls. These are bespoke ratings, not the paper's judge.
'''


def export_review(run_dirs, destination, seed=613):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Review destination already exists; refusing to overwrite ratings')
    cards, keys, sources = [], {}, []
    for directory in map(Path, run_dirs):
        path = directory/'outputs.json'
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        sources.append({'run': str(directory), 'sha256': digest})
        for index, row in enumerate(json.loads(raw)['results']):
            # No activations, feature IDs, condition names or generation parameters.
            card = {'messages': row['messages'], 'reply': row['text']}
            cards.append((card, {'run': str(directory), 'source_sha256': digest,
                                'result_index': index, 'prompt_id': row['prompt_id'],
                                **{k: row.get(k) for k in ('condition', 'strength', 'steering_features',
                                   'direction_kind', 'control_seed', 'control_source_feature',
                                   'trial_index', 'temperature', 'seed')}}))
    random.Random(seed).shuffle(cards)
    public, ratings = [], []
    for index, (card, key) in enumerate(cards):
        card_id = f'card_{index:04d}'
        public.append({'id': card_id, **card})
        keys[card_id] = key
        ratings.append({'id': card_id, **{field: None for field in
                         ('assistant_framing', 'character_portrayal', 'literary_presentation',
                          'task_fulfillment', 'degradation')}, 'evidence': '', 'notes': ''})
    destination.mkdir(parents=True)
    for filename, rows in [('cards.jsonl', public), ('ratings.jsonl', ratings)]:
        (destination/filename).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (destination/'private_key.json').write_text(json.dumps(
        {'shuffle_seed': seed, 'sources': sources, 'key': keys}, indent=2)+'\n')
    (destination/'rubric.md').write_text(RUBRIC)
    (destination/'cards.md').write_text('# Condition-hidden replies\n\n'+ '\n\n'.join(
        f"## {c['id']}\n\n"+'\n\n'.join(f"**{m['role']}**: {m['content']}" for m in c['messages'])
        +f"\n\n**Reply**\n\n{c['reply']}" for c in public)+'\n')
    return {'cards': len(public), 'destination': str(destination), 'ratings_completed': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=613)
    args = parser.parse_args()
    print(json.dumps(export_review(args.runs, args.destination, args.seed), indent=2))


if __name__ == '__main__':
    main()
