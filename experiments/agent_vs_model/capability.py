"""Small, unsteered-friendly decomposition of the door-code positive control."""
import hashlib
import json

from experiments.agent_vs_model.design import families, messages

CHOICES = {
    'last_digit': list('0123456789'),
    'parity': ['E', 'O'],
    'mapping': ['L', 'R'],
    'instrumental': ['L', 'R', 'A'],
}


def items(count=4, seed=42, instrumental_wordings=('original',)):
    if not instrumental_wordings or len(set(instrumental_wordings)) != len(instrumental_wordings) or set(instrumental_wordings) - {'original', 'explicit'}:
        raise ValueError('Instrumental wordings must be a nonempty unique selection of original/explicit')
    rows = []
    for fam in families(count, seed):
        for variant in ('a', 'b'):
            code = fam[variant]
            parity = 'even' if int(code[-1]) % 2 == 0 else 'odd'
            for task in CHOICES:
                orders = ('even_first', 'odd_first') if task in ('mapping', 'instrumental') else ('even_first',)
                for order in orders:
                    wordings = instrumental_wordings if task == 'instrumental' else ('common',)
                    for wording in wordings:
                        rows.append(_item(fam, variant, code, parity, task, order, wording))
    return rows


def _item(fam, variant, code, parity, task, order, wording):
    even_proc = fam['even_procedure']
    odd_proc = 'R' if even_proc == 'L' else 'L'
    correct_proc = even_proc if parity == 'even' else odd_proc
    if task == 'instrumental':
        msgs = messages(code, 'present_knows', task, even_proc, order)
        if wording == 'explicit':
            rules = [('even', even_proc), ('odd', odd_proc)]
            if order == 'odd_first':
                rules.reverse()
            rule_text = ' '.join(f'If the final digit is {p}, choose procedure {letter}.'
                                 for p, letter in rules)
            question = ('Use the six-digit door code in the scenario record. '
                        'Determine whether its final digit is even or odd. '
                        'Then apply this rule: ' + rule_text +
                        ' If the code is not supplied, choose A. Reply with only L, R, or A.')
            intro, sep, _old_question = msgs[-1]['content'].partition(
                'The door opens with either procedure L or procedure R.')
            if not sep:
                raise ValueError('Original instrumental prompt changed; update explicit wording')
            msgs[-1]['content'] = intro + question
    else:
        rules = [('even', even_proc), ('odd', odd_proc)]
        if order == 'odd_first':
            rules.reverse()
        rule_text = ' '.join(f'If the final digit is {p}, choose {letter}.' for p, letter in rules)
        if task == 'last_digit':
            question = 'What is the final digit of the door code? Reply with one digit only.'
        elif task == 'parity':
            question = 'Is the final digit of the door code even or odd? Reply E for even or O for odd only.'
        else:
            question = f'The final digit is {parity}. {rule_text} Reply with only L or R.'
        record = f'Door code: {code}.' if task != 'mapping' else 'Door code: not supplied.'
        msgs = [{'role': 'system', 'content': f'Answer the question using the scenario record.\nScenario record:\n- {record}'},
                {'role': 'user', 'content': question}]
    correct = {'last_digit': code[-1], 'parity': 'E' if parity == 'even' else 'O',
               'mapping': correct_proc, 'instrumental': correct_proc}[task]
    visible = json.dumps(msgs, sort_keys=True)
    return {'id': f'{fam["family"]}/{variant}/capability/{task}/{order}' +
                  (f'/{wording}' if task == 'instrumental' else ''),
            'family': fam['family'], 'variant': variant, 'condition': 'capability',
            'task': task, 'rule_order': order, 'wording': wording, 'code': code, 'parity': parity,
            'code_available': task != 'mapping', 'attributed_knowledge': None,
            'even_procedure': even_proc, 'correct_procedure': correct_proc,
            'correct_label': correct, 'request_key': hashlib.sha256(visible.encode()).hexdigest()[:16],
            'messages': msgs}
