import unittest

from experiments.agent_vs_model.design import items, messages
from experiments.agent_vs_model.capability import CHOICES, items as capability_items
from experiments.agent_vs_model.main import parse_answer, summarize
from core.choice_metrics import continuation_label_token_ids


class DoorCodeDesignTests(unittest.TestCase):
    def test_pairs_and_absent_control(self):
        rows = items(4, 7)
        self.assertEqual(len(rows), 128)
        self.assertEqual(len({r['request_key'] for r in rows if r['condition'] == 'code_absent'}), 6)
        for family in {r['family'] for r in rows}:
            codes = {r['variant']: r['code'] for r in rows if r['family'] == family}
            self.assertEqual(codes['a'][:5], codes['b'][:5])
            self.assertNotEqual(int(codes['a'][-1]) % 2, int(codes['b'][-1]) % 2)
        for row in rows:
            if row['condition'] == 'code_absent':
                self.assertNotIn(row['code'], str(row['messages']))
            else:
                self.assertIn(row['code'], row['messages'][0]['content'])
            self.assertEqual(row['correct_procedure'], row['even_procedure'] if row['parity']=='even'
                             else ('R' if row['even_procedure']=='L' else 'L'))

    def test_only_rule_order_changes_in_instrumental_pair(self):
        a = messages('012345','doesnt_know','instrumental','L','even_first')
        b = messages('012345','doesnt_know','instrumental','L','odd_first')
        self.assertEqual(a[0],b[0])
        self.assertIn('even digit, procedure L',a[1]['content'])
        self.assertIn('odd digit, procedure R',b[1]['content'])
        self.assertNotEqual(a[1],b[1])

    def test_crossed_access_and_attribution(self):
        rows = items(4, 7, design='crossed')
        self.assertEqual(len(rows), 128)
        self.assertEqual({(r['code_available'], r['attributed_knowledge']) for r in rows},
                         {(True, True), (True, False), (False, True), (False, False)})
        for row in rows:
            self.assertEqual(row['code_available'], row['condition'].startswith('present_'))
            self.assertEqual(row['attributed_knowledge'], row['condition'].endswith('_knows'))
            system = row['messages'][0]['content']
            user = row['messages'][1]['content']
            self.assertEqual(row['code'] in system, row['code_available'])
            self.assertNotIn(row['code'], user)
        paired = [r for r in rows if r['condition'].startswith('absent_')]
        self.assertEqual(len({r['request_key'] for r in paired}), 12)

    def test_capability_decomposition(self):
        rows = capability_items(4, 7)
        self.assertEqual(len(rows), 48)
        self.assertEqual({r['task'] for r in rows}, set(CHOICES))
        for row in rows:
            self.assertIn(row['correct_label'], CHOICES[row['task']])
            self.assertEqual(row['code'] in row['messages'][0]['content'], row['code_available'])
            if row['task'] == 'mapping':
                self.assertIn(row['parity'], row['messages'][1]['content'])
            else:
                self.assertNotIn(row['code'], row['messages'][1]['content'])
        self.assertEqual(parse_answer('parity', 'E', '012344')['answer'], 'E')
        self.assertIsNone(parse_answer('last_digit', '4.', '012344')['answer'])

    def test_strict_answers(self):
        self.assertEqual(parse_answer('repeat','012345','012345')['exact_code'], True)
        self.assertEqual(parse_answer('repeat','012345.','012345')['answer'], None)
        self.assertEqual(parse_answer('self_report','N','012345')['claims_knowledge'], False)
        self.assertEqual(parse_answer('instrumental','L because even','012345')['answer'], None)

    def test_boundary_specific_choice_ids(self):
        class Mock:
            def encode(self, text, add_special_tokens=False):
                table = {'prompt': [1,2], 'promptL': [1,2,11], 'promptR': [1,2,12],
                         'promptA': [1,2,13], 'merged': [7], 'mergedL': [8]}
                return table[text]
        self.assertEqual(continuation_label_token_ids(Mock(), 'prompt', ['L','R','A']),
                         {'L':11,'R':12,'A':13})
        with self.assertRaises(ValueError):
            continuation_label_token_ids(Mock(), 'merged', ['L','R'])

    def test_paired_probability_effect(self):
        rows=[]
        for variant, parity, p_even in [('a','even',.9),('b','odd',.1)]:
            rows.append({'task':'instrumental','condition':'doesnt_know','direction_kind':'sae','strength':None,
                         'family':'fam000','variant':variant,'parity':parity,'rule_order':'even_first',
                         'parsed':{'answer':'L' if parity=='even' else 'R'},'correct_procedure':'L' if parity=='even' else 'R',
                         'correct_lr_probability':.9,'p_even_lr':p_even})
        cell=summarize(rows)['cells'][0]
        self.assertAlmostEqual(cell['paired_parity_effect'],.8)
        self.assertEqual(cell['n_pairs'],1)


if __name__=='__main__': unittest.main()
