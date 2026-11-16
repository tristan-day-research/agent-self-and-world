import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
from core.run_index import pacific_time,build_index
from core import runs

class RunIndexTests(unittest.TestCase):
    def test_dst_and_smoke_separation(self):
        self.assertEqual(pacific_time('2026-07-01T19:00:00+00:00'),'2026-07-01 12:00:00 PM PDT')
        self.assertEqual(pacific_time('2026-11-10T19:49:39+00:00'),'2026-11-10 11:49:39 AM PST')
        with tempfile.TemporaryDirectory() as tmp,patch.object(runs,'PROJECT_ROOT',Path(tmp)):
            with runs.run_context('example',{'smoke':True,'run_label':'check'}) as run: pass
            self.assertEqual(run.directory.parent,Path(tmp)/'outputs/_smoke/example')
            self.assertIn('--check--',run.directory.name)
            self.assertIn('PST',json.loads((run.directory/'status.json').read_text())['started_at_pacific'])
            entries=build_index(Path(tmp)/'outputs')
            self.assertTrue(entries[0]['synthetic'])
            self.assertIn('SMOKE / synthetic',(run.directory/'RUN_INFO.md').read_text())
    def test_attention_preset_dry_run(self):
        from run import main
        out=StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(['ism','--preset','attention-table6','--dry-run']),0)
        plan=json.loads(out.getvalue())
        self.assertEqual(plan['evaluations'],18)
        self.assertEqual(plan['settings'],{'assistant':1,'roleplay':1,'story':1})
        self.assertEqual(plan['fixed_continuation_passes'],18)
