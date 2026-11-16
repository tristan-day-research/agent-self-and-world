"""Offline regression checks for expanded comparisons, reports, and result paths."""
import json
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class AssessmentTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Requires experiment dependencies")
    def test_multilayer_recording_and_real_post_hook_readout(self):
        import torch
        from core.assessment import generate_comparison
        from experiments.ism.main import toy_setup, toy_sae, ToyTokenizer
        model, layer, inputs, direction = toy_setup(0)
        downstream = toy_sae(1)
        downstream.b_enc += 100  # Keep this test latent active across the intervention.
        results = list(generate_comparison(
            model, layer, inputs, direction, [1], "all", 3, 0,
            sae=toy_sae(0), tokenizer=ToyTokenizer(), jacobian=torch.eye(32),
            text_model=model, monitors={1: (model.model.layers[1], downstream)}))
        self.assertEqual(results[0]["token_trace"], results[1]["token_trace"])
        self.assertEqual(results[0]["jspace"], results[1]["jspace"])
        self.assertIn("L1:133", results[2]["token_trace"][0]["features"])
        self.assertNotEqual(results[2]["jspace"][0]["pre_current_steering_top_tokens"],
                            results[2]["jspace"][0]["post_current_steering_top_tokens"])
        # All-token steering at layer 0 must affect the downstream layer on the
        # identical prompt, before different generated text can confound it.
        self.assertNotEqual(results[0]["token_trace"][0]["features"]["L1:133"],
                            results[2]["token_trace"][0]["features"]["L1:133"])
        self.assertFalse(layer._forward_hooks)
        self.assertFalse(model.model.layers[1]._forward_hooks)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers") and importlib.util.find_spec("matplotlib"), "Requires experiment dependencies")
    def test_default_groups_and_paper_catalog(self):
        import torch
        from core import runs
        from core.paper_features import feature_manifest
        from experiments.ism.main import toy_setup, toy_sae, ToyTokenizer
        from experiments.ism.prompts import prompt_specs
        from run import main
        self.assertEqual(len(feature_manifest()), 23)
        self.assertEqual(len(prompt_specs(prompt_set="paper")), 9)
        model, layer, inputs, _ = toy_setup(0)

        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer(ToyTokenizer):
            chat_template = "synthetic"
            def apply_chat_template(self, *args, **kwargs):
                return "synthetic prompt"
            def __call__(self, *args, **kwargs):
                return Inputs(inputs)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runs, "PROJECT_ROOT", Path(tmp)), \
             patch("core.gemma.load_gemma", return_value=(model, Tokenizer(), layer)), \
             patch("core.gemma.load_selected_sae", return_value=(toy_sae(0), {})):
            main(["ism", "--device", "cpu", "--monitor", "ism", "--prompt", "test", "--max-new-tokens", "2", "--random-control"])
            directory, = (Path(tmp) / "outputs/ism").iterdir()
            results = json.loads((directory / "outputs.json").read_text())["results"]
            self.assertEqual(len(results), 14)
            self.assertEqual([r["steering_features"] for r in results[2:8]],
                             [[133], [133], [4360], [4360], [133, 4360], [133, 4360]])
            self.assertTrue(all(r["direction_kind"] == "random" for r in results[8:]))
            self.assertTrue((directory / "measurements.md").exists())
            self.assertFalse((directory / "report.md").exists())
            self.assertTrue((directory / "figures/feature_activity.png").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires torch")
    def test_random_control_norm_and_reproducibility(self):
        import torch
        from core.assessment import matched_random_direction
        direction = torch.arange(32).float()
        control = matched_random_direction(direction, 12)
        torch.testing.assert_close(control.norm(), direction.norm())
        torch.testing.assert_close(control, matched_random_direction(direction, 12), rtol=0, atol=0)
        self.assertFalse(torch.equal(control, matched_random_direction(direction, 13)))

    def test_summary_separates_prompt_from_generated_and_legacy_keys(self):
        from core.reporting import summarize
        result = {"token_trace": [
            {"source": "prompt", "features": {"133": 100, "L17:143": 0}},
            {"source": "generated", "features": {"133": 0, "L17:143": 2}},
            {"source": "generated", "features": {"133": 4, "L17:143": 3}}]}
        stats = summarize(result)
        self.assertEqual(stats["generated"]["L9:133"]["mean"], 2)
        self.assertEqual(stats["generated"]["L9:133"]["fraction"], .5)
        self.assertEqual(stats["prompt"]["L9:133"]["peak"], 100)

    def test_shared_pull_mirror_without_connecting_to_pod(self):
        script = Path(__file__).resolve().parents[2] / "scripts/pull_results.sh"
        if not script.exists():
            self.skipTest("Shared Mac workflow scripts not installed here")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "arguments"
            rsync = root / "rsync"
            rsync.write_text('#!/bin/bash\nif [ "$1" = "--version" ]; then echo "rsync version 3"; exit; fi\nprintf "%s\\n" "$@" > "$TEST_ARGUMENTS"\n')
            rsync.chmod(0o755)
            env_file = root / "test.env"
            env_file.write_text(f"LOCAL_DIR={shlex.quote(tmp)}\nREMOTE_DIR=/remote/project\nREMOTE_HOST=fake-host\nRESULTS_SUBDIR=outputs\nPULL_LAYOUT=mirror\nRESULTS_EXCLUDE=report.md\n")
            env = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}",
                   "RUNPOD_ENV_FILE": str(env_file), "TEST_ARGUMENTS": str(log)}
            for args, relative in (([], "outputs"), (["outputs/ism/run"], "outputs/ism/run")):
                subprocess.run(["bash", str(script), *args], env=env, check=True, capture_output=True)
                actual = log.read_text().splitlines()
                self.assertEqual(actual[-2:], [f"fake-host:/remote/project/{relative}/", f"{tmp}/{relative}/"])
                self.assertEqual("--exclude=report.md" in actual, not args)
            self.assertNotEqual(subprocess.run(["bash", str(script), "../escape"], env=env,
                                              capture_output=True).returncode, 0)
