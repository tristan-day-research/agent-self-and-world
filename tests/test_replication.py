"""Replication measurements are checked offline on tiny random Gemma models."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


@unittest.skipUnless(HAS_STACK, "Requires the experiment environment")
class ReplicationTests(unittest.TestCase):
    def test_multilayer_direct_final_and_sampled_zero_control(self):
        import torch
        from core.assessment import generate_comparison
        from experiments.ism.main import toy_setup, toy_sae, ToyTokenizer
        model, layer, inputs, direction = toy_setup(0)
        results = list(generate_comparison(
            model, layer, inputs, direction, [-1, 1], "all", 4, 12,
            sae=toy_sae(0), tokenizer=ToyTokenizer(), jacobian=torch.eye(32),
            text_model=model, steering_layer_index=0, jlens_stride=1, temperature=1,
            lens_layers={0: (layer, torch.eye(32)), 1: (model.model.layers[1], None)}))
        self.assertEqual(results[0]["token_ids"], results[1]["token_ids"])
        self.assertEqual(results[0]["token_trace"], results[1]["token_trace"])
        self.assertEqual(results[0]["jspace"], results[1]["jspace"])
        baseline = results[0]
        final_point = next(p for p in baseline["jspace"] if p["layer"] == 1 and p["source"] == "prompt")
        with torch.inference_mode():
            expected = model(**inputs).logits[0, -1].topk(10).indices.tolist()
        self.assertEqual([t["token_id"] for t in final_point["top_tokens"]], expected)
        self.assertEqual(final_point["lens_kind"], "direct_final")
        points = [p for p in baseline["jspace"] if p["layer"] == 1]
        self.assertEqual([p["position"] for p in points], [2, 3, 4, 5])
        for block in model.model.layers:
            self.assertFalse(block._forward_hooks)

    def test_teacher_forcing_measures_identical_prefix_and_actual_sae_addition(self):
        import torch
        from core.assessment import assess_fixed_continuation
        from experiments.ism.main import toy_setup, toy_sae, ToyTokenizer
        model, layer, inputs, direction = toy_setup(0)
        sae = toy_sae(0)
        sae.b_enc += 100
        downstream = toy_sae(1)
        downstream.b_enc += 100
        common = dict(model=model, layer=layer, inputs=inputs, tokens=[5, 6, 7],
                      direction=direction, scope="all", sae=sae, tokenizer=ToyTokenizer(),
                      monitors={1: (model.model.layers[1], downstream)})
        baseline = assess_fixed_continuation(strength=None, **common)
        zero = assess_fixed_continuation(strength=0, **common)
        steered = assess_fixed_continuation(strength=2, **common)
        self.assertEqual(baseline, zero)
        self.assertEqual([r["token_id"] for r in baseline["token_trace"]],
                         [r["token_id"] for r in steered["token_trace"]])
        # Layer 0 pre-addition values are identical on identical text; the actual
        # post-addition encoder must reflect 2 * direction @ W_enc.
        before = baseline["token_trace"][0]["features"]["133"]
        self.assertEqual(before, steered["token_trace"][0]["features"]["133"])
        after = steered["token_trace"][0]["features_after_steering"]["133"]
        self.assertAlmostEqual(after-before, float(2 * direction @ sae.w_enc[:, 0]), places=4)
        self.assertNotEqual(baseline["token_trace"][0]["features"]["L1:133"],
                            steered["token_trace"][0]["features"]["L1:133"])

    def test_reference_peak_calibration_is_reproducible_and_checks_provenance(self):
        from core.calibration import collect_reference_peaks, validated_peaks, read_reference_prompts
        from experiments.ism.main import toy_setup, toy_sae, ToyTokenizer
        model, layer, inputs, _ = toy_setup(0)
        sae = toy_sae(0)
        sae.b_enc += 100
        class Inputs(dict):
            def to(self, _device):
                return self
        class Tokenizer(ToyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]
            def __call__(self, *args, **kwargs):
                return Inputs(inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.jsonl"
            path.write_text(json.dumps({"id": "heldout", "messages": [{"role": "user", "content": "test"}]})+'\n')
            samples, digest = read_reference_prompts(path)
            artifact = collect_reference_peaks(model, Tokenizer(), layer, sae, samples, digest,
                "cpu", 2, 0, {"model": "tiny", "layer": 0}, Path(tmp)/"calibration.json")
            peaks = validated_peaks(artifact, [133, 4360], {"model": "tiny", "layer": 0})
            self.assertFalse(artifact["exact_paper_corpus"])
            self.assertEqual(artifact["source_sha256"], digest)
            self.assertTrue(all(value > 0 for value in peaks.values()))
            for feature, value in peaks.items():
                self.assertEqual(value, artifact["samples"][0]["sample_maxima"][str(feature)])
            with self.assertRaisesRegex(ValueError, "provenance"):
                validated_peaks(artifact, [133], {"model": "different"})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validated_peaks({**artifact, "complete": False}, [133], {})

    def test_multi_lens_loader_downloads_once_and_rejects_missing_layer(self):
        import torch
        from types import SimpleNamespace
        from core.jlens import load_jacobians
        checkpoint = {"d_model": 2, "J": {0: torch.eye(2), 1: 2*torch.eye(2)}}
        with patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.hf_hub_download", return_value="fake") as download, \
             patch("torch.load", return_value=checkpoint):
            api.return_value.model_info.return_value = SimpleNamespace(sha="revision")
            matrices, metadata = load_jacobians([0, 1], 2)
            self.assertEqual(set(matrices), {0, 1})
            self.assertEqual(download.call_count, 1)
            self.assertEqual(metadata["revision"], "revision")
            with self.assertRaisesRegex(ValueError, "no layer"):
                load_jacobians([3], 2)

    def test_cli_reference_calibration_trials_and_multilayer_lens(self):
        import torch
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration, SiglipVisionConfig
        from core import runs
        from core.gemma import MODEL_ID, MODEL_REVISION, SAE_ID, SAE_REVISION, SAE_PATH
        from experiments.ism.main import toy_setup, toy_sae, ToyTokenizer
        from core.sae import SelectedJumpReLU
        from run import main
        text, _, inputs, _ = toy_setup(0)
        text.config.num_hidden_layers = 12
        text.config.layer_types = ["sliding_attention", "full_attention"] * 6
        vision = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                    num_attention_heads=2, image_size=16, patch_size=8)
        model = Gemma3ForConditionalGeneration(Gemma3Config(
            text_config=text.config.to_dict(), vision_config=vision.to_dict(),
            image_token_index=30, boi_token_index=29, eoi_token_index=28, mm_tokens_per_image=4)).eval()
        layer = model.model.language_model.layers[9]
        class Inputs(dict):
            def to(self, device):
                return self
        class Tokenizer(ToyTokenizer):
            chat_template = "tiny"
            def apply_chat_template(self, *args, **kwargs):
                return "test"
            def __call__(self, *args, **kwargs):
                return Inputs(inputs)
        def fake_sae(features, **_kwargs):
            rng = torch.Generator().manual_seed(11)
            n = len(features)
            return SelectedJumpReLU(features, torch.randn(32, n, generator=rng),
                                   torch.zeros(n), torch.zeros(n), torch.randn(n, 32, generator=rng)), {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps({"kind": "reference_set_peaks", "complete": True,
                "model": MODEL_ID, "model_revision": MODEL_REVISION, "sae": SAE_ID,
                "sae_revision": SAE_REVISION, "sae_path": SAE_PATH, "layer": 9, "dtype": "float32",
                "feature_peaks": {"133": 10, "4360": 20}}))
            with patch.object(runs, "PROJECT_ROOT", root), \
                 patch("core.gemma.load_gemma", return_value=(model, Tokenizer(), layer)), \
                 patch("core.gemma.load_selected_sae", side_effect=fake_sae), \
                 patch("core.jlens.load_jacobians", return_value=({8: torch.eye(32), 9: torch.eye(32),
                                                               10: torch.eye(32)}, {})):
                main(["ism", "--device", "cpu", "--monitor", "ism", "--prompt", "test",
                      "--feature", "133", "--coefficient-mode", "reference-alpha",
                      "--calibration-file", str(calibration), "--strengths", "-1", "1",
                      "--random-control", "--random-control-count", "2", "--sae-control-count", "1",
                      "--temperatures", "0", "1", "1", "--teacher-force-baseline",
                      "--jlens", "--save-jspace-vectors", "--jlens-layers", "8", "9", "10", "11", "--max-new-tokens", "2"])
            directory, = (root / "outputs/ism").iterdir()
            results = json.loads((directory / "outputs.json").read_text())["results"]
            self.assertEqual(len(results), 30)
            self.assertEqual(len({r["prompt_id"] for r in results}), 3)
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(len(metadata["sae_control_features"]), 1)
            self.assertEqual(len({s["seed"] for s in metadata["directions"] if s["kind"] == "random"}), 2)
            norms = [s["norm_before_strength"] for s in metadata["directions"]]
            self.assertTrue(all(abs(x-norms[0]) < 1e-4 for x in norms))
            for start in (0, 10, 20):
                self.assertEqual(results[start]["token_ids"], results[start+1]["token_ids"])
                reference = results[start]["fixed_continuation"]["token_ids"]
                for r in results[start:start+10]:
                    self.assertEqual(r["fixed_continuation"]["token_ids"], reference)
                    self.assertEqual({p["layer"] for p in r["jspace"]}, {8, 9, 10, 11})
            self.assertTrue((directory / "figures/ism_magnitude.png").exists())
            effects = json.loads((directory / "jspace_effects.json").read_text())["comparisons"]
            self.assertTrue(any(p["space"] == "fixed" and p["layer"] == 11 for p in effects))
            self.assertTrue(all(p["delta_l2"] == 0 for p in effects if p["layer"] == 8))
            self.assertFalse((directory / "report.md").exists())

    def test_multiple_controls_match_norms_and_record_source(self):
        import torch
        from core.conditions import direction_plan, direction_manifest, control_feature_ids
        from experiments.ism.main import toy_sae
        sae = toy_sae(0)
        target = torch.arange(32).float()
        plan = direction_plan([[133]], [target], sae, 0, random_count=3, sae_controls=[4360])
        self.assertEqual([p["kind"] for p in plan], ["sae", "random", "random", "random", "sae_control"])
        for spec in plan:
            torch.testing.assert_close(spec["direction"].norm(), target.norm())
        self.assertEqual(len({p["seed"] for p in plan if p["kind"] == "random"}), 3)
        self.assertEqual(direction_manifest(plan)[-1]["source_feature"], 4360)
        ids = control_feature_ids(3, 20, {0, 2, 4}, 123)
        self.assertEqual(ids, control_feature_ids(3, 20, {0, 2, 4}, 123))
        self.assertFalse(set(ids) & {0, 2, 4})

    def test_vector_effects_skip_divergent_free_text_but_compare_fixed_text(self):
        import torch
        from core.lens_analysis import save_vector_artifact, vector_effects
        def make(strength, ids, vector):
            def points():
                return [{"layer": 1, "position": position, "_vector_observed": vector.clone()}
                        for position in (1, 2)]
            return {"prompt_id": "p", "strength": strength, "input_ids": [10, 11],
                    "token_ids": ids, "jspace": points(),
                    "fixed_continuation": {"token_ids": [1, 2], "jspace": points()}}
        baseline = make(None, [1, 2], torch.tensor([1., 1.]))
        steered = make(1, [3, 4], torch.tensor([2., 1.]))
        with tempfile.TemporaryDirectory() as tmp:
            save_vector_artifact(baseline, tmp, 0)
            save_vector_artifact(steered, tmp, 1)
            json.dumps([baseline, steered])  # Tensor fields must be stripped.
            effects = vector_effects([baseline, steered], tmp)
            self.assertEqual(effects["skipped_nonmatching_prefixes"], 1)
            rows = [p for p in effects["comparisons"] if p["result_index"] == 1]
            self.assertEqual(len(rows), 3)
            self.assertTrue(any(p["space"] == "fixed" and p["position"] == 2 for p in rows))
            self.assertAlmostEqual(rows[0]["delta_l2"], 1.)
            self.assertAlmostEqual(rows[0]["cosine"], 3/(10**.5), places=6)
            steered["fixed_continuation"]["token_ids"] = [9, 9]
            with self.assertRaisesRegex(ValueError, "Fixed-continuation"):
                vector_effects([baseline, steered], tmp)


class EvidenceTests(unittest.TestCase):
    def test_evidence_never_creates_or_overwrites_report(self):
        from core.reporting import generate_evidence
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "outputs.json").write_text(json.dumps({"kind": "exploratory_ism", "results": []}))
            self.assertEqual(generate_evidence(path, plots=False).name, "measurements.md")
            self.assertFalse((path / "report.md").exists())
            (path / "report.md").write_text("My inspected scientific analysis.\n")
            generate_evidence(path, plots=False)
            self.assertEqual((path / "report.md").read_text(), "My inspected scientific analysis.\n")
            (path / "outputs.json").write_text(json.dumps({"kind": "reference_calibration", "results": [], "feature_peaks": {"133": 2}}))
            generate_evidence(path, plots=False)
            self.assertEqual((path / "report.md").read_text(), "My inspected scientific analysis.\n")
