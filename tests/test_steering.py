"""Actual CPU tensor and tiny-Gemma checks; never download pretrained weights."""

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TORCH, "Install/use the experiment environment for tensor tests")
class SteeringTests(unittest.TestCase):
    def test_selected_jumprelu_and_recording(self):
        import torch
        from core.sae import SelectedJumpReLU, flatten_record, record_sae_features

        sae = SelectedJumpReLU((10, 20), torch.eye(2), torch.tensor([0.5, 0.0]),
                               torch.tensor([1.0, 0.25]), torch.eye(2))
        hidden = torch.tensor([[[0.5, 0.5], [0.6, -1.0]]])
        # Strictly greater than threshold: first feature is zero at exactly 1.0.
        torch.testing.assert_close(sae.encode(hidden), torch.tensor([[[0.0, 0.5], [1.1, 0.0]]]))
        layer = torch.nn.Identity()
        with record_sae_features(layer, sae, capture_residuals=True) as record:
            self.assertIs(layer(hidden), hidden)
        acts, residuals = flatten_record(record)
        torch.testing.assert_close(acts, torch.tensor([[0.0, 0.5], [1.1, 0.0]]))
        torch.testing.assert_close(residuals, hidden[0].float())
        self.assertEqual(len(layer._forward_hooks), 0)

    @unittest.skipUnless(HAS_TRANSFORMERS, "Requires transformers")
    def test_jspace_readout_and_paper_prompts(self):
        import torch
        from core.jlens import jspace_top_tokens
        from experiments.ism.main import ToyTokenizer, toy_setup
        from experiments.ism.prompts import PAPER_PROMPTS

        model, _, _, _ = toy_setup(0)
        result = jspace_top_tokens(torch.eye(32), torch.ones(2, 32), model,
                                   ToyTokenizer(), top_k=3)
        self.assertEqual([len(row) for row in result], [3, 3])
        self.assertEqual([prompt["condition"] for prompt in PAPER_PROMPTS],
                         ["assistant", "roleplay", "story"])
        self.assertEqual(PAPER_PROMPTS[1]["messages"][0]["role"], "system")

    def test_addition_scope_and_zero_identity(self):
        import torch
        from core.steering import steer_layer

        layer = torch.nn.Identity()
        hidden = torch.zeros(1, 3, 2)
        direction = torch.tensor([2., -1.])
        with steer_layer(layer, direction, 3):
            torch.testing.assert_close(layer(hidden), torch.tensor([[[6., -3.]]]).expand_as(hidden))
        with steer_layer(layer, direction, -2, "last"):
            expected = hidden.clone()
            expected[:, -1, :] = torch.tensor([-4., 2.])
            torch.testing.assert_close(layer(hidden), expected)
        with steer_layer(layer, direction, 0):
            self.assertIs(layer(hidden), hidden)
        torch.testing.assert_close(hidden, torch.zeros_like(hidden))
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_tuple_output_and_exception_cleanup(self):
        import torch
        from core.steering import steer_layer

        class TupleLayer(torch.nn.Module):
            def forward(self, hidden):
                return hidden, "auxiliary"

        layer = TupleLayer()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with steer_layer(layer, torch.ones(2), 1):
                output = layer(torch.zeros(1, 1, 2))
                self.assertEqual(output[1], "auxiliary")
                torch.testing.assert_close(output[0], torch.ones(1, 1, 2))
                raise RuntimeError("interrupted")
        self.assertEqual(len(layer._forward_hooks), 0)

    @unittest.skipUnless(HAS_TRANSFORMERS, "Requires transformers")
    def test_cached_generation_and_zero_control(self):
        from experiments.ism.main import toy_setup, generate_comparison

        model, layer, inputs, direction = toy_setup(0)
        results = list(generate_comparison(model, layer, inputs, direction, [-1, 0, 1], "all", 3, 0))
        self.assertEqual(results[0]["token_ids"], results[1]["token_ids"])
        for result in results[1:]:
            self.assertEqual(result["hook"]["sequence_lengths"], [3, 1, 1])
        self.assertEqual(results[2]["hook"]["positions_steered"], 5)
        self.assertEqual(len(layer._forward_hooks), 0)

    @unittest.skipUnless(HAS_TRANSFORMERS, "Requires transformers")
    def test_complete_ism_smoke_records_sae_and_jspace(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from core import runs
        from run import main

        with tempfile.TemporaryDirectory() as tmp, patch.object(runs, "PROJECT_ROOT", Path(tmp)):
            self.assertEqual(main(["ism", "--smoke", "--device", "cpu"]), 0)
            directory, = (Path(tmp) / runs.RESULTS_SUBDIR / "_smoke" / "ism").iterdir()
            outputs = json.loads((directory / "outputs.json").read_text())
            first = outputs["results"][0]
            self.assertEqual(first["sae_record"]["feature_ids"], [133, 4360])
            self.assertTrue(first["token_trace"])
            self.assertTrue(first["jspace"])

    @unittest.skipUnless(HAS_TRANSFORMERS, "Requires transformers")
    def test_real_wrapper_hook_path_with_tiny_random_config(self):
        import torch
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration, SiglipVisionConfig
        from core.jlens import jspace_top_tokens
        from core.steering import steer_layer
        from experiments.ism.main import ToyTokenizer, generate_comparison, toy_sae, toy_setup

        text_model, _, inputs, _ = toy_setup(0)
        vision = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                    num_attention_heads=2, image_size=16, patch_size=8)
        config = Gemma3Config(text_config=text_model.config.to_dict(), vision_config=vision.to_dict(),
                             image_token_index=30, boi_token_index=29, eoi_token_index=28,
                             mm_tokens_per_image=4)
        model = Gemma3ForConditionalGeneration(config).eval()
        layer = model.get_submodule("model.language_model.layers.0")
        with torch.inference_mode():
            baseline = model(**inputs).logits
            with steer_layer(layer, torch.ones(32), 0):
                control = model(**inputs).logits
            torch.testing.assert_close(control, baseline, rtol=0, atol=0)
        readout = jspace_top_tokens(torch.eye(32), torch.ones(1, 32), model,
                                    ToyTokenizer(), top_k=2)
        self.assertEqual(len(readout[0]), 2)
        # Exercise the exact generation -> layer hook -> SAE -> J-Lens path used
        # by the real experiment, with Gemma's conditional-generation wrapper.
        results = list(generate_comparison(
            model, layer, inputs, torch.ones(32), [-1, 1], "all", 3, 0,
            sae=toy_sae(0), tokenizer=ToyTokenizer(), jacobian=torch.eye(32),
            text_model=model, jlens_top_k=2))
        self.assertEqual(results[0]["token_ids"], results[1]["token_ids"])
        self.assertTrue(all(result["jspace"] for result in results))
        self.assertEqual(len(layer._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
