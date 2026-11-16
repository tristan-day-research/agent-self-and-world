import unittest

import torch

from core.residual_experiments import capture_layers, fit_ridge_probe, patch_layer, score_probe


class ResidualExperimentTests(unittest.TestCase):
    def test_capture_patch_and_cleanup(self):
        layer = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(3))
        x = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
        with capture_layers([layer], [0]) as record:
            ordinary = layer(x)
        self.assertTrue(torch.equal(record[0], x[0, -1]))
        with patch_layer(layer, torch.tensor([9., 8., 7.])):
            changed = layer(x)
        self.assertTrue(torch.equal(changed[0, 0], x[0, 0]))
        self.assertTrue(torch.equal(changed[0, -1], torch.tensor([9., 8., 7.])))
        self.assertTrue(torch.equal(layer(x), ordinary))

    def test_probe_on_heldout_vectors(self):
        train = [torch.tensor([float(i), 1.]) for i in (-3, -2, 2, 3)]
        labels = [False, False, True, True]
        w, b = fit_ridge_probe(train, labels)
        result = score_probe(w, b, [torch.tensor([-1., 1.]), torch.tensor([1., 1.])], [False, True])
        self.assertEqual(result['accuracy'], 1.)
        self.assertEqual(result['auroc'], 1.)


if __name__ == '__main__':
    unittest.main()
