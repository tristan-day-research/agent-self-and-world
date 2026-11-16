"""Synthetic CPU end-to-end check for shared follow-up plumbing."""
from pathlib import Path

from core.runs import run_context, write_json


def main(args, stage):
    import torch
    from core.residual_experiments import capture_layers, patch_layer, fit_ridge_probe, score_probe
    from core.steering import steer_layer
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)).eval()
    with torch.no_grad():
        for layer in model:
            layer.weight.copy_(torch.eye(4))
    x = torch.tensor([[[0., 1., 0., 0.], [1., 2., 3., 4.]]])
    with capture_layers(list(model), [0, 1]) as captured:
        baseline = model(x).detach().clone()
    with patch_layer(model[0], captured[0]):
        self_patch = model(x)
    if not torch.equal(self_patch, baseline):
        raise ValueError('Synthetic self patch changed output')
    with patch_layer(model[0], torch.tensor([-1., -2., -3., -4.])):
        opposite = model(x)
    if torch.equal(opposite, baseline):
        raise ValueError('Synthetic opposite patch had no effect')
    with steer_layer(model[0], torch.ones(4), 0., scope='last'):
        zero = model(x)
    if not torch.equal(zero, baseline):
        raise ValueError('Synthetic zero steering changed output')
    with steer_layer(model[0], torch.ones(4), 1., scope='last'):
        steered = model(x)
    if torch.equal(steered, baseline):
        raise ValueError('Synthetic nonzero steering had no effect')
    train = [torch.tensor([float(i), 1.]) for i in (-3, -2, 2, 3)]
    weight, bias = fit_ridge_probe(train, [False, False, True, True])
    probe = score_probe(weight, bias, [torch.tensor([-1., 1.]), torch.tensor([1., 1.])],
                        [False, True])
    if probe['accuracy'] != 1.:
        raise ValueError('Synthetic probe failed')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != 'resume_directory'}
    with run_context(args.experiment, config) as run:
        write_json(run.directory/'outputs.json', {'synthetic': True, 'stage': stage,
                   'self_patch_exact': True, 'zero_steering_exact': True,
                   'opposite_patch_changed': True, 'nonzero_steering_changed': True,
                   'heldout_probe': probe})
        run.logger.info('Synthetic %s capture/patch/probe/steering smoke passed', stage)
    return 0
