"""Compare selected loading/encoding with a full-width official-formula reference.

Reference equation: Google DeepMind's Gemma Scope 2 tutorial, encode():
pre = input @ w_enc + b_enc; acts = (pre > threshold) * relu(pre).
No decoder-bias subtraction or input normalization. This is a formula/loading
check, NOT an independent SAELens runtime or a test of the authors' revisions.
"""
import hashlib
from pathlib import Path

REFERENCE_URL = 'https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r'


def audit_selected_sae(sae, residuals, params_path, max_positions=16):
    import torch
    from safetensors import safe_open
    samples = residuals.detach().reshape(-1, sae.w_enc.shape[0]).float().cpu()
    if not len(samples):
        raise ValueError('Audit requires actual residual positions')
    indices = torch.linspace(0, len(samples)-1, min(max_positions, len(samples))).long()
    samples = samples[indices]
    selected = sae.encode(samples.to(sae.w_enc.device)).detach().cpu()
    ids = list(sae.feature_ids)
    with safe_open(str(params_path), framework='pt', device='cpu') as weights:
        full_encoder = weights.get_tensor('w_enc').float()
        pre = samples @ full_encoder + weights.get_tensor('b_enc').float()
        expected = ((pre > weights.get_tensor('threshold').float()) * torch.relu(pre))[:, ids]
        # Decoder rows must be original values, not implicitly renormalized.
        decoder = weights.get_slice('w_dec')
        for index, feature in enumerate(ids):
            torch.testing.assert_close(sae.w_dec[index].cpu(), decoder[feature:feature+1][0].float(),
                                       rtol=0, atol=0)
    torch.testing.assert_close(selected, expected, rtol=1e-4, atol=1e-4)
    if not torch.equal(selected > 0, expected > 0):
        raise ValueError('Full/selected encoders disagree on threshold crossings')
    with Path(params_path).open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'passed': True, 'kind': 'full_width_official_formula_comparison',
            'reference_url': REFERENCE_URL, 'params_sha256': digest,
            'features': ids, 'sample_positions': indices.tolist(),
            'full_width': full_encoder.shape[1],
            'max_absolute_error': float((selected-expected).abs().max()),
            'decoder_rows_exact': True, 'threshold_masks_exact': True,
            'rtol': 1e-4, 'atol': 1e-4,
            'limitation': 'Formula/loading parity only; not independent upstream runtime or author checkpoint parity'}
