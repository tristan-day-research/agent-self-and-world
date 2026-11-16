"""Inference-only TopK-64 SAE for the public Gemma-4-31B-IT layer files."""

SAE_REPO = 'Adam1010/gemma-4-31b-sae-features'


class TopKSAE:
    def __init__(self, weights, k=64):
        import torch
        required = {'W_enc', 'b_enc', 'W_dec', 'b_dec', 'norm_mean', 'norm_std'}
        if not required <= weights.keys():
            raise ValueError(f'SAE is missing {sorted(required - weights.keys())}')
        self.weights = {name: weights[name].float() for name in required}
        w = self.weights
        if w['W_enc'].shape != (5376, 43008) or w['W_dec'].shape != (43008, 5376):
            raise ValueError('Unexpected Gemma-4 SAE shape')
        if not 0 < k <= 43008 or not all(torch.isfinite(v).all() for v in w.values()):
            raise ValueError('Invalid SAE sparsity or nonfinite weights')
        self.k = k

    def to(self, device):
        self.weights = {name: value.to(device) for name, value in self.weights.items()}
        return self

    def encode(self, hidden):
        import torch
        w = self.weights
        x = (hidden.float() - w['norm_mean']) / w['norm_std']
        pre = torch.relu(x @ w['W_enc'] + w['b_enc'])
        values, indices = pre.topk(self.k, dim=-1)
        return values, indices

    def reconstruct(self, values, indices):
        w = self.weights
        rows = w['W_dec'][indices]
        normalized = (values.unsqueeze(-1) * rows).sum(dim=-2) + w['b_dec']
        return normalized * w['norm_std'] + w['norm_mean']

    def direction(self, feature_id):
        w = self.weights
        return w['W_dec'][feature_id] * w['norm_std']


def load_sae(layer, revision=None):
    import torch
    import re
    from huggingface_hub import HfApi, hf_hub_download
    if not 0 <= layer < 60:
        raise ValueError('Gemma-4 SAE layer must be 0..59')
    api = HfApi()
    revision = revision or api.dataset_info(SAE_REPO).sha
    paths = api.list_repo_files(SAE_REPO, repo_type='dataset', revision=revision)
    candidates = [p for p in paths if p.endswith('.pt') and
                  re.search(rf'layer_0*{layer}(?:\D|$)', p)]
    if len(candidates) != 1:
        raise ValueError(f'Expected one SAE weight file for layer {layer}, found {candidates}')
    path = hf_hub_download(SAE_REPO, candidates[0], repo_type='dataset', revision=revision)
    weights = torch.load(path, map_location='cpu', weights_only=True)
    return TopKSAE(weights), {'repo': SAE_REPO, 'revision': revision, 'file': candidates[0]}
