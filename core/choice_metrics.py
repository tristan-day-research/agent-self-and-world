"""Choice-distribution uncertainty, not a direct measure of knowledge/confidence."""
import math


def label_token_ids(tokenizer, labels):
    """Explicit bare-letter convention; never truncate multi-token encodings."""
    mapping = {}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'Choice label {label!r} is not exactly one token: {ids}')
        mapping[label] = ids[0]
    if len(mapping) < 2 or len(set(mapping.values())) != len(mapping):
        raise ValueError('Need at least two distinct single-token choice labels')
    return mapping


def continuation_label_token_ids(tokenizer, prompt, labels):
    """Validate next-token labels at this exact formatted prompt boundary."""
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    mapping = {}
    for label in labels:
        joined = tokenizer.encode(prompt + label, add_special_tokens=False)
        if len(joined) != len(prefix) + 1 or joined[:-1] != prefix:
            raise ValueError(f'Label {label!r} is not one token after this prompt')
        mapping[label] = joined[-1]
    if len(mapping) < 2 or len(set(mapping.values())) != len(mapping):
        raise ValueError('Decision labels must map to distinct single tokens')
    return mapping


def distribution_metrics(logits, token_ids, correct_label=None):
    """At temperature 1, condition on the listed token events and retain their mass.

    Entropy is in nats; normalized entropy divides by log(K). A sharp conditional
    choice distribution is compatible with almost all probability being off-format.
    """
    import torch
    scores = logits.detach().reshape(-1).double()
    if not torch.isfinite(scores).all():
        raise ValueError('Logits must be finite')
    labels = list(token_ids)
    ids = list(token_ids.values())
    if len(labels) < 2 or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(scores):
        raise ValueError('Invalid or duplicate choice token IDs')
    if correct_label is not None and correct_label not in token_ids:
        raise ValueError('Correct label is not an available choice')
    selected = scores[ids]
    logp = selected.log_softmax(-1)
    probabilities = logp.exp()
    full_logp = scores.log_softmax(-1)
    full_prob = full_logp.exp()
    entropy = float(-(probabilities*logp).sum())
    ranking = sorted(labels, key=lambda k: float(selected[labels.index(k)]), reverse=True)
    top = int(scores.argmax())
    result = {'choice_logits': dict(zip(labels, selected.cpu().tolist())),
              'choice_probabilities': dict(zip(labels, probabilities.cpu().tolist())),
              'choice_entropy_nats': entropy, 'choice_entropy_normalized': entropy/math.log(len(ids)),
              'choice_probability_mass': float(full_prob[ids].sum()),
              'full_vocab_entropy_nats': float(-(full_prob*full_logp).sum()),
              'choice_argmax': ranking[0], 'unconstrained_argmax_token_id': top,
              'unconstrained_argmax_is_choice': top in ids,
              'unconstrained_argmax_label': next((k for k,v in token_ids.items() if v==top), None),
              'top_two_choice_margin': float(probabilities.topk(2).values.diff().abs()[0]),
              'n_choices': len(ids)}
    if correct_label is not None:
        index = labels.index(correct_label)
        target = torch.zeros_like(probabilities)
        target[index] = 1
        result.update(correct_choice_probability=float(probabilities[index]),
                      correct_choice_nll=float(-logp[index]),
                      multiclass_brier=float(((probabilities-target)**2).sum()),
                      choice_argmax_correct=ranking[0] == correct_label,
                      unconstrained_argmax_correct=top == token_ids[correct_label])
    return result
