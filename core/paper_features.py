"""Explicitly numbered Gemma features in Many Are My Names, not all discovered latents.

Labels are the authors' interpretations. IDs apply only to GemmaScope 2 4B-IT,
resid_post, 65k width, medium L0 at the specified zero-based layer.
"""

PAPER_FEATURES = {
    9: {
        133: ("ISM speech carrier", "Table 1 / §3.3.1"),
        4360: ("ISM launch gate", "Table 1 / §3.3.1"),
        1697: ("A-nature: overeager assistant", "Table 9"),
        2575: ("A-nature: being AI / LLM", "Table 9"),
        11963: ("A-nature: AI / LLM and Assistant personal deixis", "Table 9"),
        3274: ("A-summon: describing imagined experience", "Table 9"),
        14298: ("A-summon: providing further options", "Table 9"),
        17452: ("A-summon: describing personas / stories", "Table 9"),
        25147: ("A-summon: enthusiastic meta-commentary", "Table 9"),
        3733: ("A-traits: asking whether it could help", "Table 9"),
        440: ("Narrative", "Table 5"),
        2368: ("NNIA narrative feature", "Appendix D"),
    },
    17: {
        220: ("A-nature: being an LLM in vivid terms", "Table 9"),
        143: ("A-traits: comforting and validating", "Table 9"),
        738: ("A-traits: responding to a request", "Table 9"),
        972: ("A-traits: asking whether there are questions", "Table 9"),
        2439: ("A-traits: here to help and chat", "Table 9"),
        1526: ("Tone: childlike", "Table 5"),
        696: ("NNIA narrative feature", "Appendix D"),
    },
    22: {
        794: ("A-nature: Gemma / Google LLM", "Table 9"),
        1417: ("A-traits: processing / having a task", "Table 9"),
        3062: ("Concept: body parts", "Table 5"),
        1022: ("NNIA narrative feature", "Appendix D"),
    },
}


def feature_manifest():
    return [{"key": f"L{layer}:{feature}", "layer": layer, "feature": feature,
             "label": label, "source": source, "category": feature_category(label)}
            for layer, features in PAPER_FEATURES.items()
            for feature, (label, source) in features.items()]


def feature_category(label):
    if label.startswith('ISM'): return 'ISM'
    for prefix in ['A-summon','A-traits','A-nature','Tone','Concept']:
        if label.startswith(prefix): return prefix
    if 'narrative' in label.lower(): return 'Narrative'
    raise ValueError(f'Uncategorized paper feature: {label}')
