"""Literal generation diagnostics, not semantic immersion/coherence judgments."""
def generation_metrics(token_ids, text=""):
    windows = [tuple(token_ids[index:index+8]) for index in range(max(0, len(token_ids)-7))]
    return {"emitted_tokens": len(token_ids),
            "repeated_8token_window_fraction": 1-len(set(windows))/len(windows) if windows else 0.0,
            "opens_with_parenthesis": text.lstrip().startswith("("),
            "metric_scope": "literal repetition and opening punctuation; not an immersion or coherence score"}
