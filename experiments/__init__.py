"""Add one module and one registry entry to register an experiment.

Keep expensive imports inside experiment functions so CLI help stays lightweight.
"""

EXPERIMENTS = {
    "smoke": ("experiments.smoke.main", "Check run logging and artifacts without loading models"),
    "ism": ("experiments.ism.main", "Compare baseline with layer-9 ISM feature steering"),
    "confidence_entropy": ("experiments.confidence_entropy.main", "Measure persona × ISM effects on choice entropy and J-Lens"),
    "agent_vs_model": ("experiments.agent_vs_model.main", "Test attributed door-code knowledge against retrieval and parity use"),
    "gemma4_suite": ("experiments.gemma4_suite.main", "Gemma-4 speaker, crossed-door, persona and mechanism baseline"),
}
