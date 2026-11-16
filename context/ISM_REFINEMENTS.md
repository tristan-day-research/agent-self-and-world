# Next bounded ISM refinement

This follow-up tests the cases that showed a directional effect while checking
whether that effect survives smaller perturbations and more control directions.
See `context/ISM_REPLICATION.md` for the wider replication criteria and
`context/REPORTING.md` for agent-authored reporting.

## Scope

- **Prompts:** published Lyra roleplay, robot story, and playful Assistant prompt.
- **Direction:** joint 133+4360, with reference-alpha strengths −1, −0.5, −0.25,
  +0.25, +0.5, +1, plus baseline and zero-hook. Raw coefficients are each alpha
  times that feature's stored reference peak; these are not paper corpus peaks.
- **Controls:** two independently seeded isotropic directions and two randomly
  selected layer-9 SAE decoder directions, each norm-matched to the joint
  direction. The SAE source IDs, control seeds, and measured norms are saved.
  With the default seed 0, the selected SAE control IDs are 29413 and 3267;
  the isotropic seeds are 1000 and 1100. Both SAE IDs are excluded from the
  paper's explicitly monitored feature list.
  Random SAE features may themselves be meaningful: they are comparison
  directions, not guaranteed inert controls or a statistical null distribution.
- **Decoding:** one greedy reply, up to 192 new tokens. Each condition also has a
  fixed-baseline-continuation pass with both SAE and J-Lens measurements.
- **Output size:** 3 prompts × (2 controls + 6 strengths × 5 directions) =
  **96 evaluated completions**, each with a second fixed-text forward pass. This
  is a larger paid run than the preceding 42-completion probe; do not launch it
  without the user's explicit approval.
- **Lens:** eight blocks, stride 16, both free and fixed text. Sampled unnormalized
  transported vectors are saved in one binary artifact per completion. The
  evidence builder computes cosine, L2, and relative-L2 changes only against the
  same layer/position/phase and the identical text prefix. It skips divergent free
  prefixes and rejects differing fixed-continuation tokens. Block 33 stores the
  direct final residual, not a fitted lens. Compare within a layer; these metrics
  do not establish an ISM axis or subjective perspective.
- **Review:** separately inspect Assistant preambles, stage directions, narrative
  style, task compliance and obvious repetitive loops. The saved repeated 8-token
  fraction and opening-parenthesis flag are literal diagnostics, not classifiers.
  Do not infer reliability from two control directions or one greedy continuation.

## Launch on the same pod as the previous run

```bash
run-experiment ism --device cuda --prompt-set paper \
  --prompt-id table6_roleplay_lyra --prompt-id table6_story_robot \
  --prompt-id table7_playfulness --feature 133 4360 \
  --coefficient-mode reference-alpha \
  --calibration-file outputs/ism/20261107T014211Z-8c44c965/calibration.json \
  --strengths -1 -0.5 -0.25 0.25 0.5 1 --max-new-tokens 192 \
  --random-control --random-control-count 2 --sae-control-count 2 \
  --teacher-force-baseline --jlens --save-jspace-vectors
```

The calibration file must already be present on the pod. `outputs/` is excluded
from sync-up. If the old pod was destroyed, replace `--calibration-file ...` with
`--calibration-prompts experiments/ism/reference_prompts.jsonl`; this adds 16
unsteered reference generations and should be recorded as a new calibration.
Check actual cache/authentication and GPU memory before assuming the previous
runtime. Vector capture increases artifact size and runtime.

RunPod writes `outputs/ism/<run-id>/measurements.md` and raw artifacts; after
pulling, ask the agent to inspect them and author `report.md`. The deterministic
evidence builder never overwrites a report. The code supports arbitrary selected
published prompt IDs, control counts, lens layers and position stride without
changing the old defaults unless flags are supplied.
Default pulls exclude pod `report.md` files so old pod-generated reports cannot
overwrite local agent-authored reports. Vector `.pt` files are included.
