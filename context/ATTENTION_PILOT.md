# Prepared Table 6 attention pilot

Use `ism`, not the confidence experiment, to compare the paper's Assistant,
Roleplay and Story settings directly. Prepared 2026-11-10; not launched.

From the local repo terminal (the existing command syncs code and launches on RunPod):

```bash
run-experiment ism --preset attention-table6
```

If already in a RunPod terminal with the updated repo synced:

```bash
python run.py ism --preset attention-table6
```

Inspect scope locally without loading a model or contacting a pod:

```bash
python run.py ism --preset attention-table6 --dry-run
```

The preset lives in `experiments/ism/presets/attention-table6.json`; all expanded
parameters are saved in config.json. Explicit CLI flags after/before `--preset`
override preset defaults through argparse; inspect the dry run after overrides.

- Three exact published Table 6 prompts: Assistant self-introduction; Lyra the
  elven archivist; a story about the last robot on an abandoned space station.
- Six conditions each: baseline, exact zero hook,133−2/+2 reference-alpha,
  one residual-norm-matched isotropic random direction at both signs.
- **18 evaluated generations**, greedy, seed42, cap256 tokens, plus18
  teacher-forced passes on each prompt's same baseline continuation.
- **16 independent calibration generations**, cap50 tokens. This deliberately
  avoids relying on an ignored output folder existing on a restarted pod.
- Steering is after block9 at all positions.133 is the primary intervention;
  4360 is monitored, not separately steered. This follows the cleaner prior
  behavioral evidence for133 alone. The random control may disrupt quality;
  assess that separately rather than treating it as a clean null by default.
- All34 blocks/all heads for attention, four query positions maximum per fixed
  pass: final prompt, first continuation input, then every16 inputs. Prefill
  boundary also recorded during free generation; cached decode is not captured.
- J-Lens at8/9/10/17/22/29/32 and direct-final33, stride16, saved vectors.
- All23 explicitly numbered paper SAE features monitored.

Expected run folder: `outputs/ism/YYYY-MM-DD_HH-MM-SS_PST--attention-table6--ID/`
(or PDT in summer). This is an exploratory one-prompt-per-setting pilot, not
statistical evidence for all Assistant/Roleplay/Story behavior. Compare steering
within each exact prefix first. The three prompts differ in both content and task;
raw cross-setting differences cannot isolate perspective by themselves.

After pull, regenerate the local inventory: `python -m core.run_index`. Open
`outputs/README.md`, then the run's `RUN_INFO.md`. Attention comparisons and an
exploratory plot are generated automatically; the research report is authored
only after inspecting the actual results. No paid run was started during setup.
