# Research report workflow

Each run saves its inputs, settings, outputs, checkpoints, and optional J-Space
vectors. `python -m core.reporting <run-directory>` rebuilds `measurements.md`,
`assessment.json`, `analysis_brief.json`, `generations.md`, `magnitudes.md`,
and plots. It also creates `jspace_effects.json` if vectors were saved. These
are deterministic evidence, not a narrative report. The script never creates
or overwrites `report.md` and does not call a model or an external service.

After the user pulls a run and asks for analysis, Codex or Claude should:

1. Read status, configuration, provenance, reference calibration, analysis
   manifest, and actual outputs. A running or failed run is a partial snapshot.
2. Inspect all relevant conditions and controls, including null/contrary examples
   and behavior quality. Compute additional quantitative comparisons where they
   are useful and save the script, numerical output, and figures inside the run.
3. Compare SAE behavior on identical forced continuations and free replies;
   identify where primary-layer post-addition effects can be direct encoder
   overlap. For J-Space use matching text prefixes and within-layer comparisons;
   top-token lists alone are not full vector geometry. Identify any missing data.
4. Write a clear, concise research narrative to that run's `report.md`, in your
   own words. State author and evidence scope, distinguish observations from
   interpretation, link exact evidence and explain major limitations and next
   tests. Choose structure appropriate to the findings; there is no rigid report
   template. This is an agent-authored analysis, not independent human review.
5. Record the SHA256 from `analysis_brief.json` or independently hash
   `outputs.json` so a later pull that replaces the underlying data can be detected.
   If evidence changes, reassess before updating the report. Existing run artifacts
   must not be erased to refresh evidence.
6. Verify numerical claims and local links, visually inspect any new scientific
   figures, and update STATUS/RESEARCH/DECISIONS when warranted.

Every new research plot needs a self-contained plain-language caption inside
the image. State what was measured, axes/units, condition mappings, controls, and what the plot cannot
establish. The caption should let a reader understand a standalone PNG.

For a new run, `report.md` exists only after an agent or user writes it. The user
can simply say “I pulled the latest run; review it and write the report.” Run
generation itself does not automatically launch an agent or use an API token.
The project's default `pull-results` excludes pod `report.md` files so an old
pod-generated file cannot overwrite a locally authored report. Explicit subpath
pulls bypass shared-script exclusions; avoid explicitly pulling an old pod report
over a local authored report. Other artifacts, including `.pt` vectors, are pulled.
