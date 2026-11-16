# Start here: project handoff

Updated 2026-11-16. This is a navigation page for a new Codex or Claude
session working in `/Users/daylight/dev/code/many_are_my_names`. Read
`AGENTS.md` first, then `context/STATUS.md`, `context/RESEARCH.md`, and
`context/DECISIONS.md` as needed. Each run's `report.md` is the human-written
interpretation; `outputs.json`, metrics, code, and plots are its evidence.

## Research aim and current position

The project began by partially replicating *Many Are My Names* on its
Gemma-3-4B-IT model, especially the layer-9 SAE features 133 and 4360 that
the authors associate with Immersive Simulation Mode (ISM). The broader
question is whether speaker framing, attributed knowledge, and information
use share a mechanism, and what the paper calls the Assistant-Associated Core
(AAC) might amount to. Do not equate ISM, an SAE feature, a pronoun choice,
agenthood, or subjective perspective.

The key ISM result is a repeatable but partial change in narrative register:
negative feature-133 steering introduces an Assistant-style preface on 12/12
tested stories; positive steering often makes roleplay more literary. It does
not reliably transform ordinary Assistant replies into immersed characters.
Feature 4360 alone is weaker. See the [ISM confirmation
report](../outputs/ism/20261107T035003Z-035fea34/report.md).

Gemma-3-4B-IT failed the unsteered positive control for our door-code
instrumental protocol: it copied 8/8 codes but chose the parity-dependent
procedure only 8/16 times, with essentially no paired parity effect. Strong
133 steering also disrupted the answer format. See the [Gemma door
report](../outputs/agent_vs_model/2026-11-10_14-27-49_PST--door-code-pilot--08255c80/report.md)
and [capability diagnostic](../outputs/agent_vs_model/2026-11-10_15-53-46_PST--door-capability--7dab5efa/report.md).
This is a limitation of the tested protocol, not proof that the model can
never do such a task.

Gemma-4-31B-IT is the current capable door model. In a 16-family follow-up,
the code-present instrumental task was 29/32 correct both when the Assistant
was described as knowing and as not knowing; reports changed Y versus N.
Final-position residual patches from opposite-parity cases affected answers
mainly near layer 50. See the [door mechanism
report](../outputs/gemma4_suite/2026-11-11_12-51-53_PST--gemma4-door-mechanisms--b6b4b317/report.md).
These findings do not identify a separate inner agent or a representation of
concealment. A Gemma-4 speaker-direction discovery found a decodable
Assistant/third-person marker, but its tested interventions did not reliably
switch held-out speaker framing; no Gemma-4 ISM feature is validated. See the
[speaker report](../outputs/gemma4_suite/2026-11-11_12-54-50_PST--gemma4-speaker-discovery--899aa160/report.md).

The latest completed pilot crossed Assistant, first-person Mara roleplay,
and third-person Mara story prompts with code availability and attributed
knowledge. Free responses differed in speaking form, but the fixed door
answers matched across all three settings. Query-input speaker classification
was strong with matched task types and uneven across task types; attention
weights were descriptive, not a causal mediation test. See the [speaker and
attention report](../outputs/gemma4_suite/2026-11-11_17-05-52_PST--gemma4-door-speaker-attention--a970a5c2/report.md).

## Next research decisions

No paid run is active or approved as of this handoff. The next experiment
has **not** been selected. Strong candidates are:

1. Build a false-belief door scenario where Mara knows an old code but an
   observer knows the new one. Measure choices attributed to Mara versus
   objective door-opening, then probe held-out attention-head outputs as in
   Zhu et al. The present attention run saved Query inputs and attention
   rows, not Zhu-style head outputs.
2. Discover a Gemma-4 ISM or speaker-direction candidate across layers and
   paraphrased prompts, then test held-out directional steering with random
   and same-mode controls. Do not transfer Gemma-3 feature IDs to Gemma-4.
3. Compare code-token versus final-position patching in the same Gemma-4
   protocol, since earlier Qwen and Gemma-4 sweeps patched different sites.

Read `context/OPS.md` before any RunPod action. It requires asking the user
before `set-pod`, `run-experiment`, and `pull-results`. The earlier approvals
were for completed runs and do not authorize a new paid run.

## Files, account transition, and reproducibility

This directory currently has **no `.git` directory**. Its `outputs/` folder
is ignored by `.gitignore` and excluded from `sync-up`; reports and raw
results exist as local files on this Mac. A new OpenAI account using the
same local workspace can read them, but changing accounts does not itself
copy this directory, conversation history, attachments, pod connection,
credentials, or account settings to another computer. Back up the entire
project, including `outputs/`, before changing machines or rebuilding the
workspace. The Many Are My Names PDF is in `context/`; the Zhu paper used in
discussion is in the user's Obsidian library outside this repo.

For a new session, explicitly ask the agent to read this file and the latest
run reports. `AGENTS.md` supplies standing instructions, but the agent will
not automatically read every context file or every output artifact.
