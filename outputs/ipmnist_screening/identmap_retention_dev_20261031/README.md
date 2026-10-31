# Identifier recurrence: frozen access and relearning are different measurements

This fixed development study measures the input-remapping RLS arms on A/B/A.
The identifier improves reacquisition while its B-era frozen mapping has much
lower direct access to A than the matched plain RLS parent. This does not show
that the representation or weights erased A. It measures the actual deployed
checkpoint with its current remap, before any revisit update can repair it.

## Result (three paired development seeds)

Values below are mean +/- sample SD, never a best seed. Every arm used the same
512 held-out sentinel rows, A/B/A permutations, example orders and hardware.

| Arm | A after acquisition | A frozen after B | A after revisit | Leading250 relearning savings |
| --- | ---: | ---: | ---: | ---: |
| Parent RLS | 0.861979 +/- 0.005967 | 0.598958 +/- 0.078426 | 0.885417 +/- 0.014917 | 0.605333 +/- 0.025716 |
| Identifier 50/200/2000 | 0.861979 +/- 0.005967 | 0.110026 +/- 0.036869 | 0.895182 +/- 0.008807 | 0.642667 +/- 0.024111 |
| Identifier, trigger0.03/step0.005625 | 0.854167 +/- 0.010757 | 0.117839 +/- 0.035498 | 0.890625 +/- 0.005167 | 0.636000 +/- 0.024980 |

The predeclared primary candidate-minus-parent difference in frozen A accuracy
is **-0.488932**, SD 0.115290, SE 0.066562.
Per-seed differences are [-0.36328125, -0.513671875, -0.58984375]. All three are negative and satisfy
the declared absolute0.02 and two-SE rule. Classification:
`lower_direct_frozen_retention`. The third arm is a stronger execution-source registered
control; its detector and body-step differences are not a causal identification
contrast. Full values, all seeds and hashes are in SUMMARY.json and shards/.

This narrows the engineering decision: abrupt-stream identification scores do
not establish retained direct access under recurrence. Keep measuring frozen
access and reacquisition separately. Selecting stored mappings, latent recall
or broader continual-learning behavior remains an open question, not an
implementation or recommendation in this change.

## Named implementation need and proof

The existing grid1 (PR3076) ran MLP controls; PR3277 restored only the plain
confirmed RLS arm. Identifier arms explicitly refused probes. This was a
known capability limit, not a newly discovered bug. The retained real-MNIST
baseline refusal and failing public-adapter regression show the measurement
need before source changes. The plain parent ran before and after; its complete
report, including hashes and every metric, is exactly equal.

The new hook for the four registered identifier arms applies only the current
checkpoint remap, the inner frozen EMA and its learned RLS readout. It does not
run the identifier/Hungarian assignment, update normalization, use evaluator
permutation identity, read labels, learn or consume RNG. Existing input-only
refusal hooks and unrelated special-arm limits remain intact. No training
update, hyperparameter, registration name or promotion rule changes.

Software proof includes hand, initialized and trained checkpoints with a
nonidentity remap, eager/JIT logits recovered from the actual residual learner's
loss, explicit wrong-mapping controls, state/RNG hashes and exact unprobed
training continuation. This is actual JAX model execution, not mocked learners
or an external language-model call.

## Protocol and provenance

Read PLAN.md for the declaration made before measurement, seeds, comparison bar
and fixed compute budget. Main baseline: ea6eee9d0da3fce024028810b60edfb7fa4389b4.
Measured source/driver commit: 0d7f8810199aa191edc7170bbf761518eb653d53.
Later commits add only artifacts/documentation/verification; the measured package
and lock bytes must match that commit. Before-driver packaging used the host's
campaign path; the matched driver resolves the required repository working
directory and was formatted. This changes no data/key/model/metric operation;
exact complete parent-report equality is retained in BASELINE_EQUIVALENCE.json.

The existing OpenML cache is `outputs/upgd_ipmnist/reconstructed_openml_cache`.
The unmodified loader materializes all60000 public MNIST training rows using
its original float32 scaling and int32 labels. Exact x/y hashes are in every
shard. Independent reload reproduces them and both published grid1 sentinel
commitments; no dataset was synthesized or downloaded for this study. Data
are not copied into the packet. The driver and verifier deny Python socket
connections and refuse missing cache members.

The matched grid ran nine fresh sequential CPU processes, 82.896s total
wall time including data load/startup. Driver-only adapter time and process CPU
and peak RSS telemetry are retained separately in each shard/EXECUTIONS.json.
Timing is telemetry, not a qualified performance result. No repeated or failed
matched shard was discarded. The earlier deliberate refusal is retained.

## Reproduce and validate

From an ASI checkout containing the execution commit, existing venv and cached
public data at the path above, run from the repository root:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_DEFAULT_PRNG_IMPL=threefry2x32 JAX_RANDOM_SEED_OFFSET=0
.venv/bin/python outputs/ipmnist_screening/identmap_retention_dev_20261031/verify_study.py
.venv/bin/python -O outputs/ipmnist_screening/identmap_retention_dev_20261031/verify_study.py
```

The verifier independently re-derives the held-out rows, permutations, all
per-seed online example orders and protocol commitments, checks frozen
checkpoints, integer score counts and recurrence arithmetic, and reproduces the
full summary. It does not rerun learning or attest execution. To rerun learning,
use the exact nine commands in EXECUTIONS.json from a clean checkout of the
execution commit, with NEW output paths. Never overwrite these artifacts.
The raw adapter report stores aggregate measurements and trace commitments,
not a reconstructible per-observation trace or execution attestation.

Checks on the measured source:

- Adapter:62 passed,14.09s, including32 added cases; initial regression failed
  with the expected deliberate NotImplementedError before the hook.
- Other retention/provenance/hostile-input modules:81 passed,4.63s.
- RLS/registry regressions:54 passed,357 deselected,11.79s.
- Total unique passing software tests:197. Repository Ruff passes.
- Focused strict mypy with imported modules silent:2 files pass. Full repository
  mypy reports27 existing errors in9 unrelated files. Its complete output is
  identical with the main screening source substituted through mypy's shadow
  file option; this is not a full-typecheck pass. No unrelated repair is made.
- Evidence registry exits2 before and after: the same5 preexisting invalid
  claims, unchanged errors/artifact/protocol/seed/threshold/source bindings.
  Only its operational validation-head field advances; no evidence is promoted.
- Normal and optimized verifier reproduce SUMMARY.json. Hosted CI and the full
  repository test suite have not been run.

## Limits and attribution

One A/B/A schedule, three development seeds, one512-case held-out sentinel set
and no task identity given to the learner. These numbers are permanently
nonpromoting; they do not establish external retention, erased memory, absence
of forgetting, robotics/control transfer, whole-life quality, SOTA or scientific
acceptance. No frozen-v3/EMNIST/forager lane or retired study was resumed.

Authoring: OpenAI / gpt-6.1-sol / codex-desktop26.1027.30824.
Compute/data/packages spend:$0. No new package, paid service or model API used.

Protocol wording clarification: PLAN.md called the third arm the strongest
available registered current-source control. Its exact name was fixed before
measurement. It is a contextual control in this source revision, not a claim
that it is the global campaign winner: newer unmerged anneal/weight-decay
results were not ported or compared. This study ranks no new algorithm and
makes no best-internal-baseline improvement claim. No arm, seed, threshold,
primary metric or run was changed.

Packet integrity, from this directory: `shasum -a 256 -c MANIFEST.sha256`.
