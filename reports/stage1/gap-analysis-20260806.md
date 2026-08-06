# Stage 1 gap analysis (2026-08-06)

Snapshot: branch `feat/stage0-completion`, HEAD `975f83c`, clean worktree.
Local environment: Python 3.12.4, torch 2.10.0+cpu, package imports OK
(`param_importance_nlp`). No server SSH check and no full pytest run were
performed here; both are separate work items (server/entry gate, test baseline).

## 1. Executive summary

The Stage 1 *implementation surface* is largely present in-tree: the math core
(raw / double / equal-U / weighted-U / cross-U, FP64 oracle comparisons,
sufficient statistics, accumulator, registry, losses), the training engine with
online importance tracking, the DDP executor/reducer, the gradient
scale/clip/skip state machine, the optimizer bridge, checkpoint store, and the
task-catalog runners for `stage1.01`–`stage1.11` are all committed, and unit
tests exist for each of these.

What is **not** present is the *formal Stage 1 execution state*: no S1.1 entry
baseline snapshot, no dedicated Stage 1 branch, no Stage 1 worklog, no
stage1 gate evidence under `reports/stage1`, and no server-side evidence for
the GPU-bound gates (G1-SINGLE, G1-DDP, G1-NUMERIC, G1-RESUME). The S1.7
formal config still carries `FILL_*` asset placeholders and `estimator_decision_ref:
null`, so it cannot run before the G10 handoff is consumed and the S1.5 estimator
decision is frozen.

Bottom line: Stage 1 is at "code + unit tests written, gates unexecuted".
The fastest correct path is S1.1 → freeze contract → run CPU gates
(G1-CONTRACT…G1-STEP) → consume G10 handoff → S1.7/S1.8/S1.9 → S1.10 → S1.11.

## 2. Subtask / gate status map

| Subtask | Gate | Implementation | Formal evidence | Notes |
|---|---|---|---|---|
| S1.1 Entry baseline & math contract | G1-ENTRY, G1-CONTRACT | Contract schema exists (`schemas/stage1/contract-v1.json`, `contracts/`); math doc present | **MISSING** | No entry snapshot JSON, no stage1 worklog, no dedicated branch; server/Agent hash sync and GPU health not re-verified this run |
| S1.2 Architecture & parameter registry | G1-REGISTRY | `core/registry.py` (aliases, storage-overlap rejection, eligibility, 3 hashes); tests `test_core_registry_and_loss.py` | **NOT EXECUTED** | Registry unit coverage exists; no formal gate record |
| S1.3 Deterministic fixtures & oracles | G1-ORACLE | `core/oracles.py` (Constant/Quadratic/ZeroMeanNoise fixtures, finite differences, FP64 two-branch comparator); tests `test_core_oracles.py` | **NOT EXECUTED** | No committed fixture manifest under `fixtures/stage1` |
| S1.4 Loss reduction & gradient scale | G1-GRAD | `core/losses.py` (LossBatch, causal-LM shift, pre-shifted, classification), `runtime/gradients.py` scale/unscale/skip | **NOT EXECUTED** | Unit coverage via `test_core_*` / `test_runtime_gradients.py` |
| S1.5 raw / double / U cores | G1-EST | `core/estimators.py`, `core/sufficient_statistics.py` incl. weighted-U and cross-U; `EstimatorResult` claim guard rails | **NOT EXECUTED** | `test_core_estimators_and_accumulator.py` covers ordered-pair oracle, M=1/M=2, negative values, clip claims |
| S1.6 Step integration & accumulators | G1-STEP | `core/accumulator.py` (signed/positive/negative_mass/absolute/raw/raw_clipped/movements, v1→v2 migration), `runtime/training.py` engine | **NOT EXECUTED** | Engine + `OnlineImportanceTracker` + `StepTransaction` + skip path present |
| S1.7 Pythia-14M single-GPU | G1-SINGLE | `configs/run-ready/v2/stage1-pythia14m-formal.yaml` + layer config; providers (`huggingface_offline`, `pythia_mmap`) | **MISSING** | Config has `FILL_MODEL_REVISION`, `FILL_BASE_INITIALIZATION_HASH`, `FILL_DATA_REVISION`; requires G10 handoff IDs + healthy GPU |
| S1.8 DDP & grad accumulation | G1-DDP | `runtime/distributed_training.py`, `runtime/reducers.py`, engine microbatch collection under `no_sync`; `test_runtime_distributed.py` | **MISSING** | Requires 4 healthy GPUs + fresh NCCL smoke (server work) |
| S1.9 Precision/clip/optimizer boundaries | G1-NUMERIC | `runtime/gradients.py` (SCALED→UNSCALED→FINITE→CLIPPED/SKIPPED), `runtime/optimizer.py` (AdamW/SGD bridge, decay split), `core/estimators.global_clip_factor` | **MISSING** | BF16/GPU-scale/unscale equivalence needs server CUDA |
| S1.10 Checkpoint/resume & artifacts | G1-RESUME | `runtime/checkpoint.py`, `runtime/checkpoint_group.py`, engine save/resume incl. accumulator + RNG + cursor | **MISSING** | Fresh-process single/4-card resume evidence required on server |
| S1.11 Reporting & exit gate | G1-EXIT | `analysis/` modules + `experiments/stage01_task_runners.py` exit-report evidence path | **MISSING** | Requires all upstream gates pass |

## 3. Concrete gaps to close (dependency-ordered)

1. **S1.1 entry baseline (blocker for everything formal)**
   - Create `worklogs/2026-08-06-stage1-entry.md`; record local branch/HEAD/tree
     (done here: `feat/stage0-completion` @ `975f83c`, clean), server branch/HEAD,
     per-file SHA-256 of the five `Agent/*.md` on both ends, `$DATA_ROOT` facts,
     cache/tmp resolution, active downloads/GPU processes.
   - Re-verify healthy GPU candidates and 4-card NCCL smoke (old reports do not
     count; see plan README §11).
   - Verify the user's `docs/mathematics.md` character/EOF fix is committed and
     unchanged (git log shows it was added in `beb5c05`; review content).
   - Create the dedicated Stage 1 branch (plan: after entry baseline commit).
   - Produce machine-readable G1-ENTRY / G1-CONTRACT gate records (runner:
     `stage1.01_entry_and_contract`, see `experiments/stage01_task_runners.py`).
2. **S1.2–S1.6 CPU gates**: run the task-catalog runners/tests for
   `stage1.02`–`stage1.06` and persist gate records + tolerance tables under
   `reports/stage1`. Unit tests already cover most oracle math; the missing part
   is the formal record chain (requirements→test→artifact matrix, per-parameter
   error tables).
3. **Consume the G10 handoff**: pull environment hash, manifest refs, asset IDs
   and GPU UUIDs from the immutable `stage0-g10-stage1-handoff-v1` artifact;
   replace the `FILL_*` placeholders in `formal-stage1-pythia14m.yaml` and freeze
   the estimator decision (`estimator_decision_ref`), which the formal spec
   currently requires (`requires_estimator_decision: true`).
4. **Server GPU gates (S1.7→S1.8/S1.9→S1.10)**: single-card Pythia-14M debug
   run, 4-card DDP equivalences (route A vs D, B/C/D microbatch consistency),
   scale/unscale/clip/BF16 smoke, fresh-process single- and 4-card resume
   comparisons. These are the only blockers to claiming G1-EXIT.
5. **S1.11**: aggregate all gate records, charts from source data, whitespace/
   link/schema checks, three-end sync (local/GitHub/server at same commit),
   `Agent/*.md` hash parity, then final exit report.

## 4. Directory-convention gaps (to be frozen in S1.1)

- `tests/stage1/`, `configs/stage1/`, `fixtures/stage1/`, `reports/stage1/` do
  not exist yet; tests are flat under `tests/`, configs under `configs/run-ready/`.
  The plan treats these paths as suggestions ("实际路径在 S1.1 契约中冻结"), so the
  S1.1 contract must explicitly freeze the layout actually used.
- `fixtures/stage0/deterministic-training-v1.json` is the only committed fixture;
  `fixtures/stage1` needs the Pythia-14M manifest + fixed debug-sample IDs.

## 5. Risks / caveats

- Plan snapshot text ("repo has no src/tests/configs") is stale vs current HEAD;
  the README itself requires re-collecting state before any subtask, which this
  report does.
- No claim of pass/fail is made for any gate here: code+tests exist, but formal
  gate records are the acceptance unit, and none exist for Stage 1 yet.
- Test-baseline execution (pytest) is intentionally left to the parallel
  baseline task to avoid CPU contention.
