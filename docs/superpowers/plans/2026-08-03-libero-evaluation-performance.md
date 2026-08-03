# LIBERO Evaluation Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the single-GPU LIBERO-10 evaluator reuse one loaded policy, avoid work for completed episodes, retain vector capacity across remainder batches, cache language conditioning, and report actionable timings without changing the formal evaluation protocol.

**Architecture:** Introduce a prepared evaluation context shared by sequential tasks and a suite CLI used by the existing shell launcher. Keep each task's simulator isolated and short-lived, but reuse the checkpoint and policy; within a task, reuse one fixed-capacity vector environment and operate on active worker IDs only. Preserve the existing single-task Python API and task-local output artifacts.

**Tech Stack:** Python 3.12, PyTorch, NumPy, argparse, pytest, Bash, pinned LIBERO `SubprocVectorEnv`.

## Global Constraints

- Preserve ten tasks, 150 episodes per task, seeds `0,1,2`, `max_steps=960`, action horizon 8, task order, and single-GPU default.
- Preserve task/seed/init-state/episode-ID mapping and task-local JSON schemas; small trajectory-level floating-point differences are allowed.
- Do not modify `third_party/LIBERO` and do not add a private worker protocol, shared-memory transport, intermediate-render suppression, or multi-GPU scheduling.
- Preserve explicit `--task-name` behavior and existing simulation infrastructure exit code 3.
- Write tests before production changes and observe each focused test fail for the intended missing behavior.

---

### Task 1: Cache language conditioning and collect policy timings

**Files:**
- Modify: `src/octo_small_libero/torch_model.py`
- Modify: `src/octo_small_libero/evaluation.py`
- Test: `tests/test_octo_small_pytorch.py`
- Test: `tests/test_octo_small_evaluation.py`

**Interfaces:**
- Produces: `OctoSmallPolicy.encode_language(input_ids, attention_mask) -> torch.Tensor`.
- Produces: `OctoSmallPolicy.encode_observation(batch, *, language_embedding=None) -> torch.Tensor`.
- Produces: `EvaluationTimings` with non-negative accumulated fields and `as_dict()`.
- Produces: `LoadedCheckpointPolicy.predict_action_chunk(...)` that caches one encoded language row per instruction and adds preprocessing/inference durations to its timing accumulator.

- [ ] **Step 1: Write failing model tests for injected language embeddings**

Add a test that wraps the test model's text encoder with a call counter, calls
`sample_actions` twice with an explicitly precomputed `language_embedding`, and
asserts the text encoder is not called by either sample. Also assert that calling
`encode_language` increments the counter exactly once and returns shape
`(batch, language_tokens, hidden_size)`.

```python
language = model.encode_language(batch["language_input_ids"], batch["language_attention_mask"])
calls_after_encoding = text_encoder.calls
model.sample_actions(batch, generator=generator, language_embedding=language)
assert text_encoder.calls == calls_after_encoding
```

- [ ] **Step 2: Run the focused model test and verify RED**

Run:

```bash
pytest -q tests/test_octo_small_pytorch.py -k language_embedding
```

Expected: failure because `encode_language` and the `language_embedding` keyword do not exist.

- [ ] **Step 3: Implement the minimal model split**

Move the frozen T5 call, language projection, and language positional addition from
`encode_observation` into `encode_language`. Add an optional language embedding to
`encode_observation` and `sample_actions`; use the existing path when it is absent
so training and existing tests remain compatible.

```python
def encode_language(self, input_ids, attention_mask):
    with torch.no_grad():
        value = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    value = self.language_projection(value)
    return value + self.language_pos_embedding[:, : value.shape[1]]
```

- [ ] **Step 4: Run model tests and verify GREEN**

Run:

```bash
pytest -q tests/test_octo_small_pytorch.py
```

Expected: all Octo-small PyTorch tests pass.

- [ ] **Step 5: Write failing evaluation tests for cache and timing counters**

Add a fake tokenizer and model that count tokenization and `encode_language` calls.
Call the loaded policy twice with one instruction and once with a second instruction.
Assert counts are `2`, cached tensors are expanded to each active batch size, and
`preprocessing_seconds` / `inference_seconds` are finite and non-negative.

- [ ] **Step 6: Run the cache/timing test and verify RED**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py -k "language_cache or evaluation_timings"
```

Expected: failure because the loaded policy has no cache or timing accumulator.

- [ ] **Step 7: Implement `EvaluationTimings` and policy cache**

Add an accumulator with these exact report keys:

```python
environment_startup_seconds
settle_seconds
preprocessing_seconds
inference_seconds
simulation_step_seconds
result_io_seconds
environment_shutdown_seconds
```

Tokenize a single instruction row, move it to the policy device, encode it under
the same inference/autocast settings as sampling, store `(embedding, mask)` by
instruction, and expand both views to the current batch. Time host preprocessing
through device transfer separately from inference through `.cpu().numpy()`.

- [ ] **Step 8: Run focused evaluation tests and commit**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py -k "language_cache or evaluation_timings or model_batch"
git add src/octo_small_libero/torch_model.py src/octo_small_libero/evaluation.py tests/test_octo_small_pytorch.py tests/test_octo_small_evaluation.py
git commit -m "perf: cache LIBERO language conditioning"
```

Expected: focused tests pass and only Task 1 files are committed.

---

### Task 2: Roll out only active workers and consume step success

**Files:**
- Modify: `src/octo_small_libero/evaluation.py`
- Test: `tests/test_octo_small_evaluation.py`

**Interfaces:**
- Changes: `rollout_action_chunks(..., worker_ids: Sequence[int] | None = None, frame_callback: Callable[[Any, Sequence[int], Sequence[int]], None] | None = None)`.
- Preserves: return value `tuple[list[dict[str, Any]], Any]` and episode result fields.
- Consumes: vector environment `step(actions, id=worker_ids)` and its four- or five-element result.

- [ ] **Step 1: Replace the rollout fake with an ID-aware environment and write RED tests**

Create a fake whose `step` records worker IDs, returns success in `done`, and raises
if `check_success()` is called. Use workers with success steps 2 and 5. Assert the
policy batch sizes shrink from 2 to 1, worker 0 is absent after step 2, worker 1
continues through step 5, and recorded first-success steps remain 2 and 5.

Also add a five-element return test proving `terminated` is treated as success.

- [ ] **Step 2: Run rollout tests and verify RED**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py -k "chunk_rollout or terminated"
```

Expected: failure because rollout does not pass IDs and still calls `check_success()`.

- [ ] **Step 3: Implement active-worker scheduling**

Track active positions, worker IDs, observations, actions, and step counts as aligned
arrays/lists. On each substep, call only active worker IDs, read `done` or
`terminated`, update episode state, capture active video frames, and filter all
aligned values before the next substep or model call. Mark remaining episodes
`max_steps` when their own count reaches the limit.

- [ ] **Step 4: Run rollout and report tests and verify GREEN**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py -k "chunk_rollout or terminated or result_schema"
```

Expected: all selected tests pass with no call to `check_success()`.

- [ ] **Step 5: Commit active-worker rollout**

```bash
git add src/octo_small_libero/evaluation.py tests/test_octo_small_evaluation.py
git commit -m "perf: stop completed LIBERO workers"
```

---

### Task 3: Reuse fixed environment capacity and introduce prepared context

**Files:**
- Modify: `src/octo_small_libero/evaluation.py`
- Test: `tests/test_octo_small_evaluation.py`

**Interfaces:**
- Produces: `PreparedEvaluation` containing checkpoint, statistics, configured LIBERO paths/commit, loaded policy, package metadata, and shared setup duration.
- Produces: `prepare_evaluation(settings, *, configuration_output_dir=None) -> PreparedEvaluation`.
- Produces: `evaluate_prepared_task(prepared, settings, *, preflight_only=False) -> dict[str, Any]`.
- Preserves: `evaluate_checkpoint(settings, *, preflight_only=False)` as a wrapper.
- Changes: `settle_vector_environment(..., worker_ids: Sequence[int] | None = None)` initializes only the active IDs.

- [ ] **Step 1: Write a failing fixed-capacity regression**

Use `episodes=150`, a fake startup backoff result of capacity 12, and a fake vector
environment that records construction, reset, initialization, and step IDs. Assert
one construction for the task, active group sizes repeat as
`[12, 12, 12, 12, 2]` for every seed, and the first group of seeds 1 and 2 returns
to 12 instead of remaining at 2.

- [ ] **Step 2: Run the capacity regression and verify RED**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py -k fixed_environment_capacity
```

Expected: failure because the current loop rebuilds on size changes and permanently lowers `active_num_envs`.

- [ ] **Step 3: Implement fixed-capacity grouping**

Start one environment before the seed loop using the first full group and existing
backoff. Keep the returned capacity unchanged. For each group, select worker IDs
`range(batch_size)` and pass them to settle and rollout. Record active group sizes
separately from effective capacity and close the environment once per task.

- [ ] **Step 4: Run capacity tests and verify GREEN**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py -k "fixed_environment_capacity or automatically_reduces_parallelism"
```

- [ ] **Step 5: Write failing prepared-context reuse tests**

Patch dependency validation, checkpoint resolution, statistics loading, LIBERO
configuration, and policy loading with counters. Prepare once, evaluate two fake
tasks, and assert every shared operation count is one while task resolution and
environment construction count are two. Assert `evaluate_checkpoint` still loads
and returns one task report.

- [ ] **Step 6: Run context tests and verify RED**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py -k "prepared_evaluation or checkpoint_wrapper"
```

Expected: failure because prepared context APIs do not exist.

- [ ] **Step 7: Extract shared preparation and task execution**

Move dependency/checkpoint/statistics/LIBERO/model setup into `prepare_evaluation`.
Move task resolution, preflight, output validation, environment lifecycle, rollout,
task reporting, and task timing into `evaluate_prepared_task`. Make
`evaluate_checkpoint` prepare and delegate without changing its signature.

Instrument startup, settle, simulation, task result I/O, and shutdown using the
shared `EvaluationTimings`; include `runtime.timings` while preserving
`runtime.elapsed_seconds`.

- [ ] **Step 8: Run evaluator tests and commit**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py
git add src/octo_small_libero/evaluation.py tests/test_octo_small_evaluation.py
git commit -m "refactor: reuse prepared LIBERO evaluator"
```

Expected: the full evaluator unit test file passes.

---

### Task 4: Add suite CLI and switch the shell launcher to one process

**Files:**
- Create: `src/octo_small_libero/evaluate_suite.py`
- Modify: `src/octo_small_libero/evaluate.py`
- Modify: `scripts/evaluate_libero_octo_small.sh`
- Modify: `tests/test_evaluate_libero_script.py`
- Test: `tests/test_octo_small_evaluation.py`

**Interfaces:**
- Produces: suite CLI repeated option `--task INDEX NAME` plus existing evaluator options; in suite mode `--output-dir` is the result root.
- Produces: `SuiteTask(index: int, name: str, output_dir: Path)`.
- Produces: `evaluate_suite(settings_template, tasks, *, preflight_only=False) -> dict[str, Any]`.
- Produces: additive `<output-root>/suite_results.json` with schema version 1, ordered task entries, `shared_setup_seconds`, `total_elapsed_seconds`, and suite status.

- [ ] **Step 1: Write failing shell invocation tests**

Update the fake Python assertions so default and selected index modes expect exactly
one call beginning `-m octo_small_libero.evaluate_suite`, containing ordered
`--task <index> <name>` triples and one root `--output-dir`. Keep the explicit
task-name test expecting `-m octo_small_libero.evaluate` and its direct output path.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
pytest -q tests/test_evaluate_libero_script.py -k "default_runs or selected_indexes or explicit_task_name"
```

Expected: index tests fail because the shell starts one evaluator per task.

- [ ] **Step 3: Refactor common CLI settings and implement suite tests**

Extract common parser arguments and settings construction from `evaluate.py` so the
single and suite modules share defaults. Add suite unit tests with two tasks where
one normal task fails and the second continues, plus an infrastructure failure that
aborts immediately. Verify the suite JSON is rewritten after each status change
and task order is stable.

- [ ] **Step 4: Run suite tests and verify RED**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py -k evaluation_suite
```

Expected: failure because the suite module and API do not exist.

- [ ] **Step 5: Implement suite module and atomic suite report**

Parse task pairs, derive `task-INDEX` output paths, prepare one shared context,
print per-task progress, delegate each task, and update the suite report atomically.
Continue ordinary task failures, immediately re-raise simulation infrastructure
failures, and return/report failed task indexes so CLI exit status is 1 when needed.

- [ ] **Step 6: Switch index-mode shell execution**

Replace the shell's per-task Python loop with construction of one suite argument
array. Preserve all input validation and direct single-task `exec`. Propagate the
suite exit status without translating code 3.

- [ ] **Step 7: Run launcher and evaluator tests and verify GREEN**

Run:

```bash
pytest -q tests/test_evaluate_libero_script.py tests/test_octo_small_evaluation.py
```

Expected: all non-GPU launcher/evaluator tests pass.

- [ ] **Step 8: Commit suite evaluator**

```bash
git add src/octo_small_libero/evaluate.py src/octo_small_libero/evaluate_suite.py scripts/evaluate_libero_octo_small.sh tests/test_evaluate_libero_script.py tests/test_octo_small_evaluation.py
git commit -m "perf: reuse policy across LIBERO tasks"
```

---

### Task 5: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-03-libero-evaluation-performance-design.md` only if implementation reveals a necessary clarification, without changing approved scope.

**Interfaces:**
- Documents: unchanged commands and output layout, additive suite report, timing fields, dynamic active-worker behavior, and real smoke benchmark prerequisites.

- [ ] **Step 1: Update evaluator documentation**

Explain that index mode uses one persistent model process, tasks remain sequential,
successful episodes stop consuming compute, environment capacity survives remainder
groups, and `suite_results.json` / `runtime.timings` are available for diagnosis.

- [ ] **Step 2: Run static and focused verification**

```bash
bash -n scripts/evaluate_libero_octo_small.sh
python3 -m octo_small_libero.evaluate --help
python3 -m octo_small_libero.evaluate_suite --help
pytest -q tests/test_evaluate_libero_script.py tests/test_octo_small_evaluation.py tests/test_octo_small_pytorch.py
git diff --check
```

Expected: all commands succeed with no syntax or whitespace errors.

- [ ] **Step 3: Run the full available test suite**

```bash
pytest -q
```

Expected: all tests that do not require unavailable external data/GPU dependencies pass; expected skips are reported.

- [ ] **Step 4: Check optional real LIBERO smoke prerequisites**

Run the existing two-task GPU smoke only when `OCTO_LIBERO_EVAL_CHECKPOINT`,
`OCTO_LIBERO_EVAL_STATISTICS`, `LIBERO_ROOT`, CUDA, robosuite, MuJoCo, and EGL are
available. Otherwise record exactly which prerequisite is absent and do not claim
an end-to-end speedup.

- [ ] **Step 5: Review only task-owned changes and commit docs**

```bash
git status --short
git diff -- README.md src/octo_small_libero/evaluate.py src/octo_small_libero/evaluate_suite.py src/octo_small_libero/evaluation.py src/octo_small_libero/torch_model.py scripts/evaluate_libero_octo_small.sh tests/test_evaluate_libero_script.py tests/test_octo_small_evaluation.py tests/test_octo_small_pytorch.py
git add README.md
git commit -m "docs: explain optimized LIBERO evaluation"
```

Expected: unrelated SQCN, TDUS, and selection changes are absent from every task commit.
