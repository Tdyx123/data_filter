# LIBERO Evaluation Performance Design

## Goal

Reduce the wall-clock time of the default `scripts/evaluate_libero_octo_small.sh`
evaluation without reducing the ten-task, 150-episode-per-task protocol or changing
its three fixed seeds, 960-step limit, task order, per-task output layout, or
single-GPU default.

The optimized evaluator must preserve task, seed, initial-state, and episode-ID
mapping. Dynamic batching may produce small floating-point trajectory differences
from the previous implementation, but result schemas and success-rate semantics
must remain comparable.

## Selected Approach

Use a single-process suite evaluator that loads and verifies the checkpoint once,
then evaluates the selected tasks sequentially while reusing the loaded policy.
Within each task, keep one vector environment at its effective startup capacity,
run only unfinished workers, consume success directly from `step()`, and cache
task-invariant language features.

This approach was selected over:

- a conservative patch that retained ten model loads and most subprocess traffic;
- a custom LIBERO worker protocol that would suppress intermediate camera returns
  or add shared-memory observations, because it would depend on private pinned
  LIBERO/robosuite behavior and cannot be validated in the current environment.

The implementation must not modify `third_party/LIBERO`.

## Architecture and Interfaces

### Shared evaluation context

Split evaluation into two layers:

1. A shared preparation layer validates dependencies, resolves and hashes the
   checkpoint and statistics, configures LIBERO, loads the model/tokenizer, and
   moves the policy to the requested device exactly once.
2. A task layer resolves one LIBERO task, creates and closes its environment,
   performs its rollouts, and writes its existing task-local artifacts.

`evaluate_checkpoint(settings, preflight_only=False)` remains available and
delegates through the shared layer for backward compatibility. A suite entrypoint
accepts an ordered sequence of task/output-directory pairs and uses one shared
context for all pairs.

### Shell compatibility

`scripts/evaluate_libero_octo_small.sh` retains its existing public arguments.
Explicit `--task-name` continues to invoke the single-task path. Index mode passes
the selected tasks and their `task-N` output directories to one suite process
instead of launching one Python process per task.

The suite preserves current failure behavior:

- a normal task failure is recorded and later tasks continue;
- a simulation infrastructure or unrecoverable worker-shutdown failure aborts the
  suite with exit code 3;
- any normal task failures produce exit code 1 after all runnable tasks finish;
- argument or evaluation contract errors retain exit code 2 for single-task use.

### Output artifacts

Each task continues to produce the same `results.json`, `episodes.jsonl`,
`episodes.partial.jsonl`, `failure.json`, `preflight.json`, and optional videos in
its existing output directory.

Index mode additionally writes `<output-root>/suite_results.json` containing the
ordered task statuses, task output directories, shared setup duration, total wall
time, and final suite status. This file is additive and does not replace any
task-local result.

## Rollout and Scheduling

Create a vector environment once per task using the configured `num_envs` and
existing startup backoff. Treat the successful startup size as immutable task
capacity. For a smaller final episode group, activate only worker IDs
`0..batch_size-1`; do not rebuild the environment or lower the capacity used by
subsequent seeds.

During rollout, maintain an ordered mapping among active worker IDs, observations,
episode IDs, initial-state IDs, and per-episode step counts. At every action chunk:

1. Predict actions only for active observations.
2. Call vector `step(actions, id=active_worker_ids)` for active workers only.
3. Read success from the returned `done` array for the pinned four-element API, or
   `terminated` for a five-element API.
4. Record each newly successful episode's step and remove that worker from later
   simulation, image transfer, preprocessing, and inference.
5. Continue unfinished workers until their individual step count reaches
   `max_steps`.

The evaluator must not call `environment.check_success()` from the rollout. Video
capture receives the active episode mapping and stops recording an episode when
that episode stops running.

Settling and initialization also accept active worker IDs so remainder groups can
reuse a larger environment safely. Seed, initial-state, and episode numbering
remain identical to the current protocol.

## Language Cache and Model Behavior

Tokenize each distinct task instruction once. Under the same inference/autocast
settings used by action prediction, run the frozen T5 encoder, language projection,
and positional addition once and cache the resulting single-example GPU feature
plus attention mask.

Each action prediction expands the cached feature to the active batch without
copying it, then performs the current visual encoders, proprio projection,
Transformer, and 20-step diffusion sampler normally. The non-cached model path
remains supported for training and existing direct model tests.

The cache is scoped to the loaded evaluation policy, has at most one entry per
evaluated task instruction, and is discarded with the policy. It must not be
serialized into checkpoints.

## Performance Reporting

Preserve `runtime.elapsed_seconds` and add a `runtime.timings` object to task
reports with accumulated seconds for:

- environment startup;
- settle/reset;
- observation preprocessing and host-to-device transfer;
- policy inference and device-to-host transfer;
- simulation stepping;
- result-file I/O;
- environment shutdown.

The suite report records shared setup and total wall time. Timing must use
`time.monotonic()` and must not introduce explicit per-operation CUDA
synchronization; the existing action transfer back to CPU provides the inference
completion boundary.

## Error Handling

Every task environment is closed in a `finally` path. Partial task results and
timings are written on rollout failure when task output validation has succeeded.
Suite-level preparation failures abort before any task begins. After a recoverable
task failure, the shared policy remains loaded and the next task runs. Existing
bounded worker shutdown and startup backoff behavior remain in force.

The suite result is written atomically after every task status change so an
interrupted multi-task evaluation retains completed-task and failure information.

## Testing and Acceptance Criteria

Automated tests must prove:

- index-mode shell execution invokes one suite process while explicit task-name
  execution retains the single-task command;
- checkpoint resolution, hashing, model construction, and device transfer occur
  once for multiple tasks;
- a successful worker is never stepped or included in later policy batches;
- rollout consumes `done`/`terminated` and never calls `check_success()`;
- an effective capacity such as 12 remains 12 after a two-episode remainder and
  across later seeds, with one environment construction per task;
- episode IDs, initial-state IDs, seeds, first-success steps, and termination values
  remain correct under dynamic active batches;
- repeated predictions for one instruction tokenize and encode language once,
  while a second instruction creates one additional cache entry;
- existing single-task reports and exit codes remain compatible, and suite failure
  summaries are atomic and ordered;
- timing fields are present, finite, non-negative, and additive counters are
  accumulated across task batches.

Run the focused evaluator and launcher tests first, followed by the full available
test suite. A real LIBERO smoke comparison is required when the pinned
robosuite/MuJoCo/EGL environment is available: both implementations must use the
same three smoke seeds and task index 5, produce the same episode mapping and
schema, and have no unexpected success-rate difference. The current development
interpreter does not provide `robosuite`, so lack of that optional integration run
must be reported rather than treated as a passing benchmark.

## Out of Scope

- Reducing task count, episodes, seeds, action horizon, or maximum steps.
- Automatic or explicit multi-GPU task scheduling.
- Modifying pinned LIBERO source code.
- A private chunk-step worker command, suppressed intermediate camera rendering,
  or shared-memory observation transport.
- Claims about end-to-end speedup without a real GPU/EGL benchmark.
