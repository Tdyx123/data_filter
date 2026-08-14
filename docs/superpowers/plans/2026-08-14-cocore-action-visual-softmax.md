# Cocore Action–Visual Softmax Implementation Plan

## Global Constraints

- Work directly on `main` because the user explicitly requested it; preserve unrelated files.
- Follow strict TDD: add focused tests, observe the expected failure, then add production code.
- Prototype learning uses every stride-1 eight-frame trajectory window `[t, t+7]`.
- Window action labels use `state[t] -> state[t+7]`; window visual features are the L2-normalized mean of eight per-frame CLIP embeddings.
- Retain actions when `count >= max(40, ceil(0.005 * W))`, where `W` is the total trajectory-window count.
- A rare action maps to every retained atomic-action subset with maximum cardinality; parent probabilities are normalized retained raw counts. Fall back to `stop=1` only when no non-empty retained subset exists.
- Rare windows participate in every parent action KMeans with their parent probability as `sample_weight`.
- Effective action mass is `M_a = sum p(a|window)` and cluster count is `K_a = min(16, 1 + floor(log2(M_a)))`.
- A candidate clip unions and canonicalizes the atomic actions from `state[0] -> state[7]` and `state[7] -> state[14]`.
- Candidate visual features are the L2-normalized mean of all fifteen per-frame CLIP embeddings.
- Every action parent assigns to all its centers with stable `softmax(-squared_distance / 0.1)`; leaf weights equal action probability times conditional visual probability and sum to one.
- Schema version is 4, Cocore version is 0.8.0, and the graph directory is `graph-13-motion-softmax`.
- Replace `visual_half_embeddings.npy` with `visual_clip_embeddings.npy`; replace `half_action_labels.npy` with `clip_action_labels.npy`.
- Remove action/distance weight artifacts and exported fields. Keep prototype indices, probabilities, labels, centers, action labels, and raw clip action label.
- Schema-3 caches are incompatible and must not be read as schema 4.

## Task 1: Core action distributions and visual probability primitives

Implement the pure, independently testable behavior in `cocore/prototypes.py` and focused tests in `tests/test_cocore_prototypes.py`.

- Replace the old strict half-percent catalog and one-deletion/dominance/fallback-weight softening with the threshold and maximum-retained-subset distribution in Global Constraints.
- Add canonical atomic-action union for the two clip halves.
- Add the effective-mass logarithmic cluster-count function.
- Add stable all-center squared-distance softmax with temperature 0.1.
- Refactor the catalog dataclasses so schema-4 metadata can record raw counts, retention, parent assignments, effective mass, requested/actual centers, and leaf metadata.
- Preserve deterministic ordering and validation of malformed labels and non-finite inputs.
- Do not yet alter encoding or pipeline artifacts.

Verification: run `tests/test_cocore_prototypes.py` and record RED/GREEN evidence.

## Task 2: Full-trajectory weighted prototype learning and clip encoding

Implement the data path in `cocore/encoding.py`, `cocore/prototypes.py`, and their focused tests.

- Replace half-clip candidate embeddings with one L2-normalized fifteen-frame CLIP mean per candidate.
- Reuse complete per-episode frame caches. Generate eight-frame stride-1 means episode-by-episode using prefix sums; do not materialize a `[windows, 8, dim]` tensor or persist a global sliding-window matrix.
- Build raw action counts from every valid trajectory window, construct action distributions, and feed every window to all parent action buckets with parent probability as KMeans `sample_weight`.
- Use effective mass for K and `seed + action_id` for deterministic MiniBatchKMeans.
- Construct each candidate raw action from the canonical union of the two state deltas, compute its action distribution, apply all-center visual softmax using its fifteen-frame mean, and emit normalized leaf indices/weights.
- Reject no-window data, non-finite inputs, missing cache data, non-stop data without a retained non-stop action, empty parent distributions, and non-normalized results.

Verification: run `tests/test_cocore_encoding.py tests/test_cocore_prototypes.py` and record RED/GREEN evidence.

## Task 3: Pipeline schema-4 artifacts, outputs, and validation

Integrate the new model through `cocore/pipeline.py` and `tests/test_cocore_pipeline.py`.

- Rename the encoded candidate artifact and all fingerprints/manifests to `visual_clip_embeddings` with fifteen-frame mean semantics.
- Write graph artifacts under `graph-13-motion-softmax`, schema version 4.
- Write `clip_action_labels.npy`; stop writing half labels and action/distance weight arrays.
- Update catalog serialization, graph loading, cache requirements, fingerprints, validation/replay, selection rows, Parquet/JSONL fields, manifests, and report metadata.
- Export final prototype labels/probabilities, prototype action labels, and `raw_action_label`; remove decomposed action/distance fields.
- Validation must recompute counts, parent distributions, weighted centers/cluster metadata, clip raw actions, all-center probabilities, and normalization strongly enough to reject tampering.
- Explicitly reject incompatible schema-3 output.

Verification: run `tests/test_cocore_pipeline.py tests/test_cocore_core.py` and record RED/GREEN evidence.

## Task 4: Version, CLI/config, Bridge contract, docs, and regression

Complete the public migration across Cocore and Cocore Bridge V2.

- Bump Cocore to 0.8.0 and update CLI/config tests and shipped manifests/config contracts.
- Keep only KMeans `batch_size` and `max_iter` under prototypes; threshold, cluster formula, cap, and temperature stay fixed and are recorded in catalog/manifest rather than exposed as configuration.
- Update Cocore and Cocore Bridge V2 documentation and tests for schema 4, the new graph directory, artifacts, exported fields, cache incompatibility, and the trajectory-window/full-center-softmax algorithm.
- Update any remaining repository references that would make Cocore or Bridge load old paths/fields.
- Run the complete targeted Cocore/Bridge suite from the baseline command; then run the broader non-real-data test set if feasible.

Verification: record focused RED/GREEN evidence, targeted regression totals, and any environment-limited tests.
