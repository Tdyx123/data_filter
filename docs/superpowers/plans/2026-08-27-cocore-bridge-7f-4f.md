# Cocore Bridge 7-Frame Clips and 4-Frame Action Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change only the `bridge_v2` Cocore profile to 7-frame candidate clips and 4-frame action-clustering windows while preserving LIBERO's 15/8-frame behavior.

**Architecture:** Add one profile-resolved temporal geometry contract and thread it through candidate indexing, encoding, motion prototypes, manifests, fingerprints, replay validation, diagnostics, and documentation. Existing public helpers keep LIBERO-compatible defaults; the Bridge adapter selects the new geometry through its fixed `bridge_v2` profile.

**Tech Stack:** Python 3.12, NumPy, PyArrow, scikit-learn, pytest, YAML.

## Global Constraints

- Only `bridge_v2` changes: clip length 7, anchors `0/3/6`, half windows `[0..3]` and `[3..6]`, action window length 4, and endpoint delta `state[t] -> state[t+3]`.
- LIBERO remains clip length 15, anchors `0/7/14`, half windows `[0..7]` and `[7..14]`, and action window length 8.
- Bridge action thresholds remain unchanged; trajectory window starts retain full coverage and maximum gap 3.
- No CLI or user-configurable YAML field is added.
- Cocore remains version `0.16.0`, prototype schema 10, and graph directory `graph-18-motion-hard-nearest-pca`; the Bridge adapter becomes `0.8.0`.
- Old Bridge 15/8 artifacts must fail fingerprint/manifest/catalog validation and require `--force` rebuilds; unchanged LIBERO cache contracts remain compatible.
- No selection, retention, cluster-count, or unrelated Cocore algorithm changes.

---

### Task 1: Profile temporal geometry and core data flow

**Files:**
- Create: `cocore/temporal.py`
- Modify: `cocore/index.py`, `cocore/encoding.py`, `cocore/prototypes.py`, `libero_motion_primitives/motion_primitives.py`
- Test: `tests/test_cocore_index.py`, `tests/test_cocore_encoding.py`, `tests/test_cocore_prototypes.py`, `libero_motion_primitives/tests/test_motion_primitives.py`

**Interfaces:**
- Produce immutable `TemporalGeometry` and `resolve_temporal_geometry(profile: str)` for `libero` and `bridge_v2`.
- Preserve default 15/8 behavior for existing index, encoding, and trajectory-window helper calls; accept explicit/profile-derived lengths for Bridge.
- Make `make_bridge_v2_config().horizon == 3` without changing any Bridge thresholds.

- [ ] Write failing behavioral tests for both profile geometries, 7-frame near-uniform indexing, four-frame overlapping means, and 4-frame full-coverage starts.
- [ ] Run the focused tests and confirm they fail for missing profile-aware behavior.
- [ ] Implement the minimal temporal geometry and thread it through core index, encoding, and prototype behavior.
- [ ] Run the focused unit tests and the existing LIBERO regressions.
- [ ] Self-review and commit the task.

### Task 2: Pipeline fingerprints, manifests, replay validation, and integration

**Files:**
- Modify: `cocore/config.py`, `cocore/pipeline.py`
- Test: `tests/test_cocore_pipeline.py`, `tests/test_cocore_bridge_v2.py`

**Interfaces:**
- Consume `resolve_temporal_geometry(profile)` from Task 1.
- Record profile-specific clip length, anchors, half windows, visual pooling description, and trajectory-window constants in fingerprints/manifests/catalog validation.
- Pass `bridge_v2` into scan/encode/replay paths while retaining default LIBERO contracts.

- [ ] Write failing pipeline and synthetic Bridge tests for 7-frame artifacts and retained 15-frame LIBERO artifacts.
- [ ] Confirm failures identify fixed 15/8 assumptions.
- [ ] Replace pipeline and validator constants with profile-derived geometry, including old Bridge artifact rejection.
- [ ] Run focused pipeline and Bridge integration tests.
- [ ] Self-review and commit the task.

### Task 3: Bridge diagnostics, production baselines, versions, and documentation

**Files:**
- Modify: `cocore_bridge_v2/action_diagnostics.py`, `cocore_bridge_v2/__init__.py`, `cocore_bridge_v2/README.md`, `cocore/README.md`, `libero_motion_primitives/README.md`
- Test: `tests/test_cocore_bridge_action_diagnostics.py`, `tests/test_cocore_bridge_v2.py`, `tests/test_cocore_bridge_v2_real.py`

**Interfaces:**
- Consume the `bridge_v2` temporal geometry for diagnostic starts/endpoints and acceptance reporting.
- Set Bridge adapter version to `0.8.0` while retaining Cocore `0.16.0`.

- [ ] Write failing diagnostic/version/real-metadata tests for the new contract and exact baselines.
- [ ] Update diagnostics and acceptance thresholds to 434,370 windows, 1,399 labels, 83 retained non-stop buckets, 1,139 estimated leaves, 65.39% exact non-stop coverage, 22.36% raw stop, and at least 73.0% atomic retention quality.
- [ ] Update real metadata expectations to 2 short episodes, 202,739 candidates, and a 20,274 Top-10% budget.
- [ ] Update documentation for 7/4 geometry, 1.4-second clips, 0.6-second state deltas, rebuild requirements, and production baselines.
- [ ] Run focused tests, the complete test suite, and the real-data read-only diagnostic.
- [ ] Self-review and commit the task.
