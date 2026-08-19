# SimToolReal SAPG Code #3 Second Review Rework Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` while implementing this plan. Read root `AGENTS.md` first. Execute the checklist directly; do not return a plan. Keep all implementation changes unstaged and uncommitted for the control session.

**Goal:** Close the remaining mutation-proof and output-path gaps in the code #3 rollout/RNG oracle without changing its Source-native math or entering code #4.

**Architecture:** Keep the existing native `Runner -> A2CAgent` capture and frozen fixtures. Harden only four evidence boundaries: symlink-component validation and raw caller handoff, whole-inventory metadata gating, a genuinely counted one-call RNN delegate, and exact augmented carrier relations. Reuse the existing generator, harness, test and two generated fixtures; add no sixth implementation file.

**Tech stack:** Python 3.11, Torch 2.7.0+cu128, pytest, NumPy, vendored fixed SimToolReal RL-Games, canonical RTX 4090 fixture capture.

---

## 1. Repository and fixed baseline

Work only in:

`/home/user/ws/lemon/rlgame-unilab/UniLab`

Fixed Source:

`/home/user/ws/lemon/simtoolreal`

Expected branch:

`feat/simtoolreal-sapg-rlgames`

The parent docs HEAD before this second prompt is:

`3a09e888c625671584de6a45043d7c7986e7832a`

The implementation session may start on a later HEAD containing only this control-session docs commit. Record the actual start HEAD and require it to remain unchanged:

```bash
git rev-parse --abbrev-ref HEAD
SAPG_CODE3_REWORK2_START_HEAD=$(git rev-parse HEAD)
printf '%s\n' "$SAPG_CODE3_REWORK2_START_HEAD"
git merge-base --is-ancestor \
  3a09e888c625671584de6a45043d7c7986e7832a HEAD
git status --short
git diff --cached --name-only
git -C /home/user/ws/lemon/simtoolreal rev-parse HEAD
git -C /home/user/ws/lemon/simtoolreal status --short
git -C /home/user/ws/lemon/simtoolreal diff --stat
git -C /home/user/ws/lemon/simtoolreal diff --cached --stat
```

Expected Target status is exactly these five untracked files and no staged files:

```text
?? scripts/generate_simtoolreal_sapg_rollout_fixture.py
?? tests/algos/rlgames_sapg/source_rollout_harness.py
?? tests/algos/rlgames_sapg/test_rollout_golden.py
?? tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz
?? tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json
```

Stop with `# BLOCKED` if there is any other Target change or any new Source tracked/staged change. Preserve all pre-existing Source untracked files.

Fixed Source identity remains:

- Source HEAD: `2a9917533bfea70419ed2667a511d7238e5b3abc`
- RL-Games tree: `7a6a0bb090998d00565aaefa6ab9f2b3d356ace2`
- Train owner blob: `f363d05d4a24b190b7837703b93270d8f3fe9a9c`
- Task owner blob: `6469d46867081b70edaa589dcb31c7090b64d45e`

## 2. Platform and scope

Use only:

- Python 3.11
- Torch 2.7.0+cu128
- `cuda:0`
- the V2b canonical RTX 4090 platform

Canonical Source/Target commands must use:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128
```

Do not run Python 3.10, 3.12 or 3.13. Do not test cu126 or another CUDA build. Every Python/pytest/Ruff/audit command must be invoked through `uv run --python 3.11`; never invoke Python directly.

Only these existing five implementation paths may change:

1. `scripts/generate_simtoolreal_sapg_rollout_fixture.py`
2. `tests/algos/rlgames_sapg/source_rollout_harness.py`
3. `tests/algos/rlgames_sapg/test_rollout_golden.py`
4. `tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz`
5. `tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json`

Use `apply_patch` for all handwritten edits. The NPZ and manifest may only be written by the generator. Do not modify vendor, Source, root config, V2a/V2b, production code or docs. Do not add, commit, push, stash, reset, clean, checkout or change branch. Do not run `make test-all`.

The three handwritten files currently total 895 net non-empty/non-comment lines. This control review authorizes at most 1,000 net lines for this narrowly scoped evidence hardening. Prefer small shared validators over test-specific duplication, but do not compress code unnaturally or game the line-count metric. If the implementation exceeds 1,000, stop rather than moving logic into a sixth file or production code.

## 3. Known-good evidence that must remain unchanged

Do not redesign or reimplement these already independently verified facts:

- canonical focused replay: 8 passed;
- complete SAPG oracle: 20 passed;
- fixed Source regeneration is byte-identical to both current fixture files;
- 308 arrays and 286 floating arrays replay exactly;
- the 10-field raw `ExperienceBuffer.tensor_dict` inventory and metadata are correct;
- native raw `swap_and_flatten01` equals the eight returned base fields;
- repeat indexes `[0, 2]` and permutation `[6, 11, 1, 9, 3, 12, 7, 5, 0, 10, 2, 13, 4, 8]` are correct;
- timeout uses the action-time central value, with zero error;
- counterfactual current privileged states use Source's native time-major reshape and remain unrelabelled;
- raw intrinsic rewards are zero and `extras["mb_intr_rewards"] is None`; fixed Source provenance supplies the control-flow evidence that no intrinsic term is added;
- the current RNN wrapper code statically calls its owner once, and isolated unmasked calls consume no RNG.

This rework exists because some regressions can still pass the current tests, and because one output path escape is currently real.

## 4. Task 1 — Reject every output-path symlink component

**Files:**

- Modify: `scripts/generate_simtoolreal_sapg_rollout_fixture.py:77-110,174`
- Test: `tests/algos/rlgames_sapg/test_rollout_golden.py`

### Required RED

Add a lightweight `tmp_path` regression that creates:

```text
tmp/
  outside/nested/
    source_rollout_fp32.npz       # sentinel bytes
    source_rollout_manifest.json  # sentinel bytes
  alias -> outside
```

Call the narrow writer/path boundary with `output = alias / "nested"`.

The current implementation accepts this because `output.is_symlink()` is false and writes through the ancestor symlink. The new test must initially fail and prove both external fixture leaf sentinels changed under the old implementation. Structure the RED test or a companion RED probe so it records the exception flag and both post-call sentinel byte values before one aggregate assertion; do not let an early `pytest.raises(...): DID NOT RAISE` prevent observing the two writes.

Also add a case for a broken symlink component in the middle of the output path. Both normal and broken symlink components must be rejected before `mkdir`, NPZ write or manifest write.

### Minimal implementation contract

Harden `_validated_fixture_paths()` or a narrow helper it calls:

1. Inspect the raw, unresolved output path component-by-component.
2. Use lstat semantics such as `Path.is_symlink()` on each lexical prefix; do not use `resolve()` or `realpath()` to perform this inspection.
3. Reject if the final output or any existing ancestor component is a normal or broken symlink.
4. Reject an existing non-directory output and any normal/broken/non-regular fixture leaf as before.
5. Only after all checks may the helper create the output directory.
6. On rejection, neither external fixture sentinel may change.
7. Do not broaden this into atomic-write or TOCTOU work; those are outside code #3.

The component walk must preserve relative-path and retained `..` semantics long enough to inspect every supplied component. (`Path` may already normalize explicit `.` components; do not attempt to reconstruct them.) A suitable structure is a small helper that starts from the filesystem anchor or current working directory, advances one retained raw path component at a time, rejects a symlink immediately, and handles `..` lexically only after checking the current prefix. Do not collapse the whole path first.

Run the new ancestor/broken-component tests and retain the exact RED/GREEN outputs.

## 5. Task 2 — Lock the raw `generate()` → writer handoff

**Files:**

- Preserve and verify: `scripts/generate_simtoolreal_sapg_rollout_fixture.py:107-174`
- Test: `tests/algos/rlgames_sapg/test_rollout_golden.py`

The current implementation correctly calls `_write(output, ...)`, but the existing root test would still pass if a future mutant restored `_write(output.resolve(), ...)`, because line 110 rejects invalid roots before the writer is reached.

Add a no-CUDA/no-Source regression that proves the final writer receives the original path object/value:

1. Use a legal raw path whose spelling differs from its resolved spelling, for example an existing `sub / ".." / "out"` path with no symlink.
2. Set `UNILAB_SAPG_ORACLE_MODE=source` in the test, then stub `_run_git`, `_runner_params`, `capture_rollout`, `_verify_source_modules` and the final writer narrowly enough for `generate()` to reach the handoff without importing or executing Source RL-Games.
3. Capture the `output` passed to the writer.
4. Assert it equals the original unresolved path and is not replaced by `output.resolve()`.
5. Assert the stubbed writer is called exactly once.

Do not assert only that an earlier root precheck ran. This test specifically owns the final caller-to-writer boundary.

## 6. Task 3 — Make metadata validation a whole-inventory gate

**Files:**

- Modify: `tests/algos/rlgames_sapg/source_rollout_harness.py`
- Test: `tests/algos/rlgames_sapg/test_rollout_golden.py`

Current replay interleaves:

```text
validate(A) -> numeric diagnostics(A) -> validate(B)
```

Required order is:

```text
inventory exact -> validate metadata for every array -> compute any numeric diagnostic
```

### Required RED

Add lightweight tests that do not invoke CUDA:

1. Content-only drift: same shape and dtype, one value changed. It must raise a `content hash drift` error. This prevents removal of the SHA branch.
2. Two arrays: A is valid, B has a shape drift. Instrument the validation/numeric comparison boundary and prove no subtraction/`np.abs`/relative-error operation happens for A before B is rejected.
3. Keep the existing same-bytes reshape, dtype drift and exact-pass cases.

### Minimal implementation contract

Extract or reuse a narrow helper so replay performs:

```python
if set(actual_arrays) != set(expected_arrays):
    raise RuntimeError(...)
for name in expected_arrays:
    _validate_target_array_metadata(name, actual_arrays[name], metadata[name])
for name in expected_arrays:
    # only now calculate floating diagnostics
```

The inventory error should include sorted missing and extra names. Shape, dtype and content hash errors must continue to identify the array name and drift type. Canonical content hash remains an exact gate; tolerance diagnostics never replace it.

## 7. Task 4 — Prove one owner delegate per RNN wrapper call

**Files:**

- Modify: `tests/algos/rlgames_sapg/source_rollout_harness.py`
- Test: `tests/algos/rlgames_sapg/test_rollout_golden.py`

The current wrapper implementation is correct, but `rnn_delegate_original_calls = len(masked_records)` only counts wrapper records. Reintroducing an extra discarded `original_forward(...)` call would keep the value at four, preserve deterministic outputs and consume no RNG, so the old defect would evade every current test.

### Required RED

Build a lightweight fake owner callable with an actual call counter and a unique sentinel result. During the call, make the fake owner mutate the supplied input/state/done objects in place. Exercise the same narrow delegate-and-record helper used by the production oracle wrapper. Assert:

- fake owner call count is exactly one;
- wrapper record count is exactly one;
- returned object is the owner's exact sentinel object, not a clone or reconstruction;
- the recorded frozen input/state/done retain their pre-call values despite the fake owner's in-place mutation;
- the fake counter would become two and fail if the helper ever delegated twice; do not keep mutant code in the repository.

Do not test a separate helper that `_rms_probe()` does not use.

### Minimal implementation contract

Extract a narrow delegate-and-record function or callable that:

1. clones observation-only input/state/done before delegation;
2. calls the supplied owner forward exactly once;
3. records the returned output after delegation;
4. returns the exact owner result object;
5. increments or returns an actual owner-delegate count independent of record length.

`_rms_probe()` must use this same helper. Fail fast unless actual delegate count equals wrapper invocation count and both equal the number of masked batches. Store the actual delegate count in manifest semantics. Keep the isolated unmasked native diagnostic outside the wrapper and keep its full RNG before/after equality.

Do not add dropout, alter train inputs or change RNN mathematics.

## 8. Task 5 — Make augmented-state relabel evidence exact

**Files:**

- Modify: `tests/algos/rlgames_sapg/source_rollout_harness.py`
- Test: `tests/algos/rlgames_sapg/test_rollout_golden.py`

The current predicate only checks that each follower state carrier differs from its original carrier. An arbitrary constant or NaN can therefore be reported as “relabelled.”

Add a small relation validator used by capture and a lightweight mutation test:

1. Require every `buffer_pre_shuffle__states` carrier to be finite.
2. Require the complete 56-row state carrier to equal the same-row native relabelled observation carrier exactly.
3. Retain the follower-specific proof that the eight selected rows changed from their original base-state carriers.
4. Mutate follower state carriers to `123.0` and to `NaN`; both mutations must fail.
5. Derive `augmented_states_relabelled_for_training` only after these exact relations pass.

Do not compute expected coefficients with a copied block-roll/filter formula. The native relabelled observation carrier is the captured owner output and is the comparison oracle.

The intrinsic-reward conclusion is not part of this task. Keep the current control-flow evidence (`extras["mb_intr_rewards"] is None` plus fixed Source blob); do not invent a non-native intrinsic rollout merely to make zero rewards numerically distinguishable.

## 9. TDD execution order

- [ ] Add only the new tests in the existing `test_rollout_golden.py`.
- [ ] Run the narrow no-CUDA selection and record the genuine failures for ancestor symlink, handoff, content hash/two-phase metadata, RNN actual delegate counting and corrupted relabel carriers.
- [ ] Implement Task 1 and rerun only its tests to GREEN.
- [ ] Implement Task 2 and rerun only its test to GREEN.
- [ ] Implement Task 3 and rerun only its tests to GREEN.
- [ ] Implement Task 4 and rerun only its tests to GREEN.
- [ ] Implement Task 5 and rerun only its tests to GREEN.
- [ ] Run the complete lightweight portion of `test_rollout_golden.py` with canonical replay deselected.
- [ ] Regenerate the canonical fixture once after all handwritten changes.
- [ ] Run canonical focused and complete SAPG suites.

Use a clear `-k` expression or explicit node IDs for lightweight RED/GREEN commands. All commands still require the vendored runtime import gate:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py \
  -k 'not target_replays_canonical_source_rollout_exactly' -q
```

Ordinary pytest may write only inside pytest's `tmp_path` for these path-boundary tests. It must not call Source capture or write the repository fixture.

## 10. Canonical regeneration

Regenerate both fixture files only through:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_SAPG_ORACLE_MODE=source \
uv run --isolated --python 3.11 \
  --with gym==0.26.2 \
  --with-editable /home/user/ws/lemon/simtoolreal/rl_games \
  scripts/generate_simtoolreal_sapg_rollout_fixture.py \
  --source /home/user/ws/lemon/simtoolreal \
  --output tests/fixtures/simtoolreal_sapg
```

Update the fixed NPZ SHA256 and manifest payload SHA256 in harness/test only if regeneration changes them. Report whether the files remained byte-identical or changed, and why. Reconfirm Source HEAD/tree/owner/module provenance, canonical platform, 8 MiB budget and unchanged Source status.

## 11. Required final validation

Canonical focused:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py -q
```

Complete SAPG oracle:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg -q
```

Vendor and audit without the temporary PyTorch index:

```bash
env -u UV_INDEX uv run --python 3.11 \
  pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q

env -u UV_INDEX uv run --python 3.11 \
  scripts/audit_simtoolreal_rlgames_vendor.py
```

Static and Git boundary:

```bash
env -u UV_INDEX uv run --python 3.11 ruff check \
  scripts/generate_simtoolreal_sapg_rollout_fixture.py \
  tests/algos/rlgames_sapg/source_rollout_harness.py \
  tests/algos/rlgames_sapg/test_rollout_golden.py

env -u UV_INDEX uv run --python 3.11 ruff format --check \
  scripts/generate_simtoolreal_sapg_rollout_fixture.py \
  tests/algos/rlgames_sapg/source_rollout_harness.py \
  tests/algos/rlgames_sapg/test_rollout_golden.py

env -u UV_INDEX uv run --python 3.11 ruff check .
env -u UV_INDEX uv run --python 3.11 ruff format --check .
git diff --check
git diff --cached --name-only
git status --short
```

All required pytest runs must have 0 skip. Target must remain canonical. Every canonical metadata/content hash must be exact and all FP diagnostics must remain zero. Do not run other Python/CUDA versions or `make test-all`.

## 12. Stop conditions

Return `# BLOCKED` immediately if any of these occurs:

- a sixth implementation file or any out-of-scope edit is required;
- vendor, Source, production algorithm, runner or config must change;
- native GAE, TD, shuffle, filter, RNN reset, RMS or SAPG math must be copied or changed;
- Source and Target RL-Games must load in one process;
- canonical Python 3.11 + cu128 is unavailable;
- any required test skips;
- fixed Source identity or module blobs drift;
- an unexplained array/index/hash/RNG difference appears;
- tolerance must be loosened;
- fixture exceeds 8 MiB;
- net handwritten LOC exceeds 1,000;
- work enters loss, optimizer, AMP, checkpoint, player or code #4.

## 13. Handoff report

Return `# DONE` with:

1. Final five path sizes, net handwritten LOC and out-of-scope count.
2. Exact second-rework RED command, failures and mutation each test catches.
3. Ancestor/broken-component symlink results and proof both external fixture sentinels remain unchanged.
4. Final `generate()` → writer raw-path handoff evidence.
5. Content-only metadata drift and whole-inventory-before-numeric event-order evidence.
6. Fake-owner RNN call count, wrapper count, exact sentinel identity and runtime 1:1 fail-fast evidence.
7. Exact 56-row obs/state carrier relation and `123.0`/NaN mutation rejection.
8. Regenerated NPZ, manifest-file and canonical-payload hashes plus byte-exact/change result.
9. Canonical focused/full counts, 0 skip, metadata exact count and maximum FP errors.
10. Source provenance, canonical platform, fixture budget and unchanged Source status.
11. Repeat/permutation/rollout/timeout/counterfactual/RNG/RMS facts remain exact.
12. Vendor, audit, Ruff, format and Git boundary results.
13. Full `git status --short`, empty staged diff and unchanged start HEAD.
14. Explicit confirmation: no code #4, no `make test-all`, no add/commit/push/PR, no stash/reset/clean/checkout, no Source modification and no extra Python/CUDA matrix.

Independent control review must approve this rework before code #3 is committed.
