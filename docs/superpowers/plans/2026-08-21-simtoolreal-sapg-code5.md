# SimToolReal SAPG Code #5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This plan is executed inline by the control session; no subagent is dispatched.

**Goal:** Lock native Source/Target checkpoint restore, resume, and SAPG player routing semantics after Code #4.

**Architecture:** A Source-only generator creates one native `.pth` payload and a strict manifest. A Source/Target harness constructs the native `Runner -> A2CAgent` and `PpoPlayerContinuous` owners in separate processes/namespaces, loads the same payload, and compares complete observable load/resume state plus compact player traces. The harness reuses only source-neutral snapshot/provenance helpers from the accepted Code #4 oracle; it does not copy checkpoint or player logic and does not touch production code.

**Tech Stack:** Python 3.11, CUDA/cu128, native RL-Games `Runner`, `A2CAgent`, `PpoPlayerContinuous`, `torch.save/load`, JSON manifest, `uv run`, pytest.

---

### Task 1: Establish the Code #5 contract and RED tests

**Files:**
- Create: `tests/algos/rlgames_sapg/source_checkpoint_harness.py`
- Create: `tests/algos/rlgames_sapg/test_checkpoint_golden.py`

- [ ] **Step 1: Add loader/schema RED tests.** Assert the five top-level manifest sections (`provenance`, `payload`, `resume`, `player`, `comparison`), fixed Source/Code #4 provenance, payload regular-file/hash validation, `env_state=None`, and explicit `rng_saved=false` boundary. Add negative tests for payload hash, manifest hash, namespace root, missing keys, and player routing inventory drift.
- [ ] **Step 2: Run only the new tests.** Run `uv run pytest tests/algos/rlgames_sapg/test_checkpoint_golden.py -q`; expect collection/import or missing-symbol failures, never a schema-v1 or unrelated top-level failure.
- [ ] **Step 3: Add the minimal harness data model.** Define `CheckpointFixture`, `CheckpointReplay`, strict JSON/hash helpers, and a configurable synthetic vector environment with native `get_env_state() -> None` and `set_env_state(None)` test-boundary ABI methods.

### Task 2: Implement native checkpoint/resume capture and comparison

**Files:**
- Modify: `tests/algos/rlgames_sapg/source_checkpoint_harness.py`
- Modify: `tests/algos/rlgames_sapg/test_checkpoint_golden.py`

- [ ] **Step 1: Build the native agent through the accepted Code #3 runner params.** Use the same canonical execution/platform and deterministic parameter fill as Code #4; keep `Runner`, `A2CAgent`, central value, optimizer, GradScaler, RMS, and scheduler native. Install only the synthetic env `get_env_state/set_env_state` ABI adapter and restore it in `finally`.
- [ ] **Step 2: Capture a native payload.** Call `agent.save()`/native `torch_ext.load_checkpoint()` in a temporary directory, record payload top-level keys, recursive tensor shape/dtype/hash/signature, optimizer/scaler/RMS/LR/RNN/rollout fields, `env_state=None`, and the fact that NumPy/Torch/CUDA RNG are not checkpoint keys.
- [ ] **Step 3: Capture the resume boundary.** Set a fixed external RNG snapshot, call native `agent.restore()`, then native `init_tensors()` and `obs_to_tensors()` lifecycle conversion, and record the first `get_action_values`, `get_values`, and one native `train_actor_critic`/update return plus final model/optimizer/scaler/RMS/LR/runner-visible state. Compare Source and Target using exact hashes for discrete/structural values and `atol=1e-6, rtol=1e-5` for finite numeric signatures.
- [ ] **Step 4: Add matched and corrupted-payload tests.** Verify Source/Target same-payload replay, external RNG restoration, explicit non-bit-exact full-runtime boundary, payload byte drift rejection, and parameter/optimizer/content drift rejection.

### Task 3: Implement native player routing capture

**Files:**
- Modify: `tests/algos/rlgames_sapg/source_checkpoint_harness.py`
- Modify: `tests/algos/rlgames_sapg/test_checkpoint_golden.py`

- [ ] **Step 1: Construct `PpoPlayerContinuous` through `Runner.create_player()`.** Use the same payload and synthetic `vec_env`, set `has_batch_dimension=True` at the test boundary, call native `restore/reset/get_action`, and never implement a second inference path.
- [ ] **Step 2: Capture canonical six-env and non-six traces.** Record embedding IDs, deterministic/stochastic action arrays, RNN state transition, action clamp/rescale output, and model routing IDs for `N=6`, `N=5`, and `N=7`; assert canonical IDs `[50,40,30,20,10,0]` and Source equality/`argmax` fallback for non-six counts.
- [ ] **Step 3: Add player negative tests.** Reject action shape/dtype drift, missing checkpoint model/RMS, and changed routing inventory while preserving Source behavior for `N != 6`.

### Task 4: Add the Source-only generator and freeze the payload

**Files:**
- Create: `scripts/generate_simtoolreal_sapg_checkpoint_fixture.py`
- Generate: `tests/fixtures/simtoolreal_sapg/source_checkpoint.pth`
- Generate: `tests/fixtures/simtoolreal_sapg/source_checkpoint_manifest.json`

- [ ] **Step 1: Implement Source preflight and native capture.** Reuse the accepted fixed Source HEAD/tree/owner/module provenance checks, reject preloaded `rl_games`, run in a temporary isolated pycache namespace, call only native save/load/player owners, and restore interpreter state on all exits.
- [ ] **Step 2: Serialize safely.** Write the native payload and strict manifest through same-directory temporary regular files with fsync/replace; include the complete 27-token canonical command, payload SHA256, manifest SHA256, canonical payload SHA256, and loaded-module inventory.
- [ ] **Step 3: Run exactly one canonical generation.** Use fixed `CUDA_VISIBLE_DEVICES=0`, `UV_INDEX=cu128`, `UNILAB_REQUIRE_SAPG=1`, Source checkout `/home/user/ws/lemon/simtoolreal`, and `uv run --isolated --no-project --python 3.11`; print all anchors.
- [ ] **Step 4: Freeze printed anchors in harness/tests.** Ordinary pytest must only validate/replay and must never regenerate or mutate the fixture.

### Task 5: Final verification and Code #5 commit

**Files:** exactly the five Code #5 paths above.

- [ ] **Step 1: Run focused and full SAPG oracle tests with zero skips.** Use `PYTHONDONTWRITEBYTECODE=1 UNILAB_REQUIRE_SAPG=1 CUDA_VISIBLE_DEVICES=0 uv run pytest tests/algos/rlgames_sapg/test_checkpoint_golden.py -q` and `uv run pytest tests/algos/rlgames_sapg -q`.
- [ ] **Step 2: Run vendor suite, 72-file audit, scoped Ruff/format, and `git diff --check`.** Do not run `make test-all` for this oracle-only commit.
- [ ] **Step 3: Stage exactly five Code #5 paths, verify cached inventory, and commit:** `test: lock SAPG checkpoint and player semantics`.
- [ ] **Step 4: Run the post-commit focused gate, confirm a clean worktree, and stop at the Code #6 boundary.** Do not modify backend/assets/production paths in this batch.
