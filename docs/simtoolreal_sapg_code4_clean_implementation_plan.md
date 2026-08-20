# SimToolReal SAPG Code #4 Clean Reimplementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 撤销未接受的 Code #4 返工状态，并从干净工作树重新建立、审查、验证和提交 Source-faithful SAPG update/AMP oracle。

**Architecture:** 固定 Source checkout 在独立进程中通过原生 `Runner -> A2CAgent` 生成 source-only fixture，Target 进程通过 UniLab vendored RL-Games 重放同一路径。测试 instrumentation 只 delegate 和记录 native owner；fixture 自身 evidence invariants、Source→Target semantic comparison 与 FP32 numeric comparison 分层 fail closed。

**Tech Stack:** Python 3.11、PyTorch 2.7.0+cu128、CUDA、RTX 4090、vendored Source RL-Games、NumPy NPZ、JSON manifest、pytest、Ruff、Git。

---

## File map

控制与规格文件：

- Modify: `docs/simtoolreal_sapg_rlgames_control_handoff.md` — 删除旧返工 writer 状态，记录 clean restart 和内部 agent 控制流程。
- Create: `docs/simtoolreal_sapg_code4_clean_execution_prompt.md` — Code #4 唯一当前执行规格；不依赖被撤销的两份返工文档。
- Existing reference: `docs/simtoolreal_sapg_code4_prompt.md` — 历史原始规格，只作来源，不再直接下发。

Code #4 最终代码文件只能是：

- Create: `scripts/generate_simtoolreal_sapg_update_fixture.py` — 固定 Source provenance、调用 capture、写入两份 fixture，并拒绝不安全输出路径。
- Create: `tests/algos/rlgames_sapg/source_update_harness.py` — native update capture、fixture loader、evidence invariants、Target replay。
- Create: `tests/algos/rlgames_sapg/test_update_golden.py` — provenance、evidence、mutation、Source→Target 和路径安全测试。
- Generate: `tests/fixtures/simtoolreal_sapg/source_update_fp32.npz` — frozen tensors 和明确列入 comparison inventory 的数值证据。
- Generate: `tests/fixtures/simtoolreal_sapg/source_update_manifest.json` — provenance、platform、metadata、semantics、RNG、signatures 和 anchors。

不得修改 Code #3、vendor、Source、`src/**`、`conf/**`、root packaging 或其他测试。

### Task 1: Quiesce and quarantine the rejected working state

**Files:**

- Quarantine outside repository: the five current untracked Code #4 files
- No repository file edits in this task

- [ ] **Step 1: Recheck the exact branch and file boundary**

Run:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
git diff --name-only
git diff --cached --name-only
```

Expected: branch `feat/simtoolreal-sapg-rlgames`; tracked and staged outputs empty; untracked output contains exactly the five Code #4 paths in the file map.

- [ ] **Step 2: Detect an overlapping writer without reading implementation content**

Run two SHA snapshots five seconds apart and compare them:

```bash
code4_snapshot_a=$(mktemp /tmp/unilab-code4-snapshot-a-XXXXXX)
code4_snapshot_b=$(mktemp /tmp/unilab-code4-snapshot-b-XXXXXX)
sha256sum \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json >"$code4_snapshot_a"
sleep 5
sha256sum \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json >"$code4_snapshot_b"
cmp "$code4_snapshot_a" "$code4_snapshot_b"
printf 'snapshot_a=%s\nsnapshot_b=%s\n' "$code4_snapshot_a" "$code4_snapshot_b"
```

Expected: both snapshots are identical. If they differ, stop because another session is still writing.

- [ ] **Step 3: Move the five files to a recoverable quarantine**

Run:

```bash
code4_quarantine_dir=$(mktemp -d /tmp/unilab-code4-rework-XXXXXX)
case "$code4_quarantine_dir" in
  /tmp/unilab-code4-rework-*) ;;
  *) exit 1 ;;
esac
mkdir -p \
  "$code4_quarantine_dir/scripts" \
  "$code4_quarantine_dir/tests/algos/rlgames_sapg" \
  "$code4_quarantine_dir/tests/fixtures/simtoolreal_sapg"
mv -- scripts/generate_simtoolreal_sapg_update_fixture.py \
  "$code4_quarantine_dir/scripts/"
mv -- tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  "$code4_quarantine_dir/tests/algos/rlgames_sapg/"
mv -- tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json \
  "$code4_quarantine_dir/tests/fixtures/simtoolreal_sapg/"
find "$code4_quarantine_dir" -type f -print0 | sort -z | xargs -0 sha256sum
printf 'quarantine=%s\n' "$code4_quarantine_dir"
```

Record the printed directory and five post-move SHA256 values. Do not use a glob, recursive removal, `git clean`, `reset`, `checkout` or `stash`.

- [ ] **Step 4: Verify the repository now contains no Code #4 implementation state**

Run:

```bash
git status --short
git diff --name-only
git diff --cached --name-only
```

Expected: only already-created control/plan documentation can appear; none of the five Code #4 paths exists in the worktree.

### Task 2: Revert the two rejected review-prompt commits

**Files:**

- Delete through revert: `docs/simtoolreal_sapg_code4_review_rework2_prompt.md`
- Delete through revert: `docs/simtoolreal_sapg_code4_review_rework_prompt.md`

- [ ] **Step 1: Verify both target commits and their one-file scope**

Run:

```bash
git show --format=fuller --stat --summary 5b08333397f436feba1ad3f2376ddd96b9d2ee02
git show --format=fuller --stat --summary dbe5bf3a66055218ea109aae67f6736d87f3e4e3
```

Expected: each commit creates exactly its corresponding review prompt.

- [ ] **Step 2: Revert newest first without rewriting history**

Run:

```bash
git revert --no-edit 5b08333397f436feba1ad3f2376ddd96b9d2ee02
git revert --no-edit dbe5bf3a66055218ea109aae67f6736d87f3e4e3
```

Expected: two new revert commits; both review prompt files are absent. Do not squash these with the future Code #4 code commit.

- [ ] **Step 3: Verify no unrelated path changed**

Run:

```bash
git show --name-status --format=oneline HEAD~1..HEAD
git show --name-status --format=oneline HEAD~2..HEAD~1
git status --short --branch
```

Expected: each revert deletes exactly one docs file and the worktree is clean.

### Task 3: Publish the clean Code #4 execution contract

**Files:**

- Create: `docs/simtoolreal_sapg_code4_clean_execution_prompt.md`
- Modify: `docs/simtoolreal_sapg_rlgames_control_handoff.md`

- [ ] **Step 1: Write the clean execution prompt**

The prompt must be self-contained and contain these exact fixed identities:

```text
Source HEAD: 2a9917533bfea70419ed2667a511d7238e5b3abc
RL-Games tree: 7a6a0bb090998d00565aaefa6ab9f2b3d356ace2
train owner blob: f363d05d4a24b190b7837703b93270d8f3fe9a9c
task owner blob: 6469d46867081b70edaa589dcb31c7090b64d45e
Code #3 NPZ: 3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8
Code #3 manifest: 785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f
```

It must state the five-file boundary, canonical platform, native-delegation rule, fixture budget, required evidence, RED mutations, exact validation commands and stop conditions. It must not link to either deleted review prompt or copy any algorithm formula into the harness.

- [ ] **Step 2: Replace stale handoff state**

The handoff must state:

```text
Code #4 is reset and has no implementation files in the worktree.
The control session owns Git history and dispatches the committed clean prompt directly.
No external implementation session is an active writer.
Code #5 remains unapproved and must not start.
```

Remove all links, hashes and instructions that depend on the two deleted review prompt files or the quarantined artifacts.

- [ ] **Step 3: Self-review the documentation**

Run:

```bash
rg -n 'simtoolreal_sapg_code4_review_rework|第二次返工正在|现有实现 session' \
  docs/simtoolreal_sapg_rlgames_control_handoff.md \
  docs/simtoolreal_sapg_code4_clean_execution_prompt.md
git diff --check
```

Expected: `rg` has no stale matches and `git diff --check` passes.

- [ ] **Step 4: Commit only the clean contract and handoff**

Run:

```bash
git add -- \
  docs/simtoolreal_sapg_code4_clean_execution_prompt.md \
  docs/simtoolreal_sapg_rlgames_control_handoff.md
git diff --cached --name-status
git diff --cached --check
git commit -m "docs: restart SAPG update oracle cleanly"
```

Expected: one docs-only commit containing exactly the two declared files; clean worktree.

### Task 4: Establish Code #4 RED tests before implementation

**Files:**

- Create: `tests/algos/rlgames_sapg/test_update_golden.py`

- [ ] **Step 1: Create the test module with the required public boundary**

The module must import the vendored runtime gate before the harness and require these interfaces:

```python
from tests.algos.rlgames_sapg._runtime_requirement import require_simtoolreal_rl_games

require_simtoolreal_rl_games()

from tests.algos.rlgames_sapg.source_update_harness import (
    load_update_fixture,
    replay_update_fixture,
    validate_update_evidence_invariants,
)
```

Tests must separately cover fixture anchors/inventory, prepared dataset evidence, phase-derived batches, loss/gradient/optimizer coverage, AMP details, RNG, metadata ordering, namespace rejection and generator path safety.

- [ ] **Step 2: Run the initial collection RED**

Run:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py -q
```

Expected: collection fails because `source_update_harness` does not exist. Record this as a collection RED, not an algorithm mismatch.

- [ ] **Step 3: Add lightweight evidence-invariant RED cases**

Use a minimal complete synthetic semantics factory in the test module. Create symmetric Source/Target mutants that remove loss operations, clip events, optimizer results, prepared fields or epoch normalizer snapshots while retaining descriptive labels. Every mutant must be rejected by `validate_update_evidence_invariants`; recursive equality alone must never pass it.

- [ ] **Step 4: Add metadata-ordering and path-safety RED cases**

Install an event probe around inventory, every array metadata validation, semantic validation and numeric subtraction. Assert the only accepted order is:

```text
inventory -> shape/dtype/content for every array -> semantic invariants -> numeric
```

Add generator tests for real output directories and rejection of root/ancestor/leaf symlinks, broken symlinks, directory leaves and non-directory roots without modifying an external sentinel.

### Task 5: Implement the native capture and evidence boundaries

**Files:**

- Create: `tests/algos/rlgames_sapg/source_update_harness.py`

- [ ] **Step 1: Implement regular-file fixture loading and fixed anchors**

Define immutable constants for the final NPZ file hash, manifest file hash and canonical manifest payload hash. `load_update_fixture()` must validate regular files, fixed external anchors, exact NPZ inventory and every array's shape/dtype/content hash before returning arrays and manifest. `np.load` must use `allow_pickle=False`.

- [ ] **Step 2: Implement an unforgeable metadata gate**

Use a frozen token returned only after complete array validation:

```python
@dataclass(frozen=True)
class ValidatedUpdateArrays:
    names: tuple[str, ...]
    metadata_digest: str


def numeric_errors(
    token: ValidatedUpdateArrays,
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    comparison_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    expected_names = tuple(sorted(expected))
    actual_names = tuple(sorted(actual))
    if expected_names != token.names or actual_names != token.names:
        raise RuntimeError("numeric comparison received an unvalidated inventory")
    if len(comparison_names) != len(set(comparison_names)):
        raise RuntimeError("FP32 comparison inventory contains duplicates")
    missing = sorted(set(comparison_names) - set(token.names))
    if missing:
        raise RuntimeError(f"FP32 comparison inventory is not validated: {missing}")
    max_abs: dict[str, float] = {}
    max_rel: dict[str, float] = {}
    for name in comparison_names:
        difference = np.abs(actual[name] - expected[name])
        denominator = np.maximum(np.abs(expected[name]), np.finfo(np.float32).tiny)
        max_abs[name] = float(difference.max(initial=0.0))
        max_rel[name] = float((difference / denominator).max(initial=0.0))
    return max_abs, max_rel
```

The function must reject comparison names outside the validated inventory; no bool such as `metadata_validated=True` may authorize subtraction.

- [ ] **Step 3: Build the native agent and frozen handoff**

Construct the real agent through `Runner.load()`, `set_vec_env()`, native algo factory and `init_tensors()`. Load the Code #3 post-shuffle arrays after validating their fixed hashes. Install an identity `a2c_common.shuffle_batch` wrapper only for normal FP32/AMP `train_epoch()` calls; it must return the same object once per normal case and consume no RNG.

- [ ] **Step 4: Add delegate-only instrumentation**

Short-lived wrappers must call each native owner exactly once and restore it in `finally`. Capture native `prepare_dataset`, `PPODataset`, central/actor gradient paths, `torch.exp`, `torch.clamp`, `torch.max`, `apply_masks`, `GradScaler.scale/unscale_/step/update`, `clip_grad_norm_`, optimizer steps and scheduler/update_lr. Do not implement PPO, value, bounds, entropy, KL, optimizer or AMP formulas in test code.

- [ ] **Step 5: Map every event to case/owner/epoch/batch**

Derive `(mini_epoch, mini_batch)` from each dataset's access counter and `len(dataset)`. Both actor and central normal cases must produce:

```text
(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)
```

and batch sizes `[12, 12, 12, 20]`. Do not store hard-coded proof fields for mini-epoch counts or value-before-dataset order.

- [ ] **Step 6: Persist prepared, loss and normalizer evidence**

Store original values/returns, normalized old values/returns, native advantages, actor/central dataset handoff metadata and actual `train_value_mean_std`. For each actor batch store ratio, native surrogate/value branches, bounds, raw entropy, selected per-row entropy coefficient and product, reduced components, total loss, new mu/sigma, native KL and update-mu-sigma references. Store actor and central normalizer snapshots after prepare and after each owner/epoch.

- [ ] **Step 7: Persist gradient, optimizer, scheduler and AMP evidence**

For every real batch capture backward gradients, actor scaled/unscaled stages, clip returned norm and post-clip signature, optimizer parameter names derived from actual param groups, parameter deltas, optimizer state, and param-group LR. Capture scheduler calls once per actor epoch and LR after native `update_lr`. Capture actual autocast enabled/dtype, scaler scale and growth tracker around every scale/unscale/step/update, underlying optimizer success mask and parameter changed/unchanged relation.

- [ ] **Step 8: Implement overflow and RNG evidence**

The overflow case may mutate only `advantages[0]` on a clone of the first native prepared actor batch to `+inf`; it must run the real scaler path, skip the underlying optimizer step, keep parameters unchanged and back off the scaler. Save complete NumPy, Torch CPU and CUDA RNG state at every required phase and prove instrumentation/freeze/diagnostics do not consume additional RNG.

- [ ] **Step 9: Implement evidence invariants before equality**

`validate_update_evidence_invariants()` must independently validate Source and Target captures. It must reject missing fields, wrong counts, incomplete owner/epoch/batch coverage, inconsistent optimizer step/delta/state, incomplete normalizer transitions, absent RNG phases, wrong scheduler count/LR and AMP step/scaler/parameter contradictions. Human-readable `evidence_inventory` labels may be generated only after this validator succeeds.

- [ ] **Step 10: Implement Source→Target replay**

`replay_update_fixture()` must execute in this order:

```text
Source anchors/inventory/metadata
Source evidence invariants
Target inventory/metadata
Target evidence invariants
recursive semantic and RNG comparison
explicit FP32 numeric inventory comparison
```

The explicit FP32 inventory must be stored in the manifest and cover all NPZ normal-FP32 loss/KL/value/mu/sigma traces. AMP is compared Source→Target only for dtype/control/scaler/signature/step relations, not against FP32 values.

### Task 6: Implement source-only generation and freeze the fixture

**Files:**

- Create: `scripts/generate_simtoolreal_sapg_update_fixture.py`
- Generate: `tests/fixtures/simtoolreal_sapg/source_update_fp32.npz`
- Generate: `tests/fixtures/simtoolreal_sapg/source_update_manifest.json`

- [ ] **Step 1: Implement fixed Source provenance checks**

Before capture, verify Source HEAD, RL-Games tree and both owner blobs through Git objects. After capture, verify every loaded `rl_games.*` module lies below the expected Source package root and matches its Git blob/SHA256. Require `UNILAB_SAPG_ORACLE_MODE=source` and the exact canonical platform.

- [ ] **Step 2: Implement safe output writing**

Reject a root, ancestor or leaf symlink, broken symlink, directory leaf and existing non-directory root before any write. Preserve external sentinels. Write only the two fixed fixture filenames and use canonical JSON serialization for the payload anchor.

- [ ] **Step 3: Run all lightweight tests before CUDA regeneration**

Run the focused test selection declared in the clean prompt for invariants, metadata ordering, namespace and path safety. Expected: PASS without invoking a full Source capture.

- [ ] **Step 4: Generate exactly once from the immutable Source checkout**

Run:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
UNILAB_SAPG_ORACLE_MODE=source \
uv run --isolated --python 3.11 \
  --with gym==0.26.2 \
  --with torch==2.7.0 \
  --with numpy==2.4.4 \
  --with omegaconf==2.3.0 \
  --with-editable /home/user/ws/lemon/simtoolreal/rl_games \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  --source /home/user/ws/lemon/simtoolreal \
  --output tests/fixtures/simtoolreal_sapg
```

Expected: source-only generation succeeds on canonical RTX 4090 and does not modify Code #3 or Source.

- [ ] **Step 5: Freeze final anchors and enforce the byte budget**

Compute NPZ file SHA256, manifest file SHA256 and canonical payload SHA256 from disk, then write the exact constants into harness/tests. The combined byte size must be below `8_388_608`; reduce repeated descriptive manifest structure if necessary, never evidence coverage.

### Task 7: Prove mutations and run the full Code #4 gate

**Files:**

- Modify only during temporary mutation and restore exactly: the three Code #4 text files

- [ ] **Step 1: Run required mutation REDs**

Use `apply_patch` for each temporary mutant and record its pre-mutation SHA256. Required mutants must prove rejection of:

```text
identity shuffle removal
central-before-actor order drift
update_mu_sigma delay/removal
symmetric Source/Target evidence deletion
normal AMP step-mask/scaler falsification
overflow unconditional optimizer step
metadata validation after first subtraction
wrong expected package root
```

Restore each file with `apply_patch`, verify its SHA256 equals the pre-mutation value, and rerun the corresponding GREEN test.

- [ ] **Step 2: Run focused and full SAPG tests**

Run:

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py -q

UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg -q
```

Expected: both commands pass with zero skips.

- [ ] **Step 3: Run vendor, audit and style gates**

Run:

```bash
env -u UV_INDEX uv run --python 3.11 \
  pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q
env -u UV_INDEX uv run --python 3.11 \
  scripts/audit_simtoolreal_rlgames_vendor.py
env -u UV_INDEX uv run --python 3.11 ruff check \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py
env -u UV_INDEX uv run --python 3.11 ruff format --check \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py
env -u UV_INDEX uv run --python 3.11 ruff check .
env -u UV_INDEX uv run --python 3.11 ruff format --check .
git diff --check
```

Expected: all gates pass. Do not run `make test-all` in Code #4.

### Task 8: Independent review and the single Code #4 code commit

**Files:**

- Review and stage exactly the five Code #4 final files

- [ ] **Step 1: Stop the implementation writer and inspect all final content**

Read all three text files and independently inspect fixture metadata, file types, sizes, anchors, provenance, event coverage and comparison inventory. Confirm Code #3 hashes, vendor 72-file identities and Source working tree were not modified.

- [ ] **Step 2: Run a separate read-only spec/quality review**

The reviewer receives the clean execution prompt, final diff and validation output. It must report findings by severity and exact path/field. A green test result does not close an evidence-completeness finding.

- [ ] **Step 3: Return findings to the same implementation agent**

If findings exist, dispatch only evidence-backed corrections, then repeat Tasks 7 and 8.1–8.2. Do not edit concurrently with the implementation agent.

- [ ] **Step 4: Stage exactly five paths**

Run:

```bash
git add -- \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json
git diff --cached --name-status
git diff --cached --check
```

Expected: exactly five added files and no other staged path.

- [ ] **Step 5: Commit the accepted oracle**

Run:

```bash
git commit -m "test: lock SAPG update and AMP semantics"
```

Expected: one Code #4 code commit, separate from all docs/revert commits.

- [ ] **Step 6: Re-run the key post-commit gate**

Repeat focused Code #4, full SAPG, vendor audit and `git status --short --branch`. Expected: all pass, zero skip, and a clean worktree. Report the final commit SHA, anchors, fixture bytes, test outputs and quarantine recovery path.

- [ ] **Step 7: Stop before Code #5**

Write the Code #5 ordinary-Chinese scope summary, but do not create its execution prompt or dispatch an agent until the maintainer explicitly approves Code #5.
