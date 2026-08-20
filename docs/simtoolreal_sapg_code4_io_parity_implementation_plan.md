# SAPG Code #4 State-Transition Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unfinished forensic update oracle with a compact Source→Target golden that proves the three native update/AMP cases have identical complete inputs and observable final states.

**Architecture:** A Source-only generator and a Target replay test call one shared harness in separate processes and separate `rl_games` namespaces. The harness constructs the native Runner/A2CAgent path, injects the frozen Code #3 batch at the native rollout boundary, records initial/prepared/final state snapshots, and compares manifest structure plus NPZ arrays without tracing internal primitives.

**Tech Stack:** Python 3.11, pytest, NumPy, PyTorch 2.7.0+cu128/CUDA, vendored RL-Games, `uv run`, JSON + `np.savez_compressed`.

---

## File structure

- Rewrite `tests/algos/rlgames_sapg/test_update_golden.py`: contract RED/GREEN tests, corruption gates, external anchors, and the single native Target replay test.
- Rewrite `tests/algos/rlgames_sapg/source_update_harness.py`: native three-case capture, complete state snapshots, fixture loading/validation, and Source→Target comparison.
- Rewrite `scripts/generate_simtoolreal_sapg_update_fixture.py`: fixed Source provenance, Source-only capture, in-memory serialization, safe regular-file output, and printed anchors.
- Regenerate `tests/fixtures/simtoolreal_sapg/source_update_fp32.npz`: prepared datasets, RMS arrays, RNG bytes, and numeric native returns that cannot be represented by hashes alone.
- Regenerate `tests/fixtures/simtoolreal_sapg/source_update_manifest.json`: provenance, input/output state schema, hashes, metadata, and reproduction command.

These are the only Code #4 implementation paths. Do not modify Source, `third_party/`, Code #3 fixtures, production code, or another document. The implementation agent performs no Git staging or commits; the control session owns review and the final five-file commit.

## State contract

The manifest top-level inventory is exactly:

```python
{
    "schema_version", "generation_mode", "ordinary_pytest_regenerates",
    "provenance", "platform", "canonical_platform", "code3_anchors",
    "runner_params", "capture_contract", "cases", "npz_arrays",
    "exact_comparison_inventory", "numeric_comparison_inventory",
    "tolerances", "fixture_files", "canonical_payload_sha256",
    "generation_command",
}
```

`schema_version` is `2`, `generation_mode` is `source-only`, and `ordinary_pytest_regenerates` is false. Every capture has one `capture_contract` and three ordered cases. Each case has the following shape:

```python
{
    "name": "normal_fp32 | normal_amp | overflow_amp",
    "config": {"mixed_precision": bool, "mini_epochs": 2, "use_others_experience": "none"},
    "owners": {
        "runner": "rl_games.torch_runner.Runner",
        "agent": "rl_games.algos_torch.a2c_continuous.A2CAgent",
        "actor_dataset": "rl_games.common.datasets.PPODataset",
        "central_value": "rl_games.algos_torch.central_value.CentralValueTrain",
    },
    "input": {
        "batch": "complete SnapshotStore tree reference",
        "model": {"actor": "parameter inventory/hashes", "central": "parameter inventory/hashes"},
        "optimizer": {"actor": "groups/state keys/steps/tensor hashes", "central": "same"},
        "scaler": "complete GradScaler state_dict",
        "rms": "four roles with mean/var/count/training",
        "lr": "scheduler and optimizer learning rates",
        "rng": "complete NumPy/Torch CPU/all-CUDA state",
    },
    "execution": {
        "identity_shuffle_calls": int,
        "owner_call_order": list[str],
        "actor_update_attempts": int,
        "actor_optimizer_steps": int,
        "actor_scaler_skips": int,
        "central_optimizer_steps": int,
        "native_return": "loss/entropy/KL/LR algorithm summary tree",
        "overflow_mutation": "none or advantages[0]=+inf",
    },
    "output": {
        "prepared": {"actor": "complete values_dict", "central": "complete values_dict"},
        "model": {"actor": "parameter inventory/hashes", "central": "parameter inventory/hashes"},
        "optimizer": {"actor": "groups/state keys/steps/tensor hashes", "central": "same"},
        "scaler": "complete GradScaler state_dict",
        "rms": "four roles with mean/var/count/training",
        "lr": "scheduler and optimizer learning rates",
        "rng": "complete NumPy/Torch CPU/all-CUDA state",
    },
    "restore": {"patches": True, "hooks": True},
}
```

`normal_fp32` and `normal_amp` must report call order `prepare → central → actor`, one identity shuffle, four-item actor/central native datasets, eight actor update attempts, and central optimizer step eight (two mini-epochs × four native batches). Freeze the actual actor optimizer step/GradScaler skip count for each Source case instead of assuming every uninjected AMP attempt succeeds; the current fixed input is allowed to expose a native AMP skip. Parse the native `train_epoch()` tuple into its deterministic algorithm fields (`a_losses`, `c_losses`, `b_losses`, `entropies`, `kls`, `last_lr`, and `lr_mul`). Explicitly name `play_time`, `update_time`, and `total_time` as excluded wall-clock diagnostics rather than comparing them. `overflow_amp` must prepare the native actor dataset, clone its first native mini-batch, change only `advantages[0]` to positive infinity, and call native `A2CAgent.train_actor_critic`; actor parameters and optimizer state must remain unchanged while the scaler backs off.

### Task 1: Replace forensic tests with state-contract RED

**Files:**
- Rewrite: `tests/algos/rlgames_sapg/test_update_golden.py`
- Test: `tests/algos/rlgames_sapg/test_update_golden.py`

- [ ] **Step 1: Write the in-memory inventory RED before changing the harness**

Keep the required runtime gate before importing the harness. Replace event/capability/sentinel tests with the behavior boundaries below. Schema and drift unit tests must use a tiny in-memory schema-v2 `UpdateFixture`; they must not call `load_update_fixture()` and therefore must not be blocked by the checked-in schema-v1 artifact. Construct one 2-D floating array, its complete metadata record, and the three minimal ordered case mappings directly in the test helper.

```python
def mutated_arrays(fixture: UpdateFixture, drift: str) -> dict[str, np.ndarray]:
    arrays = {name: value.copy() for name, value in fixture.arrays.items()}
    name = sorted(arrays)[0]
    if drift == "missing":
        arrays.pop(name)
    elif drift == "extra":
        arrays["unexpected"] = np.zeros(1, dtype=np.uint8)
    elif drift == "shape":
        shaped = next(key for key, value in arrays.items() if value.ndim != 1)
        arrays[shaped] = arrays[shaped].reshape(-1)
    elif drift == "dtype":
        arrays[name] = arrays[name].astype(
            np.float32 if arrays[name].dtype == np.float64 else np.float64
        )
    elif drift == "content":
        arrays[name] = arrays[name].copy()
        arrays[name].view(np.uint8).reshape(-1)[0] ^= 1
    else:
        raise AssertionError(drift)
    return arrays


def test_in_memory_fixture_has_complete_ordered_state_contract() -> None:
    fixture = in_memory_update_fixture()
    assert [case["name"] for case in fixture.manifest["cases"]] == [
        "normal_fp32", "normal_amp", "overflow_amp"
    ]
    for case in fixture.manifest["cases"]:
        assert set(case) == {
            "name", "config", "owners", "input", "execution", "output", "restore"
        }
        assert set(case["input"]) == {
            "batch", "model", "optimizer", "scaler", "rms", "lr", "rng"
        }
        assert set(case["output"]) == {
            "prepared", "model", "optimizer", "scaler", "rms", "lr", "rng"
        }


@pytest.mark.parametrize("drift", ["missing", "extra", "shape", "dtype", "content"])
def test_fixture_array_drift_is_rejected(drift: str) -> None:
    fixture = in_memory_update_fixture()
    with pytest.raises(RuntimeError, match=drift):
        harness._validate_array_inventory(
            mutated_arrays(fixture, drift), fixture.manifest["npz_arrays"]
        )


def test_update_fixture_replays_native_target() -> None:
    result = replay_update_fixture(load_update_fixture())
    assert result.case_names == ("normal_fp32", "normal_amp", "overflow_amp")
    assert result.native_owner_paths
    assert result.numeric_array_count > 0
    assert np.isfinite(result.max_abs_error)
```

`max_abs_error` is diagnostic only. `compare_capture()` owns acceptance through per-array `np.testing.assert_allclose(atol=1e-6, rtol=1e-5)`; do not add a stricter global absolute-error gate.

Import `numpy as np`, `UpdateFixture`, and the harness module for the helper above. Also retain one wrong-namespace-root test, one ordinary symlink rejection test, the three external file/payload anchors, fixed Source/Code #3 provenance assertions, and a generator-output-root test. Do not retain private capabilities, symmetric-deletion, event-ledger, primitive, sentinel, FIFO, socket, device, or transaction-rollback tests.

- [ ] **Step 2: Observe the inventory RED, then add the next behavior REDs one at a time**

First run only the in-memory inventory/drift tests. Expected: FAIL on the first missing new harness API or schema rule, never on the old fixture loader. Implement no harness code in this task.

Then add focused tests, one behavior at a time, for `SnapshotStore` duplicate/non-finite handling, complete parameter/optimizer/RMS/RNG/LR snapshots, synthetic capture validation, numeric-vs-exact comparison, matched and mismatched `NaN`/signed-infinity patterns, overflow invariance, patch/ABI restoration, namespace rejection, and generator path safety. After each addition, run that individual test and record that it fails for the intended missing behavior before adding the next test. A syntax/import error or schema-v1 loader failure is not an acceptable RED for an in-memory behavior.

- [ ] **Step 3: Run the full focused RED**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py -q --maxfail=1
```

Expected: the new in-memory tests have already produced their individual behavior REDs, and the full file still FAILS because the checked-in intermediate fixture is schema v1 and the current harness still requires the deprecated v2 `event_ledger`. Record the exact failure; a skip is not an acceptable RED.

### Task 2: Rewrite the harness around complete initial/final snapshots

**Files:**
- Rewrite: `tests/algos/rlgames_sapg/source_update_harness.py`
- Test: `tests/algos/rlgames_sapg/test_update_golden.py`

Follow a strict RED→GREEN cycle inside this task. For each step below, enable only the already-written in-memory test for that behavior, confirm its recorded RED is still the expected one, implement the minimum harness surface, and rerun that test to GREEN before moving to the next step. Do not implement the store, all snapshots, all native cases, and the comparator as one undifferentiated change.

- [ ] **Step 1: Delete the deprecated tracing surface**

Remove `Recorder.events/event`, capability factories/tokens, `_signature_collection`, gradient snapshots, `_derive_case_views`, `_agent_and_env_with_construction_ledger`, `_case_v2`, primitive monkeypatches, the giant event-ledger validator, and forensic mutation helpers. Retain only source-neutral serialization/path helpers, exact namespace validation, Code #3 batch construction, `_agent_and_env`, deterministic parameter fill, canonical execution/platform helpers, and minimal reversible patching.

The retained data objects are exactly:

```python
@dataclass(frozen=True)
class UpdateFixture:
    manifest: dict[str, object]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class ReplayResult:
    case_names: tuple[str, ...]
    native_owner_paths: tuple[str, ...]
    exact_array_count: int
    numeric_array_count: int
    max_abs_error: float
```

The only callable test entry points are `capture_update(runner_params, expected_package_root) -> dict[str, object]`, `load_update_fixture(root=FIXTURE_ROOT) -> UpdateFixture`, and `replay_update_fixture(fixture) -> ReplayResult`. `_validate_array_inventory`, `validate_capture`, and `compare_capture` stay module-internal helpers used by those three entry points and focused tests.

- [ ] **Step 2: Implement one explicit array store**

Use a store whose only public operation recursively records tensor/NumPy leaves and returns strict-JSON references:

```python
class SnapshotStore:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, object]] = {}

    def tree(self, prefix: str, value: object, *, comparison: str) -> dict[str, object]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if prefix in self.arrays:
                raise RuntimeError(f"duplicate snapshot array: {prefix}")
            array = np.ascontiguousarray(value)
            domain = comparison if np.issubdtype(array.dtype, np.inexact) else "exact"
            if domain not in {"numeric", "exact"}:
                raise RuntimeError(f"invalid comparison domain: {domain}")
            self.arrays[prefix] = array
            self.metadata[prefix] = {
                "name": prefix,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                "comparison": domain,
            }
            return {"kind": "array", "name": prefix}
        if isinstance(value, dict):
            return {
                "kind": "dict",
                "items": {
                    str(name): self.tree(
                        f"{prefix}__{name}", item, comparison=comparison
                    )
                    for name, item in sorted(value.items())
                },
            }
        if isinstance(value, (list, tuple)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [
                    self.tree(f"{prefix}__{index}", item, comparison=comparison)
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            return {"kind": "none"}
        if isinstance(value, (bool, int, str)):
            return {"kind": "scalar", "value": value}
        if isinstance(value, float):
            if math.isnan(value):
                value = {"nonfinite": "nan"}
            elif math.isinf(value):
                value = {"nonfinite": "+inf" if value > 0 else "-inf"}
            return {"kind": "scalar", "value": value}
        raise RuntimeError(f"unsupported snapshot value at {prefix}: {type(value)!r}")
```

Supported array comparison domains are exactly `numeric` and `exact`. Every array must belong to exactly one domain: all floating Source/Target output arrays, including AMP and RMS arrays, are numerically compared within the same named case; complete input state, row IDs, discrete arrays, and RNG bytes compare exactly. JSON parameter/optimizer content hashes also compare exactly. Reject duplicate names, unsupported objects, non-finite raw JSON floats, missing/extra names, shape drift, dtype drift, and content drift before any numeric subtraction. Validation must prove the union of the two comparison inventories equals the complete NPZ inventory, so no AMP or other array can be silently skipped.

- [ ] **Step 3: Implement complete owner state snapshots**

Add concrete helpers with these outputs:

```python
snapshot_parameters(module) -> {
    "parameters": {
        name: {"shape": list, "dtype": str, "sha256": full_tensor_sha256},
    },
    "aggregate_sha256": sha256(canonical ordered parameter records),
}

snapshot_optimizer(optimizer, named_parameters) -> {
    "param_groups": ordered groups with parameter names and all non-param values,
    "state": per-parameter exact state-key inventory; every tensor has shape/dtype/hash
             and every zero-dimensional step tensor also has its integer scalar value,
    "uninitialized": sorted parameters absent from optimizer.state,
    "aggregate_sha256": sha256(canonical ordered groups/state records),
}

snapshot_rms(store, roles, prefix) -> {
    role: {
        "mean": SnapshotStore array reference,
        "var": SnapshotStore array reference,
        "count": float,
        "training": bool,
    }
}

snapshot_rng(store, prefix) -> {
    "numpy": algorithm, full uint32 keys reference, position, gaussian fields,
    "torch_cpu": full uint8 state reference,
    "torch_cuda": ordered full uint8 state references for every visible CUDA device,
}
```

The LR snapshot must describe both native owners. Actor state includes `agent.last_lr`, actor optimizer group LRs, the actor scheduler class, and strict-JSON scalar state exposed by `vars(agent.scheduler)`. Central state includes `agent.central_value_net.lr`, central optimizer group LRs, the central scheduler class, and strict-JSON scalar state exposed by `vars(agent.central_value_net.scheduler)`. Capture both owners at input and output because native `CentralValueTrain.train_net()` updates its scheduler and LR independently. GradScaler uses its public `state_dict()` plus `is_enabled()`; do not infer overflow solely from a label.

- [ ] **Step 4: Implement the three native cases**

For each case, call `configure_canonical_execution()`, copy runner params, set only `mixed_precision`, `use_others_experience="none"`, and `mini_epochs=2`, then construct the agent through `_agent_and_env`. Fill actor and central parameters deterministically and prove the fill consumes no RNG.

The only algorithm-input behavior replacements allowed are:

```text
a2c_common.shuffle_batch       -> return the same frozen batch object without RNG use
agent.play_steps               -> return a clone of the fixed Code #3 post-shuffle batch
```

The Code #3 `SyntheticVecEnv` intentionally lacks the RL-Games training-only `set_train_info` method, while native `A2CBase.train_epoch()` calls it unconditionally. Install one test-boundary ABI adapter on that synthetic instance which accepts the exact `(agent.frame, agent)` call, records it, and otherwise has no effect. Prove exactly one expected call for each normal case and remove the instance attribute in `finally`. This adapter is not an alternate update path and must not be installed on production envs.

Three observation boundaries may delegate without altering arguments or results: wrap `agent.prepare_dataset` to snapshot both complete native `values_dict` objects, wrap `agent.train_central_value` to capture its native scalar return and append the `central` order marker, and install one actor-model forward pre-hook to append the first `actor` marker and record real CUDA autocast enabled/dtype. Do not patch `PPODataset.__getitem__`, losses, optimizer, scaler, scheduler, RMS, or tensor primitives.

The `play_steps` replacement returns the complete declared input pair: a fresh clone of every frozen Code #3 batch field plus `played_frames=48` and `step_time=0.0`, and `ps_extras={"mb_intr_rewards": None, "rewards": zeros((56, 1))}`. Snapshot that entire pair under `input.batch`; do not leave the synthetic extras implicit.

Normal cases call native `agent.train_epoch()` once and store the deterministic algorithm portion of its return using this exact tuple layout:

```python
(
    _step_time,
    _play_time,
    _update_time,
    _total_time,
    a_losses,
    c_losses,
    b_losses,
    entropies,
    kls,
    last_lr,
    lr_mul,
    _extra_infos,
) = native_return
algorithm_return = {
    "a_losses": a_losses,
    "c_losses": c_losses,
    "b_losses": b_losses,
    "entropies": entropies,
    "kls": kls,
    "last_lr": last_lr,
    "lr_mul": lr_mul,
    "excluded_wall_clock_fields": ["play_time", "update_time", "total_time"],
}
```

Derive actor attempt count from `a_losses` and optimizer step/skip counts from complete before/after optimizer and scaler states. Overflow calls native `prepare_dataset`, clones `agent.dataset[0]`, asserts a tree diff of exactly `advantages[0]`, sets that value to `+inf`, then calls native `agent.train_actor_critic` once and records the loss/entropy/KL/LR portion of that native return. Always restore raw patched objects, remove the forward hook, and close the writer in `finally`.

- [ ] **Step 5: Implement validation and Source→Target comparison**

`validate_capture()` must enforce the exact state contract, owners, 56 rows/14 sequences, dataset length four and tail batch size twenty, call order, eight normal actor attempts, central optimizer step eight, recorded actor attempt/step/skip arithmetic, four distinct RMS roles with active central alias, complete optimizer/state inventories, complete RNG components, FP32/AMP scaler modes, and overflow parameter/optimizer invariance plus scaler backoff. It must not hard-code that uninjected `normal_amp` has zero skips; Source and Target must agree on the observed count.

`compare_capture()` must first compare case JSON structure and discrete/model/optimizer hashes, then exact-domain arrays, then every numeric array with `atol=1e-6, rtol=1e-5`. Before finite subtraction, require identical `NaN`, positive-infinity, and negative-infinity masks for each Source/Target numeric pair; matching non-finite positions contribute zero to the diagnostic error, mismatched masks fail, and tolerance plus maximum absolute error operate only on positions where both values are finite. If a pair has no finite positions its diagnostic maximum is `0.0`, so the aggregate `max_abs_error` is always finite. Add an in-memory RED for both accepted matching masks and rejected mismatches. Source fixture metadata must hash-validate Source arrays and Target capture metadata must independently hash-validate Target arrays; across Source/Target, numeric metadata compares name/shape/dtype/domain but deliberately does not require the two content hashes to match before tolerance comparison. It returns the ordered case names, native owner-path evidence, compared exact/numeric array counts, and maximum finite numeric absolute error. It must reject an empty or incomplete comparison inventory and must never compare Source AMP to Source FP32; comparison is Source vs Target within each named case. Source and Target module hashes are separately provenance-validated, not required to equal each other because approved compatibility patches differ.

- [ ] **Step 6: Run lightweight in-memory tests without the old fixture or generator**

Run only the individual in-memory schema/store/snapshot/validation/comparison/restore tests plus wrong-namespace rejection. Do not select any test that calls `load_update_fixture()` or imports generator serialization against the schema-v1 artifact. Expected: all selected tests pass with zero skips; frozen-fixture replay and external-anchor tests remain RED until regeneration, while generator safety tests remain for Task 3.

### Task 3: Simplify the Source-only generator

**Files:**
- Rewrite: `scripts/generate_simtoolreal_sapg_update_fixture.py`
- Test: `tests/algos/rlgames_sapg/test_update_golden.py`

- [ ] **Step 1: Preserve provenance and remove forensic transaction machinery**

Keep exact checks for:

```text
Source HEAD             2a9917533bfea70419ed2667a511d7238e5b3abc
Source RL-Games tree    7a6a0bb090998d00565aaefa6ab9f2b3d356ace2
train owner blob        f363d05d4a24b190b7837703b93270d8f3fe9a9c
task owner blob         6469d46867081b70edaa589dcb31c7090b64d45e
Code #3 NPZ             3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8
Code #3 manifest        785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f
```

Keep Source-module inventory verification and owner YAML values parsed from verified Git blobs. Remove the 8 MiB blocker, hard-link backups, pair rollback, inode transaction ledger, FIFO/socket/device handling, and declared-not-observed generation environment.

- [ ] **Step 2: Serialize fully in memory and replace regular leaves**

Validate that output is exactly `tests/fixtures/simtoolreal_sapg`, every existing path component is a real directory, and either output leaf is absent or a non-symlink regular file. Serialize NPZ and strict JSON to memory, read both back in memory, validate inventory/metadata/payload anchors, write same-directory temporary regular files, `fsync`, then `os.replace` each leaf. No ordinary pytest path invokes `generate()`.

- [ ] **Step 3: Run generator safety tests**

Run the tests covering real output root, symlink rejection, in-memory serialization round-trip, and missing/extra/shape/dtype/content corruption. They must not load or validate the old checked-in fixture. Expected: all selected tests pass with zero skips. Strict external file/payload anchors remain RED until Task 4 regenerates and freezes the schema-v2 fixture.

### Task 4: Generate Source oracle and reach GREEN

**Files:**
- Modify: `tests/algos/rlgames_sapg/source_update_harness.py`
- Modify: `tests/algos/rlgames_sapg/test_update_golden.py`
- Generate: `tests/fixtures/simtoolreal_sapg/source_update_fp32.npz`
- Generate: `tests/fixtures/simtoolreal_sapg/source_update_manifest.json`

- [ ] **Step 1: Run the one authorized Source-only generation**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
UNILAB_SAPG_ORACLE_MODE=source \
uv run --isolated --no-project --python 3.11 \
  --with gym==0.26.2 --with torch==2.7.0 --with numpy==2.4.4 \
  --with omegaconf==2.3.0 \
  --with-editable /home/user/ws/lemon/simtoolreal/rl_games \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  --source /home/user/ws/lemon/simtoolreal \
  --output tests/fixtures/simtoolreal_sapg
```

Expected: three Source cases complete on the canonical RTX 4090/cu128 platform and the command prints NPZ file SHA256, manifest file SHA256, and canonical payload SHA256.

- [ ] **Step 2: Freeze the three printed anchors**

Use `apply_patch` to place the exact printed values in both harness and test constants. Re-running ordinary pytest must only verify them; it must not update them.

- [ ] **Step 3: Run focused Target GREEN**

```bash
CUDA_VISIBLE_DEVICES=0 \
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py -q
```

Expected: all focused tests pass, zero skips, and all three Target cases execute. Every numeric array must satisfy the approved per-array `atol=1e-6, rtol=1e-5`; report maximum absolute error only as a diagnostic.

- [ ] **Step 4: Self-review the five paths**

Run `git diff --check`, confirm all five leaves are regular files, inspect fixture sizes without enforcing a byte ceiling, confirm Source and Code #3 hashes are unchanged, and confirm no `event_ledger`, capability, sentinel, primitive trace, or transaction machinery remains.

### Task 5: Controller review and acceptance gates

**Files:**
- Review only: the exact five Code #4 paths

- [ ] **Step 1: Hand off without Git writes**

The implementation agent reports the RED, generator command, printed anchors, focused test count/skip count, current diff/stat, and any warnings. It must stop writing and must not stage or commit.

- [ ] **Step 2: Run independent spec review, then code-quality review**

The spec reviewer checks same complete input, native owner path, all required final states, three branches, namespace separation, and exact five-file scope. Only after that passes, the quality reviewer checks maintainability, dead forensic code, validation order, path safety, and test quality. Important/Critical findings return to the same writer and are re-reviewed.

- [ ] **Step 3: Run fresh acceptance validation**

```bash
CUDA_VISIBLE_DEVICES=0 UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py -q

CUDA_VISIBLE_DEVICES=0 UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg -q

env -u UV_INDEX uv run --python 3.11 \
  pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q

env -u UV_INDEX uv run --python 3.11 scripts/audit_simtoolreal_rlgames_vendor.py

env -u UV_INDEX uv run --python 3.11 ruff check \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py

env -u UV_INDEX uv run --python 3.11 ruff format --check \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py
```

Expected: every command exits zero, required pytest reports zero skips, and the audit prints `72 selected Python blobs verified`. Code #4 does not run `make test-all`.

- [ ] **Step 4: Create the exact implementation commit**

The control session stages exactly the five Code #4 paths, verifies the cached path inventory and `git diff --cached --check`, then commits:

```text
test: lock SAPG update and AMP semantics
```

After a fresh post-commit focused gate, begin the separately planned Code #5 work. Do not begin Code #6.
