# SimToolReal SAPG Code #4：update oracle 第二次审查返工提示词

> **给返工 session：** 完整阅读根 `AGENTS.md`、`docs/simtoolreal_sapg_code4_prompt.md`、`docs/simtoolreal_sapg_code4_review_rework_prompt.md` 和本文件后直接执行。第一次返工修复了 frozen handoff、dataset phase、AMP step mask 和递归 Source→Target 对拍；这些成果必须保留。本轮只补齐控制审查确认仍缺失的 evidence invariants，不进入 Code #5，不提交代码。

## 1. 当前状态

工作目录：

```text
/home/user/ws/lemon/rlgame-unilab/UniLab
```

预期分支/HEAD：

```text
feat/simtoolreal-sapg-rlgames
dbe5bf3a66055218ea109aae67f6736d87f3e4e3
```

预期只有五个 Code #4 文件未跟踪，staged 和 tracked diff 都为空：

```text
?? scripts/generate_simtoolreal_sapg_update_fixture.py
?? tests/algos/rlgames_sapg/source_update_harness.py
?? tests/algos/rlgames_sapg/test_update_golden.py
?? tests/fixtures/simtoolreal_sapg/source_update_fp32.npz
?? tests/fixtures/simtoolreal_sapg/source_update_manifest.json
```

开始时记录：

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
git diff --cached --name-only
git diff --name-only
```

固定 Source 与 Code #3 provenance 不变：

- Source HEAD：`2a9917533bfea70419ed2667a511d7238e5b3abc`
- RL-Games tree：`7a6a0bb090998d00565aaefa6ab9f2b3d356ace2`
- train owner blob：`f363d05d4a24b190b7837703b93270d8f3fe9a9c`
- task owner blob：`6469d46867081b70edaa589dcb31c7090b64d45e`
- Code #3 NPZ：`3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8`
- Code #3 manifest file：`785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f`

第一次返工 artifacts 的实际 before-state：

- update NPZ：`28bb72818c27440c925742d833c54afb5100831785f7763458faf0fbebeefcce`
- update manifest file：`4029e8b60f664f6334c8f1ec096ea8ca9e1954d9b821e5d35daa18e29770e005`
- update payload：`c8ad87ca2679a4a3e4d4d1a6c6386e2597c74c8d1b9db8cd40ae135d06f65ebb`

上次报告中的 manifest file hash `bd8aa7fe...` 与工作树不符；本轮交接必须从最终文件重新计算完整 hash，不能沿用旧输出。

只能修改上述五个文件。所有手工编辑用 `apply_patch`，所有 Python 命令用 `uv run`。不得 add/commit/push/stash/reset/clean/checkout；不运行 `make test-all`；只用 Python 3.11 + canonical cu128/RTX 4090，不增加版本矩阵。

## 2. 已通过且不得回退

控制 session 已独立确认：

- identity `shuffle_batch` wrapper 实际调用，normal FP32/AMP 各一次，56 rows 原样 handoff；
- actor/central dataset phase 都是 `(0,0)..(0,3),(1,0)..(1,3)`，batch sizes 为 `[12,12,12,20]`；
- normal AMP step mask 为 `[T,T,T,T,F,T,T,T]`，scale sequence 为 `[65536,65536,65536,65536,32768,32768,32768,32768]`；
- recursive `validate_update_semantics()` 能拒绝单边 normalizer/AMP 字段漂移；
- fixed NPZ/payload constants 已存在；
- focused `13 passed`、full SAPG `43 passed`、vendor `37 passed`、audit/Ruff 通过。

不要撤销这些实现，也不要把 Source 的自然 AMP skip 改成八次成功。

## 3. 为什么仍不能提交

当前 recursive validator 只证明“Source 和 Target 记录了相同字典”，不证明该字典包含 Code #4 承诺的证据。控制 session 将 Source/Target 两边同时删除以下内容后，validator 仍通过，而七个 `evidence_inventory` 标签保持不变：

- 71/72 个 loss/clip/optimizer events；
- actor/central 两个 mini-epoch 的 normalizer snapshots；
- optimizer step result 与 per-step delta。

当前具体缺口：

1. `evidence_inventory` 是标签集合，不验证每类 evidence 的字段、数量、phase/batch coverage。
2. `prepare_dataset_inputs` 与 `prepare_dataset_values` 虽在临时 `traces` 中产生，却没有进入 NPZ 或 manifest，最终 fixture 中不存在 prepared old values/returns/advantages handoff。
3. `gradient_stages` 和详细 `scaler_transitions` 同样只存在临时 `traces`，没有进入最终 semantics/fixture；标签不能证明 scaled→unscaled gradient 或逐 batch scaler 边界。
4. fixture 中没有 native total loss、per-row selected entropy coefficient、autocast enabled/dtype、GradScaler growth tracker、scheduler 后 optimizer param-group LR、normal AMP skipped batch 的参数 unchanged 关系。
5. `normalizer_boundary.value_before_dataset=True`、`actor_mini_epochs=2`、`actor_batches_per_epoch=[4,4]` 仍是硬编码结论。
6. `normalizer_mini_epoch_boundaries` 标签只检查 `normalizer_prepare_after` 是否存在，即使所有 epoch snapshots 缺失仍会声明覆盖。
7. FP32 direct numeric comparison 仍只由名字筛选 21/45 arrays，没有显式 comparison inventory/count。
8. `metadata_validated_before_numeric=True` 与 manifest `numeric_subtraction_after_complete_validation=True` 仍是直接赋值；没有能杀死 gate reorder mutant 的执行顺序测试。
9. `_capture_update(..., expected_package_root)` 未使用参数；传入错误 namespace root 不会由 update harness 自身 fail closed。
10. generator output symlink/ancestor/broken-leaf safety没有 Code #4 近风险测试。

## 4. 先补 RED：验证 evidence，不验证标签

先只修改 `test_update_golden.py`，建立 lightweight RED。至少包含：

### 4.1 Evidence invariant mutant

构造 Source/Target 两份相同的 semantics copy，同时：

- 删除 `actor_loss_exp/clamp/max` 与 `critic_loss_max`；
- 删除所有 `clip_grad_norm`、`optimizer_step_result`；
- 删除 actor/central epoch normalizer snapshots；
- 保留原七个 `evidence_inventory` 字符串。

新增的 `validate_update_evidence_invariants()` 必须拒绝；不能只调用 recursive Source→Target diff。

### 4.2 Prepared dataset mutant

删除 prepared dataset 的 `old_values`、`returns` 或 `advantages` 任一字段，validator 必须报告具体字段。当前 fixture没有该 evidence，RED 必须真实失败。

### 4.3 AMP detail mutant

分别删除/篡改：

- actor autocast enabled/dtype；
- scaler growth tracker transition；
- normal AMP 第五个 batch的 step/skip parameter-changed relation；
- overflow `scaler.step` 调用证据。

每类 mutation 必须 fail closed，错误带 case/epoch/batch/field。

### 4.4 Metadata ordering mutant

对 metadata validator 和第一次 numeric subtraction 安装 lightweight event recorder。正常顺序必须是：

```text
inventory -> every shape/dtype/content validation -> semantic validation -> numeric
```

临时将一次 subtraction 移到完整 metadata gate 前，测试必须 RED。不得用 manifest bool 或函数中的 `metadata_validated=True` 作为证据。

### 4.5 Namespace 与 generator path mutant

- `_capture_update` 或其可独立调用的 namespace guard 对错误 `expected_package_root` 必须拒绝；
- generator 正常 output directory通过；root/ancestor/broken symlink、existing non-directory root、normal/broken/directory leaf均在写前拒绝；外部 sentinel 不变。

先运行 lightweight tests，记录当前实现的真实 RED；不要先 regenerate fixture。

## 5. Evidence invariant contract

实现一个独立、纯验证的 `validate_update_evidence_invariants(semantics, rng_states, ...)`。它验证 Source fixture 自身和 Target capture各自满足不变量；调用顺序应为：

1. Source fixture anchor/inventory/metadata；
2. Source evidence invariants；
3. Target capture inventory/metadata；
4. Target evidence invariants；
5. Source→Target recursive semantic/RNG diff；
6. FP32 numeric comparison。

至少逐 case 验证：

- dataset batch phase/row identity和完整 `[12,12,12,20]`；
- normal cases各一次 identity freeze、相同对象、完整 before/after RNG hash；overflow 不调用；
- actor/central每个实际 batch有 forward、loss、backward/gradient、clip和相应 step/skip evidence；
- normal FP32 actor 8 个成功 step，normal AMP step mask与实际底层 step events一致，overflow actor 0 个底层 step；
- optimizer step result确实包含每参数 delta与optimizer state，参数集合来自实际 optimizer param groups而非简单复制 `model.named_parameters()`；
- actor/central epoch normalizer snapshots各两份，count transition证明 first epoch更新、second epoch冻结；
- required RNG phases完整，freeze/diagnostic不消费额外 RNG；
- scheduler恰好每 actor mini-epoch一次且 optimizer LR在 native `update_lr` 后与scheduler output一致。

`evidence_inventory` 可以保留用于人类阅读，但只能由 invariant validator验证成功后生成，不能再作为 coverage gate。

从真实 events 推导 mini-epoch/batch counts；删除或替换以下硬编码证明：

```python
"actor_mini_epochs": 2
"actor_batches_per_epoch": [4, 4]
"value_before_dataset": True
```

## 6. 补齐仍缺失的 native evidence

继续使用 delegating instrumentation，不复制算法公式。

### 6.1 Prepare/normalizer

把 `prepare_dataset` wrapper 已记录的输入和 native dataset outputs真正保存进 semantics：

- original values/returns；
- normalized old_values/returns；
- native advantages；
- actor/central dataset handoff字段、shape/dtype/hash；
- 实际 `train_value_mean_std` 参数；
- prepare前后及四个 owner/epoch normalizer snapshots。

### 6.2 Loss branches与 total loss

保留现有 native `exp/clamp/max/apply_masks` recorder，并增加 evidence invariant，明确把 events映射到每个 actor batch的：

- ratio；
- unclipped/clipped surrogate与native max；
- unclipped/clipped value loss与native max；
- bounds loss；
- raw entropy、selected per-row entropy coefficient、coefficient×entropy；
- apply_masks denominator和四项reduced loss；
- native total loss。

不要手算 total loss。可在 native `GradScaler.scale(loss)` delegating wrapper中记录传入的 loss；FP32 disabled scaler同样会经过该 owner调用。

selected entropy coefficient必须来自 owner执行路径的 delegating capture，例如记录 native block-index routing结果及 `agent.intr_reward_coef`；不能只引用 owner config中的全局 `entropy_coef=0.0`。

### 6.3 AMP/scaler/parameter relation

- actor forward期间记录实际 CUDA autocast enabled和dtype；central forward必须显示普通FP32；
- scaler的 `scale/unscale_/step/update` 全链路记录 state_dict中的 scale/growth tracker；
- `scaler.step` wrapper在每个batch前后记录actor parameter hash，即使底层 optimizer step被skip；
- normal AMP自然skip必须证明该batch参数不变，其余成功step参数改变；
- overflow记录真实 `scale(loss).backward → unscale → clip → scaler.step → scaler.update`，参数不变与backoff。

### 6.4 Optimizer与scheduler

- 从 optimizer param groups反向映射实际参数名，拒绝 unknown/missing/duplicate；
- 保留per-step delta、optimizer state、clip前后gradient；
- 给相关event添加明确 case/epoch/batch；
- delegating wrap native `update_lr`，记录调用后actor optimizer param-group LR；central scheduler/LR也记录实际 owner结果。

## 7. Metadata与numeric domain

让 `validate_update_arrays` 返回不可伪造的 validated result/token，或采用同等可测试结构，使 `_numeric_errors` 只能在 Source和Target完整inventory/metadata通过后调用。删除直接赋值的成功布尔：

```python
metadata_validated = True
```

manifest/test显式保存和断言 FP32 direct comparison names。不得继续用：

```python
name.startswith("input__") or "normal_fp32" in name
```

作为唯一coverage定义。comparison inventory至少覆盖所有保存进NPZ的 normal FP32 loss/KL/value/mu/sigma trace；manifest-only gradient/optimizer/normalizer evidence由semantic signature gate覆盖，并在manifest分开列明。

canonical loader必须直接执行固定 NPZ/payload anchors，而不只依赖另一个pytest函数事后assert。

update namespace在 capture结束前调用现有 `_assert_namespace(expected_package_root)` 或等价 fail-closed guard。Source generator仍须在 capture后做固定 Git-object module provenance。

## 8. Fixture预算

当前实际大小：

- NPZ：74,891 bytes；
- manifest：7,147,332 bytes；
- 合计：7,222,223 bytes，约为8 MiB上限的86.1%。

新增 evidence 后仍必须 `< 8,388,608` bytes。若接近上限，压缩 manifest中重复的描述性结构，复用按 case/epoch/batch索引的signature；不得删除 evidence、放宽hash或提交完整模型/Adam tensor。维护者只豁免了手写LOC，没有豁免8 MiB fixture gate。

## 9. Regeneration和验证

lightweight RED/GREEN全部完成后，只做一次 canonical Source regeneration：

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

然后只运行既定 Python 3.11/cu128 gates：

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
git diff --cached --name-only
git status --short
```

所有 required pytest必须0 skip。不要运行其他Python/CUDA版本，也不要运行 `make test-all`。

## 10. 交接报告

返回 `# DONE` 时必须给出：

1. 五个文件字节/LOC、fixture总字节和按8 MiB计算的百分比；
2. 最终 NPZ、manifest file、payload完整SHA256；
3. evidence invariant mutation RED，包括对称删除Source/Target evidence仍被拒绝；
4. prepared dataset字段与normalizer count transition；
5. 每batch loss branch/total loss/entropy coefficient coverage；
6. actor/central gradient、clip、optimizer/delta和actual param-group集合；
7. normal FP32/AMP/overflow的autocast、step mask、parameter relation、scale/growth tracker；
8. metadata validation → semantic validation → numeric ordering mutation；
9. namespace与generator path safety tests；
10. FP32 comparison inventory及exact max abs/rel；
11. focused/full/vendor/audit/Ruff实际结果；
12. branch/HEAD/status/staged=0，并确认Code #3/vendor/Source/production未改、未进Code #5、未运行make test-all、未增加版本矩阵。

若补齐 evidence 后 fixture超过8 MiB、必须修改第六个文件或需要复制算法公式，返回 `# BLOCKED`，不要用声明标签代替证据。
