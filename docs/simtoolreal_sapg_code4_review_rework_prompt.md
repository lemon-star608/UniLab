# SimToolReal SAPG Code #4：update oracle 审查返工提示词

> **给返工 session：** 完整阅读本文件、`docs/simtoolreal_sapg_code4_prompt.md` 和根目录 `AGENTS.md` 后直接执行。当前 Code #4 尚未通过控制审查；不要提交现有结果，也不要进入 Code #5。本轮只修复五个 Code #4 文件，使 frozen-input update/AMP oracle 真正验证 Source → Target 行为。

## 1. 结论与目标

当前 9 个 focused tests 虽然通过，但不能证明 Code #4 达成 O1b。控制审查已经复现以下事实：

1. `test_only_overrides.freeze_shuffle_batch=true` 只写入 manifest，没有安装 identity wrapper。`A2CAgent.train_epoch()` 仍在 `a2c_common.py:1382-1383` 调用 native `shuffle_batch`，把 Code #3 已经 post-shuffle 的 56 rows 再随机打乱一次。
2. `normal_amp` 的 8 个 actor batches 实际只有 7 次底层 optimizer step；GradScaler scale trace 是 `[65536, 65536, 65536, 32768, 32768, 32768, 32768, 32768]`。这是 Source 的真实观测，不能隐藏成“normal AMP 全部正常 step”，也不能人工改掉；Target 必须逐 batch 对齐 Source 的 step/skip mask 和 scaler transition。
3. `replay_update_fixture()` 只比较 central-before-actor、second-epoch reference、scheduler epoch 编号和 overflow 的三个布尔值。控制审查把 Target capture 中的 normalizer、gradient signatures、optimizer 参数集合、RNG、event log、normal AMP enabled/step 数、scheduler LR，以及 overflow scaler 字段全部篡坏后，replay 仍返回 `normal_amp_semantics_match=True` 和 `overflow_amp_semantics_match=True`。
4. `normal_amp_semantics_match` 与 `overflow_amp_semantics_match` 当前只是 case 名字符串检查。
5. FP32 direct numeric error 只覆盖 21/45 arrays；其余关键证据有的只存在 Source manifest，有的完全未记录。当前 exact content-hash gate 又先保证数组相同，使 `max_abs=0` 本身不能说明 loss/gradient/optimizer requirements 都进入了对拍。
6. 当前没有 ratio、clipped/unclipped surrogate、clipped/unclipped value branch、per-row entropy product/coefficient、clip 前后 gradient、逐 step parameter delta、optimizer state/LR、normalizer mini-epoch boundary 或 central clip 的完整证据。
7. actor dataset 事件的 `(mini_epoch, batch)` 当前为 `[(0,0),(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]`；central 两个 mini-epoch 被标成 batch `0..7` 且 mini_epoch 恒为 0。硬编码的 `actor_mini_epochs=2` 与 `[4,4]` 不能替代真实事件推导。
8. Code #4 fixture 没有像 Code #3 一样由测试中的固定 NPZ/payload SHA256 anchor 锁住，manifest 与其自带 hash 可以协同漂移。

返工目标不是另写一份 PPO update，而是补齐 delegating instrumentation、Target semantic validation 和 mutation tests。Source native Runner/A2CAgent/PPODataset/CentralValueTrain/loss/optimizer/GradScaler 仍是唯一算法 owner。

## 2. 仓库状态与固定边界

工作目录：

```text
/home/user/ws/lemon/rlgame-unilab/UniLab
```

预期分支和 HEAD：

```text
feat/simtoolreal-sapg-rlgames
b17f3d865675eb0cb91923c012d25149d8212c1c
```

开始时必须验证：

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
git diff --cached --name-only
```

预期只有以下五个未跟踪文件，staged 必须为空：

```text
?? scripts/generate_simtoolreal_sapg_update_fixture.py
?? tests/algos/rlgames_sapg/source_update_harness.py
?? tests/algos/rlgames_sapg/test_update_golden.py
?? tests/fixtures/simtoolreal_sapg/source_update_fp32.npz
?? tests/fixtures/simtoolreal_sapg/source_update_manifest.json
```

只能修改这五个文件。不得修改 Code #3、vendor、Source、`src/**`、`conf/**`、root packaging 或已有文档。维护者已批准 Code #4 超过原约 900 行的规模豁免；不要为了压行数删除证据，也不要增加第六个实现文件。

所有手工编辑使用 `apply_patch`；所有 Python 命令使用 `uv run`。不得执行 `git add`、commit、push、stash、reset、clean、checkout 或切换分支。不要运行 `make test-all`。

只验证 Python 3.11 + canonical cu128/RTX 4090。不要运行 Python 3.10/3.12/3.13、cu126 或其他矩阵。

固定 Source：

- repo：`/home/user/ws/lemon/simtoolreal`
- HEAD：`2a9917533bfea70419ed2667a511d7238e5b3abc`
- RL-Games tree：`7a6a0bb090998d00565aaefa6ab9f2b3d356ace2`
- train owner blob：`f363d05d4a24b190b7837703b93270d8f3fe9a9c`
- task owner blob：`6469d46867081b70edaa589dcb31c7090b64d45e`

固定 Code #3 inputs：

- NPZ SHA256：`3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8`
- manifest file SHA256：`785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f`

返工开始时的 Code #4 artifacts 仅作为 before-state 记录，返工后必须重新生成并更新固定 anchors：

- update NPZ SHA256：`d51bae2e0fc9d5e5fa34eacd48371341fed5d57d4f90138a08b8195a98253260`
- update manifest file SHA256：`401566b745cce3a1c126fc59a4823185d99bfcb740a1f56ee818d567cffa4388`
- update manifest canonical payload SHA256：`8ec6c6a2860d40cdde5c3b0d5de95545d0ba4ac050330706ebe888e2ffb9947f`

## 3. 先建立真实 RED

先只改 `test_update_golden.py`，增加近风险测试；不要先改 harness。至少让当前实现出现下列 RED：

1. frozen input rows 与 normal FP32/AMP 第一 central mini-epoch、第一 actor mini-epoch拼接后的 rows 不相等；当前实现应失败并显示发生了第二次 permutation。
2. actor/central `(mini_epoch,batch)` 不是各自两轮 `[0,1,2,3]`；当前实现应失败。
3. pure semantic validator 尚不存在，或无法拒绝 Target normalizer/gradient/optimizer/RNG/AMP/scheduler/scaler 漂移；当前实现应失败。
4. required update evidence inventory 缺少 loss branches、gradient stages、parameter delta、optimizer/scaler/normalizer fields；当前实现应失败。
5. 固定 update fixture NPZ/payload anchors 尚未由测试常量校验；当前实现应失败。

推荐先运行不触发完整 Target replay 的 lightweight RED：

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py \
  -k 'freeze or phase_indices or semantic_validator or evidence_inventory or fixed_fixture_anchor' -q
```

报告真实失败数和每类 root cause，不得把 collection/import error 冒充这些 RED。

## 4. 修复 frozen handoff，禁止第二次 shuffle

Code #3 的 `buffer_post_shuffle__*` 已经是 native shuffle 后的 frozen update universe。本轮 normal FP32/AMP 调用 native `train_epoch()` 时，在 test boundary 临时替换 `rl_games.common.a2c_common.shuffle_batch`：

- wrapper 必须只记录调用并直接返回同一个 batch object；
- 不得生成 permutation、clone/reorder tensor 或消费 RNG；
- 每个 normal case 必须调用恰好一次；
- overflow 的手动 prepare path 不得假装调用；
- `finally` 中恢复原函数；
- `freeze_shuffle_batch=true` 只有在 wrapper 实际安装时才能写入 manifest。

记录 wrapper 前后的 row-keyed hashes 和完整 NumPy/Torch CPU/CUDA RNG state。严格断言 normal FP32 与 normal AMP：

- central mini-epoch 0 拼接 rows == frozen 56 rows；
- actor mini-epoch 0 拼接 rows == frozen 56 rows；
- 每条 4-step sequence 保持完整；
- 两个 owner 的 batch sizes 都是 `[12,12,12,20]`；
- wrapper 前后 RNG 完全相同。

修复 dataset spy 的 phase accounting。不要依赖 `train_actor_critic()` 在 `dataset[i]` 返回之后才更新的计数。由每个 dataset 自身的 access counter 和 `len(dataset)` 推导 `(mini_epoch,batch)`，normal cases 必须得到：

```text
central: (0,0)..(0,3),(1,0)..(1,3)
actor:   (0,0)..(0,3),(1,0)..(1,3)
```

这些结论必须由 event log 推导，禁止再硬编码 `actor_mini_epochs=2`、`actor_batches_per_epoch=[4,4]` 后直接作为证明。

## 5. 补齐 native update evidence

只允许 delegating instrumentation；不得在 harness 重新实现 PPO、value、bounds、entropy、gradient clipping 或 optimizer 公式。短生命周期 wrapper 可以记录 native owner 的输入、输出和 native Torch op 结果，但必须调用原 owner 恰好一次并在 `finally` 恢复。

### 5.1 prepare 与 normalizer

记录并让 Target 对拍：

- prepare 前后的原始 values/returns、normalized old_values/returns、advantages；
- actor/central dataset handoff 的完整 inventory、shape、dtype、hash；
- actor input RMS、actor shared value RMS、central input RMS、central value RMS；
- prepare 前/后、central mini-epoch 0/1 后、actor mini-epoch 0/1 后的 mean/var/count/training；
- `train_value_mean_std=true` 的实际调用参数；
- input RMS 第一 mini-epoch 更新、第二 mini-epoch冻结的真实 count transition。

删除 `value_before_dataset=True` 这类无调用证据的常量。结论必须从记录的 call/event/snapshot 推导。

### 5.2 actor/value/bounds/entropy/KL

每个 actor batch 至少保存 native 小 tensor 或完整 signature：

- old/new neglogp、advantage、ratio；
- unclipped surrogate、clipped surrogate、最终 per-row actor loss；
- old/new value、return、value-clipped branch、unclipped/clipped squared error、native max result；
- mu、soft bound、per-row bounds loss；
- raw entropy、native per-row entropy product/coefficient evidence、reduced entropy numerator；
- apply_masks 输入四项、mask/None、sum_mask/mean denominator、reduced四项；
- native total loss；
- new mu/sigma、native KL；
- update_mu_sigma 前后的 range、row-keyed hash；
- scheduler 每 epoch 的 batch KL、mean KL、input/output LR/entropy、optimizer param-group LR。

若某个 intermediate 不是 native function 返回值，使用只在该 owner 调用期间生效的 delegating Torch-op recorder 捕获原执行产生的结果；不得在 owner 返回后用复制公式重算。

第二 mini-epoch old reference 必须按 row/range 对应比较，不只把前四个 tensor `cat` 后与后四个碰巧同序的 tensor比较。

### 5.3 gradients、clip、optimizer

对 central 和 actor 的每个实际 step 分开记录：

- backward 后 gradient signature 与 absent list；
- actor AMP scaled gradient；
- scaler.unscale_ 后、clip 前 gradient norm；
- native `clip_grad_norm_` returned total norm 与 clip 后 signature/norm；
- central FP32 backward、clip 前/后和 returned norm；
- optimizer step 前/后 parameter hashes；
- 每参数 delta 的 norm/sum/max/max_abs/name-seeded sentinels；
- optimizer state keys/step、param-group LR/eps/weight_decay；
- actor/central optimizer parameter name set 和排序 hash；
- `backward → unscale → clip → scaler.step/optimizer.step → scaler.update` 的真实 event order。

当前只在全部更新完成后读取一次 final gradient/parameter signature 不够。不要提交完整模型参数或完整 Adam state。

### 5.4 AMP 与 overflow

三个 case 仍保持：`normal_fp32`、`normal_amp`、`overflow_amp`。不得为了让名称好看而改 Source 的真实行为。

`normal_amp` 当前 Source 观测到一次自然 skip。返工应记录并 Target 对拍：

- mixed_precision/autocast enabled 与实际 autocast dtype；
- scaler enabled、initial scale、每 batch scale/growth tracker；
- 每 batch scaler.step 调用和底层 optimizer.step success mask；
- 当前 Source 的 8-batch success mask及 `65536 → 32768` transition；
- parameter changed/unchanged relation。

如果 identity shuffle 修复后 Source 的自然 step mask发生变化，以重新 capture 的 Source 事实为准，报告新序列，不得硬编码旧 7/8。

`overflow_amp` 仍只把第一个 actor batch clone 的 `advantages[0]` 改为 `+inf`，并调用 native scaler path。Source/Target 必须逐项比较 scaler enabled、scale before/after、growth tracker、scaler.step called、底层 optimizer step skipped、参数不变、central FP32。不能只比较三个布尔值，也不能用 case 名作为 match。

## 6. Target replay 必须验证全部语义

增加一个可单独测试的 pure semantic validation boundary。它接收 Source manifest evidence 与 Target capture evidence，按稳定字段逐项 fail closed；错误必须指出 case、phase、batch 和 field。

至少比较：

- 完整 event sequence及每个 phase/batch identity；
- freeze wrapper call count、row hashes和 RNG unchanged；
- prepared dataset与全部 normalizer snapshots；
- loss branch、KL、update_mu_sigma和scheduler traces；
- actor/central gradient、clip、optimizer/parameter delta signatures；
- normal FP32、normal AMP、overflow AMP control/scaler/step semantics；
- actor/central optimizer parameter sets；
- 所有要求 phase 的完整 RNG states/component hashes。

删除以下弱逻辑：

```python
capture.semantics["cases"]["normal_amp"]["case"] == "normal_amp"
capture.semantics["cases"]["overflow_amp"]["case"] == "overflow_amp"
```

`UpdateReplay` 的 match 字段必须来自上述 validator，不能来自字符串或硬编码 `True`。

Target namespace 要实际使用 `expected_package_root` 做 fail-closed 检查；不能保留未使用参数。

RNG validation 不能只证明额外的 zero-mask diagnostic 没消费 RNG。Source 与 Target 每个要求 phase 的 NumPy/Torch CPU/CUDA state必须对拍，freeze/spy 自身的 before/after 也必须证明无额外消费。

## 7. Array、metadata 与 fixture anchors

区分两个 gate：

1. fixture loader 先用固定外部 anchors 验证 Source NPZ 和 canonical manifest payload，再做完整 inventory、每数组 shape/dtype/content SHA256；
2. Target capture 先完成 inventory、shape、dtype和适用的 signature/hash validation，再进入 FP32 numeric subtraction。

FP32 direct numeric set 必须显式列入 manifest/test，覆盖 frozen inputs 之外的所有 FP32 loss、KL、normalizer、gradient和parameter-delta numeric evidence。测试断言 comparison-name inventory，不能只看 `max(default=0)`。

AMP 不与 FP32 case 比较；AMP Source/Target之间比较 control flow、dtype口径、step/skip、scaler state和明确列出的统计/signature。

`metadata_validated_before_numeric` 不能直接赋值为 `True`。使用可测试的 phase/event gate证明 complete inventory与所有 metadata validation在第一次 subtraction/`np.abs`/relative error之前完成。

重新生成 fixture 后，把新的固定值写入 harness/test 常量并至少校验：

- NPZ file SHA256；
- canonical manifest payload SHA256；
- manifest file SHA256可记录在报告中；
- 51 loaded Source modules 的 path/blob/SHA256；
- owner blobs和 canonical platform。

同时补 generator output root/ancestor/leaf symlink、broken symlink、directory leaf 的 lightweight tests；不能只测 loader。

## 8. 必须做的 mutation RED

至少完成以下四类临时 mutation；每次都用 `apply_patch`，记录失败，再精确恢复并核对文件 hash回到 mutation 前：

1. 删除/绕过 identity shuffle wrapper：row identity或 freeze RNG test 必须失败。
2. pure semantic validator 中跳过 normalizer、gradient/optimizer、RNG 任一类比较：对应 mutation test 必须失败。
3. 把 normal AMP 的 step mask/scaler transition 改成全成功或只保留 case 名：AMP test 必须失败。
4. 把 metadata gate 后移到第一次 subtraction之后：whole-inventory ordering test 必须失败。

另保留原三类 mutation 的有效覆盖：central-before-actor、update_mu_sigma、overflow scaler/optimizer。无需增加 Python/CUDA版本矩阵。

## 9. Source regeneration 与验证命令

只在全部 lightweight tests 变绿后做一次 canonical Source regeneration：

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

确认 regeneration 没有修改 Code #3 hashes或 Source。然后运行：

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

所有 required pytest 必须 0 skip。上游 AMP deprecation warning 可以如实记录；其他 warning/error/mismatch 必须解释或修复。

## 10. 交接报告

完成时返回 `# DONE`，至少报告：

1. 五个文件、字节数、净手写 LOC、fixture预算和 out-of-scope path count；
2. 新的 NPZ/file/payload SHA256与固定 Source/Code #3 hashes；
3. identity shuffle wrapper调用、56-row exact handoff、sequence完整性和 RNG unchanged；
4. actor/central真实 `(mini_epoch,batch)`、event order及 `[12,12,12,20]`；
5. prepared dataset、normalizer各 boundary 的 count/hash；
6. loss branches、mask denominator、KL/reference、scheduler/LR；
7. 每 batch gradient/clip/optimizer/parameter delta evidence；
8. normal FP32、normal AMP和overflow AMP逐 batch step/scaler结果；如 normal AMP仍自然 skip，必须明确列出；
9. Target semantic mutation拒绝证据、metadata-before-numeric证据和 direct numeric comparison inventory/count；
10. focused/full SAPG、vendor/audit、Ruff/format/whitespace 的实际结果；
11. branch、HEAD、`git status --short`、staged=0；
12. 明确确认未进入 Code #5、未修改 Code #3/vendor/Source/生产代码、未运行 `make test-all`、未增加 Python/CUDA矩阵、未执行 Git 破坏操作。

若必须修改第六个实现文件、vendor/Source/Code #3/生产代码，或需要手写算法公式才能得到 intermediate，立即返回 `# BLOCKED`，不要删减 evidence 或放宽 gate。
