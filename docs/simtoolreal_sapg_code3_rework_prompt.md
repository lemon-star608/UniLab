# SimToolReal SAPG Code #3 Review Rework Prompt

> 实现 session：完整阅读本文件和根目录 `AGENTS.md` 后直接按 TDD 返工。不要只给计划。完成后保持未暂存、未提交并返回 `# DONE`；触发停止条件则返回 `# BLOCKED`。

你负责修正 code commit #3 / O1a rollout-RNG oracle 的独立审查阻断项。当前数值 replay 已经在 canonical 平台 exact，但证据边界仍不完整，所以本轮不是 code #4，也不是重做算法；只把现有五个 code #3 文件收严到可提交状态。

最终 code commit 仍为：

`test: lock SAPG rollout and RNG semantics`

## 一、仓库、基线与当前状态

工作目录：

`/home/user/ws/lemon/rlgame-unilab/UniLab`

固定 Source：

`/home/user/ws/lemon/simtoolreal`

预期分支：

`feat/simtoolreal-sapg-rlgames`

固定 code baseline：

`1adb159e9fb82ff322653a3533e9b2f32c3f862a`

本返工 prompt 的父级 docs HEAD 为：

`897c9c2710b154f06254c725cb2ec1127f54004f`

实现 session 的实际 HEAD 可以在上述 commit 之上仅包含控制 session 的 docs commit。开始时记录：

```bash
git rev-parse --abbrev-ref HEAD
SAPG_CODE3_REWORK_START_HEAD=$(git rev-parse HEAD)
printf '%s\n' "$SAPG_CODE3_REWORK_START_HEAD"
git merge-base --is-ancestor \
  1adb159e9fb82ff322653a3533e9b2f32c3f862a HEAD
git merge-base --is-ancestor \
  897c9c2710b154f06254c725cb2ec1127f54004f HEAD
git status --short
git diff --cached --name-only
git -C /home/user/ws/lemon/simtoolreal rev-parse HEAD
git -C /home/user/ws/lemon/simtoolreal status --short
git -C /home/user/ws/lemon/simtoolreal diff --stat
git -C /home/user/ws/lemon/simtoolreal diff --cached --stat
```

预期只有以下五个未跟踪文件，暂存区为空：

```text
?? scripts/generate_simtoolreal_sapg_rollout_fixture.py
?? tests/algos/rlgames_sapg/source_rollout_harness.py
?? tests/algos/rlgames_sapg/test_rollout_golden.py
?? tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz
?? tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json
```

若存在其他 tracked/untracked/staged 变化，停止并返回 `BLOCKED`。不得删除或覆盖其他 session 的工作。

固定 Source identity 不变：

- Source HEAD：`2a9917533bfea70419ed2667a511d7238e5b3abc`
- RL-Games tree：`7a6a0bb090998d00565aaefa6ab9f2b3d356ace2`
- Train owner blob：`f363d05d4a24b190b7837703b93270d8f3fe9a9c`
- Task owner blob：`6469d46867081b70edaa589dcb31c7090b64d45e`

## 二、唯一平台口径

本轮只使用：

- Python 3.11
- Torch 2.7.0+cu128
- `cuda:0`
- V2b manifest 锁定的 RTX 4090 canonical platform

需要解析 canonical Torch 的 Source capture 和 Target SAPG pytest 命令必须显式带：

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128
```

Target replay 必须得到 `is_canonical_platform=True`，不得走 tolerance fallback，不得 skip。

不要运行 Python 3.10、3.12、3.13，不要测试 cu126 或其他 CUDA 版本，不要增加版本矩阵。本轮所有 Python、pytest、Ruff、audit 命令都使用 `uv run --python 3.11`，禁止直接运行 `python`。

## 三、唯一允许的文件范围

只允许修改现有三个手写文件并由 generator 重新生成两个 fixture：

1. `scripts/generate_simtoolreal_sapg_rollout_fixture.py`
2. `tests/algos/rlgames_sapg/source_rollout_harness.py`
3. `tests/algos/rlgames_sapg/test_rollout_golden.py`
4. `tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz`
5. `tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json`

不得新增第六个文件，不得修改 docs、vendor、root config、V2a/V2b 文件或任何生产代码。手写编辑使用 `apply_patch`；两个 fixture 只能由 generator 生成。

不得 add、commit、push、stash、reset、clean、checkout 或切换分支。不得运行 `make test-all`。

## 四、必须修复的四个提交阻断项

### 4.1 冻结完整原始 ExperienceBuffer owner boundary

当前 fixture 只保存了 `play_steps()` 已经 `swap_and_flatten01` 的返回值，没有保存 `agent.experience_buffer.tensor_dict` 的原始 `[time, env, ...]` storage。这不满足原 prompt §6.1。

必须：

1. 在 `agent.play_steps()` 返回后立刻、任何 delta probe 或 augmentation 之前，对完整的 `agent.experience_buffer.tensor_dict` 做 detached deep clone。
2. 递归序列化该 snapshot 的全部 tensor leaf，使用清晰且稳定的 `buffer_raw_experience__...` 前缀；不能只挑选若干字段。
3. 固定并测试本配置下的完整字段 inventory。本配置的 exact key set 是：
   - `actions`
   - `mus`
   - `sigmas`
   - `neglogpacs`
   - `values`
   - `obses`
   - `states`
   - `dones`
   - `rewards`
   - `intr_rewards`
4. 固定每个 raw tensor 的 shape、dtype 和 SHA256，并证明首两维是 `[4, 12, ...]`。本固定配置的预期 owner storage 是：

   | 字段 | raw shape | dtype |
   |---|---:|---|
   | `actions` | `[4, 12, 29]` | `float32` |
   | `mus` | `[4, 12, 29]` | `float32` |
   | `sigmas` | `[4, 12, 29]` | `float32` |
   | `neglogpacs` | `[4, 12]` | `float32` |
   | `values` | `[4, 12, 1]` | `float32` |
   | `obses` | `[4, 12, 141]` | `float32` |
   | `states` | `[4, 12, 163]` | `float32` |
   | `dones` | `[4, 12]` | `uint8` |
   | `rewards` | `[4, 12, 1]` | `float32` |
   | `intr_rewards` | `[4, 12, 1]` | `float32` |

   若原生 runtime 实际 inventory、shape或dtype不同，不得修改预期来蒙混通过；先核对固定 Source owner，无法解释则返回 `BLOCKED`。
5. 使用当前加载的原生 `rl_games.common.custom_utils.swap_and_flatten01` 对 cloned raw tensors 做 observation-only transform。对 `agent.tensor_list` 中实际由 `play_steps()` 返回的 `actions/neglogpacs/values/mus/sigmas/obses/states/dones`，逐项证明 native raw transform exact 等于 returned `base`；不得在 harness 手写 transpose/reshape 来替代 owner 函数。
6. manifest显式记录 raw 前两轴为 `["time", "env"]`。raw obs row-ID matrix应为 `t0=[0,8,...,88]`、`t1=[1,9,...,89]`、`t2=[2,10,...,90]`、`t3=[3,11,...,91]`；flattened row identity保持 `e0t0,e0t1,e0t2,e0t3,e1t0,...,e11t3`。
7. `rewards/intr_rewards` 不在 `agent.tensor_list`，所以不得声称 `play_steps()` returned `base` 中存在对应字段。它们必须直接冻结 raw storage；可以另存原生 `swap_and_flatten01` diagnostic 来锁布局，但要明确这是 diagnostic，而不是 returned batch。
8. 明确锁住：即使 `intr_reward_model is None`、`extras["mb_intr_rewards"] is None`，ExperienceBuffer 仍然存在原始 `intr_rewards` tensor；fixture 必须直接包含它，不能只从 reward-shaper spy 间接推断。
9. tail obs/value、stored rewards、done 以及已有 rollout/augmentation/shuffle 证据继续保留。

这里允许调用并记录 Source 原生 owner 方法；不允许复制 `swap_and_flatten01` 公式或修改 Source。

### 4.2 Target replay 必须先校验每个数组的完整 metadata

当前 Target replay 只比较裸 bytes hash 和数值误差。相同 bytes 的 `(8,)` 与 `(1, 8)` 可能逃过 hash，并通过 NumPy broadcast 得到零误差。

必须在任何减法、broadcast 或数值容差比较之前，对每个 Target array 执行等价于：

```python
array_metadata(actual) == fixture.manifest["npz_arrays"][name]
```

完整 metadata 至少包括：

- shape
- dtype
- canonical content SHA256

要求：

1. inventory 不同立即失败；
2. shape 或 dtype 不同立即失败，不能进入 broadcast；
3. canonical content hash 不同立即失败；
4. 只有 metadata 合法后才能计算并报告 FP32 max absolute/relative error；
5. 错误消息包含数组名和 drift 类型。

在现有 `test_rollout_golden.py` 增加轻量 regression test，至少证明：

- 相同 raw bytes 但 reshape 后的数组被拒绝；
- dtype drift 被拒绝；
- 合法 metadata 通过。

可提取一个窄的 validation helper 供 replay 和轻量测试共用；不要为了测试复制 replay 逻辑。

### 4.3 Generator output/root/leaf symlink 必须 fail closed

当前 `generate()` 把 `output.resolve()` 传给 `_write()`，导致 `_write()` 检查时已经丢失 output root 的 symlink 身份；两个 fixture leaf 也没有写前检查。

必须：

1. 在任何 `resolve()`、`mkdir()`、遍历或写入之前检查调用方传入的原始 output path。
2. output root 是正常 symlink或 broken symlink时都拒绝。
3. output 已存在但不是真实目录时拒绝。
4. 在写任何一个 fixture 前，分别检查：
   - `source_rollout_fp32.npz`
   - `source_rollout_manifest.json`
5. 任一 leaf 是正常 symlink、broken symlink或其他非 regular file时拒绝。
6. 已存在的 regular fixture file可以由 canonical regeneration 覆盖。
7. 不得再通过 `_write(output.resolve(), ...)` 消除身份。将root/leaf检查集中到一个窄validator/writer boundary，`generate()`必须把原始CLI `output` 交给该boundary并使用其验证结果；不能在caller中先resolve。
8. 失败不得改写 symlink target 或另一个 fixture leaf。

在现有 `test_rollout_golden.py` 用 `tmp_path` 增加轻量 regression tests。root cases必须覆盖接收原始CLI path的 `generate()` → validator/writer wiring，或等价地证明 `generate()` 第一时间把未经resolve的path交给同一个被测boundary；不能只直接调用当前本来就会拒绝symlink的 `_write(symlink)`。允许为这一纯路径测试stub capture/provenance重活，但不得加载Source runtime。至少覆盖：

- 真实 output directory正常通过窄的写入 helper；
- output root symlink拒绝；
- broken output root symlink拒绝；
- NPZ leaf symlink拒绝；
- manifest leaf symlink拒绝；
- leaf directory等非 regular target拒绝；
- 被指向的外部 sentinel bytes保持不变。

不要因此引入新的依赖或第六个测试文件。原子写入不是本轮新增 contract；若实现会显著扩大范围，不要顺手加入。

这是原 prompt“普通pytest只读canonical fixture”的一个窄例外：pytest只允许在pytest管理的 `tmp_path` 中测试path validator/writer，禁止调用Source capture、禁止运行完整 `generate()` 重活、禁止写或重生成仓库中的canonical fixture。

### 4.4 RNN instrumentation 必须真正 delegate-only

当前 `rnn_spy` 在一次 wrapper 调用中先执行 masked native forward，又额外执行 unmasked native forward。固定单层无 dropout LSTM 下虽然没有观察到额外 RNG 消费，但它违反“spy只调用一次并原样返回”的 oracle 边界。

必须：

1. `rnn_spy` 每次只调用一次 `original_forward(input, states, done_masks, bptt_len)`。
2. spy只记录 frozen input/state/done、masked output，并原样返回这一次调用的原对象；不能在 wrapper 内做第二次 native forward。
3. 恢复 `native_rnn.forward` 后，再用已捕获的 cloned frozen inputs/states做明确命名的 isolated unmasked diagnostic。
4. unmasked diagnostic 必须直接调用恢复后的原生 RNN forward，不能复制 RNN/reset 数学。
5. 记录 `before_unmasked_rnn_diagnostic` 与 `after_unmasked_rnn_diagnostic` 的完整 NumPy、Torch CPU、Torch CUDA RNG state，并 assert/manifest 证明三者都未消费。
6. 保留 masked/unmasked returned hidden/cell 不同的现有证据，但在 manifest 中明确它来自隔离 diagnostic，而不是 delegating spy。
7. 审计本文件中的其他 spy，确保每个 wrapper 对 owner 函数都恰好调用一次并原样返回；不得为了预知结果额外调用 RNG 或 native model。

## 五、native inventory 与语义自证

manifest 的 `native_calls` 必须显式列出，而不是只靠上层方法间接暗示：

- `ExperienceBuffer.tensor_dict` raw snapshot
- `swap_and_flatten01` raw-to-flatten transform
- `filter_leader` native path used by `augment_batch_for_mixed_expl`
- 原来已经记录的 `play_steps`、`discount_values`、augmentation、shuffle、dataset、RNN和RMS owner calls

不要为列 inventory 额外 monkeypatch `filter_leader`；固定 Source module identity和原生 augmentation 调用链已足够，除非现有轻量 instrumentation 可以无行为变化地证明它。

以下现有 semantic 字段不能仅靠 literal 自报。用已捕获 arrays/call records推导并 fail fast：

- counterfactual current + tail 的实际 call count和phase；
- counterfactual value前 privileged state carrier未 relabel；
- 返回给训练的 augmented states carrier已 relabel；
- raw intrinsic reward存在，但 follower TD没有使用 intrinsic项，因为 native extras为 `None`。

不得在这里手写 TD、GAE、filter 或 carrier relabel公式；只比较原生输入、输出与已捕获row/carrier关系。

## 六、TDD返工顺序

先只修改现有 `test_rollout_golden.py`，增加能在当前实现上暴露缺口的轻量 tests：

1. raw ExperienceBuffer inventory/layout缺失；
2. same-bytes reshape metadata逃逸；
3. dtype drift逃逸；
4. output root/leaf symlink绕过；
5. RNN delegate-only和isolated diagnostic contract缺失。

执行最窄RED；新增测试整体必须失败，但不要求每个case都独立为RED，例如当前 `_write()` 直接接收root symlink时本来就会拒绝。关键是root wiring test必须能暴露 `generate()` 预先 `resolve()` 的caller漏洞。命令必须使用 Python 3.11。若测试需要加载vendored runtime，使用 canonical index：

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py -q
```

记录真实失败数量和每项准确原因。不要伪造“旧测试不存在”的第一次 code #3 RED；本轮报告的是 review-rework RED。

然后按 4.1 → 4.2 → 4.3 → 4.4 顺序逐项最小修复。helper-only tests应逐项GREEN；依赖新fixture的raw/RNN assertions在最后一次canonical regeneration前可以保持预期RED。全部手写修改完成后只需按下一节执行一次canonical regeneration，再跑完整GREEN；不要为了分步GREEN反复覆盖canonical fixture。

## 七、重新生成 Source fixture

所有手写修改完成后，用唯一 canonical 命令重新生成两个 fixture：

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

必须更新 harness和test中的固定 NPZ SHA256、manifest canonical payload SHA256。不得手工编辑NPZ或manifest。

重新生成后必须确认：

- Source HEAD/tree/owner blobs仍匹配；
- loaded modules全部来自固定 Source package root并匹配Git objects；
- raw ExperienceBuffer inventory/metadata完整；
- fixture合计仍不超过8 MiB；
- Source工作树没有本轮新增tracked/staged变化；
- Source既存untracked文件未被依赖、修改或删除。

## 八、必跑验证

### 8.1 Canonical Target focused replay

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py -q
```

### 8.2 完整SAPG oracle

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg -q
```

两项都必须 `is_canonical_platform=True`、0 skip。所有可严格映射数组必须metadata exact，全部canonical computed hashes必须exact。FP32 max absolute/relative error仍作为diagnostic报告，但exact hash成立时应全部为0；原容差不能替代exact gate，也不能放宽。

### 8.3 Vendor与audit

这些命令不要继承PyTorch download index：

```bash
env -u UV_INDEX uv run --python 3.11 \
  pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q

env -u UV_INDEX uv run --python 3.11 \
  scripts/audit_simtoolreal_rlgames_vendor.py
```

### 8.4 Ruff与Git边界

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

不要测试其他Python/CUDA版本，不要运行额外cu128 replay下载诊断，不要运行 `make test-all`。

## 九、停止条件

出现任一情况立即返回 `# BLOCKED`：

- 需要修改五个允许路径之外的文件或新增第六个文件；
- 需要修改vendored runtime、Source或生产算法；
- 需要重写GAE、TD、shuffle、filter、RNN reset、RMS或任何SAPG数学；
- 需要让Source和Target两套`rl_games`在同一进程加载；
- canonical Python 3.11 + cu128不可用或Target不是canonical；
- required pytest出现skip；
- Source identity/module blob漂移；
- Source/Target出现未解释metadata、tensor、index、hash或RNG差异；
- 必须放宽容差；
- fixture超过8 MiB；
- 三个手写文件的net handwritten LOC超过900；当前返工增量应通过窄helper控制在原“约800 LOC”附近，生成的NPZ/manifest不计入；
- 需要进入loss、backward、optimizer、AMP、checkpoint、player或code #4。

## 十、交接格式

完成后返回 `# DONE`，至少包含：

1. 五个允许路径的最终byte size和out-of-scope path count。
2. review-rework TDD RED命令、exit code、失败数量及五类失败原因。
3. 新NPZ SHA256、manifest文件SHA256、manifest canonical payload SHA256。
4. fixture合计大小及8 MiB gate。
5. raw ExperienceBuffer精确字段inventory，以及每个字段的shape/dtype摘要。
6. raw `[time,env,...]` → native flattened `[env,time,...]` exact关系与row labels。
7. `intr_rewards` raw storage和`extras["mb_intr_rewards"] is None`的直接证据。
8. Target逐数组metadata exact结果，以及reshape/dtype regression tests结果。
9. root、broken-root、NPZ leaf、manifest leaf、non-regular leaf symlink tests结果；外部sentinel未变。
10. RNN wrapper一次delegate证据、isolated unmasked diagnostic结果及前后完整RNG exact结果。
11. 更新后的native-call inventory和非literal semantic关系摘要。
12. Source capture完整命令、Source provenance和canonical platform。
13. canonical focused/完整SAPG pytest实际数量、0 skip和最大FP32误差。
14. 既有repeat index、permutation、48+8=56、dataset batches、timeout/GAE/TD/RMS证据是否保持exact。
15. vendor suite、audit、Ruff、format、diff检查结果。
16. net handwritten LOC。
17. 完整 `git status --short` 和空的 `git diff --cached --name-only`。
18. 当前HEAD仍等于 `SAPG_CODE3_REWORK_START_HEAD`，给出完整SHA。
19. 明确确认未进入code #4、未运行`make test-all`、未add/commit/push、未stash/reset/clean/checkout、未修改Source。

独立控制审查通过前，不得自行进入 code #4。
