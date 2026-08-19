# SimToolReal SAPG Code #3 Implementation Prompt

> 实现 session：请完整阅读本文件，然后严格执行。不要只给计划；直接按 TDD 开始工作。遇到停止条件时返回 `BLOCKED`，不得自行扩大范围。

你负责 UniLab SimToolReal Source-faithful SAPG migration 的 code commit #3：

`test: lock SAPG rollout and RNG semantics`

这是 O1a frozen-input differential oracle。只建立 Source → Target 的 rollout、return、augmentation、shuffle、RNG 与 RNN 数值证据，不修改生产代码，不进入 loss、optimizer、AMP、checkpoint 或 player。

## 一、仓库与当前基线

工作目录：

`/home/user/ws/lemon/rlgame-unilab/UniLab`

固定 Source：

`/home/user/ws/lemon/simtoolreal`

开始前完整阅读根目录 `AGENTS.md`。

预期分支：

`feat/simtoolreal-sapg-rlgames`

固定 code baseline：

`1adb159e9fb82ff322653a3533e9b2f32c3f862a`

当前 HEAD 可以在该 baseline 之上包含本 prompt 的独立 docs commit。实现 session 开始时必须记录当前 HEAD，并在交接时证明 HEAD 未变。预期 Target 工作树干净。

固定 Source identity：

- Source HEAD：`2a9917533bfea70419ed2667a511d7238e5b3abc`
- RL-Games tree：`7a6a0bb090998d00565aaefa6ab9f2b3d356ace2`
- Train owner：`isaacsimenvs/cfg/train/SimToolRealSAPG.yaml`
- Train owner blob：`f363d05d4a24b190b7837703b93270d8f3fe9a9c`
- Train owner SHA256：`04f30820094b062412541764b3feeb1492097e75afe5ad0df3fd0e2853496d34`
- Task owner：`isaacsimenvs/cfg/task/SimToolReal.yaml`
- Task owner blob：`6469d46867081b70edaa589dcb31c7090b64d45e`
- Task owner SHA256：`9d2bf514f75cc8c72b20da1e8ec971163bbd4cbdf6fc74812aa4a509340acb5e`

先执行：

```bash
git rev-parse --abbrev-ref HEAD
SAPG_CODE3_START_HEAD=$(git rev-parse HEAD)
printf '%s\n' "$SAPG_CODE3_START_HEAD"
git merge-base --is-ancestor \
  1adb159e9fb82ff322653a3533e9b2f32c3f862a HEAD
git status --short
git -C /home/user/ws/lemon/simtoolreal rev-parse HEAD
git -C /home/user/ws/lemon/simtoolreal rev-parse \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games
```

Target 若不是上述分支、固定 code baseline 不是当前 HEAD 的 ancestor，或工作树不干净，立即停止。Source 可以有既存 untracked 文件，但不得修改、删除或依赖它们；所有 Source owner 和模块字节必须对照固定 Git objects。

## 二、解释器和平台口径

本轮只使用：

- Python 3.11
- Torch 2.7.0+cu128
- V2b manifest 已定义的 canonical CUDA 平台
- `cuda:0`
- RTX 4090 / compute capability 8.9

必须复用 `tests/fixtures/simtoolreal_sapg/source_network_manifest.json` 中的完整 canonical platform identity 和 deterministic flags。Target replay 必须明确得到：

`is_canonical_platform=True`

不得使用 non-canonical tolerance fallback，不得 skip。

这是本轮唯一的解释器和平台口径；不要增加任何版本矩阵。migration plan 中与此冲突的旧验证要求已被 maintainer 本 prompt 覆盖。

所有 Python、pytest、Ruff、audit 命令都必须使用 `uv run --python 3.11`。禁止直接运行 `python`。

## 三、唯一允许的文件范围

只允许新增以下五个文件：

1. `scripts/generate_simtoolreal_sapg_rollout_fixture.py`
2. `tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz`
3. `tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json`
4. `tests/algos/rlgames_sapg/source_rollout_harness.py`
5. `tests/algos/rlgames_sapg/test_rollout_golden.py`

不得修改任何已有文件，包括：

- `third_party/simtoolreal_rl_games/**`
- `pyproject.toml`
- `uv.lock`
- V2a/V2b generator、harness、test、fixture或manifest
- 任何生产算法、runner、config、文档

不得新增第六个文件。若确实需要，停止并报告。

手工文件编辑一律使用 `apply_patch`。两个 fixture 文件只能由本轮 generator 生成。禁止 add、commit、push、stash、reset、clean、checkout或切换分支。

目标规模约 800 net handwritten LOC；生成的 NPZ和manifest不计入手写行数。若明显超过，应停止说明原因。

## 四、必须使用的 Source 原生调用链

不能用只带若干字段的 fake agent 调用 unbound methods，也不能在 harness 重新实现 Source 数学。

应从固定 Source owner 构造原生：

`Runner -> A2CAgent`

测试仅允许在边界提供：

- deterministic synthetic vecenv/env
- observer/writer/test output directory
- 只记录后 delegate 的 instrumentation

建议原生流程：

1. 从固定 Git object 读取并解析 Source train/task owner。
2. 在内存中的 owner copy 上施加下面明确列出的 synthetic overrides。
3. `Runner.load(...)`
4. `Runner.set_vec_env(synthetic_vecenv)`
5. 通过原生 algo factory 创建 continuous `A2CAgent`
6. `agent.init_tensors()`
7. 原生 `agent.env_reset()` / `agent.play_steps()`
8. 原生 `agent.augment_batch_for_mixed_expl(..., repeat_idxs=None)`
9. 原生 `rl_games.common.custom_utils.shuffle_batch(...)`
10. 原生 `agent.prepare_dataset(..., train_value_mean_std=False)`
11. 从原生 actor/central `PPODataset` 取 mini-batch

agent 的 train/log目录必须指向临时目录，结束时关闭 writer，不得在仓库留下 runs、event、checkpoint或cache。

以下实现必须来自当前加载的 RL-Games：

- `A2CAgent.play_steps`
- `discount_values`
- `ExperienceBuffer`
- `swap_and_flatten01`
- `augment_batch_for_mixed_expl`
- `filter_leader`
- `shuffle_batch`
- `PPODataset`
- `RnnWithDones`
- `RunningMeanStd`

严禁在 harness 重写：

- timeout bootstrap
- delta/GAE/return
- follower choice
- block roll/filter
- counterfactual TD target
- trajectory shuffle
- dataset slicing
- RNN done reset
- RMS公式

允许 observation-only instrumentation，但 wrapper 必须调用原函数并原样返回，不能改变结果或额外消费 RNG。可以观测：

- `np.random.choice`
- `torch.randperm`
- `agent.get_action_values`
- `agent.get_values`

如需要单独捕获 delta，可在 cloned frozen inputs 上临时将同一个 native agent 的 `tau` 设为 `0.0`，调用原生 `discount_values`，然后用 `try/finally` 恢复 `tau=0.95`。正式 advantage 必须来自原配置 tau 的 native 调用。不能自己写 delta公式。

## 五、固定 synthetic contract

测试专用配置固定为：

- num_envs / num_actors：12
- expl_coef_block_size：2
- blocks：6
- coefficient IDs：`[50, 40, 30, 20, 10, 0]`
- horizon_length：4
- seq_length：4
- actor obs：140维，另带1维coefficient carrier
- privileged state：162维，另带1维coefficient carrier
- actions：29
- use_others_experience：`lf`
- off_policy_ratio：`1.0`
- actor test-only minibatch_size：12
- central-value test-only minibatch_size：12

必须在 manifest 中把 Source owner 默认值和 test-only overrides 分开记录，不能把以下值伪装成已改变的 owner：

- Source 24576 env / 4096 block
- Source horizon/sequence 16
- Source minibatch 98304

固定结果：

- base rollout：12 × 4 = 48 rows
- native repeat：`[0, k]`
- leader保留完整48 rows
- follower只保留一个2-env block，即8 rows
- augmented batch：56 rows
- actor native dataset batches：`[12, 12, 12, 20]`
- central native dataset batches：`[12, 12, 12, 20]`

synthetic obs/state/reward/done/timeout 必须是有限、确定且带唯一 time/env row identity 的输入。done pattern 至少包含：

- 普通 non-terminal
- timeout + done
- done without timeout
- horizon 中足够早的 done，使后续一步能观察 RNN reset

不要使用物理引擎，不依赖 IsaacSim或MuJoCo。

## 六、必须锁定的 Source 行为

### 6.1 Rollout layout

记录并对拍：

- ExperienceBuffer 的原始 `[time, env, ...]`
- `swap_and_flatten01` 后的 `[env, time, ...]`
- flattened row identity 必须是 `e0t0,e0t1,e0t2,e0t3,e1t0,...`
- actions、mu、sigma、neglogp、values、obses、states、dones
- rewards、intr_rewards和tail obs/value

### 6.2 Timeout / GAE / return

记录：

- raw env reward
- reward shaper 后、bootstrap 前 reward
- timeout mask
- action 前 `get_action_values()` 返回且已被 central critic覆盖的 value
- timeout bonus
- ExperienceBuffer 中最终 stored reward
- rollout `mb_fdones`
- rollout结束后的 `fdones`
- tail value
- native delta
- native GAE/advantage
- return

必须锁住 Source 的 done定义：buffer中的 done 是“本步 action 前 done”；GAE 中间步读取 `mb_fdones[t+1]`，尾步读取 rollout 后 `fdones`。

timeout bootstrap 必须证明使用 action/env.step 前的 central value，而不是 next-state value或actor shared value。

### 6.3 RNN

同时覆盖：

- initial hidden/cell
- 每步传入 actor 的 hidden/cell
- actor model 返回的 pre-reset hidden/cell
- env done 后 `play_steps` 清零后的 hidden/cell
- `rnn_state_buffer`
- `mb_rnn_states`
- final/returned hidden/cell
- native dataset中的 rnn_states
- 带 sequence-internal done mask 的 native train-mode RNN forward
- train-mode forward 返回的 hidden/cell

必须证明长度4的sequence内 done boundary 真实触发 hidden/cell reset。不能只检查 LSTM output。

### 6.4 Follower selection / counterfactual

调用：

`augment_batch_for_mixed_expl(..., repeat_idxs=None)`

不能显式传 follower index绕过 RNG。

用 delegating spy 记录：

- candidate集合：`[1,2,3,4,5]`
- size：1
- replace：False
- native choice结果
- 完整 `repeat_idxs=[0,k]`
- follower来自哪个原始 block/env rows
- off_policy_mask

用 delegating `get_values` spy 记录 counterfactual current/tail调用的真实输入、RNN state与输出。本例 current 48 rows小于8192，因此应观察 current call + tail call。

必须锁住 Source 当前的一个重要行为，不能擅自修正：

- counterfactual阶段只改 `mb_obs` / `last_obs["obs"]` 的coefficient carrier；
- 用于 central value 的 privileged `states` 没有在该次 value计算前relabel；
- 但返回给训练的 augmented `states` buffer 又确实经过了 block relabel。

fixture必须忠实记录这个 Source行为。不要在 harness“修好”它。

由于 Source entropy配置下 `intr_reward_model is None`，follower one-step TD target不包含 intrinsic reward项；必须通过 native output记录，不能手算。

### 6.5 Shuffle / dataset

在 shuffle 前 clone全部 deterministic buffer。

记录原生 `torch.randperm` 的14条trajectory排列，并证明：

- 14条trajectory整体重排
- 每条trajectory内部4个时间步顺序不变
- env/source identity未拆散
- rnn_states按相同sequence索引重排

对 augmentation前、shuffle前、shuffle后所有 tensor/list buffer记录：

- shape
- dtype
- canonical SHA256

至少包括：

- actions
- neglogpacs
- values
- mus
- sigmas
- obses
- states
- dones
- returns
- off_policy_mask
- rnn_states

`played_frames`精确记录。`step_time`是wall-clock metadata，明确排除，不保存、不比较。

从 actor和central原生dataset读取实际row identity和batch size，必须得到两组：

`[12, 12, 12, 20]`

不能假设尾批为12，也不能丢掉最后20 rows。

## 七、RNG golden

沿用 Source `Runner.load_config` 的原生seed行为。至少记录这些phase的完整状态和SHA256：

- after_runner_seed
- after_agent_initialization
- before_play
- after_play
- before_augment
- after_augment
- before_shuffle
- after_shuffle
- after_prepare
- after_input_rms_probe

每个phase记录：

- NumPy global RNG完整 state：algorithm、keys、position、has_gauss、cached_gaussian
- Torch CPU RNG完整 byte state
- `torch.cuda.get_rng_state_all()` 的全部 CUDA states
- NumPy exact version
- Torch exact version
- visible CUDA device identity

必须证明消费归属：

- stochastic action sampling消费 CUDA RNG
- follower choice消费 NumPy RNG
- `shuffle_batch` 中无device参数的 `torch.randperm`消费 CPU Torch RNG
- GAE/get_values/prepare/RMS probe不应额外消费这些RNG

spy必须只在对应调用附近安装，delegate原函数，并原样返回结果；不能为了“预测”结果提前调用 RNG。

## 八、normalizer边界

本轮不得进入真实 loss/backward/optimizer，但要补齐 V2b 未覆盖的 input RMS数值证据。

1. 保存 actor和central input RMS初始 `running_mean`、`running_var`、`count`。
2. 证明 play_steps、augmentation、shuffle及 `prepare_dataset(..., train_value_mean_std=False)` 阶段 input RMS保持冻结。
3. 使用 native actor/central dataset的四个真实mini-batch做隔离的 forward-only input RMS probe：
   - central native model train forward依次读取四批；
   - actor native model train forward依次读取四批；
   - actor forward使用真实 seq_length、dones和rnn_states；
   - 只执行 model forward；
   - 不计算loss；
   - 不backward；
   - 不调用optimizer或GradScaler；
   - 一轮后将normalizer切回eval。

记录：

- 每次forward实际row identity
- batch sizes `[12,12,12,20]`
- actor physical obs rows
- central privileged state rows
- 更新后的 running_mean/running_var/count
- count应由1更新到57
- coefficient carrier不得进入 input RMS统计

这部分必须在manifest中命名为类似 `diagnostic_first_miniepoch_forward`，不能声称它已经验证完整 `train_epoch` 更新顺序。central-before-actor orchestration、首mini-epoch后freeze、value RMS、loss和optimizer由code #4验证。

为了不提前混入 value normalization，dataset wiring调用：

`prepare_dataset(batch_dict, train_value_mean_std=False)`

并记录这是test-only boundary。value RMS必须保持未更新；默认 `train_value_mean_std=True` 的Source语义明确留给code #4。

## 九、fixture和provenance

Source capture与Target replay必须是两个独立进程。任何一个进程同时加载Source和vendored两套 `rl_games`，证据无效。

`source_rollout_harness.py` 对 `rl_games` 使用lazy import，使：

- Source generator只加载 `/home/user/ws/lemon/simtoolreal/rl_games`
- Target pytest只加载 `third_party/simtoolreal_rl_games`

Source generator必须：

- 只在 `UNILAB_SAPG_ORACLE_MODE=source` 下工作
- 验证Source HEAD和RL-Games tree
- 从固定 Git objects读取owner YAML
- 验证实际加载的每个 `rl_games.*` 模块都位于Source package root
- 对每个loaded module比较工作树字节与固定 Git blob
- manifest记录module/path/blob/SHA256 inventory
- 拒绝从其他distribution或路径捕获
- 不修改Source工作树

fixture loader必须：

- 要求NPZ和manifest都是regular file且非symlink
- 使用 `np.load(..., allow_pickle=False)`
- 校验schema version
- 校验manifest canonical payload SHA256
- 校验NPZ文件SHA256
- 校验array inventory、shape、dtype和content SHA256
- 在test中固定预期 NPZ SHA256和manifest canonical payload SHA256
- 用代码常量固定Source identity与canonical platform，不能只相信manifest自报字段

manifest至少包含：

- Source/owner/module provenance
- owner defaults与test-only overrides
- canonical platform和deterministic flags
- synthetic contract
- native call/phase inventory
- RNG states
- NPZ array inventory
- buffer hashes
- tolerances
- `ordinary_pytest_regenerates=false`

普通pytest只能读取fixture，绝不能自动调用generator。

两个fixture文件合计不得超过8 MiB。超出立即停止，不得静默删除必需证据或提交大文件。

数值规则：

- FP32：`atol=1e-6, rtol=1e-5`
- index、row identity、done、mask、repeat、permutation、hash、RNG state：exact
- canonical平台computed tensor hashes：exact
- Target必须 `is_canonical_platform=True`

## 十、TDD顺序

第一步只新增：

`tests/algos/rlgames_sapg/test_rollout_golden.py`

test必须先调用现有：

`require_simtoolreal_rl_games()`

再导入尚不存在的 `source_rollout_harness`，与V2b import顺序一致。

运行RED：

```bash
UNILAB_REQUIRE_SAPG=1 uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py -q
```

预期：

- exit 2
- collection error
- 原因是 `source_rollout_harness` 不存在
- 0 skip
- 没有错误加载其他RL-Games distribution

保存准确RED输出摘要。

随后实现harness和generator，生成Source fixture。

Source capture命令：

```bash
UNILAB_SAPG_ORACLE_MODE=source uv run --isolated --python 3.11 \
  --with gym==0.26.2 \
  --with-editable /home/user/ws/lemon/simtoolreal/rl_games \
  scripts/generate_simtoolreal_sapg_rollout_fixture.py \
  --source /home/user/ws/lemon/simtoolreal \
  --output tests/fixtures/simtoolreal_sapg
```

Source legacy runtime eager-import `gym`，所以显式 `gym==0.26.2` 是允许的capture依赖；不得把它作为新的root dependency。

## 十一、必跑验证

Target focused replay：

```bash
UNILAB_REQUIRE_SAPG=1 uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py -q
```

完整SAPG oracle回归：

```bash
UNILAB_REQUIRE_SAPG=1 uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg -q
```

Vendor回归：

```bash
uv run --python 3.11 \
  pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q
```

Audit：

```bash
uv run --python 3.11 \
  scripts/audit_simtoolreal_rlgames_vendor.py
```

Scoped Ruff：

```bash
uv run --python 3.11 ruff check \
  scripts/generate_simtoolreal_sapg_rollout_fixture.py \
  tests/algos/rlgames_sapg/source_rollout_harness.py \
  tests/algos/rlgames_sapg/test_rollout_golden.py

uv run --python 3.11 ruff format --check \
  scripts/generate_simtoolreal_sapg_rollout_fixture.py \
  tests/algos/rlgames_sapg/source_rollout_harness.py \
  tests/algos/rlgames_sapg/test_rollout_golden.py
```

Root静态检查：

```bash
uv run --python 3.11 ruff check .
uv run --python 3.11 ruff format --check .
git diff --check
git diff --cached --name-only
git status --short
```

所有required pytest必须0 skip。

不要运行任何额外解释器或平台矩阵，也不要运行 `make test-all`。

`make test-all`留给控制session完成commit后的最终gate。

## 十二、立即停止条件

出现任一情况立即停止并返回 `BLOCKED`：

- 需要修改五个允许路径之外的文件
- 需要修改vendor或任何SAPG公式
- 需要第六个文件
- 无法使用原生A2CAgent/path取得证据
- 必须复制GAE、shuffle、choice、filter或TD公式
- Source和Target必须在同一进程加载
- Source HEAD/tree/owner/module blob漂移
- canonical Python 3.11 + V2b CUDA平台不可用
- required pytest出现skip
- Source/Target出现未解释tensor/index/hash/RNG差异
- 必须放宽容差或手工修改Target输出
- fixture超过8 MiB
- 手写规模明显超过约800 LOC
- 需要进入loss、backward、optimizer、AMP、overflow、checkpoint或player
- 需要修改code #2 fixture/runtime来让本轮通过

不得“先修一下”再汇报。

## 十三、交接要求

完成后保持所有变更未暂存、未提交，返回：

`# DONE`

或：

`# BLOCKED`

DONE报告必须包含：

1. 五个路径逐项byte size。
2. NPZ SHA256。
3. manifest文件SHA256。
4. manifest canonical payload SHA256。
5. fixture合计大小，证明不超过8 MiB。
6. net handwritten LOC和out-of-scope path count。
7. TDD RED命令、exit code和准确失败原因。
8. Source capture完整命令。
9. Source HEAD/tree/owner/loaded-module/platform证据。
10. Target focused和完整SAPG测试数量、0 skip。
11. canonical exact-hash结果及所有FP32 tensor最大absolute/relative error。
12. 实际 repeat index、candidate集合和follower source block。
13. 实际14-entry permutation。
14. 48 base + 8 follower = 56 rows证据。
15. actor和central两组 `[12,12,12,20]`。
16. timeout/GAE/TD/RNN reset摘要。
17. NumPy/Torch CPU/CUDA各phase RNG hash和消费归属摘要。
18. input RMS probe结果，以及value RMS/code #4边界说明。
19. vendor audit、Ruff、diff检查结果。
20. 完整 `git status --short`。
21. `git diff --cached --name-only`为空。
22. HEAD仍等于本轮开始时记录的 `SAPG_CODE3_START_HEAD`，并同时报告其完整SHA。
23. 明确确认：
    - 未进入code #4；
    - 未运行`make test-all`；
    - 未add/commit/push；
    - 未stash/reset/clean/checkout；
    - 未修改Source工作树。
