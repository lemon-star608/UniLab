# SimToolReal SAPG Code #4：update 与 AMP oracle 执行提示词

> **给实现 session：** 完整阅读本文件和仓库根目录的 AGENTS.md，然后严格执行。不要重新规划，也不要进入 code #5 或真实环境接线。本轮只建立 O1b frozen-input update oracle；所有改动保持未暂存、未提交，完成后把证据交回控制 session。

**Goal:** 在固定 Source RL-Games SAPG owner 上锁定 central-before-actor、loss/minibatch、KL/reference、normalizer、optimizer、scheduler、AMP/GradScaler 和真实 overflow-skip 语义，并让 Target replay 在同一 canonical CUDA 平台上逐项对拍。

**Architecture:** Source 的 Runner、A2CAgent、PPODataset、CentralValueTrain、native loss、scheduler 和 optimizer 仍是唯一算法 owner。generator/harness 只在输入/调用边界做冻结和 delegating instrumentation，不在测试代码中重写任何 loss、return、梯度或更新公式。五个新文件分别负责 Source capture、replay、测试和 fixture；不修改生产算法、vendor、Source 或 code #3。

**Tech Stack:** Python 3.11、PyTorch 2.7.0+cu128、CUDA 12.8、固定 SimToolReal RL-Games fork、NumPy frozen arrays、pytest。只使用 canonical cuda:0；不增加 Python/CUDA 版本矩阵。

---

## 1. 本轮结论边界

本轮是 code commit #4 / O1b，目标提交标题为：

test: lock SAPG update and AMP semantics

必须覆盖：

- frozen post-rollout mini-batch 的 row identity、dataset slicing 和 [12, 12, 12, 20] 尾批；
- prepare_dataset 后的 value/advantage/normalizer 状态；
- central critic 全部更新先于 actor 更新；
- actor 两个 mini-epoch、每个 mini-batch 的 native call；
- ratio、unclipped/clipped PPO surrogate；
- unclipped/clipped value loss；
- bounds loss、entropy coefficient、entropy numerator 和分母；
- native KL、PPODataset.update_mu_sigma 后第二 mini-epoch 的 old reference；
- scheduler 更新时机、KL average、learning rate before/after；
- actor/central optimizer 参数集合、backward gradient、unscale/clip、step 前后摘要；
- actor autocast、GradScaler、正常 step 和真实 overflow skip；
- central 的普通 FP32 backward/clip/step；
- 每个相关 phase 的 NumPy/Torch CPU/CUDA RNG state；
- Source module provenance、完整 fixture inventory、shape/dtype/content hash。

本轮明确不覆盖：

- checkpoint、resume、player（code #5/O1c）；
- rollout、GAE、augmentation、shuffle 或 RNG 生成规则的重新 capture（code #3 已锁定）；
- MuJoCo/MuJoCoUni、NpEnv、Hydra、CLI、async runner、生产算法；
- vendor 或 Source 的任何字节修改；
- 训练曲线、物理后端差异或不同 batch size 的等价声明。

AMP 的数值不得与 FP32 逐元素比较。FP32 Source/Target 对拍使用 atol=1e-6、rtol=1e-5；AMP 只比较 Source/Target 的 AMP 语义、step/skip、scaler state 和有明确 dtype 口径的统计量。

## 2. 仓库、分支和固定 provenance

工作目录必须是：

/home/user/ws/lemon/rlgame-unilab/UniLab

预期分支：

feat/simtoolreal-sapg-rlgames

本 prompt 建立时的固定 HEAD：

fbb12266d19b5cb2b0f4e73690c73785956421a0

实现 session 开始时先运行并记录：

~~~bash
git rev-parse --abbrev-ref HEAD
SAPG_CODE4_START_HEAD=$(git rev-parse HEAD)
printf '%s\n' "$SAPG_CODE4_START_HEAD"
git merge-base --is-ancestor \
  fbb12266d19b5cb2b0f4e73690c73785956421a0 HEAD
git status --short
git diff --cached --name-only
~~~

若分支不对、固定 HEAD 不是 ancestor、或工作树不干净，立即返回 # BLOCKED。实现 session 不得执行 git add、git commit、git push、stash、reset、clean、checkout 或切换分支。

Source oracle：

- 仓库：/home/user/ws/lemon/simtoolreal
- Source HEAD：2a9917533bfea70419ed2667a511d7238e5b3abc
- Source RL-Games parent tree：7a6a0bb090998d00565aaefa6ab9f2b3d356ace2
- train owner：isaacsimenvs/cfg/train/SimToolRealSAPG.yaml
- train owner blob：f363d05d4a24b190b7837703b93270d8f3fe9a9c
- task owner：isaacsimenvs/cfg/task/SimToolReal.yaml
- task owner blob：6469d46867081b70edaa589dcb31c7090b64d45e

开始时还要记录：

~~~bash
git -C /home/user/ws/lemon/simtoolreal rev-parse HEAD
git -C /home/user/ws/lemon/simtoolreal rev-parse \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games
~~~

Source 工作树可以有既存 untracked 文件，但不得修改、删除、stash、reset、clean 或依赖它们。Source owner 和加载的 rl_games.* module 必须通过固定 Git object/blob 校验；不能把官方 pip RL-Games 当作 oracle。

### 2.1 唯一验证平台

只允许：

- Python 3.11.15；
- Torch 2.7.0+cu128；
- CUDA build 12.8，runtime 13020；
- cuDNN 90701；
- NVIDIA GeForce RTX 4090，compute capability [8, 9]；
- driver 580.173.02；
- cuda:0；
- deterministic algorithms=true；
- cudnn_deterministic=true、cudnn_benchmark=false；
- TF32=false；
- float32 matmul precision=highest。

应复用 code #3/source_network_harness.py 的 canonical configuration 和 platform schema，并在 manifest 中保存完整 identity。Target replay 必须得到 is_canonical_platform=true。没有 CUDA、设备不是上述 canonical identity、或 AMP 被 disabled 时不得 skip-pass，必须 # BLOCKED。

## 3. 不可修改范围与五个文件边界

本轮只能新增以下五个 regular files：

1. scripts/generate_simtoolreal_sapg_update_fixture.py
2. tests/algos/rlgames_sapg/source_update_harness.py
3. tests/algos/rlgames_sapg/test_update_golden.py
4. tests/fixtures/simtoolreal_sapg/source_update_fp32.npz
5. tests/fixtures/simtoolreal_sapg/source_update_manifest.json

不得修改：

- scripts/generate_simtoolreal_sapg_network_fixture.py；
- scripts/generate_simtoolreal_sapg_rollout_fixture.py；
- tests/algos/rlgames_sapg/source_network_harness.py；
- tests/algos/rlgames_sapg/source_rollout_harness.py；
- 任何 code #3 NPZ/manifest；
- third_party/simtoolreal_rl_games/**；
- src/**、conf/**、pyproject.toml、uv.lock、现有测试或 Source。

手工编辑一律使用 apply_patch。两个 fixture 只能由本轮 generator 生成；普通 pytest 不得自动 regenerate。目标为约 800 行净手写代码；若为了满足 gate 必须明显超过约 900 行、需要第六个文件或 fixture+manifest 超过 8 MiB，立即停止并报告，不要压缩或静默删掉证据。

## 4. 复用并冻结 code #3 输入

code #3 fixture 是只读输入，不得重生或改 hash：

- tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz
  - SHA256：3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8
- tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json
  - 文件 SHA256：785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f
  - canonical payload SHA256：7d88cb01dce4607391a39d1fb31b21d8366d2bdadae2e0dce6eb02323c06901d

generator 启动时必须先以 regular-file、inventory、shape、dtype、content hash gate 读取这两个文件。使用 buffer_post_shuffle__* 作为 frozen update batch，保持：

- 56 rows = 48 leader rows + 8 follower rows；
- 14 条完整 sequence，每条 4 个 time step；
- post-shuffle row identity 完整保存；
- actor 和 central dataset 的 native batch sizes 都是 [12, 12, 12, 20]；
- actions、old values、old neglogpacs、old mu/sigma、obses、privileged states、dones、returns、off_policy_mask、rnn_states 的原始 dtype/shape/row 顺序；
- rnn_masks 若 Source fixture 中不存在就保持 None，不能为了测试方便伪造；若 native prepare_dataset 产生它，则完整记录。

不得根据 code #3 的最终数组手算 advantage、ratio、loss 或 optimizer delta。所有 update output 必须由 Source native owner 产生。

### 4.1 test-only 配置

manifest 必须把 Source owner defaults 和本轮 test-only overrides 分开记录。固定 synthetic overrides：

| 字段 | Source owner default | 本轮 frozen update |
|---|---:|---:|
| num_actors / env rows | 24576 | 12 个 synthetic actor |
| expl_coef_block_size | 4096 | 2 |
| blocks | 6 | 6 |
| horizon/sequence | 16 | 4 / 4 |
| actor minibatch_size | 98304 | 12 |
| central minibatch_size | 98304 | 12 |
| mini_epochs | 2 | 2 |
| e_clip | 0.1 | 0.1 |
| critic_coef | 4.0 | 4.0 |
| bounds_loss_coef | 0.0001 | 0.0001 |
| entropy_coef | 0.0 | 0.0 |
| learning_rate | 1e-4 | 1e-4 |
| lr_schedule / schedule_type | adaptive / standard | adaptive / standard |
| kl_threshold | 0.016 | 0.016 |
| normalize_input/value/advantage | true / true / true | unchanged |
| truncate_grads / grad_norm | true / 1.0 | unchanged |

此外必须从 Source owner 原样保留 gamma=0.99、tau=0.95、value_bootstrap=true、ppo=true、reward_shaper.scale_value=0.01、mixed_precision=true、clip_value=true；central_value_config 的 learning_rate=1e-4、mini_epochs=2、clip_value=true、normalize_input=true、truncate_grads=true 也必须单独记录。SAPG coefficient IDs 仍为 [50, 40, 30, 20, 10, 0]，expl_type=mixed_expl_learn_param、expl_reward_type=entropy、expl_reward_coef_scale=0.002、expl_reward_coef_embd_size=32、use_others_experience=lf、off_policy_ratio=1.0。为把已经 augmented/shuffled 的 code #3 batch 送入 native update，可在 test boundary 将 use_others_experience 设为 none 并用 identity delegating wrapper 固定 shuffle_batch；该 test-only freeze override 必须写入 manifest，不能进入生产配置，也不能改变 actor/central update owner。

## 5. 必须调用 Source native update owner

Source 固定代码的关键 owner 和行段（实现 session 必须用 nl -ba 再核对）：

- rl_games/common/a2c_common.py:1370-1465：continuous train_epoch，包括 prepare_dataset、central-before-actor、mini-epoch、update_mu_sigma、standard scheduler 和 input-RMS freeze；
- rl_games/common/a2c_common.py:1467-1532：value normalization、advantage normalization、actor/central dataset handoff；
- rl_games/algos_torch/a2c_continuous.py:105-230：autocast、actor/value/bounds/entropy loss、apply_masks、backward、GradScaler step、KL 和 result；
- rl_games/common/a2c_common.py:360-393：all-reduce、unscale、clip、scaler.step、scaler.update；
- rl_games/algos_torch/central_value.py:223-292：central train loop、普通 backward、clip、optimizer.step；
- rl_games/common/datasets.py:25-80：mini-batch slicing、tail batch、update_mu_sigma；
- rl_games/common/common_losses.py:10-48：PPO actor/value clipping；
- rl_games/algos_torch/torch_ext.py:29-38,143-154：policy KL 和 mask denominator；
- rl_games/common/schedulers.py:20-35：adaptive KL scheduler。

推荐的 native invocation：

1. 在固定 Source package 下通过 Runner.load()、set_vec_env()、native algo factory 构造 A2CAgent，调用 init_tensors()；不得只构造一个 fake object 再调用 unbound method。
2. 使用 code #3 的 name-seeded deterministic parameter fill（只读调用现有 helper）覆盖 actor/central 参数；记录 native initialization hash 和 deterministic parameter hash。fill 操作不消费 RNG。
3. 在 test boundary 提供具备 set_train_info、get_env_info、get_env_state、set_env_state 的 synthetic vecenv。它不执行物理 step；play_steps 由 delegating freeze wrapper 返回 code #3 batch 的 clone 和 ps_extras。
4. 让 Source 的原生 agent.train_epoch() 执行 prepare_dataset、central train、actor 两个 mini-epoch 和 scheduler。execution-freeze wrapper 只允许包住 play_steps 与 a2c_common.shuffle_batch；另可对 train_epoch、central/actor calc_gradients、PPODataset、loss、optimizer、scaler 和 scheduler 安装纯记录用的 delegating spy。所有 wrapper/spy 都不能写 loss/return/gradient 公式，不能额外消费 RNG，必须在 finally 恢复原函数。
5. overflow case 可以在 native prepare_dataset 完成后，对一个 clone 的第一个 actor mini-batch 的 advantages[0] 注入 +inf，再调用 native agent.train_actor_critic(batch)；central 和其他 actor update 不得被伪造。该 case 必须在 manifest 明确标为 test-only input mutation。

不得在 harness 中重写或复制：

- train_epoch 的 central/actor 循环；
- prepare_dataset、advantage/value normalization；
- actor/value/bounds/entropy loss；
- apply_masks、ratio、KL、scheduler；
- gradient clipping、GradScaler overflow 判定、optimizer step；
- PPODataset slicing 或 update_mu_sigma。

## 6. 必须记录的 update evidence

所有记录同时保存 shape、dtype、canonical SHA256；大 tensor 不完整写入 NPZ 时，manifest 保存 norm、sum、max、max_abs 和 64 个 name-seeded sentinel coordinates。禁止提交完整 model parameter 或完整 Adam state/delta。

### 6.1 调用顺序和 row identity

记录一个带 phase、mini_epoch、mini_batch、row IDs 的 event log，至少包含：

prepare_dataset:start/end → central:train_net:start/end → 每个 central mini-epoch/batch 的 model/gradient/optimizer events → actor:mini_epoch_0 的四个 batch → scheduler update 0 → actor:mini_epoch_1 的四个 batch → scheduler update 1。

严格断言 central 的最后一个 optimizer step 发生在 actor 第一个 forward/backward 之前。记录 actor/central 实际参数名集合及排序后的 hash；central 和 actor 参数集合不得交叉或遗漏。

每个 actor/central dataset batch 保存：

- batch index、mini_epoch、row identity；
- row count，必须得到 [12, 12, 12, 20]；
- obs/state/actions/returns/old values/old mu/old sigma 的 metadata；
- native rnn_states、dones、rnn_masks（若有）；
- PPODataset.last_range 和 update_mu_sigma 的 start/end。

### 6.2 Normalizer boundary

在 prepare_dataset 前后、central 每个 mini-epoch 后、actor 每个 mini-epoch 后分别保存：

- actor input RMS running_mean/running_var/count；
- central input RMS running_mean/running_var/count；
- actor shared value RMS；
- central value RMS；
- training/eval 状态。

必须证明：

- prepare_dataset 默认 train_value_mean_std=true，先更新 value normalizer，再建立 actor/central dataset；
- input RMS 在第一组真实 train forward 中更新；
- 每个 owner 在第一个 mini-epoch 后按 Source 进入 eval/frozen boundary，第二 mini-epoch 不再更新 input count；
- value RMS 与 input RMS 的 owner、count 和更新顺序不能混淆。

### 6.3 Actor loss、value loss、bounds、entropy

通过 delegating wrapper 记录 native 输入/输出，不要在 harness 另写公式：

- old/new neglogp、native ratio 输入和 actor_loss 输出；
- unclipped surr1、clipped surr2、最终 per-row actor loss；
- old values、new values、returns、clip range、unclipped/clipped value loss、native max 结果；
- mu、soft bound 1.1、native bounds loss 和 bounds_loss_coef；
- 每行 entropy、mixed-expl 的 per-row entropy coefficient、entropy_coef * entropy numerator；
- actor shared-value output、central-value output 及其各自 value normalizer 的输入；
- native apply_masks 的 mask 是否为 None、mask shape、sum_mask 或 mean denominator，以及四类 loss 的 reduced result；
- 总 loss 必须能由 native result 的四项 trace 对应到：

 a_loss + 0.5 * c_loss * critic_coef - entropy_loss + b_loss * bounds_loss_coef

当前 synthetic Source update 通常没有 rnn_masks；此时必须记录 mask=None 和 native mean denominator。另做一个不参与 optimizer 的 native apply_masks diagnostic，使用固定含零 mask，证明 Source 分母是 mask.numel() 而不是 mask.sum()。diagnostic 不得改写 owner 或消耗 RNG。

### 6.4 KL、old reference 和 scheduler

每个 actor mini-batch 记录：

- batch 前 old mu/sigma 的 row-keyed hash；
- native new mu/sigma；
- native policy KL（RNN mask 为 None 时的 reduce 语义；有 mask 时记录 mask.numel() 分母）；
- PPODataset.update_mu_sigma 调用前后范围和 content hash。

必须证明第二 mini-epoch 的 old mu/sigma 对每一行来自第一 mini-epoch 对应 batch 的新输出，而不是初始 rollout 副本。测试必须对“删除 update_mu_sigma 或把它延后到 epoch 末”的临时 mutation 失败。

schedule_type=standard 时记录每个 mini-epoch 的 batch KL 列表、native mean_list 结果、scheduler 输入 LR/entropy/KL、输出 LR/entropy 和 optimizer param-group LR。必须证明 scheduler 每个 mini-epoch 末更新一次，不在每个 mini-batch 更新；adaptive threshold 使用 0.016。

### 6.5 Gradients、optimizer 和 central/actor 差异

对 actor 和 central 分别记录：

- backward 后每个参数的 gradient signature 和 absent list；
- actor AMP 下 scaled gradient；
- actor scaler.unscale_ 后、clip 前的 gradient norm；
- clip_grad_norm_ 后的 norm 和 returned total norm；
- central 普通 FP32 clip 前/后 norm；
- optimizer step 前后参数 hash、每参数 delta 的 norm/sum/max/max_abs/sentinels；
- optimizer state key/step、param-group LR/eps/weight_decay；
- actor 与 central 的 optimizer parameter name set。

all_grads、unscale、clip、step 的事件顺序必须直接来自 Source delegating spy。不得用手写的 second optimizer 或根据最终参数反推 gradient。

### 6.6 AMP、GradScaler 和真实 overflow

必须生成三个 case，并在 manifest 分开记录：

1. normal_fp32：同一 frozen input、同一 deterministic weights，test-only mixed_precision=false，仍在 cuda:0；用于 FP32 loss/gradient/parameter update golden。
2. normal_amp：Source owner mixed_precision=true，真实 torch.cuda.amp.autocast 和 enabled GradScaler；记录 autocast dtype、scaler enabled、scale/growth/backoff state、step/update 事件。
3. overflow_amp：由 native prepare_dataset 产生的第一个 actor batch clone 仅把一个 advantage 设为 +inf；调用真实 scaler.scale(loss).backward()、unscale_、scaler.step、scaler.update。必须观察到：
   - underlying optimizer 参数不变；
   - scaler 确实 backoff，scale/growth tracker 改变；
   - optimizer 的底层 step 没有被错误地当成成功 step；
   - central 普通 update 不被 AMP 包装；
   - 没有用 fake scaler、手动跳过 optimizer 或 CPU fallback 伪造 overflow。

Target 必须在同一 canonical CUDA 上重放三个 case。AMP case 不与 FP32 case 逐元素比较；要求 Source/Target 的 enabled/disabled、step/skip、scaler state transition 和参数不变/变化关系一致。

### 6.7 RNG

至少保存以下 phase 的完整 NumPy global、Torch CPU、全部 CUDA RNG state 及 component hash：

- after_runner_seed
- after_agent_initialization
- after_deterministic_parameter_fill
- before_prepare
- after_prepare
- before_central
- after_central
- before_actor_epoch_0
- after_actor_epoch_0
- before_actor_epoch_1
- after_actor_epoch_1
- before_overflow_step
- after_overflow_step

每个 delegating spy 必须证明参数/输入 clone、loss/gradient signature、metadata validation 和 diagnostic 没有额外消费 RNG。update fixture 不得把 rollout 的 follower/trajectory RNG 重新归因于 update。

## 7. Fixture 和 manifest contract

source_update_fp32.npz 只保存不可由 manifest 充分表达的 finite frozen inputs、row identity、native small tensors、overflow probe inputs 和必要的 loss/gradient trace；source_update_manifest.json 保存：

- schema_version、fixture 文件 hash、canonical payload hash；
- Source HEAD、RL-Games tree、train/task owner blob/SHA256；
- loaded Source module path/blob/SHA256 清单；
- canonical platform 完整 identity；
- Source owner defaults 与 test-only overrides；
- code #3 输入 fixture 的两个固定 hash；
- case inventory：normal_fp32、normal_amp、overflow_amp；
- 每个 array 的 exact {shape,dtype,sha256}；
- event sequence、row/batch metadata、native call names；
- loss/KL/LR/gradient/optimizer/scaler signatures；
- normalizer snapshots 和 RNG states；
- tolerances 及 FP32/AMP numeric domain；
- generation mode=source-only，ordinary pytest 不 regenerate。

NPZ 必须用 allow_pickle=False 读取；manifest/NPZ/output leaf 必须是 regular file，root、ancestor、leaf symlink、broken symlink、directory leaf 都要 fail closed。Target replay 的顺序必须是：

1. provenance/distribution gate；
2. root/leaf regular-file gate；
3. complete inventory gate；
4. every-array shape/dtype/content hash validation；
5. event/semantic validation；
6. 最后才做 numeric subtraction、absolute/relative error。

任何 missing/extra、reshape、dtype drift 或 content-only drift 都要给出明确字段名；numeric 代码不得在 metadata 全量验证前运行。Fixture + manifest 总字节数必须小于 8 MiB。

## 8. TDD、mutation gate 和测试文件

实现 session 先运行未创建实现文件时的 RED：

~~~bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py -q
~~~

真实 collection error 必须记录；不得把初始 RED 伪造成算法失败。

最终 test_update_golden.py 至少包含以下近风险测试：

1. fixture provenance/platform/inventory/hash/schema；
2. frozen row identity、actor/central [12,12,12,20] 和 complete metadata；
3. FP32 Target replay：loss、ratio、clip、bounds、entropy denominator、KL/LR、normalizer、gradient/optimizer signatures 在 atol=1e-6、rtol=1e-5 内；
4. central-before-actor event order 和 central/actor optimizer parameter sets；
5. second mini-epoch old mu/sigma reference 与 scheduler-per-epoch timing；
6. normal AMP control flow；
7. real overflow AMP：step skip、parameter unchanged、scaler backoff；
8. whole-inventory gate、symlink/reshape/dtype/content mutation rejection；
9. RNG state/hash and no-extra-consumption assertions。

至少做三次临时 mutation RED，再用 apply_patch 精确恢复并证明最终文件 hash 回到 mutation 前：

- 在 central/actor event wrapper 中交换或伪造顺序；central-before-actor 测试必须失败；
- 绕过 PPODataset.update_mu_sigma 或把它移到第二 epoch 之后；old-reference 测试必须失败；
- 将 overflow harness 的真实 scaler.step/update 替换成无条件 optimizer.step，或把 metadata gate 后移到 numeric subtraction；对应测试必须失败。

mutation 只允许在本轮新 harness/test 中进行，不能修改 vendor、Source 或 code #3；每次必须用 apply_patch 恢复，最终不得留下 mutant。

## 9. Source capture 和 Target replay 命令

Source fixture 只在 canonical platform、固定 Source checkout 上生成。Target runtime gate 只在 vendored replay 中执行；generator 的 source mode 不得导入会强制要求 vendored distribution 的测试模块，但仍必须显式保留本节的 source provenance checks：

~~~bash
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
~~~

Source capture 必须在 generator 内 fail closed 检查 Source HEAD、tree、owner blobs、loaded namespace 和 canonical platform；不得读取 Source 工作树中的未跟踪替代文件。若 source capture 改变 code #3 fixture 或任何已有路径，立即停止。

Focused Target replay：

~~~bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py -q
~~~

完整 SAPG oracle：

~~~bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg -q
~~~

Vendor、audit、Ruff 和 whitespace：

~~~bash
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
~~~

本轮实现 session 不运行 make test-all；控制 session 在 code #4 审查/提交后运行，并继续显式设置 UNILAB_REQUIRE_SAPG=1。不要运行 Python 3.10/3.12/3.13、cu126、CPU AMP fallback 或其他 CUDA matrix。

## 10. 停止条件

出现任一情况立即返回 # BLOCKED，不要通过放宽容差或删除证据继续：

1. 需要修改 code #3、vendor、Source、生产算法、配置或第六个文件；
2. native Source update 不能在 frozen input 上运行，必须手写 loss/GAE/optimizer；
3. central-before-actor、second-epoch reference、normalizer boundary 或 denominator 出现无法解释的 mismatch；
4. AMP/GradScaler/overflow 只能在 CPU、disabled scaler、fake spy 或 skip 上“通过”；
5. canonical Python 3.11 + cu128 + RTX 4090 不可用；
6. Source/Target loaded distribution、module path/blob 或 owner provenance 漂移；
7. fixture inventory、shape/dtype/content hash gate 不完整，或 numeric 在 metadata gate 前执行；
8. fixture+manifest 超过 8 MiB、净手写代码明显超过约 900 行、或测试需要额外长期依赖；
9. required pytest 出现 skip、未解释 warning/error、未解释数值差异或 is_canonical_platform=false；
10. 需要进入 checkpoint/player、真实 MuJoCo、async/distributed/compile/export 或 code #5。

## 11. 交接报告格式

完成时只返回 # DONE，并逐项列出：

1. 五个最终文件、regular-file 检查、net handwritten LOC 和 fixture/manifest 字节预算；
2. 起始/结束 branch、HEAD、git status --short、staged=0；
3. Source HEAD/tree/owner blobs、loaded module provenance 和 canonical platform；
4. TDD 初始 RED、每个 mutation RED、恢复后的最终 GREEN；
5. frozen row identity、actor/central batch sizes、event sequence 和 central-before-actor 证据；
6. loss/clip/bounds/entropy denominator、KL/reference、scheduler/LR、normalizer snapshots；
7. actor/central parameter sets、gradient/clip/optimizer signatures；
8. normal FP32、normal AMP、overflow AMP 的实际 step/skip/scaler 结果；明确 AMP 未与 FP32 逐元素比较；
9. 每个 phase 的 NumPy/Torch CPU/CUDA RNG hash 和 no-extra-consumption 结果；
10. focused/full SAPG、vendor、audit、Ruff、format、git diff --check 的实际命令与结果；
11. 明确确认：未修改 code #3/vendor/Source/生产代码，未运行 make test-all，未增加版本矩阵，未执行任何 Git 破坏操作，未进入 code #5。

控制 session 会独立审查五个文件和 fixture provenance，确认所有 required tests 为 0 skip 后再提交 code #4；实现 session 不得自行 commit。
