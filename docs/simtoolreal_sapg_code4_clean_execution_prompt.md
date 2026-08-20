# SimToolReal SAPG Code #4 clean execution prompt

> 给新的实现 agent：本文件是 Code #4 当前唯一执行规格。先完整阅读仓库根目录
> AGENTS.md，再按本文从干净工作树重新实现。不要读取、恢复或复用任何被隔离的旧
> Code #4 产物；它们不是 oracle、fixture seed 或实现参考。不要重新规划，不要进入
> Code #5。完成后保持全部改动未暂存、未提交，并把实际证据交回控制 session。

## 1. 本批唯一结果

本批只建立 Code #4 / O1b frozen-input update oracle：

~~~text
immutable Source checkout
  -> native Runner -> A2CAgent
  -> native prepare_dataset / PPODataset / CentralValueTrain
  -> native actor loss / optimizer / GradScaler / scheduler
  -> source-only NPZ + manifest

vendored Target checkout（独立进程、独立 rl_games namespace）
  -> 同一 native owner path
  -> 先独立验证 Target evidence invariants
  -> 再做 Source/Target semantic、RNG 和 FP32 numeric comparison
~~~

Source 的 Runner、A2CAgent、PPODataset、CentralValueTrain、loss、optimizer、
GradScaler 和 scheduler 是唯一算法 owner。测试 instrumentation 只能 delegate、记录，
并在 finally 中恢复原对象；不得在 generator、harness 或 test 中复制 PPO surrogate、
value clipping、bounds、entropy、KL、normalization、gradient clipping、optimizer、
GradScaler overflow 或 scheduler 公式。

本批必须锁定：

- code #3 frozen post-shuffle handoff、row identity、完整 sequence 和 native dataset slicing；
- prepare_dataset 的 old values、returns、advantages、value normalization 与两个 dataset handoff；
- central 全部更新严格先于 actor，central/actor 各两个 mini-epoch；
- 每 batch native loss branches、normalizer、KL/reference、gradient、optimizer 和 LR；
- normal_fp32、normal_amp、overflow_amp 三种真实 native path；
- NumPy、Torch CPU 与所有 CUDA RNG state；
- Source 和 Target 各自完整、不可由“相同残缺字典”伪造的 evidence invariants；
- file anchors、namespace、metadata-before-numeric 和 generator 路径安全。

目标代码提交标题是：

~~~text
test: lock SAPG update and AMP semantics
~~~

但实现 agent 不执行 git add、commit、push、PR、stash、reset、clean、checkout 或切分支。

明确不做 checkpoint/resume/player、MuJoCo/MuJoCoUni、NpEnv、Hydra、CLI、async、
distributed、compile、export、生产算法或配置修改；这些都不属于 Code #4。

## 2. 仓库、起点和固定身份

唯一工作目录：

~~~text
/home/user/ws/lemon/rlgame-unilab/UniLab
~~~

预期分支：

~~~text
feat/simtoolreal-sapg-rlgames
~~~

固定 Git 身份分两层：

~~~text
lineage base:    ba16f5b490c2fcf1bf3bd81a03314b3f57d19770
correction base: 910a4309918b1dd2fadc60c43f4250d03d84153a
~~~

`910a4309918b1dd2fadc60c43f4250d03d84153a` 是已提交的初版 docs commit；控制 session
提交本轮 authoritative semantic correction 后，实际 dispatch HEAD 必须是 correction base
的 exact single-parent one-commit child。不要猜测该 correction commit 的未来 SHA。开始时
运行并记录：

~~~bash
(
set -e
set -o pipefail
git rev-parse --abbrev-ref HEAD
SAPG_CODE4_START_HEAD=$(git rev-parse HEAD)
printf '%s\n' "$SAPG_CODE4_START_HEAD"
git rev-parse 910a4309918b1dd2fadc60c43f4250d03d84153a^
git rev-list --count \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..910a4309918b1dd2fadc60c43f4250d03d84153a
git diff --name-status \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..910a4309918b1dd2fadc60c43f4250d03d84153a
git merge-base --is-ancestor \
  910a4309918b1dd2fadc60c43f4250d03d84153a HEAD
git rev-list --count \
  910a4309918b1dd2fadc60c43f4250d03d84153a..HEAD
git show -s --format=%P HEAD
git rev-list --count \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..HEAD
git diff --name-status \
  910a4309918b1dd2fadc60c43f4250d03d84153a..HEAD
git diff --name-status \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..HEAD
git status --short --branch
git diff --name-only
git diff --cached --name-only
for code4_file in \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json
do
  test ! -e "$code4_file" || exit 1
done
test "$(git rev-parse --abbrev-ref HEAD)" = \
  feat/simtoolreal-sapg-rlgames
test "$(git rev-parse 910a4309918b1dd2fadc60c43f4250d03d84153a^)" = \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770
test "$(git rev-list --count \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..910a4309918b1dd2fadc60c43f4250d03d84153a)" \
  -eq 1
test "$(git rev-list --count \
  910a4309918b1dd2fadc60c43f4250d03d84153a..HEAD)" -eq 1
test "$(git show -s --format=%P HEAD)" = \
  910a4309918b1dd2fadc60c43f4250d03d84153a
test "$(git rev-list --count \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..HEAD)" -eq 2
test "$(git diff --name-status \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..910a4309918b1dd2fadc60c43f4250d03d84153a)" = \
  "$(printf 'A\tdocs/simtoolreal_sapg_code4_clean_execution_prompt.md\nM\tdocs/simtoolreal_sapg_rlgames_control_handoff.md')"
test "$(git diff --name-status \
  910a4309918b1dd2fadc60c43f4250d03d84153a..HEAD)" = \
  "$(printf 'M\tdocs/simtoolreal_sapg_code4_clean_execution_prompt.md\nM\tdocs/simtoolreal_sapg_rlgames_control_handoff.md')"
test "$(git diff --name-status \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..HEAD)" = \
  "$(printf 'A\tdocs/simtoolreal_sapg_code4_clean_execution_prompt.md\nM\tdocs/simtoolreal_sapg_rlgames_control_handoff.md')"
code4_worktree_status=$(git status --short)
test -z "$code4_worktree_status"
code4_staged_paths=$(git diff --cached --name-only)
test -z "$code4_staged_paths"
)
~~~

所有层次必须同时精确成立：

- `910a4309^` 的输出必须 exactly 是 lineage base，且 lineage base 到 correction base 的
  rev-list count 必须恰为 1；
- correction base 必须是 HEAD ancestor，correction base 到 HEAD 的 count 必须恰为 1，
  `git show -s --format=%P HEAD` 必须只输出完整 correction base SHA，不能有第二 parent；
- lineage base 到 HEAD 的 cumulative count 必须恰为 2；
- lineage base 到 correction base 的 name-status 必须恰为：

~~~text
A	docs/simtoolreal_sapg_code4_clean_execution_prompt.md
M	docs/simtoolreal_sapg_rlgames_control_handoff.md
~~~

- correction base 到 HEAD 的 correction diff 必须恰为：

~~~text
M	docs/simtoolreal_sapg_code4_clean_execution_prompt.md
M	docs/simtoolreal_sapg_rlgames_control_handoff.md
~~~

- lineage base 到 HEAD 的 cumulative diff 必须仍恰为：

~~~text
A	docs/simtoolreal_sapg_code4_clean_execution_prompt.md
M	docs/simtoolreal_sapg_rlgames_control_handoff.md
~~~

开始实现前还必须是正确分支、工作树干净、staging 为空，而且上述五个 Code #4 文件都
不存在。仅证明 ancestor、某一个 count 或 clean tree 不够；任一 parent、count、diff、
branch、clean/staging 或五文件 absence 不精确都返回 # BLOCKED，不得自行清理、切换或覆盖。

固定 Source：

~~~text
Source checkout:    /home/user/ws/lemon/simtoolreal
Source HEAD:        2a9917533bfea70419ed2667a511d7238e5b3abc
RL-Games tree:      7a6a0bb090998d00565aaefa6ab9f2b3d356ace2
train owner:        isaacsimenvs/cfg/train/SimToolRealSAPG.yaml
train owner blob:   f363d05d4a24b190b7837703b93270d8f3fe9a9c
task owner:         isaacsimenvs/cfg/task/SimToolReal.yaml
task owner blob:    6469d46867081b70edaa589dcb31c7090b64d45e
~~~

开始时必须从 Git object database 核对：

~~~bash
git -C /home/user/ws/lemon/simtoolreal rev-parse HEAD
git -C /home/user/ws/lemon/simtoolreal rev-parse \
  HEAD:rl_games/rl_games
git -C /home/user/ws/lemon/simtoolreal rev-parse \
  HEAD:isaacsimenvs/cfg/train/SimToolRealSAPG.yaml
git -C /home/user/ws/lemon/simtoolreal rev-parse \
  HEAD:isaacsimenvs/cfg/task/SimToolReal.yaml
~~~

Source 工作树中的 untracked/dirty 文件不是 oracle。不得修改、删除、stash、reset、clean
Source，也不得从工作树直接相信 owner 或 loaded module 的身份；必须以固定 Git blob
重读、hash 并验证。

### 2.1 唯一 canonical platform

Source capture 和 Target replay 都必须 fail closed 得到：

~~~text
Python:                    3.11.15
PyTorch:                   2.7.0+cu128
CUDA build:                12.8
CUDA runtime:              13020
cuDNN:                     90701
GPU:                       NVIDIA GeForce RTX 4090
compute capability:        [8, 9]
driver:                    580.173.02
device:                    cuda:0
deterministic algorithms:  true
cudnn deterministic:       true
cudnn benchmark:           false
TF32:                      false
float32 matmul precision:  highest
~~~

复用现有 source_network_harness.py 的 canonical configuration 与 platform schema，
但只读调用，不能修改它。manifest 必须保存完整实际 identity；Target 必须得到
is_canonical_platform=true。CUDA 不可用、设备身份漂移、Torch 不是
2.7.0+cu128 或 canonical flags 不匹配时返回 # BLOCKED，不得 skip、CPU fallback 或换
CUDA/Python matrix。autocast/GradScaler enabled 要求只适用于 normal_amp 与
overflow_amp；这两个 case 任一 disabled 都阻塞。normal_fp32 必须反向证明 test-only
mixed_precision=false，actor autocast 与 scaler 均 disabled。central 只在 normal_fp32
和 normal_amp 两个 normal cases 中运行，且始终走普通 FP32
backward/clip/optimizer path，不得被 actor autocast/scaler 包装；overflow_amp 是单个
prepared actor batch probe，不运行 central loop。

## 3. 五文件边界和预算

只能新建以下五个 regular files：

1. scripts/generate_simtoolreal_sapg_update_fixture.py
2. tests/algos/rlgames_sapg/source_update_harness.py
3. tests/algos/rlgames_sapg/test_update_golden.py
4. tests/fixtures/simtoolreal_sapg/source_update_fp32.npz
5. tests/fixtures/simtoolreal_sapg/source_update_manifest.json

前三个文本文件的所有手工编辑，包括临时 mutation 和恢复，都必须使用 apply_patch。
后两个文件只能由本轮 source-only generator 生成；ordinary pytest 永远不得 regenerate。

不得修改任何现有路径，尤其是：

- scripts/generate_simtoolreal_sapg_network_fixture.py；
- scripts/generate_simtoolreal_sapg_rollout_fixture.py；
- source_network_harness.py、source_rollout_harness.py 和任何现有测试；
- code #3 NPZ/manifest；
- third_party/simtoolreal_rl_games/**；
- src/**、conf/**、pyproject.toml、uv.lock；
- 固定 Source checkout。

三个文本文件的净手写规模应保持在本 batch 已批准的约 800 LOC 量级。若需要第六个文件、
长期依赖、生产改动或 fixture+manifest 合计达到 8,388,608 bytes，立即 # BLOCKED。
不得为满足预算压缩掉、摘要掉或省略必需 evidence；也不得提交完整 model parameters 或
完整 Adam tensors。

本批的永久维护成本就是这一个 generator、一个 harness、一个 focused test 和两份小型
frozen artifact；以后 Source/vendor/Python/Torch/CUDA identity 变化都要显式重跑 provenance、
mutation 与 canonical capture gate。它不新增 production runtime owner、公共 contract 或
常规 CI。

## 4. Source/Target namespace 必须隔离

Source capture 和 Target replay 必须是两个全新、互不 import 的进程。一个进程只要同时
加载过 Source 与 vendored Target 的 rl_games namespace，其结果立即无效。

Source 进程必须：

- 要求 UNILAB_SAPG_ORACLE_MODE=source；
- 使用 expected_package_root =
  /home/user/ws/lemon/simtoolreal/rl_games/rl_games；
- 在 import 前核对 Source HEAD、RL-Games tree 和两个 owner blobs；
- capture 后枚举所有 loaded rl_games.* modules；
- 对每个 module 验证 regular-file path 位于 expected_package_root 下，且相对路径对应
  固定 tree 中的 Git blob，当前 bytes 的 SHA256 与 Git blob bytes 一致；
- generator 可以 import 仓库内 source-neutral 的
  tests.algos.rlgames_sapg.source_update_harness 复用 capture implementation，但该 module
  在 import-time 不得调用 Target runtime gate，也不得 eager import rl_games；
- Source 进程不得加载 vendored Target namespace，也不得在同一进程做 Source/Target
  comparison。

Target 进程必须先调用现有 require_simtoolreal_rl_games()，并验证：

- distribution name 是 unilab-simtoolreal-rl-games；
- expected_package_root =
  /home/user/ws/lemon/rlgame-unilab/UniLab/third_party/simtoolreal_rl_games/rl_games；
- 每个 loaded rl_games.* module 都在该 root 下；
- 当前 module bytes 符合 vendored patched manifest，而不是官方 pip package 或 Source path。

expected_package_root 必须真正参与 capture/replay 的 fail-closed 检查，不能只作为未使用
参数、记录字段或事后 label。测试必须把它改成一个错误但存在的目录并观察拒绝。

## 5. 固定 code #3 输入与 native handoff

code #3 只读 anchors：

~~~text
tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz
SHA256 3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8

tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json
SHA256 785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f
~~~

generator 必须先验证两份文件及其 root/ancestor/leaf 都不是 symlink，leaf 是 regular file，
然后依次验证完整 inventory、每个 array 的 shape、dtype 和 content hash。只有这个 gate
完成后才能读取 buffer_post_shuffle__* 作为 update 输入。

冻结事实：

- 56 rows = 48 leader rows + 8 follower rows；
- 14 条完整 sequence，每条 4 time steps，不能拆散；
- post-shuffle row identity、sequence identity 和 row order 必须完整保存；
- actor 与 central 的 native dataset 尾批都必须由 events 推导为 [12, 12, 12, 20]；
- actions、old values、old neglogpacs、old mu/sigma、obses、privileged states、dones、
  returns、off_policy_mask、rnn_states 保持 code #3 原 dtype/shape/order；
- rnn_masks 在 frozen input 不存在时保持 None；若 native prepare_dataset 产生，则记录
  native 结果，不能人为制造。

normal_fp32 和 normal_amp 的 train_epoch boundary 分别安装一次 identity
a2c_common.shuffle_batch freeze wrapper。每个 case 必须恰好调用一次，返回同一输入对象、
不改 row order、不消费 RNG，并在 finally 恢复。调用零次、两次或回到随机 shuffle 都是
失败。overflow_amp 只能从 native prepare 后的第一个 actor batch clone 开始，不重新
shuffle 或生成 rollout。

manifest 把 Source owner defaults 与 test-only overrides 分开。固定 test-only case 为
12 synthetic actors、block size 2、6 blocks、horizon/sequence 4、actor/central
minibatch_size 12、mini_epochs 2；其余 Source fields 包括 e_clip=0.1、
critic_coef=4.0、bounds_loss_coef=0.0001、entropy_coef=0.0、
learning_rate=1e-4、adaptive/standard scheduler、kl_threshold=0.016、
normalize_input/value/advantage=true、truncate_grads=true、grad_norm=1.0、
gamma=0.99、tau=0.95、value_bootstrap=true、ppo=true、mixed_precision=true 和
clip_value=true 均从固定 train owner 核对并记录。为接入已经 augmented/shuffled 的 batch，
只允许在 test boundary 把 use_others_experience 冻结为 none；该 override 不得进入生产。

## 6. 必须使用的 native owner

先从固定 Source commit 的 Git object database 读取 owner blobs、用 `nl -ba` 核对实现，再
通过真实对象调用。以下命令不读取或信任 Source worktree bytes：

~~~bash
(
set -e
set -o pipefail
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/common/a2c_common.py \
  | nl -ba | sed -n '360,393p;429,437p;1370,1532p'
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/algos_torch/a2c_continuous.py \
  | nl -ba | sed -n '14,79p;105,234p'
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/algos_torch/central_value.py \
  | nl -ba | sed -n '207,292p'
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/algos_torch/models.py \
  | nl -ba | sed -n '36,62p;245,295p;440,474p'
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/algos_torch/running_mean_std.py \
  | nl -ba | sed -n '10,94p'
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/common/datasets.py \
  | nl -ba | sed -n '25,80p'
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/common/common_losses.py \
  | nl -ba | sed -n '10,48p'
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/algos_torch/torch_ext.py \
  | nl -ba | sed -n '29,38p;143,154p'
git -C /home/user/ws/lemon/simtoolreal show \
  2a9917533bfea70419ed2667a511d7238e5b3abc:rl_games/rl_games/common/schedulers.py \
  | nl -ba | sed -n '20,35p'
)
~~~

- rl_games.common.a2c_common 的 set_train、continuous train_epoch 与 prepare_dataset；
- rl_games.algos_torch.a2c_continuous 的 central constructor/value alias、
  A2CAgent.calc_gradients 与 train_actor_critic；
- rl_games.common.datasets.PPODataset slicing 与 update_mu_sigma；
- rl_games.algos_torch.central_value.CentralValueTrain.train_critic 与 train_net；
- rl_games.algos_torch.models 的 input norm_obs、continuous value forward/denorm path 与
  ModelCentralValue.Network.forward；
- rl_games.algos_torch.running_mean_std 的 training-conditional forward/update；
- rl_games.common.common_losses 的 actor/value loss；
- rl_games.algos_torch.torch_ext 的 apply_masks 与 policy_kl；
- rl_games.common.schedulers 的 native scheduler；
- agent 和 central_value_net 实际 optimizer；
- actor 的真实 autocast 与 torch GradScaler scale/unscale_/step/update。

构造必须经过 Runner.load()、set_vec_env()、native algo factory、A2CAgent 和
init_tensors()。可以在 test boundary 提供只实现 native ABI 的 synthetic vecenv，并让
play_steps delegate freeze 返回 code #3 batch clone 与 ps_extras；不得用 fake agent、
unbound method、第二个手写 optimizer 或自建训练循环替代 native train_epoch。

deterministic parameter fill 必须复用 code #3/network 已有 name-seeded helper，记录 native
initialization hash 与 fill 后 hash，并证明 fill 不消费 RNG。

instrumentation 规则：

1. wrapper 进入后调用被包裹 native owner 恰好一次，再记录其真实 inputs/outputs/events；
2. 不额外执行一次 native loss、backward、optimizer 或 RNG operation 来获得“方便的”
   expected value；
3. 对 torch.exp/clamp/max、apply_masks、loss、dataset、optimizer、GradScaler、
   clip_grad_norm_、scheduler/update_lr 的观察必须限定在当前 tagged native call context；
4. monkeypatch、hook、module train/eval 状态和 global function 都必须在 finally 恢复；
5. wrapper 抛异常时也要恢复；测试必须检查恢复后的 object identity；
6. 除明确批准的 play_steps 与 identity shuffle freeze boundary 外，所有 spy 只 delegate，
   不改变输入、返回或调用次数。

禁止在 harness/test 中写出或重算 surrogate、value clip、bounds、entropy、KL、mask
denominator、normalizer、total loss、gradient clip、Adam、overflow 或 LR 公式。所需
branches 和 total loss 必须从 native 调用的参数、返回值及传给 backward/scaler 的实际
tensor直接记录。

## 7. 三个 case 与完整 evidence

### 7.1 case inventory

必须 source capture 并 Target replay：

1. normal_fp32：仍在 cuda:0，同一 frozen input/weights，test-only
   mixed_precision=false；用作 FP32 numeric golden。
2. normal_amp：Source owner mixed_precision=true，真实 autocast 与 enabled GradScaler；
   step/skip mask、scale 和 growth tracker 必须从本次 Source capture 得到。
3. overflow_amp：从 native prepare_dataset 产生的第一个 actor batch clone，仅把
   advantages[0] 设为 +inf，然后调用真实 native calc_gradients/scaler path。

不得在 prompt、代码或测试中预设未知的 normal_amp step mask、scaler sequence、最终
artifact hashes或 pytest test counts。它们都必须由新的固定 Source capture 或实际 test
输出产生，再作为 fixture evidence 冻结。

AMP 不与 FP32 做逐元素比较。normal_amp/overflow_amp 只比较 Source/Target 的实际
autocast dtype、enabled state、scale/growth transition、underlying optimizer step mask、
parameter change relation、overflow skip 和签名语义。

### 7.2 native event order、epochs 和 batches

每条 event 至少含 case、owner、phase、dataset access counter、native dataset length、
derived mini_epoch、derived mini_batch、row IDs 和 row count。不能在 manifest 手写
mini_epochs=2、batch_count=4 或 value-before-dataset=true 作为证据。

对 actor 和 central 分别从 PPODataset access events 推导且严格覆盖：

~~~text
(0,0), (0,1), (0,2), (0,3),
(1,0), (1,1), (1,2), (1,3)
~~~

每个 epoch 的 batch sizes 必须从 row IDs 得到：

~~~text
[12, 12, 12, 20]
~~~

event validator 必须证明：

- prepare_dataset start/end 先于两个 dataset 的消费；
- central 的最后一个 real optimizer result 先于 actor 第一个 forward/backward；
- central 与 actor 都真实完成两个 epoch、八个 batch；
- scheduler/update_lr 在每个 actor epoch 末各一次，不在每 batch 调用；
- identity shuffle 对每个 normal case 恰好一次；
- overflow 是单独的 prepared actor batch probe，不伪造 central loop。

### 7.3 prepared data 与 dataset handoff

必须把下列 native tensors/metadata持久化，而不是只留在临时 trace：

- prepare 前 original old_values 与 returns；
- prepare 后 normalized old_values、normalized returns 和 native advantages；
- actual train_value_mean_std；
- actor dataset 与 central dataset 的完整 handoff inventory、row-key mapping、shape/dtype/hash；
- obs/states/actions/dones/rnn_states/rnn_masks、old neglogp、old mu/sigma；
- PPODataset.last_range、access range 与 update_mu_sigma start/end range/content hash。

validator 要从 native events 证明 value normalizer update 发生在 dataset handoff 前。不得从
最终数组手算 advantages、ratio、loss 或 optimizer delta。

### 7.4 四类 normalizer

每个 normal case 必须先在各自进程内 fail closed 建立以下固定 role→object mapping，再记录
after_prepare、每个 central epoch 后、每个 actor epoch 后和 final 的 snapshot：

1. actor input RMS = `agent.model.running_mean_std`；
2. central input RMS = `agent.central_value_net.model.running_mean_std`；
3. actor-model value RMS = `agent.model.value_mean_std`，它是 distinct object；
4. active/central value RMS = `agent.value_mean_std`，并且必须以 runtime
   `agent.value_mean_std is agent.central_value_net.model.value_mean_std` 证明 alias。

alias 只允许在 Source 进程和 Target 进程内分别用 `is` 验证；不得跨进程比较，也不得把
raw `id()` 持久化为 fixture/manifest identity。四个 role 都要分别保存完整
running_mean、running_var、count、training mode，以及每次 forward 的 owner context、输入
row count、进入/退出时 mode、state/count before/after 和是否实际 update。mode 为
`training=true` 只表示允许 update，不等于已经发生 forward/update。

固定 native transition contract 是：

- 初始四对象都是 count=1、training=true。after_prepare 时 actor input、central input、
  actor-model value 仍是 count=1、training=true；active/central value 经 values 与 returns
  两次 native forward/update，count 按 1→57→113，然后 native eval，故 count=113、
  training=false。该 update 必须发生在两个 dataset handoff 前。
- central input 的每个 batch 都经过 `train_critic()->self.train()`，所以 epoch 0 的四个
  train-forward 将 count 从 1 更新到 57，epoch 0 尾 native eval；epoch 1 第一 batch 又由
  `self.train()` 恢复 training=true，四个 train-forward继续把 count 更新到 113，epoch 1
  尾再次 eval。final 为 count=113、training=false。epoch 0 尾 eval 不是永久 freeze。
- actor input 在 actor epoch 0 的四个真实 batch forward 更新 count 1→57，epoch 0 尾
  native eval；epoch 1 仍完整执行四个 batch forward 和 optimizer step，但 input RMS
  training=false，mean/var/count 均不变，epoch 1 尾再次 eval，final 仍 count=57、
  training=false。
- actor-model value RMS 不参与这条 native update path 的 forward/update；它始终 count=1、
  training=true、forward count=0。必须把它自己的 mode 与 mean/var/count 独立记录，不能
  编造 first update 或 freeze，也不能与 active value RMS 合并。
- active/central value RMS 只在 prepare_dataset 通过上述 alias 对 values 与 returns 做两次
  native forward/update，随后没有 value-RMS forward。central 的 `self.train()` 会在后续
  batch 把其 training flag 设回 true，所以 central epoch 0、epoch 1 与 final 都是
  count=113、training=true；central train-forward 不调用 value RMS，mean/var/count 不再
  更新。

Source capture 与 Target replay 必须各自先验证 role→object identity/alias、全部
owner-specific mode transition 与 forward/update event、mean/var/count transition；两边
各自 invariants 完整通过后才允许 semantic 或 numeric comparison。任一 role/object、alias、
event 或 snapshot 字段缺失时，即使 Source 与 Target 同时缺失也必须失败。

### 7.5 每个 actor batch 的 native update evidence

八个 normal actor batches 都必须记录：

- old/new neglogp 与 native ratio；
- native unclipped/clipped surrogate branches 及 selected per-row actor loss；
- old/new values、returns、clip range、native unclipped/clipped value branches 与 selected result；
- native mu、soft bounds、bounds branch 和 coefficient；
- raw per-row entropy、mixed-expl 选中的 per-row entropy coefficient、coefficient/product；
- native apply_masks 的 mask=None 或实际 mask、shape、mask cardinality、sum 和真正 reduction
  denominator，以及 actor/value/bounds/entropy reduced outputs；
- native total loss tensor（实际送入 scaler/backward 的值），以及其四类 native component
  references；不得在 harness 重算 total loss；
- new mu/sigma、native policy KL；
- update_mu_sigma 调用前后的 row-keyed old mu/sigma reference。

第二 mini-epoch 每一行的 old mu/sigma 必须由第一 epoch 对应 batch 的 new mu/sigma 经
native PPODataset.update_mu_sigma 更新。删除、延后或错 row 更新必须 RED。

如果 normal input 的 rnn_masks 为 None，记录 native mean denominator。另做一个完全不
进入 optimizer、无 RNG 消耗的 native apply_masks diagnostic，用固定含零 mask capture
Source 的真实 denominator 行为；不要在测试代码重写 apply_masks。

### 7.6 central evidence、gradients 与 optimizer

每个 central batch 保存 native model/value/loss、backward、clip 和 optimizer events。
actor 与 central 分别保存：

- actual optimizer.param_groups 经 parameter object identity 映射得到的参数名、group index、
  LR、eps、weight_decay 和其他实际 hyperparameters；
- 未映射、重复映射或 actor/central 参数交叉时 fail closed；
- backward 后每个参数 gradient signature 与 absent list；
- actor AMP 的 scaled gradients；
- scaler.unscale_ 后、clip 前的 unscaled gradients 和 total norm；
- clip_grad_norm_ 返回值与 clip 后 signature/norm；
- central 普通 FP32 backward、clip 前/后 signature/norm，且没有被 actor AMP 包装；
- real optimizer step 前后 parameter signatures；
- 每参数 delta 的 shape/dtype/hash/norm/sum/max/max_abs 和 name-seeded sentinels；
- optimizer state keys、step 与 state tensor signatures；
- underlying optimizer 是否真的被 GradScaler 调用，而不是根据 scaler.step 返回值猜测。

大 tensor 只保存 canonical SHA256、shape/dtype、norm/sum/max/max_abs 与 64 个
name-seeded sentinel coordinates；不得保存完整 model/Adam state。

### 7.7 scheduler、update_lr 与 old reference

每个 actor epoch 保存 batch KL list、native mean input、scheduler 的 LR/entropy/KL 输入、
输出，以及 native update_lr 后所有 optimizer param-group 的实际 LR。validator 必须从
event order 证明每 epoch 一次、总计两次，并验证 threshold 0.016 来自 owner config。

不能只记录 scheduler 返回值；必须记录 update_lr delegate 的调用和调用后的 optimizer
state。不能把第二 epoch old mu/sigma 继续指向 rollout 副本。

### 7.8 autocast、GradScaler 与真实 overflow

每个 actor batch 保存实际 autocast enabled/device/dtype，以及 GradScaler 在
scale/unscale_/step/update 前后的 enabled、scale、growth_factor、backoff_factor、
growth_interval、growth tracker、found-inf/相关 native state。

normal_amp 的实际 step mask完全由 Source capture 决定。对每一 batch，invariants 独立验证：

- 记录的 underlying optimizer call 与 parameter delta/change relation一致；
- skipped batch 参数保持不变、optimizer state step 不伪增；
- successful batch 存在相应真实 call 和一致 delta/state；
- scale/growth transition 与真实 scaler event sequence 自洽。

overflow_amp 必须观察：

- advantages[0]=+inf 是唯一 test-only input mutation；
- 真实 scaler.scale(loss).backward()、unscale_、step、update 均被调用；
- underlying optimizer.step 没有执行；
- 参数和 optimizer step/state保持不变；
- scale backoff、growth tracker/found-inf state发生 Source 所定义的转换；
- 没有 fake scaler、手动 if 跳过、无条件 optimizer.step 或 CPU fallback。

### 7.9 RNG

必须无损保存 NumPy global state、Torch CPU state 和 torch.cuda.get_rng_state_all() 的完整
bytes/arrays及各 component hash，至少覆盖：

~~~text
after_runner_seed
after_agent_initialization
after_deterministic_parameter_fill
before_prepare
after_prepare
before_central
after_central
before_actor_epoch_0
after_actor_epoch_0
before_actor_epoch_1
after_actor_epoch_1
before_overflow_step
after_overflow_step
~~~

只存 digest 不算“完整 RNG state”。每个 phase 同时记录 component hash 便于诊断。
instrumentation、metadata validation、signature/sentinel selection、identity freeze 和
apply_masks diagnostic 必须使用局部确定性规则且不额外消费 global RNG；用前后完整 state
直接证明，而不是用布尔 label 声称。

## 8. Fixture、anchors 和 evidence invariants

source_update_fp32.npz 保存必要 frozen arrays、小型 native tensors、row identity、完整 RNG
state、overflow probe 和 numeric traces。source_update_manifest.json 保存 provenance、
platform、owner/default overrides、module inventory、events、metadata、signatures、
normalizers、role/object mapping、process-local alias proof、owner-specific mode transitions 与
forward/update events、optimizer/scaler relations、comparison inventory 和 generation command。

manifest 至少包含 schema_version、generation_mode=source-only、固定 Source HEAD/tree、
train/task owner path/blob/SHA256、两个 Code #3 file anchors、loaded Source module
path/blob/SHA256 inventory、canonical platform、Source defaults/test-only overrides、
三 case inventory、NPZ array inventory、event/invariant schema、normalizer/RNG/scaler
evidence、四 RMS role/object mapping 与 distinctness、process-local active/central value alias
proof、owner-specific training transitions/forward-update events、tolerances、FP32 comparison
inventory、NPZ file SHA 和 canonical payload SHA。schema 不得持久化 raw `id()` 或用它做
跨进程 identity comparison。

每个 NPZ array 在 manifest 中有且只有一个 exact：

~~~text
{name, shape, dtype, sha256, semantic_role, comparison_domain}
~~~

NPZ 用 allow_pickle=False 读取。missing/extra name、reshape、dtype drift 或 content-only
drift 都必须在错误中指出具体字段；不允许先让 NumPy broadcasting 或 subtraction 给出
间接错误。

### 8.1 三个独立外部 anchors

最终 Source capture 后，必须重新从磁盘计算并冻结：

1. update NPZ file SHA256；
2. update manifest file SHA256；
3. canonical manifest payload SHA256。

payload 使用定义明确的 canonical JSON serialization，并只排除自引用的
canonical_payload_sha256 字段。manifest 内记录 NPZ SHA 和 payload SHA；harness/test 中
另有固定常量保存 NPZ、manifest file 和 payload 三个值。loader 不得从被验证的同一
manifest 读取“expected”值来自证；普通 test 不得自动更新 constants。这里禁止使用任何
旧 Code #4 artifact hash，最终值现在未知，必须以新 Source capture 为准。

### 8.2 不可伪造 metadata token 和固定执行顺序

loader/replay 只能按：

~~~text
namespace/provenance and external file anchors
  -> exact inventory
  -> every array shape/dtype/content validation
  -> Source evidence invariants
  -> Target exact inventory and every-array metadata validation
  -> Target evidence invariants
  -> recursive semantic/RNG comparison
  -> explicit FP32 numeric comparison
~~~

完成全部 array metadata 后，由 validator 返回一个绑定完整 sorted inventory、
metadata digest、当前 arrays object identities 和 module-private capability 的 frozen token。
numeric comparison 必须要求这个 token，并重新核对 actual/expected inventory 与 digest。
调用者不能用 metadata_validated=true、公开 dataclass constructor、伪 event label 或
部分 names list 授权 subtraction。

Source NPZ 的 content hash 必须逐项等于 manifest。Target capture 也必须先为每个实际
array 计算并回验其自身冻结 metadata/content digest，证明比较输入在 gate 后没有变化；
FP32 Target content hash 不要求在 numeric gate 前等于 Source hash，否则会绕开声明的
atol/rtol 口径。离散、identity、RNG bytes 和明确 exact-hash domain 仍按各自 semantic
contract exact 比较。

numeric function 还必须拒绝：

- actual/expected names 与 token 不完全相同；
- FP32 comparison inventory 重复；
- comparison name 未经过 metadata gate；
- manifest 中标为 normal_fp32 numeric 的承诺 array 未列入 comparison inventory；
- numeric subtraction 在 Source/Target semantic invariants 前发生。

### 8.3 显式 FP32 comparison inventory

manifest 必须有唯一、稳定排序的 fp32_comparison_inventory。它逐项列出所有需要
Source→Target atol=1e-6、rtol=1e-5 比较的 normal_fp32 arrays，至少覆盖 prepared
old_values/returns/advantages、每 batch loss/ratio/surrogate/value/bounds/entropy、
new mu/sigma、KL、normalizer、gradient/clip、optimizer delta/state numeric traces 和
scheduler/LR。所有标成 comparison_domain=normal_fp32 的 NPZ array 必须恰好出现一次；
排除项只能是明确的 identity/control/hash/RNG bytes 或 AMP domain，不能靠名字前缀隐式
筛选。

AMP 不进入这个 FP32 inventory；它通过独立 Source→Target dtype/control/scaler/step 和
signature relation验证。

### 8.4 Source 和 Target 各自 evidence invariants

validate_update_evidence_invariants(capture) 必须先分别作用于 Source capture 和 Target
capture，再允许递归相等。它至少独立检查：

- 三 case 完整且名字唯一；
- identity shuffle 次数、56 rows、14 sequences；
- actor/central 两 epoch、八 batches、完整 derived tuple 与 [12,12,12,20]；
- central-before-actor 和 prepare/dataset顺序；
- prepared tensors、dataset handoff、四 RMS role/object mapping、distinctness、进程内 alias、
  owner-specific transitions/forward-update events 与完整 phase snapshots；
- exact RMS transition contract；必须拒绝 central input epoch 1 伪 freeze、actor input
  epoch 1 伪继续 update、伪造 active/central value alias，以及给 actor-model value RMS
  编造 forward/update；
- 每 actor/central batch 的 native loss/gradient/clip/optimizer coverage；
- update_mu_sigma row reference 与 scheduler/update_lr coverage；
- autocast/scaler/growth/step mask/parameter delta relations；
- overflow真实 skip/parameter/optimizer state relation；
- 所有 RNG phases 与完整 component state；
- no-extra-RNG 和 finally restore evidence。

human-readable evidence_inventory 只能在 validator 成功后由真实 event coverage生成，不能
是预先写好的标签。即使 Source 与 Target 同时删除相同 loss events、prepared field、
normalizer role/object/alias/transition/forward-update event、optimizer result 或 scaler
transition，也必须在 equality 前失败。

## 9. File/path 安全

loader 与 generator 都必须用 lstat 逐段检查，不可先 resolve 后掩盖 symlink：

- 拒绝 filesystem root 作为 output root；
- 拒绝任一 existing ancestor 是 symlink，包括指向有效目录的 symlink；
- 拒绝 broken ancestor/root/leaf symlink；
- output root 必须是 existing real directory，non-directory root 拒绝；
- 两个固定 leaf 若存在，只能是 non-symlink regular file；directory、FIFO/device/socket、
  symlink 和 broken symlink 都拒绝；
- loader 对 input root/ancestor/leaf 使用同样 regular-file gate；
- 所有 provenance、inventory、budget 和安全校验必须在第一次 write/replace 前完成；
- 只允许写两个固定 leaf，使用同目录安全临时 regular file 再原子 replace；
- 任一失败都不能修改 root 外 sentinel 或已有 fixture。

test 必须覆盖 real directory success，以及 root/ancestor/leaf symlink、broken symlink、
directory leaf、non-directory root 和外部 sentinel不变。

## 10. TDD 与必须观察的 mutation RED

先用 apply_patch 只创建 test_update_golden.py 的 public import boundary：

~~~python
from tests.algos.rlgames_sapg._runtime_requirement import require_simtoolreal_rl_games

require_simtoolreal_rl_games()

from tests.algos.rlgames_sapg.source_update_harness import (
    load_update_fixture,
    replay_update_fixture,
    validate_update_evidence_invariants,
)
~~~

然后运行初始 RED：

~~~bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_update_golden.py -q
~~~

预期 collection 只因
tests.algos.rlgames_sapg.source_update_harness 不存在而失败；relative-import/package
context 错误不是有效 RED。失败如实记录，不得称为算法 mismatch。

最终 test module 至少必须有以下固定 node IDs，供 lightweight gate 精确运行：

~~~text
test_evidence_invariants_reject_symmetric_source_target_deletion
test_metadata_validation_precedes_semantic_and_numeric_comparison
test_wrong_expected_package_root_is_rejected
test_generator_rejects_unsafe_output_paths_without_touching_sentinel
~~~

在第一次 CUDA regeneration 前运行：

~~~bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest \
    tests/algos/rlgames_sapg/test_update_golden.py::test_evidence_invariants_reject_symmetric_source_target_deletion \
    tests/algos/rlgames_sapg/test_update_golden.py::test_metadata_validation_precedes_semantic_and_numeric_comparison \
    tests/algos/rlgames_sapg/test_update_golden.py::test_wrong_expected_package_root_is_rejected \
    tests/algos/rlgames_sapg/test_update_golden.py::test_generator_rejects_unsafe_output_paths_without_touching_sentinel \
    -q
~~~

完整实现后，至少逐项观察以下 mutation RED：

1. 移除或绕过 normal case identity shuffle freeze；
2. 交换 central/actor order，或伪造 order labels；
3. 删除/延后 PPODataset.update_mu_sigma；
4. 对 Source 和 Target 对称删除同一 evidence，包括 loss/optimizer/normalizer/prepared field；
5. 篡改 normal_amp step mask、GradScaler scale 或 growth tracker，但保留描述标签；
6. overflow 路径改成无条件 underlying optimizer.step；
7. 把第一次 numeric subtraction 移到 complete metadata/semantic gate 前；
8. 把 expected_package_root 改成错误 namespace。
9. 把 central input RMS 的 epoch 1 events/count 篡改为 epoch 0 后永久 freeze；
10. 把 actor input RMS 的 epoch 1 四次 forward 篡改为继续更新 mean/var/count；
11. 伪造 `agent.value_mean_std` 与 central model value RMS 的进程内 alias，或以 raw `id()`
    冒充可跨进程 identity；
12. 给 actor-model value RMS 编造 native forward/update、first update 或 freeze。

还必须有 inventory missing/extra、reshape、dtype、content-only、symlink/path cases 的永久
mutation tests。任何改变文件的临时 mutant 都必须用 apply_patch；每次先记录文本文件
SHA256，观察对应测试失败，再用 apply_patch 精确恢复，确认 SHA256 回到 mutation 前并
重跑 GREEN。数据 mutation 可以留作永久测试，但报告仍须列出实际 RED 命令/失败原因。
不得修改 vendor、Source、code #3 或 fixture 来制造 RED。

## 11. Source generation、freeze 和验证命令

### 11.1 source-only generation

只在全部 lightweight gates 通过后运行一次：

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

generator 必须在进程内执行固定 Source Git/provenance、namespace、canonical platform、
code #3 anchors、path safety 和 native evidence gate。它只能写两份本轮 fixture；若修改
Source、code #3 或其他路径，立即停止。

生成后重新冻结外部 anchors，并运行：

~~~bash
sha256sum \
  tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json
wc -c \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json
file \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json
~~~

五个 leaf 都必须是 regular file，fixture+manifest 总数严格小于 8,388,608 bytes。将新
NPZ、manifest file、canonical payload SHA256 用 apply_patch 固定到 harness/test 后，
ordinary tests 只能验证，不能 rebaseline。

### 11.2 required Target gates

Focused Code #4：

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

Vendor 与 audit：

~~~bash
env -u UV_INDEX uv run --python 3.11 \
  pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q
env -u UV_INDEX uv run --python 3.11 \
  scripts/audit_simtoolreal_rlgames_vendor.py
~~~

Scoped 与 root style：

~~~bash
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
~~~

最终 Git 边界：

~~~bash
rg -n '[[:blank:]]+$' \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py
git diff --check
git diff --cached --name-only
git status --short
git diff --stat
~~~

实现期间五个 Code #4 files 都是 untracked，staging 必须为空；因此 git diff --check
只检查 tracked diff，不覆盖这五个文件。上面的 rg 是三个 untracked text files 的显式
trailing-whitespace gate，预期 0 matches（rg 因无匹配返回 1 是本 gate 的成功语义）；
两个 binary/JSON-generated fixture 由 generator metadata/hash gate 覆盖。实现 agent 的
交接必须分别记录 rg 的 0 matches、git diff --check 的 tracked-only 结果、五个 untracked
paths 和 staged=0。

控制 session 接受实现后才精确 stage 五个声明路径，并必须运行：

~~~bash
git diff --cached --name-status
git diff --cached --check
~~~

git diff --cached --name-status 用于核对完整五路径边界；git diff --cached --check 才覆盖
staged 的三个手写文本和 generated JSON manifest 的 whitespace diagnostics（NPZ 为
binary）。两项结果都由控制 session 单独记录，不能用实现期的 git diff --check 替代。
不得为获得 cached gate 让实现 agent stage，也不得增加第六个实现文件。

所有 required pytest 必须实际通过且 0 skip。不要在规格中预设最终 passed 数；报告本次
命令的真实 counts。audit 必须实际通过并打印固定 Source/72-file identity。

固定 Source 在 Torch 2.7 上由 native torch.cuda.amp.GradScaler 和
torch.cuda.amp.autocast 发出的 deprecation FutureWarning 是唯一预先允许的 warning 组；
必须原样保留、逐条记录并解释其 Source-owner provenance。不得修改 Source/vendor native
owner、包裹 warning filter 或屏蔽 warning 来消除它们。除这组已知 owner warnings 外，
任何 warning 都必须解释并修复后才能 GREEN；skip、failure 或
is_canonical_platform=false 始终不算 GREEN。

本 Code #4 实现 session 明确不运行 make test-all，也不运行 Python 3.10/3.12/3.13、
cu126、CPU AMP 或其他 device/version matrix。

## 12. 停止条件

出现任一情况立即停止写入并返回 # BLOCKED：

1. 起点 branch/ancestor/clean-tree/five-file boundary 不符；
2. 需要第六个文件或修改 code #3、vendor、Source、生产 code/config/package；
3. native Runner/A2CAgent/PPODataset/CentralValueTrain path 无法使用，必须复制算法公式；
4. delegate instrumentation 不能 finally 恢复或必须改变 native inputs/outputs/call count；
5. 56-row/sequence、central-before-actor、epoch/batch、prepared/normalizer、
   update_mu_sigma、loss/optimizer/scheduler evidence 无法由 native events证明；
6. Source 或 Target 自身 evidence invariants 不完整，或对称删除仍能通过；
7. metadata token、explicit FP32 inventory、namespace/expected root 或 path safety不能
   fail closed；
8. canonical Python 3.11 + Torch 2.7.0+cu128 + RTX 4090 不可真实执行；
9. normal AMP/overflow 只能通过 disabled/fake scaler、CPU fallback、手动 skip 或
   无条件 optimizer step 模拟；
10. Source/Target provenance、owner blobs、code #3 anchors 或 loaded module identity漂移；
11. fixture+manifest 达到 8 MiB，或只能删除必要 evidence 才满足预算；
12. required test 有 skip/failure、除已列 native AMP deprecation FutureWarning 外仍有
    warning、未解释 mismatch，或 mutation没有真实 RED；已列 warning 未原样记录和解释
    也同样阻塞；
13. 需要进入 checkpoint/player、MuJoCo、async/distributed/compile/export 或 Code #5。

不要放宽 tolerance、重生 code #3、使用旧 artifact anchor、伪造 label 或扩大 scope 来绕过
停止条件。

## 13. 实现 agent 交接格式

成功时只以 # DONE 开头，并逐项报告：

1. 五个最终文件、regular-file/path-safety结果、文本净 LOC、两个 fixture bytes 与
   小于 8 MiB 的计算；
2. 起始/结束 branch 和 HEAD、git status --short、git diff --stat、staged=0；
3. Source HEAD/tree、train/task owner blobs、loaded Source module blob/SHA256 inventory；
4. Target distribution/root/patched-hash provenance；
5. Python/Torch/CUDA/cuDNN/driver/GPU/flags 的完整实际 canonical identity；
6. code #3 两个固定 hashes，以及重新 capture 的 update NPZ/manifest/payload 三 anchors；
7. 初始 collection RED、十二类 required mutation RED、每次恢复 hash 与最终 GREEN；
8. 56 rows/14 sequences、每 normal case identity shuffle 次数、actor/central derived
   epochs/batches/order；
9. prepared tensors、dataset handoff、四 RMS 的 role→object mapping、process-local alias proof、
   owner-specific training transitions/forward-update events 和 mean/var/count snapshots；
10. 每 batch loss branches、entropy coefficient/product、apply_masks denominator、native
    total loss、new mu/sigma、KL/update_mu_sigma；
11. actor/central actual param groups、scaled/unscaled/clip、delta/state、scheduler/update_lr 后 LR；
12. normal_fp32、normal_amp、overflow_amp 本次 Source capture 的实际 autocast dtype、
    step mask、scaler/growth tracker 和 parameter/optimizer relation；明确没有把 AMP 与
    FP32 逐元素比较；
13. 每个 phase 的完整 NumPy/Torch CPU/CUDA RNG state/hash 与 no-extra-consumption；
14. explicit FP32 comparison inventory count/coverage、metadata token/order、Source/Target
    各自 evidence invariant结果；
15. source generation、focused/full SAPG、vendor、audit、Ruff、format 的每条实际命令、
    exit status、真实 pass/skip 数；
16. untracked 三文本 rg trailing-whitespace gate 的 0 matches，以及 git diff --check 的
    tracked-only 结果；另列五个 untracked paths、git diff --cached --name-only 为空和
    staged=0；
17. 明确说明 git diff --cached --check 只能由控制 session 精确 stage 后执行并记录；
    同时确认未修改五文件外路径、未运行 make test-all、未执行 Git 写操作、未进入 Code #5。

阻塞时只以 # BLOCKED 开头，给出：停止条件编号、最后一个成功 gate、失败命令及完整关键
输出、当前五文件状态、git status、是否有部分新 fixture。不要继续写入或自行清理。

无论 # DONE 或 # BLOCKED，报告后停止；控制 session 独立审查、验证、stage 和 commit。
