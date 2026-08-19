# SimToolReal SAPG Source-Fidelity Migration Implementation Plan

> **For agentic workers:** This umbrella is planning-only. After the maintainer approves one named child issue, write/execute only that child's standalone plan with `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`; never execute this umbrella sequentially. Checkboxes below describe the first three child plans, not blanket authorization.

**Goal:** 在 UniLab 的 MuJoCo SimToolReal 任务上直接运行固定 Source fork 的 RL-Games SAPG，使 SAPG 网络、rollout、augmentation、更新、AMP、checkpoint 和 player 语义以 Source 为唯一 oracle；算法边界以下只保留已经明确批准的 MuJoCo/task/resource 差异。

**Architecture:** 完整保留 Source 的 `rl_games` Python runtime，由原生 `Runner -> A2CAgent` 唯一拥有 rollout、env step/reset、central critic、actor update、checkpoint 和 player lifecycle。UniLab 只负责 Hydra owner、创建满足 `NpEnv` contract 的 MuJoCo SimToolReal env、通过一个同步 `IVecEnv` adapter 转换 obs/action/done/info，以及接入现有实验目录和 tracker；不再在 RSL-RL 中仿写 SAPG，也不新增 async collector/learner 协议。

**Tech Stack:** Python 3.10–3.13、PyTorch 2.7、Hydra/OmegaConf、Gymnasium、固定 SimToolReal RL-Games fork、UniLab `NpEnv`/MuJoCo backend、MuJoCoUni、pytest、NumPy frozen fixtures。

---

## Owner summary

决策记录（2026-08-19）：maintainer 已接受“固定 Source RL-Games runtime + 同步 `NpEnv` adapter”成为 SAPG 唯一路径，并接受对应的长期 fork 成本。本路线只做 MuJoCo SimToolReal SAPG，复用 donor task/assets 并以 frozen oracle 锁定算法与 task 数学；不做 PPO、Motrix、sim2sim、async、distributed 或 export。当前明细为 37 个 UniLab PR，另有 MuJoCoUni `M0-dev` 与 `M0-release` 两个外部 gate。永久成本是维护 72-file fork、兼容补丁、adapter、task 与 golden tests。

本文件是可跟踪的 umbrella roadmap；上述架构决策不构成 37 个 child 的批量实施授权。任何 child issue 开始前仍需按 `AGENTS.md` 单独确认；完成一个 child 后不得自动进入下一个。

## Execution and review protocol

每个 child 使用两个串行角色，且同一工作树任一时刻只有一个 writer：

1. 控制/审查 session 固定 child scope，生成只覆盖该 child 的完整 prompt，并独占 branch、staging area、commit 与 PR 历史。启动实现 session 后，控制 session 在收到交接前不得同时编辑工作树。
2. 实现 session 只允许修改 prompt 明列的文件并执行 focused tests；所有 Python 命令使用 `uv run`，所有手工编辑使用 `apply_patch`。它不得执行 `git add`、`git commit`、`git push`、创建 PR、`stash`、`reset`、`clean` 或切换 branch，也不得自动进入下一 child。
3. 实现 session 完成后必须停止写入并报告：改动文件、`git status --short`、`git diff --stat`、完整验证命令及结果、已知缺口和 blocker。测试失败、skip 或未验证平台必须如实列出，不能用预期结果代替实际结果。
4. 控制 session 独立阅读完整 diff，按 child spec 做 scope/spec review，再做代码质量、provenance 与近风险验证。发现问题时先把具体反馈交回同一实现 session 修正并重新交接；有未关闭问题时不得提交。
5. 只有控制 session 可以精确 stage 本 child 的文件并 commit。最终 commit 后、创建或更新 PR 前，由控制 session 按根 `AGENTS.md` 运行 `make test-all`、复跑 child 的关键 gate 并确认工作树干净；任何后续改动都会使旧验证失效。

子代理或额外 reviewer 可以辅助 spec/code-quality review，但不能代替控制 session 对最终 diff、测试证据和 commit 内容负责。用户把某个 child prompt 交给实现 session，即只授权该 child；不会连带授权依赖链中的下一个 child。

## 1. 固定基线与已有证据

### 1.1 仓库快照

| 角色 | 路径 | 固定提交 | 用途 |
|---|---|---|---|
| Clean target | `/home/user/ws/lemon/rlgame-unilab/UniLab` | `60c2ce7ce13a6d5078b342d598d590e1023a5f76` | 新实现落点 |
| Mature donor | `/home/user/ws/lemon/UniLab` | `3d479690cc26b1bbe39e7c7b3b71ebc7821e1650` | MuJoCo env、600 tools、assets 和测试 |
| Source oracle | `/home/user/ws/lemon/simtoolreal` | `2a9917533bfea70419ed2667a511d7238e5b3abc` | IsaacSim task config 与定制 RL-Games SAPG |

Source RL-Games 固定信息：

- Source parent-tree OID：`7a6a0bb090998d00565aaefa6ab9f2b3d356ace2`（完整 tree 为 72 `.py` + 122 `.yaml`；本计划只选择其中 72 个 Python blobs）
- `rl_games/pyproject.toml` blob：`185e2b8f8b4b7437344026216e241562c49b698b`
- `rl_games/LICENSE` blob：`313ca229e6ca879466f94bff49362fb65667e22f`
- Python runtime：72 个 `.py` 文件，MIT license
- 上游 RL-Games `v1.6.1`：`f5bd8f2a0022220a1109200a3da47d2e96cb9aa1`
- 当前 owner config：`isaacsimenvs/cfg/train/SimToolRealSAPG.yaml`

普通 `rl-games>=1.6` 不是 oracle。Source 要求 editable 安装仓库内 fork；官方包会解析到另一套代码。

### 1.2 已证明的问题

1. Donor 的 RSL-RL SAPG audit 已证明 actor 激活、observation RMS、完整 augmented universe、mini-batch partition、KL/reference、AMP/overflow、RNG、resume 和 play 存在实质差异。
2. Source 的 SAPG 修改不只在四个文件中。`ExperienceBuffer`、`PPODataset`、model wrapper、central value、player、checkpoint 和 runner 都参与算法语义。
3. RL-Games `torch_runner` eager import closure 是 51/72 个 Python 模块；裁剪后的长期白名单比保留完整 Python package 更脆弱。
4. Clean target 的 `mujoco-uni-runtime==0.3.1` 在真实 12-distribution model oracle 上失败：`ValueError: models are not compatible: model[0] and model[7]`。本地 0.4 mixed-layout build 对同一 oracle 通过。
5. Donor 600-tool catalog 有 1/2/3 tool-geom 三类 topology；mixed-layout 不是可跳过的边角能力。
6. Root Ruff 未排除新 vendor；对 Source runtime 执行 format check 时 72 个 Python 中 63 个会被改写。V1 必须同时落精确 vendor exclusion，否则 `make test-all` 会破坏 pristine hashes。

### 1.3 为什么值得做

- 直接服务 UniLab 的 contract-driven RL 目标：同一 task/env contract 可以由不同算法 runtime 消费，同时不用在 UniLab 内重写第三方算法。
- 不做的代价是继续维护一份已经证明不等价的 RSL-RL SAPG fork，并为每个 Source quirk 重复实现和对拍。
- 最小方案是“固定 Source runtime + 一个 adapter”；不是重写通用 runner，也不是引入第二套 collector/learner 协议。
- 新增的长期责任是真实的：第三方 fork provenance/安全升级、Python/Gym/Torch 兼容、RL-Games adapter、`.pth` checkpoint/player、专属 parity suite。

## 2. “只存在后端差异”的准确含义

### 2.1 必须与 Source 相同

以下内容由 vendored Source runtime 与原生 owner config 直接拥有，不允许在 UniLab 重新实现：

- block 编号、leader/follower 定义、coefficient IDs `[50, 40, 30, 20, 10, 0]`
- entropy/exploration coefficient、learnable embedding、conditional per-block sigma
- actor/shared value/central critic 网络、RNN/LN、normalizer
- rollout storage、time/env layout、trajectory shuffle、RNN done reset
- follower selection 的 NumPy RNG、augmentation universe、counterfactual TD target
- GAE、advantage、PPO ratio/clip/value/bounds/entropy denominator
- central-before-actor 顺序、dataset/minibatch 尾批、mini-epochs、KL scheduler
- optimizer parameter sets、gradient clipping、AMP/autocast/GradScaler/overflow
- checkpoint schema、RNN/rollout state、player block 与 stochastic/deterministic 语义
- backend-neutral task semantics：action scale/delay 顺序、140/162 observation feature 顺序与缩放、reward term 公式/raw scale、termination 与 DR 数学；reset 保持 `source_random` 与 Source ranges，只允许 manifested MuJoCo table-reference mapping

Source 当前行为即使看起来像 bug，也先作为 parity baseline 原样保留：

- timeout reward 使用 action 前的 `res_dict["values"]`，不对 final observation 重新估值；
- follower return 使用 one-step reward + bootstrap，不重算 GAE；
- checkpoint 不保存 Python/NumPy/Torch RNG state；
- current IsaacSim wrapper 的 `get_env_state()` 返回 `None`；
- player 保留 Source 的六 ID 与 owner YAML 中 `deterministic: false`。

任何修正都必须在 source-exact baseline 达成后另立 issue，且不能继续标为 Source parity。

### 2.2 允许存在的差异

| 边界 | 允许差异 | 约束 |
|---|---|---|
| Simulator/backend | IsaacSim/PhysX → UniLab/MuJoCo/MuJoCoUni | 只能留在 env/backend/adapter；不得进入 SAPG runtime |
| Env carrier | Isaac Lab dict/tensor → `NpEnvState.obs` dict/NumPy | adapter 显式映射 `obs`/`critic`，不探测 backend 私有能力 |
| Tool pool | Source 12×100 → donor 12×50 = 600 | 已批准的资源/后端差异；distribution、seed、固定 assignment 保持 |
| Reset physics | SAPG owner 保持 `source_random`、full-SO(3)、x/y=`0.1`、z noise=`0.02`、arm=`0.1`；固定 MuJoCo table 下 `object_spawn_z_reference_range=0.0`，并使用 `multiccd=disable` | Source table reference jitter 为 `0.01`；这两个字段逐项 manifest，不允许继承通用 PPO owner 的 horizontal/.575/.05/.025 profile |
| Parallel scale | Source `24576/4096`；资源 profile `12288/2048` | 都保持 6 blocks；缩放 profile 不用于训练轨迹等价声明 |
| Play scale | Source `train.py --test` 继承实际 task env 数；UniLab canonical profile 使用 6 envs | 这是显式 resource/play 差异；不修改 Source player 对任意 N 的原始路由 |
| Packaging | Python metadata、Gymnasium/NumPy/Torch 兼容 | 每个 patch 记录 provenance，并通过 frozen parity |
| Tracking paths | Hydra run dir、UniLab metadata sidecar | 不改变 RL-Games callback/update/checkpoint payload |

“backend 差异”只允许改变物理引擎产生的状态、接触、积分误差和吞吐；不能借此改变给定同一 backend-neutral state tensor 时的 action、observation、reward、termination 或 DR 数学。600-tool pool 与 12k resource profile 是上表单独批准的资源差异，不能混称为算法或 backend parity。

数值对拍统一使用 `12 env / block size 2 / 6 blocks`，隔离算法语义。Source-scale owner 保留 `24576/4096/minibatch_size=98304`；资源 owner 只允许在独立 YAML 中覆盖规模字段，不能由训练脚本翻译。

### 2.3 明确禁止

- 不把 RSL-RL SAPG 文件迁入 clean target。
- 不让 UniLab async runner、RSL runner 或新 collector 包裹 RL-Games rollout。
- 不在 `scripts/` 中维护 Source → Target 超参数翻译表。
- 不因 UniLab 有 final observation 就修改 Source timeout bootstrap。
- 不把 Source stochastic player 改成 donor 的 leader deterministic play。
- 不把官方 pip RL-Games 当作 Source fork。
- 不直接复制 donor 的整个 backend 文件、Motrix side tables 或旧 `mujoco/xml.py`。
- 不用不同 env/batch size 的训练曲线证明算法等价。

## 3. 最小运行边界

```text
Hydra SAPG owner YAML
  ├─ training/env/reward/MuJoCo fields ──> UniLab registry.make() ──> NpEnv
  └─ native params.{algo,model,network,config}
                                      |
                                      v
                            Source RL-Games Runner
                                      |
                         A2CAgent owns rollout/update
                                      |
                                      v
                         RlGamesNpEnvAdapter (sync)
                                      |
                      obs/action/done/info conversion only
                                      |
                                      v
                           SimToolRealEnv -> SimBackend
                                      |
                                      v
                              MuJoCo/MuJoCoUni
```

| 责任 | Owner |
|---|---|
| rollout/storage/dataset/minibatch | Source RL-Games |
| env step/reset 调用时机 | Source RL-Games |
| actor/central update、AMP、scheduler | Source RL-Games |
| task 创建、reward/env 配置 | Hydra owner + UniLab registry |
| obs/action/device/space 转换 | `RlGamesNpEnvAdapter` |
| physics、tool models、wrench/autoreset | UniLab MuJoCo backend |
| `.pth` payload | Source RL-Games |
| run dir、metadata、W&B lifecycle bridge | UniLab observer/tracker；只能有一个 owner |
| play action/RNN state | Source RL-Games player |

Adapter 必须提供 Source 实际依赖的 ABI：

```text
reset() -> {"obs": Tensor[N,140], "states": Tensor[N,162]}
step(actions) -> obs_dict, reward, done, info
done = terminated | truncated
info["time_outs"] = truncated
get_env_info(), get_number_of_agents(), set_train_info()
get_env_state() -> None
set_env_state(None) -> no-op
vec_env.env.device -> RL device
```

第一次 reset 使用 `NpEnv.init_state()` 已完成的初始化结果，不能再做一次全量随机 reset。Action 在 adapter 中只做一次 Torch→NumPy 转换；返回 tensor 在 RL device 上构造。Observation/action spaces 使用 Gymnasium finite `Box`，bounds 分别来自 owner 的 `clip_observations=10`、`clip_actions=1`。

## 4. 配置 owner

新增独立算法配置组，不复用 `conf/ppo`：

```text
conf/rlgames_sapg/config.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco_12k.yaml
```

公共 CLI 使用 `--algo sapg`，内部 runtime 目录使用 `rlgames_sapg` 说明 provenance：

```bash
uv run train --algo sapg --task simtoolreal --sim mujoco
uv run train --algo sapg --task simtoolreal --sim mujoco --profile 12k
```

规则：

1. `cfg.rl_games.params` 直接保持 Source `params` schema，入口只做 `OmegaConf.to_container(resolve=True)`。
2. 允许入口写入的 runtime handle 只有 `train_dir`、`device/device_name`、`vec_env` 和已经由 adapter 生成的 `env_info`；它们不是算法超参数。
3. Source-scale owner 使用 seed 42、24576/4096、horizon/sequence 16、minibatch 98304、2 mini-epochs、mixed precision、adaptive KL 0.016。
4. `mujoco_12k.yaml` 只覆盖 `num_actors=12288`、env count 和 block size 2048；actor/central 的 `minibatch_size=98304` 保持 Source 值。229376 个 augmented rows 因此严格按 Source `PPODataset` 形成 `[98304, 131072]` 两个 batch，而不是重新翻译成四个等分 batch。Source-scale 对称计算为 `24576×16 + 4096×16 = 458752` rows，原生尾批结构是 `[98304,98304,98304,163840]`。
5. Env reward 使用 Source raw scales `200/20/300/50/1000/0.03/0.003`，仅由 RL-Games `reward_shaper.scale_value=0.01` 缩放一次。不能复制 donor RSL owner 中已经乘过 0.01 的值。
6. 下载的历史 `pretrained_policy/model.pth` 内嵌 coefficient scale 是 `0.005`，而 Source HEAD owner 是 `0.002`；当前训练配置以 HEAD YAML 为准。
7. Canonical play owner 显式使用 `training.play_env_num=6`。Source `BasePlayer` 按实际 `env.num_envs` 生成 linspace embedding，而 `PpoPlayerContinuous` 的 conditional parameter IDs 固定为 `[50,40,30,20,10,0]`；6 envs 令两者逐项对应。不得修改 player 来“支持”任意 N；额外测试必须证明 N≠6 时仍保留 Source 的 equality/`argmax` fallback 行为。
8. SAPG owner 必须显式设置 `env.domain_randomization.enable_object_wrench=true`。Source 每步无条件进入 wrench DR；donor 旧 owner 的 `false` 是稳定性实验，不属于本路线允许的 backend 差异。B2 与 E8b 必须在 R2 前完成，并由 config test 和 fixed-RNG wrench trace 证明 force/torque path 实际激活。
9. `cpu_ids` 不是 SimToolReal task 或 SAPG 算法 contract。开发 owner 必须显式设置 `env.cpu_ids=null`；M0-dev 下由操作系统调度 MuJoCo worker，只允许吞吐/线程迁移统计变化，不允许将 CPU affinity 缺失解释为算法差异。

## 5. 最终文件边界

### 5.1 Third-party runtime

```text
third_party/simtoolreal_rl_games/
├── LICENSE
├── README.md
├── UPSTREAM.md
├── PATCHES.md
├── pyproject.toml
├── source_manifest.json
└── rl_games/**/*.py
```

distribution 使用唯一名称 `unilab-simtoolreal-rl-games`，import namespace 保持 `rl_games`。最终由 root `sapg` extra 精确引用本地 distribution；不包含 122 个示例 YAML、notebooks、Source tests 或根目录示例 runner。

### 5.2 UniLab RL-Games owner

```text
src/unilab/algos/torch/rlgames_sapg/
├── __init__.py
├── dependency.py
├── env_adapter.py
├── observer.py
├── checkpoint.py
├── runtime.py
└── player.py

scripts/train_rlgames_sapg.py
```

`scripts/train_rlgames_sapg.py` 只 compose、建 env/adapter/tracker、注入 runtime handles 并调用原生 `Runner`。

公共入口与安装面只做路由/依赖声明，不承载算法翻译：

```text
pyproject.toml                         # `sapg` optional extra
uv.lock                                # pinned fork/dependencies
src/unilab/cli.py                      # `--algo sapg` train/eval route
src/unilab/tools/completion.py         # discover the new owner group
conf/rlgames_sapg/config.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco_12k.yaml
tests/test_cli.py
tests/test_completion.py
tests/scripts/test_train_rlgames_sapg.py
```

`eval --algo sapg` 仍路由到同一个 entrypoint，再由 `training.play_only=true` 选择 Source `Runner.run_play()`；不另写一套推理循环。`algo.load_run` 只用于 UniLab 的可信 run 解析，解析结果作为 runtime checkpoint path 注入，不改写 `params` 内算法字段。

### 5.3 Mature task reuse

近原样移植 donor 最终态：

```text
src/unilab/envs/manipulation/simtoolreal/
├── __init__.py
├── action_pipeline.py
├── config.py
├── constants.py
├── delay_buffer.py
├── dr_provider.py
├── dr_wrench.py
├── env.py
├── episode_lifecycle.py
├── goal_sampling.py
├── keypoints.py
├── observations.py
├── rewards.py
├── tool_assets.py
└── tool_catalog.py
```

机械 assets 例外：

```text
src/unilab/assets/robots/kuka_sharpa/kuka_sharpa.xml
src/unilab/assets/robots/kuka_sharpa/scene.xml
src/unilab/assets/robots/kuka_sharpa/LICENSE.simtoolreal.txt
src/unilab/assets/robots/kuka_sharpa/LICENSE.kuka_iiwa.txt
src/unilab/assets/robots/kuka_sharpa/ASSET_PROVENANCE.md
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/*.STL
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/{collision,visual}/*.stl
```

Donor 的 `build_simtoolreal_assets.py` 为 936-line 冷路径再生成工具，不是 shipped training runtime 的依赖，首轮不迁移。若后续要求从 URDF/USD 重建这两个 XML，必须另立 generator child，而不能塞入 asset 机械复制 PR。

Backend 只能摘取最小 hunk：

```text
src/unilab/dr/types.py
src/unilab/base/backend/base.py
src/unilab/base/backend/mujoco/backend.py
```

所需能力仅为 `ModelVariantSpec.source_model_file`、完整 source-model direct compile、public body wrench、public step-autoreset mask。禁止覆盖 target 已有 CPU affinity、terrain、render、MjSpec 或 newer XML 行为。

### 5.4 明确不迁移

```text
conf/ppo/task/simtoolreal*
scripts/train_rsl_rl.py
src/unilab/algos/torch/sapg/**
src/unilab/algos/torch/rsl_rl_ppo.py
src/unilab/algos/torch/rsl_rl_runtime.py
src/unilab/training/rsl_rl.py
src/unilab/training/sim2sim.py
src/unilab/structured_configs.py
src/unilab/base/backend/motrix/backend.py
```

首轮也不迁移 donor 的 collision tuning XML/viewer、DexToolBench、Motrix、RSL-specific normalization tests、benchmark research docs 或 support matrix 声明。

## 6. 分阶段外部前置：MuJoCoUni

`cpu_ids` 是通用 MuJoCo worker 性能/调度接口，不是 SimToolReal task、SAPG 数学或 Source parity 的依赖。当前 SimToolReal owner 未设置它，`EnvCfg.cpu_ids` 默认也是 `None`；UniLab backend 只在非 `None` 时才向 `BatchEnvPool` 传该参数。因此本路线先用固定 0.4 runtime 跑通任务，再恢复 0.3.1 affinity compatibility。

### 6.1 M0-dev：解除开发与 smoke 阻塞

M0-dev 固定使用 `mujoco-uni-runtime==0.4.0.dev0` 的 Git 提交 `7205e070e983df90d520f0f8593853013e976746`（下文简称 `7205e07`），它必须提供：

- dominated mixed-data-layout allocation；
- per-env `was_autoreset`/warning surface。

版本字符串本身不能证明身份；manifest 与 lock 必须记录上述完整 Git SHA。若依赖解析到由该提交构建的 sdist/wheel，还必须同时记录具体文件名、artifact SHA256，并证明 artifact provenance 指回同一提交。B1 是第一个真实 600-model consumer，由 B1 把 root MuJoCo extra 从 registry `0.3.1` 切到该固定 dev source/artifact；不得提交未固定的 `0.4.0.dev0`，也不得把 dirty sibling path `../mujoco_uni` 当作可复现依赖。改变到 `7205e07` 的任意 descendant 必须先更新 manifest，并重跑 mixed-layout/autoreset focused tests。

M0-dev 下 SAPG owner 必须显式 `env.cpu_ids=null`。MuJoCo 仍按 `nthread` 建 worker pool，只是不固定 CPU；这可能改变吞吐、线程迁移和性能方差，但不改变给定 frozen tensor 时的 SAPG、task math 或 env/backend contract。`geom_size/geom_pos/mocap_pos` side tables 不是本任务的 support claim。

M0-dev 允许 B1/B3、E9、真实 train/play smoke 与 12288/2048 profile 前进；它不允许宣称通用 MuJoCo CPU affinity support。

### 6.2 M0-release：最终发布 gate

M0-release 在任务 train/play 和 parity 跑通后处理：以固定 0.4 代码线为基础恢复 0.3.1 的 `cpu_ids`/`worker_cpu_ids` contract，形成可从 clean checkout 安装的正式 MuJoCoUni sdist/wheel。最终 artifact 必须同时具备 mixed-layout、autoreset 和 CPU affinity，记录正式版本与 SHA256，并通过三项组合回归。

M0-release 不阻塞算法 oracle、env 接线或 S1 smoke，只阻塞 D1 的最终依赖晋升和 support claim。D1 将 root lock 从 M0-dev SHA 切换到正式 artifact并重跑完整 gate。若 MuJoCoUni 需要生产修改，必须在 MuJoCoUni owner 仓库另写/批准 roadmap；本 umbrella 不隐含授权该外部开发。

## 7. Child issue 路线图

### 7.1 最近三个可执行 issue

| 顺序 | 单一结果 | 规模 | 依赖 |
|---|---|---:|---|
| V1 | 固定 Source RL-Games pristine snapshot、provenance 与 formatter isolation | 72 个机械复制 Python + 6 metadata + root `pyproject.toml` + 2 audit/test；≤800 net 手写 LOC，1 PR | 无 |
| V2a | Python 3.10–3.13/Gymnasium compatibility + pristine/patched hash audit | ≤15 total files、≤700 net 手写 LOC、1 PR；14 named paths，最多 1 个 contingency path | V1 |
| V2b | 锁定 network/config golden | 6 files、≤800 net 手写 LOC、1 PR；fixture bytes 单列且≤8 MiB | V2a |

这三个 issue 结束时还没有 production training path，也不会修改 root optional dependencies。

### 7.2 后续方向与启动条件

| 组 | Child issues | 单一结果 | 启动条件 |
|---|---|---|---|
| Runtime prerequisite | M0-dev | 固定 0.4 `7205e07`，提供 mixed-layout + autoreset，`cpu_ids=null` | 不计 UniLab PR；真实 B1/B3 与 S1 前 |
| Runtime prerequisite | M0-release | 正式 artifact 补齐 0.3.1 CPU affinity contract | 不计 UniLab PR；只阻塞 D1/final support |
| Algorithm oracle | O1a/O1b/O1c | rollout/return/shuffle/RNG；loss/update/AMP；checkpoint/resume/player parity | V2b 后顺序执行；每项≤5 files、≤800 net 手写 LOC、1 PR；fixture cap 依次为 8/8/2 MiB |
| Assets | A0a/A0b | meshes + license/provenance；两个生产 XML + compile validation | A0a 仅 meshes 为机械例外；不迁移 generator |
| Backend | B1/B3 | source-model variant；autoreset mask | M0-dev 是真实路径 gate；每个独立 PR |
| Backend | B2 | public body wrench + MuJoCo `xfrc_applied` | 不依赖 M0-dev/M0-release；独立 PR |
| Task oracle | T0/T1 | 捕获 backend-neutral Source task fixture；验证 donor action/obs/reward/termination/reset/DR parity | T0 可与 V1 并行但需要可运行 Source；T1 依赖 T0 + E10 |
| Env foundations | E1a/b、E2/E3、E4a/b、E5、E6a/b、E7a/b/c、E8a/b | config/constants、catalog/materializer、delay/action、goal/lifecycle、keypoints、reward、obs、reset/DR | A0b；按 donor tests 分片 |
| Env owner | E9/E10 | `SimToolRealEnv`；registry/Hydra task integration | B1–B3 + 全部 env foundations |
| Adapter | R1 | fake `NpEnv` 下完整 RL-Games ABI | V2b；可与 O1 并行，不等真实 env |
| Config | R2 | Source-native Hydra config + 12k resource profile | O1c + E10 |
| Runner | R3a/R3b/R3c | 原生 Runner path；tracker/W&B lifecycle；可信 `.pth` resolver/resume modes | R1/R2；每项独立 PR且新 training path 单独批准 |
| Player | R4 | Source-native player/play 与 RNN state | R3a + R3c；不含 export |
| Acceptance | S1 | MuJoCo train/play smoke、finite/loss/profile evidence | M0-dev + R3b + R4 + T1 |
| Docs/support | D1 | 正式 MuJoCoUni pin、用户命令、已知限制、support claim | M0-release + S1 + `make test-all` + maintainer approval |

完整依赖链如下；`child <- prerequisites` 表示右侧全部满足后才能启动左侧。每个 child ID 恰好对应一个 PR：

```text
V1: none
V2a <- V1
V2b <- V2a

O1a <- V2b
O1b <- O1a
O1c <- O1b

A0a: none
A0b <- A0a

B1 <- M0-dev
B2: none
B3 <- M0-dev

T0 <- runnable Source oracle
T1 <- T0 + E10

E1a <- A0b
E1b <- A0b + E1a
E2 <- A0b
E3 <- A0b + E1b
E4a <- A0b
E4b <- A0b + E1b + E4a + E8a
E5 <- A0b + E1b
E6a <- A0b
E6b <- A0b + E6a
E7a <- A0b + E1b + E3 + E5
E7b <- A0b + E7a
E7c <- A0b + E7b
E8b <- A0b + B2
E8a <- A0b + E8b + E4a + E7c
E9 <- B1 + B2 + B3 + all env foundations
E10 <- E9

R1 <- V2b
R2 <- O1c + E10
R3a <- R1 + R2 + maintainer execution-path approval
R3b <- R3a
R3c <- R1 + R2
R4 <- R3a + R3c
S1 <- M0-dev + R3b + R4 + T1
D1 <- M0-release + S1 + make test-all + maintainer support approval
```

Donor env 拆分预算：

| Issue | 生产 owner | 预计规模 |
|---|---|---:|
| E1a | `constants.py` + constant/layout tests | 280 production + ≤300 test LOC |
| E1b | `config.py`、staged `__init__.py` + config validation tests | 460 production + ≤300 test LOC |
| E2 | `tool_catalog.py`、`tool_assets.py` + tests | 4 files，约 690 LOC |
| E3 | `delay_buffer.py`、`action_pipeline.py` + `test_action_pipeline.py` | 3 files，761 donor LOC；依赖 E1b（test imports config/constants） |
| E4a | `goal_sampling.py` + `test_goal_sampling.py` | 2 files，273 donor LOC |
| E4b | `episode_lifecycle.py` + `test_episode_lifecycle.py` | 2 files，612 donor LOC；依赖 E1b + E4a + E8a，因为 `advance_goal_on_success()` 运行时读取 `dr_provider.DSTAR_SENTINEL` |
| E5 | `keypoints.py` + `tests/envs/test_simtoolreal_keypoints.py` | 2 files，645 donor LOC；依赖 E1b；只覆盖 primitive geometry |
| E6a | `rewards.py` primitive slice + matching tests | ≤305 production + ≤360 test LOC；实现 lifting/distance/keypoint/action/goal primitives |
| E6b | append reward aggregation + matching tests | ≤115 production + ≤260 test LOC；实现 `compute_rewards`、phase gating、term dict、in-place d-star |
| E7a | `observations.py` helper slice + staged primitive tests | ≤295 production + ≤180 test LOC；依赖 E1b + E3 + E5；实现 normalize/quaternion/noise/stack helpers |
| E7b | append actor/critic assembly + matching tests | ≤256 production + ≤500 test LOC；实现 `build_observations` 与 `_assemble_observations` |
| E7c | append reset observation path + pure-mock test slice | ≤64 production + 71 test LOC；同两文件≤135 added LOC；只搬 `test_observations.py::TestBuildResetObservations`，不得提前搬真实-env reset suite |
| E8b | `dr_wrench.py` + tests | ≤3 files，≤800 LOC；依赖 A0b + B2 |
| E8a | `dr_provider.py` + provider-only reset tests | 2 files、≤725 net LOC、1 PR；依赖 A0b + E8b + E4a + E7c；不 compose donor `conf/ppo` owners |
| E9 | `env.py` + 两个 env-composition regression tests | 3 files、1,285 donor mechanical LOC、≤200 net handwritten adaptation LOC、1 PR；799-line env final blob + 107-line keypoint + 379-line real reset tests |
| E10 | package registration + task integration | ≤5 files，约 660 LOC |

E6/E7 的每个 child 都同时落 implementation slice 与最近风险测试；最终函数体必须与 donor 固定 blob 对应，允许的机械变化只有为分阶段追加而调整定义顺序。T1 task golden 在 E10 后覆盖最终组合，禁止把 later child 变成纯 test backfill。

Donor 的 `test_keypoint_geometry.py` 不是 E5 primitive test：它顶层导入并执行 `SimToolRealEnv`、reward 与 episode lifecycle，必须延后到 E9 以避免 E5↔E9 循环。Donor 的 379-line `test_reset_observations.py` 同样会创建真实 MuJoCo env、import `dr_provider` 并允许依赖缺失时 skip，不能放进 E7c；E7c 只搬 `test_observations.py` 中 71-line pure-mock `TestBuildResetObservations` slice。E9 在 M0-dev、B1–B3、assets 和全部 foundations 到位后机械搬入这两个 composition tests，focused gate 必须实际 collect 且 0 skip。禁止为了让早期 child collect 而添加临时 production stub。

E8a 只保留 `test_reset_distribution.py` 中使用 `SimToolRealCfg`/fake env 的 provider 数学测试；删除该 donor test 对 `conf/ppo/task=simtoolreal*` 的 compose helper 和三个 shipped-owner assertions。通用 PPO horizontal owner 不在本路线，donor 的 `enable_object_wrench=false` assertion 还与本计划的 SAPG Source contract 相反。SAPG owner 的 `source_random` ranges、z-reference mapping 与 `enable_object_wrench=true` 统一由 R2 Source-config golden 验收，不能在 E8a 临时创建旧 PPO YAML 来让 donor test 通过。

不能直接 cherry-pick donor 的 `8a4f5ccc`/`32a5cd38`/`4bc37203`：共同祖先 `8313b4cd…` 之后，target/donor 分别已有 147/51 个独立提交，backend、XML、CPU affinity 和 visualization 已分叉。Task-owned final blobs可以作为 port source；backend 必须逐 hunk 重放。

## 8. Detailed plan — V1: pristine Source runtime

**Files:**

- Create: `third_party/simtoolreal_rl_games/LICENSE`
- Create: `third_party/simtoolreal_rl_games/README.md`
- Create: `third_party/simtoolreal_rl_games/UPSTREAM.md`
- Create: `third_party/simtoolreal_rl_games/PATCHES.md`
- Create: `third_party/simtoolreal_rl_games/pyproject.toml`
- Create: `third_party/simtoolreal_rl_games/source_manifest.json`
- Create mechanically: `third_party/simtoolreal_rl_games/rl_games/**/*.py`
- Modify: `pyproject.toml` (`tool.ruff.extend-exclude` only)
- Create: `scripts/audit_simtoolreal_rlgames_vendor.py`
- Test: `tests/vendor/test_simtoolreal_rl_games_vendor.py`

- [ ] **Step 1: Write the manifest/hash test before adding runtime files**

```python
import json
from pathlib import Path


SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
SOURCE_PARENT_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party/simtoolreal_rl_games"


def test_vendor_manifest_pins_source_parent_tree():
    vendor_manifest = json.loads((VENDOR_ROOT / "source_manifest.json").read_text())
    assert vendor_manifest["source_head"] == SOURCE_HEAD
    assert vendor_manifest["source_parent_tree"] == SOURCE_PARENT_TREE
    assert len(vendor_manifest["python_files"]) == 72


def test_pristine_vendor_has_no_compatibility_patches():
    patches_text = (VENDOR_ROOT / "PATCHES.md").read_text()
    assert "No compatibility patches are applied in V1." in patches_text


def test_root_ruff_excludes_pristine_vendor():
    root = VENDOR_ROOT.parents[1]
    config_text = (root / "pyproject.toml").read_text()
    ruff_section = config_text.split("[tool.ruff]", maxsplit=1)[1].split("\n[", maxsplit=1)[0]
    assert "third_party/simtoolreal_rl_games" in ruff_section
```

- [ ] **Step 2: Run the test and verify it fails because the vendor does not exist**

Run:

```bash
uv run pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q
```

Expected: FAIL while resolving `third_party/simtoolreal_rl_games/source_manifest.json`.

- [ ] **Step 3: Add the exact 72-file snapshot, nested MIT license and provenance**

`source_manifest.json` must contain relative path, Git blob SHA and SHA256 for every selected Python file. `UPSTREAM.md` records Source HEAD、parent tree（包含未迁移的 122 YAML）和 RL-Games/SAPG lineage。`PATCHES.md` contains exactly the V1 no-patch statement. Root `pyproject.toml` 的唯一变化是把 `third_party/simtoolreal_rl_games` 加入 Ruff `extend-exclude`；本 issue 不 import package、不增加 root dependency。

- [ ] **Step 4: Add a fail-closed audit command**

The audit must reject missing/extra `.py` files, hash drift, a changed Source identity, missing license, or non-empty compatibility allowlist.

Run:

```bash
uv run scripts/audit_simtoolreal_rlgames_vendor.py
uv run pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q
```

Expected: audit prints Source HEAD/parent tree and `72 selected Python blobs verified`; pytest passes.

- [ ] **Step 5: Prepare the V1 handoff without staging or committing**

```bash
git diff --check
git status --short
git diff --stat
uv run scripts/audit_simtoolreal_rlgames_vendor.py
uv run pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q
```

The implementation session stops here and returns the required handoff report; it must leave all V1 changes unstaged. The control session independently reviews and reruns the gates, then stages only the declared V1 paths and commits `vendor: pin SimToolReal RL-Games runtime`. After that final commit it runs `make test-all`, reruns the audit and confirms a clean worktree before any PR. Stop if any selected Python blob is not byte-identical to Source, if the vendored package has an extra/missing `.py`, if Ruff touches vendor bytes, or if the required license/provenance cannot be stated precisely. Do not claim the selected package has the full parent-tree identity because the 122 Source YAML files are intentionally absent.

## 9. Detailed plan — V2a: compatibility plus dual-hash audit

**Files:**

- Modify: `third_party/simtoolreal_rl_games/pyproject.toml`
- Modify: `third_party/simtoolreal_rl_games/PATCHES.md`
- Modify: `third_party/simtoolreal_rl_games/source_manifest.json` to record pristine and patched hashes separately
- Modify: `scripts/audit_simtoolreal_rlgames_vendor.py`
- Modify: `tests/vendor/test_simtoolreal_rl_games_vendor.py`
- Modify: `third_party/simtoolreal_rl_games/rl_games/common/a2c_common.py`
- Modify: `third_party/simtoolreal_rl_games/rl_games/common/env_configurations.py`
- Modify: `third_party/simtoolreal_rl_games/rl_games/common/experience.py`
- Modify: `third_party/simtoolreal_rl_games/rl_games/common/player.py`
- Modify: `third_party/simtoolreal_rl_games/rl_games/common/vecenv.py`
- Modify: `third_party/simtoolreal_rl_games/rl_games/common/wrappers.py`
- Modify: `third_party/simtoolreal_rl_games/rl_games/algos_torch/players.py`
- Create: `tests/algos/rlgames_sapg/_runtime_requirement.py`
- Create: `tests/algos/rlgames_sapg/test_import.py`

- [ ] **Step 1: Extend the V1 tests to require a reviewed patch allowlist**

`source_manifest.json` gains an exact list of `{path, pristine_blob, pristine_sha256, patched_sha256, reason, covering_test}` records. The audit and vendor test must verify both pristine provenance and current patched bytes, reject an unlisted changed file, and reject an allowlist entry whose current file is still pristine. This deliberately replaces V1's “no compatibility patches” assertion; V2a must update the audit and its test in the same PR.

Because root install does not include the future `sapg` extra yet, every `tests/algos/rlgames_sapg` module calls `_runtime_requirement.require_simtoolreal_rl_games()` before importing a harness. When `UNILAB_REQUIRE_SAPG` is unset, that helper skips if `rl_games` is absent or resolves to any distribution/path other than this vendor. When `UNILAB_REQUIRE_SAPG=1`, absence of the distribution, a distribution name other than `unilab-simtoolreal-rl-games`, a loaded `rl_games.__file__` outside `third_party/simtoolreal_rl_games/rl_games`, or a current-module hash inconsistent with the patched manifest must raise rather than skip. This environment switch makes the focused gate fail closed instead of relying on pytest's zero exit status for an all-skipped module.

`test_import.py` must include both provenance and behavior checks: assert native `Runner`, continuous agent, central value, model builder and player resolve from the vendored path, then construct `ExperienceBuffer` on CPU with exact Gymnasium `Box` observation/action spaces (`num_actors=2`, `horizon_length=4`) and assert `is_continuous`, action count and `[4, 2, ...]` buffer shapes. Import success alone does not validate the patched `type(space) is gym.spaces.Box` path.

- [ ] **Step 2: Run import tests and confirm the pristine snapshot fails at legacy Gym/Python boundaries**

Run:

```bash
UNILAB_REQUIRE_SAPG=1 uv run --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_import.py -q
```

Expected before compatibility patch: FAIL at the first unsupported Python/Gym/NumPy/Torch boundary, not a silent fixture mismatch.

- [ ] **Step 3: Apply the minimum compatibility patch set**

Allowed changes are limited to distribution metadata, `gym`/Gymnasium imports and exact space-type handling, removed NumPy aliases, and explicit trusted checkpoint loading. Every changed Source file gets one entry in `PATCHES.md` and the manifest. The 14 named paths leave room for at most one additional file under the 15-file PR cap; if compatibility needs more, stop and split another child. No SAPG, GAE, timeout, normalizer, scheduler, AMP or player decision may change.

该 contingency path 在编辑前必须先写入 V2a standalone issue 的 Files 列表；未更新 scope summary 时不得触碰匿名第 15 个文件。

- [ ] **Step 4: Run the patched-hash audit and imports on Python 3.10–3.13**

```bash
uv run scripts/audit_simtoolreal_rlgames_vendor.py
UNILAB_REQUIRE_SAPG=1 uv run --python 3.10 --with-editable ./third_party/simtoolreal_rl_games pytest tests/algos/rlgames_sapg/test_import.py -q
UNILAB_REQUIRE_SAPG=1 uv run --python 3.11 --with-editable ./third_party/simtoolreal_rl_games pytest tests/algos/rlgames_sapg/test_import.py -q
UNILAB_REQUIRE_SAPG=1 uv run --python 3.12 --with-editable ./third_party/simtoolreal_rl_games pytest tests/algos/rlgames_sapg/test_import.py -q
UNILAB_REQUIRE_SAPG=1 uv run --python 3.13 --with-editable ./third_party/simtoolreal_rl_games pytest tests/algos/rlgames_sapg/test_import.py -q
uv run ruff check scripts/audit_simtoolreal_rlgames_vendor.py tests/vendor/test_simtoolreal_rl_games_vendor.py tests/algos/rlgames_sapg
git diff --check
```

Expected: native `Runner`, continuous agent, central value, model builder and player import on all four interpreters. If an interpreter cannot be provisioned, record it as unverified and do not claim 3.10–3.13 support.

- [ ] **Step 5: Prepare the independent V2a handoff**

```bash
git diff --check
git status --short
git diff --stat
```

The implementation session leaves changes unstaged and returns its report. After independent review and verification, only the control session may stage the declared V2a paths and commit `fix: make SimToolReal RL-Games runtime compatible`. Stop if compatibility requires changing a tensor formula, RNG call, update order, checkpoint payload or player decision. That discovery returns to the maintainer instead of being hidden as a compatibility fix.

## 10. Detailed plan — V2b: network/config golden

**Files:**

- Create: `scripts/generate_simtoolreal_sapg_network_fixture.py`
- Create: `tests/fixtures/simtoolreal_sapg/source_network_fp32.npz`
- Create: `tests/fixtures/simtoolreal_sapg/source_network_manifest.json`
- Create: `tests/algos/rlgames_sapg/test_source_config.py`
- Create: `tests/algos/rlgames_sapg/source_network_harness.py`
- Create: `tests/algos/rlgames_sapg/test_network_golden.py`

- [ ] **Step 1: Write config/network tests before adding the harness and fixture**

```python
import numpy as np

from ._runtime_requirement import require_simtoolreal_rl_games

require_simtoolreal_rl_games()

from .source_network_harness import (
    load_network_fixture,
    load_source_owner_contract,
    replay_network_fixture,
)


def test_source_owner_shape_contract():
    source_owner = load_source_owner_contract()
    assert source_owner["num_envs"] == 24576
    assert source_owner["block_size"] == 4096
    assert source_owner["actor_obs"] == 140
    assert source_owner["critic_obs"] == 162
    assert source_owner["actions"] == 29
    assert source_owner["embedding_shape"] == [6, 32]
    assert source_owner["sigma_shape"] == [6, 29]


def test_network_matches_source_fixture():
    fixture = load_network_fixture()
    actual = replay_network_fixture(fixture)
    np.testing.assert_allclose(actual.mu, fixture.mu, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(actual.value, fixture.value, atol=1e-6, rtol=1e-5)
    assert actual.actor_gradient_signature.matches(fixture.actor_gradient_signature)
    assert actual.central_gradient_signature.matches(fixture.central_gradient_signature)
```

Run:

```bash
UNILAB_REQUIRE_SAPG=1 uv run --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_source_config.py \
         tests/algos/rlgames_sapg/test_network_golden.py -q
```

Expected: FAIL because `source_network_harness`/fixture does not exist.

- [ ] **Step 2: Generate a compact fixture from the immutable Source checkout**

Source capture and Target replay must run in separate isolated processes because both distributions own the `rl_games` namespace. The Source process must verify Source HEAD, parent-tree provenance and every loaded Python blob, and assert that every imported `rl_games.*.__file__` is below `/home/user/ws/lemon/simtoolreal/rl_games/rl_games`; the Target process must independently assert the vendored path and patched hashes. A process that has imported both namespaces is invalid oracle evidence.

The Source generator records native seed-based initialization hashes/RNG post-state, then fills every named parameter deterministically and records fixed obs/block/RNN inputs; embedding/LSTM/LN/MLP intermediates; `mu`, `sigma`, value, log-prob and entropy. For actor/central gradients it stores every tensor's shape, exact canonical FP32 SHA256, norm/sum/max and 64 name-seeded coordinates, not full weight/gradient arrays.

Canonical execution 使用 Python 3.11、Torch 2.7 和 CUDA，并在 manifest 固定 device、GPU 型号、compute capability、Torch exact version/CUDA build、CUDA runtime、cuDNN、driver，以及 TF32、matmul precision、deterministic-algorithm/cuDNN flags。确定性填充的 weights/inputs 与 Source blobs 在所有平台都要求 exact hash；计算得到的 intermediate/output/gradient exact hash 只在完全匹配该 canonical platform 时要求，其他 Python/GPU 平台检查 shapes 与 numerical signatures，使用 `atol=1e-6, rtol=1e-5`。Manifest 不得把 CPU replay 标成 canonical Source execution。

The committed `.npz` + manifest must total at most 8 MiB. If complete evidence cannot fit, stop and request explicit LFS/external-artifact approval; do not commit a >100 MB fixture or silently drop a tensor.

Run:

```bash
UNILAB_SAPG_ORACLE_MODE=source uv run --isolated \
  --with-editable /home/user/ws/lemon/simtoolreal/rl_games \
  scripts/generate_simtoolreal_sapg_network_fixture.py \
  --source /home/user/ws/lemon/simtoolreal \
  --output tests/fixtures/simtoolreal_sapg
```

The command above captures only Source outputs and exits. Target replay happens later under `UNILAB_REQUIRE_SAPG=1 --with-editable ./third_party/simtoolreal_rl_games`; the generator may not import the patched vendor to “compare in process.”

- [ ] **Step 3: Implement replay only through native Source builders**

`source_network_harness.py` must instantiate native Source `ModelBuilder` and central-value builders, load the deterministic parameter state, execute public forward/backward paths, and return the same trace schema. Forward hooks may capture intermediates; the harness may not reproduce a network/loss formula.

- [ ] **Step 4: Run network parity on Python 3.10–3.13 and enforce the byte budget**

```bash
UNILAB_REQUIRE_SAPG=1 uv run --python 3.10 --with-editable ./third_party/simtoolreal_rl_games pytest tests/algos/rlgames_sapg/test_source_config.py tests/algos/rlgames_sapg/test_network_golden.py -q
UNILAB_REQUIRE_SAPG=1 uv run --python 3.11 --with-editable ./third_party/simtoolreal_rl_games pytest tests/algos/rlgames_sapg/test_source_config.py tests/algos/rlgames_sapg/test_network_golden.py -q
UNILAB_REQUIRE_SAPG=1 uv run --python 3.12 --with-editable ./third_party/simtoolreal_rl_games pytest tests/algos/rlgames_sapg/test_source_config.py tests/algos/rlgames_sapg/test_network_golden.py -q
UNILAB_REQUIRE_SAPG=1 uv run --python 3.13 --with-editable ./third_party/simtoolreal_rl_games pytest tests/algos/rlgames_sapg/test_source_config.py tests/algos/rlgames_sapg/test_network_golden.py -q
uv run ruff check scripts/generate_simtoolreal_sapg_network_fixture.py tests/algos/rlgames_sapg
git diff --check
```

Expected: canonical hashes exact, mapped numeric tensors/signatures within tolerance, actual fixture byte count ≤8 MiB. An unavailable interpreter is recorded as unverified and removed from the support claim.

- [ ] **Step 5: Prepare the independent V2b handoff**

Fixture regeneration requires the explicit command, exact Source checkout and reviewed manifest diff. Normal test execution never regenerates fixtures.

```bash
git diff --check
git status --short
git diff --stat
```

The implementation session leaves changes unstaged and returns its report. After independent review and verification, only the control session may stage the declared V2b paths and commit `test: lock RL-Games SAPG network fidelity`. Stop on any unexplained network/config mismatch or if replay requires changing runtime code. Do not loosen tolerances to make the test pass.

## 11. 后续 child issue 的验收边界

### 11.1 Remaining algorithm oracles

- O1a 单独创建 rollout/return/shuffle/RNG generator、harness、test、fixture 和 manifest，≤5 total files、≤800 net handwritten LOC、1 PR。Canonical synthetic case 固定为 12 env、block size 2、6 blocks、horizon/sequence length 4：48 个 base rows 加 8 个 follower rows；actor 与 central-value 两处 test-only `minibatch_size` 都必须显式设为 12，以避免 central-value dataset 初始化沿用生产值并除零，同时验证 Source 原生尾批 `[12, 12, 12, 20]`。Fixture 必须记录 action 前 timeout value、delta/GAE/return/advantage、done/tail value、follower repeat index、候选集合、relabel next value/one-step TD target、env permutation、每个重排 buffer 的 canonical SHA256，以及 NumPy/Torch/CUDA RNG 前后状态。提交的 fixture + manifest 总计不得超过 8 MiB。
- O1b 单独创建 update generator、harness、test、fixture 和 manifest，≤5 total files、≤800 net handwritten LOC、1 PR；记录 central-before-actor 顺序、second-epoch KL reference、ratio、clipped/unclipped surrogate、value/bounds/entropy/KL/LR、pre/post-clip gradient norm、逐参数/optimizer delta、scaler state 和 overflow-skip。不得提交完整参数或完整 Adam delta；每个 tensor 保存 canonical SHA256、shape/dtype、norm/sum/max 和 64 个 name-seeded sentinel coordinates。FP32 使用 `atol=1e-6, rtol=1e-5`；AMP 只对拍 Source step/skip/scaler 行为，不与 FP32 逐元素比较；fixture + manifest 总计不得超过 8 MiB。AMP 必须在 V2b manifest 所定义的 canonical CUDA platform 上实际执行；无该平台时本 child 与最终 AMP fidelity 保持未验证，不能用 CPU/skip 作为通过。
- O1c 单独创建 checkpoint/resume/player generator、harness、test、fixture 和 manifest，≤5 total files、≤800 net handwritten LOC、1 PR；记录 Source payload keys、model/optimizer/normalizer/scaler/RNN/rollout fields、`env_state=None`、未保存 RNG、外部恢复 RNG 后的首个 action/value/update，以及 canonical 6-env 与 N≠6 player routing。Checkpoint 在 pytest 临时目录中运行时生成；仓库只提交 schema、逐 tensor hash、数值 signature/sentinel 和 fixture manifest，总计不得超过 2 MiB。不得把 Source 未保存的状态包装成 bit-exact resume 声明。
- O1a、O1b 与 O1c 各自拥有独立 fixture manifest、生成命令和 commit；普通测试不自动 regenerate，任何 rebaseline 都需要 exact Source checkout 与 reviewed manifest diff。任一 fixture 无法在上述预算内保存充分证据时立即停止，请求 LFS/外部 artifact 授权；不得提交超过 100 MB 的 fixture，也不得为满足预算静默删掉需验收的 tensor。
- O1a/O1b/O1c 若实际需要第 6 个文件或超过 net LOC 预算，必须先更新对应 standalone issue 的 scope；不得把额外 harness/helper 匿名塞进同一 PR。

### 11.2 Task mathematics oracle

- T0 从固定 Source checkout 生成 backend-neutral fixture：固定 joint/object/tool state、action、delay history、goal、episode counters 与显式 random draws，记录 action target、actor 140 维 obs、critic 162 维 state、逐 reward term/raw total、termination/reset mask 和 DR 参数。manifest 必须记录 Source HEAD、task config blob、字段顺序、单位和 dtype。
- Source fixture generator 可以读取 IsaacSim tensor，但 fixture 必须同时保存公式所需的全部 primitive inputs；不得只保存无法解释的最终输出。接触力等 simulator output 作为已命名输入固定，避免把 PhysX/MuJoCo 物理差异误判为公式差异。
- T1 用 donor 的 SAPG `source_random` owner 和同一 primitive inputs 重放 Source reset/action/obs/reward/termination/DR；FP32 使用 `atol=1e-6, rtol=1e-5`，离散 mask/index 必须 exact。Shipped owner 的 reset manifest 只允许固定 MuJoCo table reference jitter `0.0` 对 Source `0.01`，另允许 XML `multiccd=disable`；full-SO(3)、x/y `0.1`、z noise `0.02`、arm/finger `0.1`、velocity `0.5` 与 active object-wrench DR 必须保持。除 tool-pool cardinality、backend state acquisition 与这两个 backend mappings 外，不允许 reward double scaling、obs 重排、action delay 重排、wrench disable 或 DR range 漂移。
- 如果没有可运行的 Source oracle且现有 fixture/provenance 不足，T0/T1 保持未验证，S1 不得宣称“训练只存在后端差异”。

### 11.3 Backend、assets 与 mature env

- A0a 只复制 42 个 mesh，并创建 `LICENSE.simtoolreal.txt`、`LICENSE.kuka_iiwa.txt` 和 `ASSET_PROVENANCE.md`，逐目录映射 Source blob 与许可证；Sharpa mesh 权属若不能从 Source provenance 明确证明则停止。A0b 只复制 415 LOC 的两个生产 XML并增加 compile/mesh-reference validation；不夹带 936-line generator。
- B1：只增加 `source_model_file` variant 和 direct compile；real 12-distribution oracle 必须通过。B1 同时把 MuJoCo extra 固定到 M0-dev 的完整 source SHA；若使用 artifact，还要固定文件名、SHA256 和 source provenance。它是第一个真实 consumer，不得使用模糊版本或 dirty sibling checkout。
- B2：只增加 public body-wrench contract 和 MuJoCo `xfrc_applied` 实现；force/torque row isolation 必须通过，不依赖 M0-dev/M0-release。
- B3：只增加 public step-autoreset mask；多 substep 必须 OR-latch，并在 M0-dev 上通过真实 autoreset oracle；不得为了兼容 0.3.1 而伪造 mask。
- E1a/b、E2/E3、E4a/b、E5、E6a/b、E7a/b/c、E8a/b、E9/E10：每个 donor implementation/test slice 按预算独立；reward owner 使用 raw scales，不迁移 RSL value-normalization/bounds tests。
- 600-tool acceptance 必须验证 catalog 长度、固定 assignment、完整 source model parity、reset stability、冷路径 asset access 和无 backend-private feature leakage。

### 11.4 Adapter、Runner 与 config

- R1 adapter 使用 fake `NpEnv` 证明 actor/critic routing、single conversion、done/timeout、first reset、space bounds、alias/device 和 `env_state=None`。
- R2 config test 逐字段与 Source owner 比较；只允许 manifest 中列出的 backend/resource path 不同。它必须断言 SAPG owner 使用 `source_random` 与 Source reset ranges、仅 `object_spawn_z_reference_range=0.0`，不能从通用 PPO owner 继承 `horizontal_near_table`，并断言 `env.domain_randomization.enable_object_wrench=true`、`env.cpu_ids=null`。Canonical play config 必须断言 `training.play_env_num=6`，并将它记录为 play resource 差异而非 Source algorithm 字段。
- R3a 原生 Runner 必须通过 `Runner.load()` + `set_vec_env()` 注入 adapter并形成唯一 train/play execution path；禁止新 rollout loop。
- R3b 单独接入 run-dir、observer 与 ExperimentTracker；不得同时初始化第二个 W&B run，RL-Games observer callback timing保持 Source。
- R3c 单独实现 `.pth` resolver 与 resume/weights modes；resolver 独立于现有 RSL `.pt` resolver，可信本地 legacy checkpoint 显式 `weights_only=False`，非可信 checkpoint fail closed。
- Resume 验收只声称 Source checkpoint schema parity。由于 Source 不保存 RNG 且 current env state 为 `None`，不得声称 bit-exact full-runtime continuation。

### 11.5 Player 与真实环境 smoke

- R4 保留 Source player 的 coefficient ID、input RMS、RNN state、done reset、action clamp/rescale 和 `deterministic: false`。Canonical 6-env trace 必须得到 embedding/conditional IDs `[50,40,30,20,10,0]`；另用 N≠6 trace 锁定 Source 当前 equality/`argmax` fallback，禁止在 adapter 或 player wrapper 中静默修正。
- 旧 pretrained checkpoint config `.005` 只作 artifact evidence，不覆盖 HEAD owner `.002`。
- S1 在 M0-dev 上先跑小规模 MuJoCo finite smoke，再跑 12288/2048 profile；fixed-RNG smoke 必须观察到 active force/torque wrench rows。只比较 runtime trace、finite metrics、吞吐和资源，不比较 IsaacSim/MuJoCo reward curve 等价；吞吐报告必须注明 CPU affinity disabled，不能外推为最终 affinity 性能。
- 正式 support claim 需要 `make test-all`、真实 train/play 证据和 maintainer 单独批准；不自动新增常规 CI。通过全部 gate 后只能表述为：“SAPG algorithm Source-exact；task 差异仅限 manifest 中的 MuJoCo/table/tool/resource mappings。”不得笼统表述为 IsaacSim 与 UniLab 的全部训练只剩物理后端差异。

## 12. Parity acceptance matrix

| Gate | 输入 | 必须相同 | 允许差异 |
|---|---|---|---|
| Vendor | Source Git blobs | 72 selected Python blob hashes、license、source parent-tree provenance | recorded packaging metadata；122 YAML intentionally absent |
| Config | Source HEAD YAML | 所有 SAPG/network/update/player fields | backend paths；显式 resource profile |
| Network | fixed FP32 weights/obs/block/RNN | embedding、latent、mu/sigma/value/logprob/entropy/grad | 无 |
| Rollout | scripted 12/2 VecEnv | buffers、timeout旧语义、GAE、augmentation、TD | env carrier conversion |
| Update | frozen mini-batch | rows、loss、KL/LR、grad norm、Adam delta、AMP step/skip | documented dtype tolerance only |
| RNG/shuffle | fixed NumPy/Torch/CUDA states | repeat indices、permutation、post-state、hashes | 无 |
| Checkpoint | scripted runtime | Source payload/schema/model/optimizer/RMS/scaler/RNN buffers | filesystem/run metadata |
| Resume | Source-supported state | restored stored fields和外部恢复RNG后的首步 | env state `None`；不声称完整连续性 |
| Player | fixed checkpoint/obs/RNN | block/ID、stochastic action path、value、state advance | rendering/backend |
| Task math | fixed backend-neutral state/action/random draws | source_random action/obs/reward/termination/reset/DR formulas and ordering | named simulator-query inputs；600-tool cardinality；table-reference jitter 0.0；`multiccd=disable` |
| MuJoCo smoke | real SimToolReal env on M0-dev | finite rollout/update、shape/contract、可保存/加载 | physics/reward trajectory/throughput；`cpu_ids=null` |

## 13. 总体验证命令

所有 Python 命令必须使用 `uv run`。每个 child issue 先运行自己的近风险测试；最终 commit 完成后、创建或更新该 child PR 前，还必须按根 `AGENTS.md` 运行 `make test-all` 并确认工作树干净。由于 `sapg` 是 optional extra，plain suite 永久允许在未安装 extra 时 SKIP；声明 root extra 并不代表普通 `uv run pytest` 或 CI 会自动安装它，plain skip 也永远不算 parity 证据。从 V2a 起，每个 SAPG child 和最终 support gate 都必须设置 `UNILAB_REQUIRE_SAPG=1`，在 extra 落地前用 `--with-editable ./third_party/simtoolreal_rl_games`，落地后用 `--extra sapg`，并让 absence/wrong distribution/wrong path/hash drift 直接失败；required test set 不得包含 skip。下面是累计全集；support gate 只在合并序列改变或 maintainer 要求时再次运行，不能替代每个 PR 自己的 gate：

```bash
uv run scripts/audit_simtoolreal_rlgames_vendor.py
UNILAB_REQUIRE_SAPG=1 uv run --extra sapg pytest tests/algos/rlgames_sapg -q
uv run pytest tests/envs/manipulation/simtoolreal tests/simtoolreal -q
uv run pytest tests/base/backend/test_mujoco_model_source_variants.py \
              tests/base/backend/test_mujoco_autoreset_real_pool.py -q
uv run ruff check src/unilab/algos/torch/rlgames_sapg scripts/train_rlgames_sapg.py \
                  tests/algos/rlgames_sapg
uv run ruff format --check src/unilab/algos/torch/rlgames_sapg \
                         scripts/train_rlgames_sapg.py tests/algos/rlgames_sapg
git diff --check
make test-all
```

真实 smoke 的最终命令形态：

```bash
uv run train --algo sapg --task simtoolreal --sim mujoco \
  rl_games.params.config.max_epochs=1
uv run eval --algo sapg --task simtoolreal --sim mujoco \
  --load-run -1
```

这里的 `-1` 只解析前一条 smoke 命令在本地生成的最新可信 run；resolver 不接受任意不可信 pickle checkpoint。

## 14. Stop conditions

出现任一情况立即停止当前 child issue并回到 maintainer：

1. compatibility patch 需要改变 SAPG tensor公式、RNG、update、AMP、checkpoint 或 player 语义；
2. adapter 需要调用 backend 私有方法、读取 XML/asset 或新增 collector thread；
3. owner config 需要 Python 长期翻译算法字段；
4. 单个常规 issue 超过 15 files、800 net handwritten LOC，或需要第 2 个 PR；
5. donor port 会覆盖 target CPU affinity、terrain、render、XML/MjSpec 或其他已合入能力；
6. M0-dev 只能解析到 dirty/unversioned sibling MuJoCoUni checkout，lock 未记录完整 Git SHA，或所用 artifact 缺少文件名、SHA256 与对应 source provenance；
7. exact resume 需要新增公共 env snapshot contract；该 contract 必须另立 ADR/issue；
8. 需要新增 distributed、async、sim2sim、export 或常规 CI 才能让基础 SAPG path 工作；
9. frozen parity 出现无法解释的 Source/Target mismatch；
10. SAPG owner 继承 `horizontal_near_table` reset profile，或 `env.domain_randomization.enable_object_wrench` 不是显式 `true`；
11. O1b 无法在 manifest 固定的 canonical CUDA platform 上实际执行 Source mixed-precision、GradScaler 与 overflow case；此时保持未验证并阻塞 D1 support，而不是 skip-pass；
12. M0-dev 期间任何 shipped SAPG owner 把 `env.cpu_ids` 设为非 `null`，或代码假定 0.4 提供 affinity surface；
13. D1 无法把 dev SHA 替换为 clean-install M0-release artifact并通过 mixed-layout/autoreset/affinity 组合回归；
14. 实际规模超出本 umbrella 的长期维护预估，或 maintainer 无法复述当前 child 的单一结果。

## 15. Non-goals 与永久维护成本

本 umbrella 不交付：

- 独立 PPO 或官方 RL-Games 通用支持；
- Motrix、MJWarp、Drake 或跨 backend play；
- UniLab async SAPG、distributed/multi-GPU、ROCm、`torch.compile`；
- ONNX/export、sim2sim、DexToolBench；
- Source 算法 bug 修复；
- IsaacSim/MuJoCo 接触轨迹或训练曲线逐步等价；
- bit-exact full resume；
- 自动更新上游 fork或永久 benchmark/CI infrastructure。

合并后仓库永久承担：

- 72-file third-party runtime、nested MIT notice、provenance/patch manifest；
- 约 15 个 SimToolReal task modules、42 个 mesh、2 个生产 XML 和 asset provenance；
- 3 个 backend contract hunk、约 7 个 RL-Games integration modules、独立 Hydra/CLI path；
- network/rollout/update/RNG/checkpoint/player golden fixtures；
- 每次 Python/Gym/Torch/MuJoCoUni 升级时对 compatibility 与 parity gates 的复验。

当前明细恰好列出 37 个 UniLab child ID，每个 ID 对应且只对应 1 个 PR；需要第二个 PR 时必须先拆出新 child ID并重新计数。V1 的 72 Python blobs 与 A0a 的 42 meshes 是两个纯机械 file-count 例外；M0-dev/M0-release 是 MuJoCoUni 外部 gate，不计入 37。这个成本是选择“核心算法以 Source runtime 为 owner”带来的长期 fork 成本，不应在完成后被描述成一个零维护的薄插件。
