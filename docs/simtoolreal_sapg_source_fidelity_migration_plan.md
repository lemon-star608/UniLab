# SimToolReal SAPG Source-Fidelity 迁移总指导

> 本文件是这条迁移路线唯一的长期指导与状态文档。每完成一个 Code，都必须在同一
> commit 或紧随其后的 docs commit 中更新这里的状态、commit、验证证据和已知缺口。
> 阶段 prompt、handoff 和临时设计稿只服务当期执行，不作为长期事实来源。

最后更新：2026-08-21

当前分支：feat/simtoolreal-sapg-rlgames

本次整理前基线：6e1087f62cd17d196d67ccbd4ea880d0341cf6b5

## 1. 普通中文路线摘要

这条路线只做一件事：让 UniLab 的 MuJoCo SimToolReal task 使用固定 Source fork 的
原生 RL-Games SAPG 训练和播放，而不是继续维护 UniLab 中不完全等价的 RSL-RL SAPG
仿写。

当前算法迁移、算法回归基线、MuJoCo backend public contracts、训练资产、task foundations、
真实 MuJoCo env 和 native RL-Games production vertical slice 已经完成；正式依赖和 support
promotion 尚未开始：

~~~text
Code 1-5  固定并验证 Source SAPG runtime                 已完成
Code 6    补齐 MuJoCo backend public contracts            已完成
Code 7    迁移 assets 和 task primitives，完成 T0          已完成
Code 8    组合真实 MuJoCo env，完成 T1                     已完成
Code 9    接 native Runner、adapter、tracker、pth 和 CLI    已完成
Code 10   真实 smoke、正式依赖和 support promotion         未开始
~~~

后续不要求 IsaacSim/PhysX 与 MuJoCo 的真实轨迹、随机数、接触状态、reward curve 或训练
曲线 bit-exact。Code #1-#5 的 frozen Source fixtures 继续作为算法回归护栏；Code #6-#10
重点转为 UniLab backend、task、env 和 training pipeline 的 contract 接线。

Code #6-#10 不能混成一个无边界的大改动。它们分别拥有 backend public surface、
backend-neutral task math、真实 env composition、production execution path 和 release
support 五个不同风险边界。

## 2. 当前总进度

| Code | 主要结果 | commit | 状态 |
|---:|---|---|---|
| 1 | 固定 72-file Source RL-Games runtime | ed9c0ae5f6e0d31550d8152e48f213f13505025f | 已完成 |
| 2 | compatibility、dual-hash、network/config oracle | 1adb159e9fb82ff322653a3533e9b2f32c3f862a | 已完成 |
| 3 | rollout、GAE、augmentation、shuffle、RNG oracle | 3a712a97ff43563225d657de0b5181a34f4e0974 | 已完成 |
| 4 | update、AMP、optimizer、GradScaler oracle | 2e1c7874d4a550a63834325b6a9a8b078304ba6a | 已完成 |
| 5 | checkpoint、resume boundary、player oracle | 6e1087f62cd17d196d67ccbd4ea880d0341cf6b5 | 已完成 |
| 6 | MuJoCo backend public contracts | 31583cae7a4084258d28e330ed301c8dc4240c38 | 已完成 |
| 7 | assets、task foundations、T0 | af5c3401cecf280fc641f48e9c3ae4a134260ac7 | 已完成 |
| 8 | 真实 MuJoCo env composition、T1 | cc9a4fdea72b40716a861611cbca0fac874c7ce4 | 已完成 |
| 9 | Source RL-Games SAPG production path | af01194e1074c9c6bc0938a66352198d5a36c12f | 已完成 |
| 10 | release、dependency lock、support promotion | — | 未开始 |

Code #5 提交后的已记录验证：

- focused Code #5 gate：11 passed；
- 完整 SAPG oracle：186 passed；
- vendor suite：37 passed；
- 72-file vendor audit：通过；
- scoped Ruff、format check 和 git diff check：通过；
- required SAPG tests：0 skip；
- Code #3-#5 没有修改 Source、vendored runtime 或生产 training runtime；
- 当前 oracle-only 阶段没有运行 make test-all。

Code #6 提交后的已记录验证：

- 代码 commit：31583cae7a4084258d28e330ed301c8dc4240c38；
- focused backend gate：24 passed、0 skipped；
- existing source-model slow regressions：2 passed、0 skipped；
- 邻近 backend regressions：65 passed、11 skipped，skip 全部来自未安装的 optional Motrix；
- 完整 non-slow `make test`：1784 passed、60 skipped、273 deselected、1 xfailed、0 failed、
  85 个已记录的 Gymnasium/XML/ONNX warnings；
- `make check`：Ruff 通过，mypy 对 229 个 source files 无错误，pyright 为 0 errors 和
  3 个既有 optional Motrix import warnings；
- `uv lock --check`、`git diff --check` 和 post-commit focused rerun 通过；
- 根目录没有 `MUJOCO_LOG.TXT`；没有运行 `make test-all`，没有进入 Code #7。

Code #7 提交后的已记录验证：

- 代码 commit：af5c3401cecf280fc641f48e9c3ae4a134260ac7；
- focused task/T0 gate：90 passed、0 skipped、0 warnings；Code #6 邻近 source-model gate：
  3 passed；
- T0 独立 regeneration 的 NPZ 和 manifest 均逐字节一致；
- Ruff 和 format check 通过，mypy 对 243 个 source files 无错误，pyright 为 0 errors 和
  3 个既有 optional Motrix import warnings；
- 40-mesh closure、license/provenance、XML/代表 tool compile、cold-path 和 backend capability
  audit 通过；post-code-commit focused rerun 仍为 90 passed；
- `git diff --check` 通过，根目录没有 `MUJOCO_LOG.TXT`；没有运行 `make test-all`，没有进入
  Code #8。

Code #8 提交后的已记录验证：

- 代码 commit：cc9a4fdea72b40716a861611cbca0fac874c7ce4；
- clean registry bootstrap、唯一 MuJoCo owner 和
  `registry.make("SimToolReal", sim_backend="mujoco", num_envs=6)` 真实构造通过；
- focused task/env/T0/T1 gate：101 passed、0 skipped；Code #6 邻近 backend gate：37 passed；
- 600 个 materialized source XML、600 个 physics variants、fixed assignments 和
  box_box/capsule_box/box_only=`250/300/50` census 通过；完整 source XML 为
  `nq/nv/nu/nmesh=36/35/29/40`，既有 `discardvisual=true` physics pool 为
  `36/35/29/19`；
- 真实 reset/partial reset、64 steps、raw reward、success、timeout/final observation、public
  wrench handoff、engine autoreset exact-row/latch/cleanup 和 construction-failure cleanup 通过；
- T1 独立 regeneration 的 NPZ 和 manifest 均逐字节一致；NPZ/manifest SHA256 分别为
  `228b704e0a5b8e94269ce4b4da29cff4e51bb57338390d79453fe0d921cfb760` 和
  `6b87220134e2711939bad47d8ae64c0fa8820e5731b887c751d7646a061a5fdb`；
- `uv lock --check`、Ruff、format check 通过，mypy 对 244 个 source files 无错误，pyright
  为 0 errors 和 3 个既有 optional Motrix import warnings；
- cold-path/backend-private/source-access audit 和 `git diff --check` 通过，根目录没有
  `MUJOCO_LOG.TXT`；没有修改 backend、Source、vendor、MuJoCoUni、dependency 或 conf，
  没有运行 `make test-all`，没有进入 Code #9。

Code #9 提交后的已记录验证：

- 代码 commit：af01194e1074c9c6bc0938a66352198d5a36c12f；实际 scope 为 32 paths、
  2913 insertions、18 deletions，其中包含 generated `uv.lock`，净手写约 2.85k 行；
- root `rlgames-sapg` optional extra 精确安装
  `unilab-simtoolreal-rl-games==1.6.1+simtoolreal.2a991753.compat2`，editable source 指向仓库内
  vendor；base+mujoco exact sync 下 integration package 可 import，但 distribution 和
  `rl_games` 均不存在；Linux/aarch64 和错误 identity fail closed；
- `conf/rlgames_sapg` 的 base/12k owners 分别固定 `24576/4096/6` 和 `12288/2048/6`，
  Source-native R2 field-by-field、raw reward × native 0.01、unsupported config preflight 通过；
- `RlGamesNpEnvAdapter` 锁定 first-reset、`obs/states`、一次 action transfer、
  done/timeout/device/dtype/shape/finite 和 `env_state=None` contract；唯一 train 调用链为
  `Runner.load -> set_vec_env -> run_train -> A2CAgent.train`，没有第二套 rollout/update；
- `ExperimentTrackerObserver`、native run directory、trusted `.pth` resolver、
  resume/weights boundary 和 `PpoPlayerContinuous` bridge 已接通；W&B 仍只有
  `ExperimentTracker` 一个 lifecycle owner，不声明 env/RNG/trajectory bit-exact resume；
- MuJoCo cold-path visual materializer 已按 assigned physics model 同步 contact topology；真实
  indexes 0/1/7 均保留完整 `36/35/29/40` visual model，并覆盖 box_box、capsule_box 和
  box_only；
- 完整 SAPG oracle：186 passed、11 条既有 AMP deprecation warnings；Code #9 focused：
  117 passed、0 skipped；real N=6 vertical slice：1 passed、2 条既有 AMP warnings；vendor：
  37 passed，72+7 audit 通过；Code #8：101 passed；邻近 backend：52 passed、7 deselected、
  5 条既有 XML/Gymnasium warnings；
- 真实 CLI train/eval 产生 101,989,547-byte native checkpoint 和 45,351-byte MP4；checkpoint
  为 outer rank 0、epoch 1、frame 24、非空 actor/central optimizer state、`env_state=None`；
- Ruff、format check、`uv lock --check` 通过，mypy 对 252 个 source files 无错误，pyright 为
  0 errors 和 3 个既有 optional Motrix import warnings；cold-path/private/source-access、
  single-W&B、no-second-loop 和 cleanup audit 通过；
- 没有修改 Source、donor、vendor bytes、Code #1-#8 fixtures、env/task formulas 或 shared
  sim2sim；没有运行 `make test-all`，没有进入 Code #10。

以上证据证明已接受的算法/runtime、backend、task、真实 env 和 production vertical slice
回归边界。当前仍然没有：

- 持续 M0-dev S1 和真实 `12288/2048` profile 证据；
- clean-install M0-release artifact 和正式 dependency promotion；
- `make test-all`、final-current-head CI 和 maintainer support judgment。

## 3. 目标、owner 和非目标

### 3.1 最终目标

最终在 UniLab 内形成以下唯一 SAPG 路径：

~~~text
Hydra SAPG owner
  ├─ task/reward/backend fields -> UniLab registry.make() -> NpEnv
  └─ native Source params       -> vendored RL-Games Runner
                                        |
                                A2CAgent owns rollout/update
                                        |
                                RlGamesNpEnvAdapter
                                        |
                             SimToolRealEnv -> SimBackend
                                        |
                                  MuJoCo/MuJoCoUni
~~~

责任分工：

| 能力 | 唯一 owner |
|---|---|
| rollout、storage、augmentation、dataset、minibatch | vendored Source RL-Games |
| env step/reset 调用时机 | native Runner/A2CAgent |
| actor/central update、AMP、scheduler | vendored Source RL-Games |
| task 创建、reward/env 配置 | Hydra owner + UniLab registry |
| obs/action/device/space 转换 | 同步 RlGamesNpEnvAdapter |
| physics、tool models、wrench、autoreset | UniLab MuJoCo backend |
| pth payload、resume 和 player lifecycle | vendored Source RL-Games |
| run directory、metadata、W&B bridge | UniLab tracker，且只能有一个 lifecycle owner |
| 相机、视频、viser/interactive 外壳 | UniLab visualization |

### 3.2 明确非目标

- 不迁移或继续完善 RSL-RL SAPG 仿写；
- 不新增第二套 rollout loop、collector、learner 或同步协议；
- 不接 UniLab async runner；
- 不做 PPO、Motrix、sim2sim、distributed、ROCm、torch.compile 或 export；
- 不修复 Source 中看起来像 bug 的既有语义；
- 不做 IsaacSim/MuJoCo 物理轨迹或训练曲线逐步等价；
- 不声称 bit-exact full-runtime resume；
- 不做通用 RL-Games support；
- 不让部署环境依赖外部 Source checkout 或 IsaacSim。

## 4. 固定仓库与 provenance

| 角色 | 路径 | 用途 |
|---|---|---|
| Clean target | /home/user/ws/lemon/rlgame-unilab/UniLab | 本路线唯一写入仓库 |
| Mature donor | /home/user/ws/lemon/UniLab | 参考成熟 MuJoCo env、600-tool、assets、backend 和测试 |
| Source oracle | /home/user/ws/lemon/simtoolreal | 离线生成算法/task fixtures 和做 provenance 审计 |

固定 Source identity：

- Source reference commit：2a9917533bfea70419ed2667a511d7238e5b3abc；
- RL-Games parent tree：7a6a0bb090998d00565aaefa6ab9f2b3d356ace2；
- train owner：isaacsimenvs/cfg/train/SimToolRealSAPG.yaml；
- train owner blob：f363d05d4a24b190b7837703b93270d8f3fe9a9c；
- task owner：isaacsimenvs/cfg/task/SimToolReal.yaml；
- task owner blob：6469d46867081b70edaa589dcb31c7090b64d45e；
- parent tree 包含 72 个 Python 和 122 个 YAML，本路线只 vendor 72 个 Python；
- 普通 pip rl-games 不是算法 oracle。

目标仓库内的实际 runtime 位于：

~~~text
third_party/simtoolreal_rl_games/
~~~

Source checkout 只用于 fixture 生成、显式 rebaseline 和审计。训练或 play 若仍需要访问
/home/user/ws/lemon/simtoolreal，说明 production 接线不符合本设计。

Vendor 永久维护规则：

- source_manifest 同时记录 pristine Source identity 和当前 patched identity；
- 7 个 compatibility patch 必须逐文件记录原因和覆盖测试；
- 未列入 PATCHES 的 65 个 Python 文件必须保持 Source identity；
- vendor 不由 Ruff/formatter 改写；
- Source/vendor identity 改变时必须重跑全部算法 oracle，不能静默 rebaseline。

## 5. 必须与 Source 相同的语义

以下内容由 vendored runtime 和原生 owner config 直接拥有，不允许在 UniLab 重写：

- block 编号、leader/follower 定义、coefficient IDs [50,40,30,20,10,0]；
- entropy exploration、learnable embedding、conditional per-block sigma；
- actor/shared value/central critic、RNN、layer norm 和 normalizer；
- rollout storage、time/env layout、trajectory shuffle 和 RNN done reset；
- follower selection RNG、augmentation universe 和 counterfactual TD；
- timeout bootstrap、GAE、advantage、PPO ratio/clip/value/bounds/entropy；
- central-before-actor、dataset 尾批、mini-epochs 和 adaptive KL；
- optimizer parameter sets、gradient clipping、AMP、GradScaler 和 overflow；
- checkpoint schema、RNN/rollout state 和 player routing；
- backend-neutral action、observation、reward、termination、reset 和 DR 数学。

Source 当前行为即使不理想，也先作为 baseline 保留：

- timeout reward 使用 action 前 value，不对 final observation 重估；
- follower return 使用 one-step reward 加 bootstrap，不重算 GAE；
- checkpoint 不保存 Python、NumPy、Torch RNG；
- current env_state 为 None；
- player owner 中 deterministic 为 false，并保留 N 不等于 6 的原生 fallback。

任何算法行为修正必须在 source-exact baseline 之后另立 issue，且不能继续标为 Source
parity。

## 6. 允许差异与配置原则

允许差异必须显式写入 manifest：

| 边界 | 允许差异 | 约束 |
|---|---|---|
| Simulator | IsaacSim/PhysX 到 MuJoCo/MuJoCoUni | 差异只能留在 env/backend/adapter |
| Env carrier | Isaac Lab tensor 到 NpEnv NumPy dict | adapter 显式映射，不探测 backend 私有能力 |
| Tool pool | Source 12x100 到 UniLab 12x50，共 600 | distribution 和固定 assignment 可审计 |
| Reset mapping | fixed MuJoCo table reference、multiccd | Source reset ranges 其余保持 |
| Resource profile | 24576/4096 到 12288/2048 | 都保持 6 blocks |
| Play scale | UniLab canonical 6 envs | 不修改 Source player 的任意 N 路由 |
| Packaging | Gymnasium、NumPy、Torch、build metadata | 每个 patch 有 dual hash 和 regression |
| Tracking | run dir 和 metadata sidecar | 不改变 callback、update 或 checkpoint payload |

不能借 backend 差异改变：

- observation 字段顺序或缩放；
- action scale/delay 顺序；
- raw reward term、reward double scaling；
- termination、timeout、done 或 autoreset 语义；
- reset/DR 范围；
- object wrench 是否激活；
- 给定同一 backend-neutral input 时的 task math；
- native Runner、checkpoint 或 player lifecycle。

生产 owner 的关键配置：

- actor observation 140、critic state 162、action 29；
- actor LSTM + MLP，central critic MLP；
- 6 blocks、32 维 embedding；
- gamma 0.99、tau 0.95、e_clip 0.1、critic_coef 4.0；
- learning rate 1e-4、adaptive LR、KL threshold 0.016；
- input/value/advantage normalization 和 mixed precision；
- exploration scale 0.002；
- reward_shaper.scale_value 0.01；
- env 使用 Source raw reward scales 200/20/300/50/1000/0.03/0.003，只缩放一次；
- reset 使用 source_random、full SO(3)、x/y 0.1、z 0.02、arm/finger 0.1、
  velocity 0.5；
- 固定 MuJoCo table 下 object_spawn_z_reference_range 为 0.0；
- object wrench DR 必须显式为 true；
- M0-dev 的 cpu_ids 必须显式为 null；
- canonical play_env_num 为 6。

资源 profile：

| profile | actors/envs | block size | blocks | horizon | 用途 |
|---|---:|---:|---:|---:|---|
| Source owner | 24576 | 4096 | 6 | 16 | Source 配置和生产语义基线 |
| UniLab M0-dev | 12288 | 2048 | 6 | 16 | 真实 smoke 和资源 profile |
| Code #3/#4 oracle | 12 | 2 | 6 | 4 | frozen rollout/update fixture |
| Code #5 oracle | 6 | 1 | 6 | 4 | checkpoint/player fixture |

小型 oracle profile 是 test-only boundary，不得进入生产 owner。

## 7. Code #1-#5 已完成内容

### Code #1：固定 Source runtime

Commit：

~~~text
ed9c0ae5f6e0d31550d8152e48f213f13505025f
vendor: pin SimToolReal RL-Games runtime
~~~

主要结果：

- vendor 固定 72 个 Source Python runtime 文件；
- 保存 nested MIT license、UPSTREAM、source_manifest 和 package metadata；
- 建立独立 distribution 名 unilab-simtoolreal-rl-games；
- 增加 read-only vendor audit；
- 通过 path-scoped 配置保护 Source bytes 不被 formatter 或 whitespace gate 改写。

Code #1 只建立算法 owner，未接 UniLab env 或生产训练入口。

### Code #2：兼容和 network/config oracle

Commit：

~~~text
1adb159e9fb82ff322653a3533e9b2f32c3f862a
fix: make RL-Games compatible and lock network fidelity
~~~

主要结果：

- 7 个最小 Gymnasium、NumPy 和 packaging compatibility patch；
- pristine/patched dual hash 和 fail-closed import gate；
- Source 与 Target 使用独立进程和独立 rl_games namespace；
- 固定 actor/critic shape、RNN/MLP、embedding、conditional sigma；
- 验证 forward、value、log-prob、entropy 和 gradient signatures。

Code #2 回答的是兼容补丁是否改变网络/config 语义，不是完整 training pipeline。

### Code #3：rollout、GAE、augmentation、shuffle、RNG

Commit：

~~~text
3a712a97ff43563225d657de0b5181a34f4e0974
test: lock SAPG rollout and RNG semantics
~~~

Canonical synthetic case：

- 12 env、block size 2、6 blocks；
- horizon/sequence length 4；
- 48 base rows + 8 follower rows = 56 rows；
- actor/central native dataset batches [12,12,12,20]；
- 确定性 obs、reward、done 和 timeout；
- 真实 native play_steps、GAE、augmentation、shuffle 和 dataset owner。

验证边界：

- action 前 timeout value；
- delta、GAE、return 和 advantage；
- follower selection、repeat indices 和 counterfactual TD；
- post-shuffle row identity；
- RNN done reset；
- NumPy、Torch CPU/CUDA RNG 前后状态。

这些不是 MuJoCo 或 IsaacSim 的真实训练数据。

### Code #4：complete update 和 AMP

Commit：

~~~text
2e1c7874d4a550a63834325b6a9a8b078304ba6a
test: lock SAPG update and AMP semantics
~~~

Code #4 直接消费 Code #3 的 frozen 56-row post-shuffle batch，不重新采样环境。

验证 normal FP32、normal AMP 和 overflow AMP 三个 native case：

- prepare_dataset 和 central-before-actor；
- central/actor 各两个 mini-epoch；
- PPO/value/bounds/entropy loss；
- KL reference、adaptive LR 和 tail minibatch；
- optimizer、gradient clipping 和 parameter delta；
- input/value RMS；
- autocast、GradScaler step/skip 和 overflow backoff；
- 最终 model、optimizer、scaler、RMS、LR 和 RNG。

FP32 case 关闭 AMP、identity shuffle freeze 和 overflow 注入均为记录过的 test-only
override，不得进入生产配置。

### Code #5：checkpoint、resume boundary 和 player

Commit：

~~~text
6e1087f62cd17d196d67ccbd4ea880d0341cf6b5
test: lock SAPG checkpoint and player semantics
~~~

Code #5 使用 6 actors、block size 1、MLP [32,32,16,16] 和 RNN 16 的小型 native profile，
只为减小 fixture，不改变 checkpoint/player owner。

验证：

- native Source pth payload schema；
- model、optimizer、RMS、scaler、RNN/rollout 和 tracker fields；
- env_state=None；
- Source 不保存 RNG 的真实边界；
- 外部恢复 RNG 后的首个 action/value/update；
- canonical 6-env player routing；
- N=5 和 N=7 的原生 equality/argmax fallback；
- deterministic 和 stochastic player path。

这个 profile 不代表生产网络被缩小。

## 8. Frozen fixture anchors

普通 pytest 只能重放，不能生成或更新 fixtures。显式 rebaseline 必须固定 Source checkout、
独立 namespace、canonical platform 和 reviewed manifest diff。

| Artifact | SHA256 |
|---|---|
| source_network_fp32.npz | 4becec09cea5d3a81cc0061f8c4821c6bf8fde2982ad85b37f3c29f10672eba6 |
| source_network_manifest.json | 444056a59eb50bc0632703d72ed820ab3e07b08e3c2a355a24de06af0ef46aa2 |
| source_rollout_fp32.npz | 3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8 |
| source_rollout_manifest.json | 785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f |
| source_update_fp32.npz | df58bb09d67edd24a19f2a164a4851fa24b9f2d305e9826c10433635cee78463 |
| source_update_manifest.json | 748be517553df7689ee4a06991241e37fc205336f6a5638f2bdd168735d57e45 |
| source_checkpoint.pth | bbe577dc7efed068bb38ce6f268e849de6a41e8ab6bb4a78fabeed9b0d7b5e02 |
| source_checkpoint_manifest.json | 8d55469d09095827587d502758d477913c76f13e8e9cd0baa23cb142d518c946 |
| source_t0_fp32.npz | 5393583ed7a424910b24622867785e6d6431e29570da997a8064f654ec70624d |
| source_t0_manifest.json | d90453bec0db06046aa832615f52c9b8499bee735dc62b4ed9d0d7f107e387b6 |

Manifests 内另有 canonical payload、Source owner、loaded module 和 platform hashes；上表是
Git 工作树文件的外层 anchor。

## 9. Code #6-#10 详细执行边界

### Code #6：MuJoCo runtime public contracts

唯一主要结果：让 SimToolReal 所需的 backend 能力成为 SimBackend public contract。

状态与代码提交：

~~~text
已完成
31583cae7a4084258d28e330ed301c8dc4240c38
feat(backend): add SimToolReal MuJoCo runtime contracts
~~~

只做：

- ModelVariantSpec.source_model_file 和 source model direct compile；
- public apply_body_wrench；
- public get_step_autoreset_mask；
- 多 substep autoreset mask OR-latch；
- M0-dev dependency identity、lock 和 provenance；
- source-model、row-isolated force/torque、真实 autoreset 近风险测试。

不做：

- task/assets/env/Runner 接线；
- env 调 backend 私有方法；
- script feature leakage；
- 新的 backend capability；
- MuJoCoUni owner 仓库中的未批准生产修改。

预计规模：约 11 paths、约 800 行手写实现/测试加 generated lock。

永久成本：3 个 public backend contracts 和 M0-dev/M0-release provenance。

实际 scope 为 11 paths、504 insertions、13 deletions。原 prompt 的 10 个允许路径全部按
计划落地；第 11 个路径 `tests/base/test_mujoco_batch_env_randomization.py` 经 maintainer
明确授权，只把 M0-dev 新增的 `geom_size` 和 `geom_pos` 纳入安装态 compatibility 断言。

已完成的 contract 和证据：

- `ModelVariantSpec.source_model_file` 在 materialization 冷路径直接编译完整 source model，
  并保留 geom-size-only 原路径；
- 12 个独立 source XML 使用 `model_assignments=np.arange(12)` 真实 materialize 和 step，
  index 7 的 dominant layout 在 `ngeom/nbvh/nC/nbuffer` 上支配 small variant；
- `SimBackend.apply_body_wrench` 由 MuJoCo 按 env row、body 6D block 累加 world-frame
  force/torque，并对两类 shape mismatch fail closed；
- `SimBackend.get_step_autoreset_mask` 区分 unknown 与 no-reset，MuJoCo direct path 和
  pre-step-control 多 substep path 都读取 public `BatchEnvPool.was_autoreset` 并 OR-latch；
- real-pool 测试固定 baseline、env 1 exact mask、env 2 首个/四个 substep latch 和下一步
  clear；
- production 热路径没有 `getattr/hasattr` capability probe，也不读取 XML 或 asset metadata。

依赖固定为 `mujoco-uni-runtime==0.4.0.dev0`，Git URL 是
`https://github.com/lemon-star608/mujoco_uni.git`，source/lock/安装态 commit 都是
`7205e070e983df90d520f0f8593853013e976746`。本批没有使用 sibling checkout 或 artifact，
没有修改 MuJoCoUni owner 仓库。M0-dev 仍没有 `cpu_ids/worker_cpu_ids` ABI；CPU affinity
和正式 dependency promotion 继续属于 Code #10，Code #6 不构成正式 support 声明。

### Code #7：assets、task foundations 和 T0

唯一主要结果：把成熟 SimToolReal 资源和 backend-neutral task math 放入 target，并得到
可解释的 Source T0 oracle。

状态与代码提交：

~~~text
已完成
af5c3401cecf280fc641f48e9c3ae4a134260ac7
feat(simtoolreal): add task foundations and Source T0
~~~

只做：

- 40 个由生产训练 XML 实际引用的 mesh、2 个生产 XML；其中 16 个 KUKA
  collision/visual mesh 和 24 个 Sharpa hand mesh，明确不复制未引用的
  `left_hand_C_MC_visual_.STL` 与 `left_thumb_MC_modified.STL`；
- LICENSE.simtoolreal、LICENSE.kuka_iiwa 和 ASSET_PROVENANCE；
- 约 14 个 task modules；
- action、delay、goal、observation、reward、termination、reset、DR primitives；
- focused task tests；
- T0 generator、fixture 和 manifest；
- asset/XML 只在 init、materialization 或 cache 冷路径访问。

不做：

- 注册或运行真实 env；
- native Runner/tracker/CLI；
- RSL-RL SAPG；
- donor 的 collision research、viewer、DexToolBench 或 Motrix；
- 把 task keyframe 放进 robot.xml。

预计规模：机械迁移 40 meshes、2 XML、约 7-8k donor LOC；手写 provenance/adaptation
约 650-1000 行。

永久成本：14 个 package/task modules、assets/licenses 和一个 T0 golden。

实际 scope 为 78 paths、8890 insertions：原计划的 77 个 asset/task/test/fixture paths，加
根 `.gitattributes` 的 1 条精确 license whitespace rule。该规则只让 Git 在保留固定 BSD
license 原始 CRLF/尾空格 bytes 时不误报 whitespace；license blob 仍为
`46670489513480eff80b81e3ec780abf29e347bd`。

已完成的 contract 和证据：

- 生产 XML 的闭包恰好是 40 meshes：16 个 KUKA collision/visual 和 24 个 Sharpa hand；
  `left_hand_C_MC_visual_.STL` 与注释候选 `left_thumb_MC_modified.STL` 没有迁移；
- robot XML 真实编译为 `nq/nv/nu/nmesh=29/29/29/40`，box_box、capsule_box、box_only
  代表 scene 均真实编译为 `36/35/29/40`；keyframe 只在 task-level scene XML；
- 特殊 `left_hand_C_MC_visual.STL` 是 Source 5964-triangle ASCII STL 到 donor binary STL 的
  deterministic、geometry-preserving adaptation，不是另一套几何；Source/target 分别为
  1254651/298284 bytes，target SHA256 为
  `2104aad51f03537a3458e3afcf5f5c7532b8a3ef0abba08b4425b3bd31e4f55f`；
- `ASSET_PROVENANCE` schema 1 固定 Source/donor/XML/license/40-mesh inventory、39 identical
  + 1 adaptation 和排除清单；外层 SHA256 为
  `ad12eeec35d7e33e8f4a00011aaa56a56b04acaf87af45c84697b3e9b94b4b42`；
- package 在没有 env.py/registry owner 时可导入；config 固定 action 29、actor 140、critic 162、
  600 steps，env/raw reward 保持 200/20/300/50/1000/0.03/0.003，只有 critic reward feature
  乘 0.01；
- deterministic catalog 为 12 distributions × 50 = 600，topology census 为
  box_box=250、capsule_box=300、box_only=50；Source-native 12×1 representative URDF 与
  Target ToolSpec 的 geometry、mass、COM、diagonal inertia 和 object scale 已比较；
- focused tests 覆盖 non-identity canonical/backend permutation、delay、goal/keypoint、
  observation、raw reward/lifecycle、full SO(3) reset、fixed table mapping 和 wrench DR；
- T0 固定 N=6、CPU FP32、8 个 Source-native modules、74 arrays 和 17 个 exact discrete
  fields，覆盖 non-identity permutation、delay/noise、goal/keypoint、raw reward trackers、
  termination/reset、wrench 及 tool distribution/ToolSpec；NPZ/manifest SHA256 见第 8 节；
- focused gate 为 90 passed，Code #6 邻近 gate 为 3 passed；独立 regeneration 两个 cmp
  通过，Ruff/format/mypy/pyright 通过；
- production 中仅 `tool_assets.py` materialization 和 `dr_provider.py` trajectory-cache 两个
  冷路径读取 XML/metadata；没有 env.py、registry import、backend `getattr/hasattr` probe、
  Source/donor/vendor/MuJoCoUni 修改，也没有进入 Code #8。

### Code #8：真实 MuJoCo env composition 和 T1

状态：已完成。代码 commit 为 cc9a4fdea72b40716a861611cbca0fac874c7ce4。8A
registry/construction、8B NpEnv lifecycle、8C real 600-tool/wrench/autoreset integration 和
8D Target-only T1 均已落地；T1 不访问 Source checkout，也不做跨 simulator 轨迹对拍。

唯一主要结果：组合、注册真实 SimToolReal MuJoCo env，并锁定 NpEnv contract。

实际 scope 为 12 paths、1702 insertions、6 deletions。`env.py` 是固定 donor blob 的机械移植
和当前 Target contract 适配，最终 793 行；没有修改 Code #7 task formulas。

已完成的 contract 和证据：

- `SimToolRealCfg` 和 `SimToolRealEnv` 分别注册 config/mujoco env owner，clean bootstrap 后没有
  第二个 backend owner；
- action/actor/critic/episode contract 固定为 `29/140/162/600`，真实 scene 为
  `nq/nv/nu=36/35/29`；
- deterministic catalog、600 complete XML、600 compiled physics variants 和 fixed env
  assignments 均进入真实 MuJoCo runtime；完整 XML 的 40 meshes 与 physics pool 的 19 meshes
  是 manifest 中显式记录的 `discardvisual=true` mapping，不是资产缺失；
- NpEnv reset/step、partial-row scatter、raw reward、success、timeout、terminal observation、
  public body wrench 和 engine-autoreset lifecycle 已由 real N=6 integration 锁定；
- close、重复 close 和 construction failure 均清理 materialized tool temp root，divergence
  warning 被限制在 pytest tmp path；
- T1 固定 N=6、H=8、seed=20260821、50 arrays 和 13 个 exact discrete fields；普通 replay
  校验 fixture 外层 hashes、当前 production hashes、runtime identity、array inventory 和真实
  capture；
- required focused gate 为 101 passed、0 skipped；T1 deterministic cmp、Code #6 邻近 gate、
  Ruff/format/mypy/pyright/lock 均通过。

明确仍未做：

- native Runner、tracker、pth 或 CLI；
- IsaacSim/MuJoCo 轨迹 bit-exact；
- support promotion。

永久成本：真实 task/env owner、registry 和 T1 regression。

### Code #9：Source RL-Games SAPG production path

状态：已完成。代码 commit 为 af01194e1074c9c6bc0938a66352198d5a36c12f。9A dependency/
config、9B adapter/Runner、9C tracker/checkpoint 和 9D player/CLI/visual/real slice 均已落地。
执行 prompt 为 `docs/simtoolreal_sapg_code9_prompt.md`。

Phase 0 用 `uv run --with-editable` 临时加载 vendor 时，overlay 会独立解析 vendor 的
`torch==2.7.0`，必须同时显式增加 `--with 'torch==2.7.0+cu128'` 和
`--index pytorch-cu128=https://download.pytorch.org/whl/cu128`。未加这两个约束时从 PyPI
得到 cu126 只是临时 overlay 选源错误，不是 frozen oracle/runtime drift，不得因此放宽或
重生成 fixture。

唯一主要结果：建立一个可 train、resume、weights-load 和 play 的 native vertical slice。

已完成：

- 修正Code #7增加KUKA license whitespace attribute后，既有vendor audit仍锁旧单行内容的
  compatibility drift；不修改`.gitattributes`或vendor bytes；
- root sapg optional extra 和 pinned vendor dependency；
- 独立 conf/rlgames_sapg owner group；
- RlGamesNpEnvAdapter；
- Runner.load 加 set_vec_env 的唯一 train/play path；
- observer、run directory 和 ExperimentTracker bridge；
- pth resolver 和 resume/weights modes；
- PpoPlayerContinuous；
- CLI train/eval route；
- fake env ABI、config、Runner、tracker、checkpoint、player 和真实 smoke tests；
- player/video 必须验证每个 env 的 assigned-tool visual mapping；Code #8 只锁定了 19-mesh
  physics variant，不能把它当作完整 40-mesh visual playback 已经成立的证据；已知现有
  size-only materializer会让capsule_box/box_only复用tool 0 topology，9D在既有MuJoCo cold-path
  playback helper内做generic topology同步，不新增backend public contract，也不改env/T1。

实际 scope 为 32 paths、2913 insertions、18 deletions，包含 generated `uv.lock`；净手写约
2.85k 行，没有单个 production 文件达到 800 行。主要 production owners 为 8 个
`rlgames_sapg` modules、3 个 Hydra YAML、一个 325 行 training script、CLI/completion route
和既有 MuJoCo cold-path playback helper。

已锁定的关键 contract：

- base install 不携带 vendor runtime，选择 extra 后只接受仓库内 exact editable identity；
- native config 直接由 Hydra compose，不在 Python 维护第二份算法翻译；
- adapter 只做同步 ABI 转换，native Runner/A2CAgent 独占 rollout/update/checkpoint；
- `.pth` 只从可信 task root 解析，resume 与 weights 分离，`env_state=None` 边界显式；
- player 使用 native normalizer、RNN、coefficient routing、action 和 done reset；UniLab 只保留
  env、tracker、camera 和 video shell；
- assigned visual playback 保持 40 meshes，并按真实 tool variant 同步 primitive contact
  topology；physics pool 仍保持 Code #8 的 19-mesh `discardvisual=true` contract。

明确仍未做：

- 新 rollout loop；
- Python 层算法配置翻译；
- 第二个 W&B lifecycle；
- RSL-RL checkpoint schema conversion；
- async/distributed/export；
- 反向修改 vendor 或 rebaseline Code #1-#5。
- 持续 S1、真实运行 12k profile、M0-release、完整 suite、CI 或 support promotion。

永久成本：8 个以内职责单一的 integration modules、3 个 Hydra owners、一个薄 training
script、独立 CLI route、pth resolver、native player path 和一个既有cold-path visual helper
的topology regression。

### Code #10：release 和 support promotion

唯一主要结果：把已经真实验证的开发路径晋升为可安装、可维护的正式 support。

按顺序只做：

1. M0-dev 小规模 S1 finite train/play smoke；
2. 12288/2048 profile；
3. 外部 clean-install M0-release artifact；
4. dependency lock 从 dev identity 晋升为正式 artifact；
5. mixed-layout、autoreset、affinity 组合回归；
6. audit、docs、support matrix、make test-all；
7. final-current-head CI；
8. maintainer support judgment。

没有 M0-release、真实 train/play、完整测试和最终产品判断时，Code #10 保持未完成。

永久成本：正式 MuJoCoUni dependency、release provenance、support docs 和升级复验。

## 10. T0 和 T1

### 10.1 T0：backend-neutral task oracle

T0 在 Code #7 完成，不依赖真实 MuJoCo trajectory。

固定 Source generator 构造并保存：

- joint、object、tool state；
- action、delay history 和固定 non-identity canonical/backend permutation；
- goal 和 episode counters；
- 显式 random draws；
- 必要的命名 simulator-query inputs，例如 body pose/velocity 和 fingertip distance；
- 12 个 Source-native distribution representative URDF 的 geometry/mass metadata。

Source 原生 task math 产生：

- action target；
- actor 140 维 observation；
- critic 162 维 state；
- 每个 reward term 和 raw total；
- termination/reset mask；
- DR parameters；
- 代表 ToolSpec 的 geometry、mass、COM、inertia 和 scale；
- 字段顺序、单位和 dtype。

fixture 必须同时保存公式所需 primitive inputs，不能只存无法解释的最终 tensor。T0 的
目的，是把 reward scaling、obs ordering、action delay 和 termination 公式错误与物理
引擎差异分离。

### 10.2 T1：real env integration oracle

T1 已在 Code #8 完成。它使用同一组 task primitives，验证它们接入真实 MuJoCo env、
backend、600-tool pool 和 NpEnv lifecycle 后仍保持 task contract。

T1 验证：

- registry、reset 和 step；
- obs/action shape、dtype 和字段顺序；
- reward、termination、timeout、autoreset 和 info；
- reset randomization、delay、wrench DR；
- tool assignment 和 model variant。

离散 mask/index 必须 exact；浮点 task math 使用已批准 tolerance。只允许 manifest 中的
backend、table、tool 和 resource mapping 差异。T0 未通过时不得开始 T1。

固定 capture 为 N=6、H=8、CPU FP32、seed=20260821，通过 registry composition 创建真实
env；事件依次为 init、4 steps、rows `[1,4]` selected reset、3 steps、row 2 exact timeout。
fixture 包含 50 arrays 和 13 个 exact discrete fields，普通 pytest 不运行 generator、不访问
Source/donor。完整 source XML 的 `nmesh=40` 与 physics pool 的 `nmesh=19` 在 manifest 中显式
区分。最终 hashes：

~~~text
target_t1_fp32.npz
228b704e0a5b8e94269ce4b4da29cff4e51bb57338390d79453fe0d921cfb760

target_t1_manifest.json
6b87220134e2711939bad47d8ae64c0fa8820e5731b887c751d7646a061a5fdb
~~~

## 11. Production adapter、config、checkpoint 和 play

### 11.1 Adapter ABI

adapter 只转换 observation、action、done 和 info：

~~~text
reset() -> {"obs": Tensor[N,140], "states": Tensor[N,162]}
step(actions) -> same observation dict, reward, done, info
done = terminated OR truncated
info["time_outs"] = truncated
get_env_info()
get_number_of_agents()
set_train_info()
get_env_state() -> None
set_env_state(None) -> no-op
vec_env.env.device -> RL device
~~~

约束：

- NpEnvState.obs 必须是 dict；
- 第一次 reset 复用 init_state，不能二次全量随机 reset；
- action 只做一次 Torch 到 NumPy 转换；
- 返回 tensor 在 RL device 上构造；
- observation/action spaces 使用 Gymnasium finite Box；
- adapter 不读取 XML/assets，不探测 backend 私有能力。

### 11.2 Hydra owner

生产配置使用独立组，不复用 conf/ppo：

~~~text
conf/rlgames_sapg/config.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco_12k.yaml
~~~

cfg.rl_games.params 直接保持 Source params schema。入口只允许注入 train_dir、device、
vec_env 和 env_info 等 runtime handles，不长期解释算法超参数。

R2 config gate 必须逐字段比较 Source owner，仅允许 manifest 中的 backend/resource
差异。

### 11.3 Checkpoint

SAPG 生产 checkpoint 使用 native RL-Games pth schema；现有 RSL-RL 和其他算法继续使用
各自的 pt schema。后缀只是约定，真正的边界是 payload 和 runner/player lifecycle：

- pth 由 native A2CAgent 保存和恢复；
- resume 恢复 Source 保存的状态；
- weights mode 只加载模型权重和可支持的 central value state；
- 可信本地 legacy pth 可显式 weights_only=false；
- 不可信 pickle checkpoint 必须 fail closed；
- 不靠改后缀把 pt 伪装为 pth；
- 本路线不做 RSL-to-RL-Games schema converter。

### 11.4 Play 和渲染

UniLab 保留：

- Hydra compose 和 env 创建；
- MuJoCo camera；
- video/frame recording；
- viser/interactive rendering；
- 通用统计和可视化外壳。

SAPG 分支替换策略加载和推理 owner：

~~~text
UniLab rendering shell
  -> SAPG pth resolver
  -> vendored PpoPlayerContinuous
  -> RlGamesNpEnvAdapter
  -> Source player action
  -> MuJoCo step and render
~~~

这里的 Source player 是目标仓库内 vendored 的 RL-Games player，不是运行时访问外部
Source 仓库。input RMS、LSTM state、done reset、block ID、action clamp/rescale 和
stochastic/deterministic decision 必须由 native player 拥有。

## 12. MuJoCoUni 的 M0-dev 和 M0-release

registry mujoco-uni-runtime 0.3.1 在真实 mixed-layout model oracle 上失败；开发阶段使用：

~~~text
mujoco-uni-runtime 0.4.0.dev0
Git URL https://github.com/lemon-star608/mujoco_uni.git
source SHA 7205e070e983df90d520f0f8593853013e976746
~~~

Code #6 已把 pyproject 和 uv.lock 固定到上述 HTTPS Git URL 与完整 SHA；安装态
`direct_url.json` 也记录同一 requested revision 和 commit id，`BatchEnvPool.was_autoreset`
为真实 public property。该 identity 只用于开发阶段，不是 M0-release artifact。

M0-dev 必须提供：

- mixed-data-layout allocation；
- per-env autoreset reporting。

如果使用 wheel/sdist，还必须记录 artifact filename、SHA256 和指向同一 source SHA 的
provenance。不得依赖 dirty sibling checkout。

M0-dev 的 cpu_ids=null 允许任务和 smoke 前进，但不能声称 CPU affinity support。

M0-release 在真实 task train/play 后完成：

- 以固定 0.4 代码线产生 clean-install 正式 artifact；
- 恢复并验证 cpu_ids/worker_cpu_ids ABI；
- 同时通过 mixed-layout、autoreset 和 affinity；
- 用正式 artifact 替换 dev dependency；
- 阻塞最终 support，但不阻塞 Code #1-#9 的开发。

MuJoCoUni owner 仓库中的生产修改需要独立普通中文 roadmap 和明确授权，本路线不自动
授权外部源码修改。

## 13. 执行和验证协议

### 13.1 每个 Code 的执行规则

1. 开始前用普通中文说明只做什么、不做什么、规模、永久成本和近风险 gate；
2. 得到当前 Code 的 execution approval；
3. 同一共享工作树任一时刻只有一个 writer；
4. 实现先建立真实 RED，再做最小 GREEN；
5. 所有 Python 命令使用 uv run；
6. 手工文件编辑使用 apply_patch；
7. fixture 只能由固定 generator 生成，普通 pytest 不 rebaseline；
8. 独立检查 scope、provenance、代码质量和 focused tests；
9. 精确 stage 当前 batch 文件并 commit；
10. post-commit fresh validation 和 clean worktree 后才能进入下一 Code。

阶段 prompt 可以在 Code #6-#10 执行期间作为独立 docs commit 保留；每个 Code 完成后，
最终事实必须同步回本文件。整条路线完成后再统一清理阶段 prompt，不重写已完成的 Code
commit 历史。

### 13.2 近风险 gate

Code #6：

- source-model direct compile；
- 12-distribution mixed-layout；
- force/torque row isolation；
- multi-substep autoreset mask；
- real pool autoreset。

Code #7：

- asset catalog、mesh references、XML compile；
- license/provenance；
- task primitive unit tests；
- T0 Source generation 和 Target replay；
- cold-path asset access。

Code #8：

- registry 和 NpEnv contract；
- 600-tool integration；
- reset/action/obs/reward/termination/DR；
- T1；
- real finite env steps。

Code #9：

- adapter fake-env ABI；
- Source owner field-by-field config；
- native Runner train/play lifecycle；
- single tracker/W&B owner；
- pth resume/weights；
- 6-env 和 N 不等于 6 player；
- real small train/play vertical slice。

Code #10：

- M0-dev S1；
- 12288/2048 profile；
- clean-install M0-release；
- dependency audit；
- make test-all；
- final-current-head remote CI。

### 13.3 Required SAPG mode

required SAPG tests 必须显式 fail closed：

~~~text
UNILAB_REQUIRE_SAPG=1
~~~

在 root sapg extra 落地前，测试使用 editable third_party distribution；extra 落地后使用
root optional extra。absence、错误 distribution、错误 import path 或 hash drift 必须
失败，不能 skip-pass。

production path 暴露前和最终 PR 前必须运行 make test-all。最终 PR 必须满足：

- final commit 完成；
- 工作树干净；
- Validation 如实记录；
- 当前 HEAD 的所有远程 CI 完成且通过；
- old-head success、pending、in-progress 或挂起 job 不算通过。

## 14. 停止条件

出现以下任一情况，停止当前 Code 并回报 maintainer：

- compatibility 或接线需要改变 SAPG tensor formula、RNG、update、AMP、checkpoint 或
  player 语义；
- adapter/env 需要调用 backend 私有方法；
- step/reset/DR 热路径需要解析 asset/XML；
- 需要在 script 中长期翻译算法 config；
- 需要新 collector、async protocol、distributed 或 export 才能工作；
- 当前 Code 引入未批准 public owner、execution path 或 support scope；
- Source/Target 或 T0/T1 出现无法解释 mismatch；
- canonical AMP、真实 MuJoCo 或 M0-release 不能实际执行；
- required tests 有 skip、failure 或未解释 warning；
- Source/vendor/M0-dev/M0-release provenance 不完整；
- 只能使用 dirty sibling checkout；
- exact resume 需要新 public env snapshot contract；
- 当前 Code 超出已批准的文件、规模或永久维护成本；
- 共享工作树出现 writer overlap。

不得通过放宽 tolerance、关闭 wrench、改变 reset profile、删除证据或扩大 scope 绕过停止
条件。

## 15. 完成和 support wording

整条路线只有在以下事实全部有当前证据时完成：

- Code #1-#10 都有范围正确的代码 commit；
- Source/vendor identity 和 frozen oracles 无未声明漂移；
- backend contracts、T0、T1 和 production vertical slice 通过；
- 真实 M0-dev train/play 和 12288/2048 profile 通过；
- M0-release 是 clean-install、身份固定的正式 artifact；
- make test-all 通过；
- final-current-head CI 全部通过；
- maintainer 明确批准正式 support。

最终允许的准确表述：

~~~text
SAPG algorithm Source-exact；task 差异仅限 manifest 中的 MuJoCo、table、tool、
resource 和 packaging mappings。
~~~

不能写成“IsaacSim 与 UniLab 的全部训练只剩物理差异”，也不能用算法 oracle 代替真实
pipeline 或产品 support 判断。

## 16. 本文件维护规则

每完成一个 Code，必须更新：

- 总进度表中的状态和完整 commit SHA；
- 实际 scope 和与计划的偏差；
- focused/full validation 命令、实际 pass/skip/fail 数；
- 新增或变化的 fixture/file anchors；
- Source、vendor、donor、MuJoCoUni 和 artifact provenance；
- 已关闭风险、剩余 blocker 和下一个 execution approval 点；
- 当前可声明和不可声明的 support 边界。

如果本文件中的历史状态与只读 Git、fixture manifest 或当前测试结果冲突，以可复验事实
为准，并在下一 docs commit 修正本文件。测试和 AI review 不能替代 maintainer 的产品
判断。
