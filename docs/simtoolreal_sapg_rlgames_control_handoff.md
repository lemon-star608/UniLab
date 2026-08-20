# SimToolReal SAPG RL-Games 迁移控制交接

> 本文面向新的控制/审查 session。它记录 2026-08-20 的 Git 实况、已批准决策、当前阻塞点和后续控制流程。算法与各批次的完整要求仍以链接的计划和执行提示词为准；若历史文档中的“当前状态”与本文或 Git 实况冲突，以 Git 实况为准。

## 1. 新控制 session 从这里开始

工作仓库不是 mature donor，而是：

```text
/home/user/ws/lemon/rlgame-unilab/UniLab
```

接管后先完整阅读：

1. 根目录 [`AGENTS.md`](../AGENTS.md)；
2. 总体计划 [`simtoolreal_sapg_source_fidelity_migration_plan.md`](simtoolreal_sapg_source_fidelity_migration_plan.md)；
3. 本文；
4. 当前 Code #4 的原始提示词 [`simtoolreal_sapg_code4_prompt.md`](simtoolreal_sapg_code4_prompt.md)、第一次返工提示词 [`simtoolreal_sapg_code4_review_rework_prompt.md`](simtoolreal_sapg_code4_review_rework_prompt.md) 和最新第二次返工提示词 [`simtoolreal_sapg_code4_review_rework2_prompt.md`](simtoolreal_sapg_code4_review_rework2_prompt.md)。

然后只读核对：

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
git diff --name-only
git diff --cached --name-only
git log --oneline --decorate -18
```

交接基准应为：

```text
branch: feat/simtoolreal-sapg-rlgames
handoff parent HEAD: 5b08333397f436feba1ad3f2376ddd96b9d2ee02
tracked diff: empty
staged: empty
untracked: exactly five Code #4 files listed in section 5
```

本文及其交接修订以 docs-only commits 提交，因此新 session 的实际 HEAD 应是 `5b083333...` 的后继，且中间只能是这些交接文档提交，而不是仍等于该 parent。若 branch、ancestor、tracked diff、staging area 或未跟踪文件范围不同，先调查差异；不得用 `stash`、`reset`、`clean`、`checkout` 或删除文件来“恢复”预期状态。

最新返工提示词自身记录的 expected HEAD 是它提交前的 `dbe5bf3a...`，这是历史锚点，不是新 session 应切回的提交。实际实现 HEAD 应包含 `dbe5bf3a...`、`5b083333...` 和最新交接 docs commits 作为 ancestors，同时仍保持 tracked/staged 为空及五文件边界。

Code #4 第二次返工已经由现有实现 session 同步执行。新控制 session 的当前动作不是再次下发 prompt，也不是产出下一份返工 prompt，而是完整理解本文与总体计划、只读核对 Git 状态，然后等待现有实现 session 的返工报告。等待期间不要编辑或检查出 Code #4 的中间态结论；收到完整交接后再按 section 7.3 独立审查。

在 Code #4 返工被独立审查、接受并提交前，不得进入 Code #5。

## 2. 最终目标与已批准架构

最终目标是在 UniLab 的 MuJoCo SimToolReal 任务上运行固定 Source fork 的 RL-Games SAPG，使 UniLab/MuJoCo 与原 IsaacSim+SAPG 训练在算法层只剩已声明的 backend、task-resource 和规模差异。这个目标目前尚未达成；现在只完成了 vendor、兼容/network oracle 和 rollout oracle。

已经确认且后续不得悄悄重议的架构决策：

- 固定 Source vendored RL-Games 是 SAPG 的唯一算法 owner；普通 pip 版 RL-Games 不是 oracle。
- 原生 `Runner -> A2CAgent` 唯一拥有 env step/reset 时机、rollout、storage、augmentation、dataset/minibatch、central/actor update、AMP、checkpoint 和 player lifecycle。
- UniLab 只提供 Hydra owner、同步 `NpEnv`/`IVecEnv` adapter、MuJoCo task/backend、run directory 与 tracker bridge。
- adapter 只转换 observation、action、done 和 info，不重写算法，不另建 collector/learner lifecycle，也不接 UniLab async runner。
- 不继续完善或迁移 RSL-RL SAPG 仿写。
- 本路线只做 MuJoCo SimToolReal SAPG；不做 PPO、Motrix、sim2sim、distributed、`torch.compile`、export 或通用 RL-Games support。
- 文档、计划和审查提示词 commit 不计入 maintainer 要求的约十个代码型 commit。

最小责任边界：

```text
Hydra SAPG owner
  ├─ UniLab task/backend/reward config -> registry.make() -> NpEnv
  └─ Source params.{algo,model,network,config}
                                      |
                                      v
                           Source RL-Games Runner
                                      |
                         A2CAgent owns rollout/update
                                      |
                                      v
                        synchronous IVecEnv adapter
                                      |
                                      v
                           SimToolRealEnv -> MuJoCo
```

“只存在后端差异”不允许修改给定相同 frozen tensors 时的网络、RNG、rollout、loss、update、AMP、checkpoint 或 player 行为。`24576/4096` 与 `12288/2048` 是显式资源规模差异；二者都保持六个 block，不能用不同规模训练曲线宣称算法等价。

## 3. 固定仓库与 provenance

| 角色 | 路径/身份 | 用途 |
|---|---|---|
| Clean target | `/home/user/ws/lemon/rlgame-unilab/UniLab` | 本路线唯一写入仓库 |
| Mature donor | `/home/user/ws/lemon/UniLab` | 参考成熟 600-tools、task/assets 和 MuJoCo 实现；不作为 SAPG owner |
| Source oracle | `/home/user/ws/lemon/simtoolreal` | 固定 IsaacSim task owner 与定制 RL-Games SAPG |

固定 Source 身份：

```text
Source HEAD:       2a9917533bfea70419ed2667a511d7238e5b3abc
RL-Games tree:     7a6a0bb090998d00565aaefa6ab9f2b3d356ace2
train owner blob:  f363d05d4a24b190b7837703b93270d8f3fe9a9c
task owner blob:   6469d46867081b70edaa589dcb31c7090b64d45e
runtime selection: 72 Python blobs, MIT license
```

Source package 的 SAPG 语义横跨 `ExperienceBuffer`、`PPODataset`、model、central value、player、checkpoint 和 runner；`torch_runner` eager import closure 已覆盖 51/72 个模块。因此选择完整 72-file Python runtime，而不是只复制几个名为 SAPG 的文件。122 个 Source YAML 没有进入 vendor。

`third_party/simtoolreal_rl_games/{README.md,UPSTREAM.md,PATCHES.md,source_manifest.json}` 是 vendor provenance 与七个兼容补丁的当前记录。任何新补丁或 identity 变化都必须重新审计，不能静默 rebaseline。

Source 与 donor 工作树可能有其他 session 的既存未跟踪文件。它们不是本任务清理对象；不得修改、删除、stash、reset 或 clean。

## 4. 已完成的代码提交与真实路线状态

| Code # | 当前状态 | Commit/预期结果 | 主要内容 |
|---:|---|---|---|
| 1 | 已提交 | `ed9c0ae5` `vendor: pin SimToolReal RL-Games runtime` | 固定 72-file Source runtime、license、manifest 和隔离审计 |
| 2 | 已提交 | `1adb159e` `fix: make RL-Games compatible and lock network fidelity` | 七个兼容补丁、dual-hash/import gate、config/network FP32 golden |
| 3 | 已提交 | `3a712a97` `test: lock SAPG rollout and RNG semantics` | rollout、GAE、augmentation、shuffle、RNG、RNN、RMS oracle |
| 4 | 未接受、未提交 | `test: lock SAPG update and AMP semantics` | 五个文件处于第二次控制审查返工 |
| 5 | 未开始；下发前需 maintainer 确认 | `test: lock SAPG checkpoint and player semantics` | checkpoint/resume boundary、6-env 与 `N != 6` player routing |
| 6 | 未开始；下发前需 maintainer 确认 | `feat(backend): add SimToolReal MuJoCo runtime contracts` | M0-dev、source-model、body-wrench、autoreset public contracts；另有 public-contract gate |
| 7 | 未开始；下发前需 maintainer 确认 | `feat(simtoolreal): add assets, task foundations, and Source oracle` | 600-tool assets、task primitives、T0；尚不注册真实 env |
| 8 | 未开始；下发前需 maintainer 确认 | `feat(simtoolreal): compose MuJoCo env and lock task parity` | 真实 env composition、registry、`NpEnv` contract、T1 |
| 9 | 未开始；下发前需 maintainer 确认 | `feat: integrate Source RL-Games SAPG runtime` | `sapg` extra、Hydra、adapter、native Runner、tracker、checkpoint/player、CLI；另有 production-path gate |
| 10 | 未开始；下发前需 maintainer 确认 | `release: promote SimToolReal SAPG support` | S1、M0-release、final lock/docs/support；另有 support-promotion gate |

Code #3、#4、#5 三个 oracle 不能 squash。Roadmap/umbrella approval 不等于后续每个 code batch 的 execution approval；完成一批后可以起草下一份 prompt，但必须先让 maintainer 看过并明确确认，才能下发实现。除此之外，Code #6、#9、#10 分别涉及新 backend public contract、production execution path 和 support promotion，还必须关闭相应的独立产品 gate。“大步推进”和约十个 commit 的目标不扩大这些授权。

相关 docs 时间线：

```text
60bc034a docs: plan source-faithful SAPG migration
846b4b49 docs: reframe SAPG migration as ten code commits
897c9c27 docs: add SAPG rollout oracle execution prompt
3a09e888 docs: add SAPG rollout oracle review rework prompt
1127d2f4 docs: add second SAPG rollout oracle rework plan
d4561cdf docs: add final SAPG rollout oracle test-hardening prompt
b17f3d86 docs: add SAPG update and AMP oracle prompt
dbe5bf3a docs: add SAPG update oracle review rework prompt
5b083333 docs: harden SAPG update evidence review
```

## 5. Code #4 当前工作树

当前只有以下五个未跟踪文件；它们是第一次返工产物，不是已接受结果：

```text
scripts/generate_simtoolreal_sapg_update_fixture.py
tests/algos/rlgames_sapg/source_update_harness.py
tests/algos/rlgames_sapg/test_update_golden.py
tests/fixtures/simtoolreal_sapg/source_update_fp32.npz
tests/fixtures/simtoolreal_sapg/source_update_manifest.json
```

2026-08-20 第二次返工开始前的实际 anchors：

```text
update NPZ SHA256:
28bb72818c27440c925742d833c54afb5100831785f7763458faf0fbebeefcce

update manifest file SHA256:
4029e8b60f664f6334c8f1ec096ea8ca9e1954d9b821e5d35daa18e29770e005

update canonical payload SHA256:
c8ad87ca2679a4a3e4d4d1a6c6386e2597c74c8d1b9db8cd40ae135d06f65ebb

fixture bytes:
74,891 + 7,147,332 = 7,222,223 bytes (< 8,388,608)
```

Code #3 的固定 anchors 不得变化：

```text
rollout NPZ:
3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8

rollout manifest:
785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f
```

旧实现报告曾声称 update manifest file hash 是 `bd8aa7fe...`，但该值与返工开始前的工作树不符；不得引用它。现有实现 session 正在修改这五个文件，因此新控制 session 看到不同的中间 hash 不构成异常，也不得要求回退；第二次返工完整交接后必须从最终文件重新计算全部 hash。

第一次返工中已有且必须保留的事实：

- normal FP32/AMP 的 identity `shuffle_batch` wrapper 各调用一次，并保持 frozen 56-row handoff 与 RNG；
- actor/central phase 均为 `(0,0)..(0,3),(1,0)..(1,3)`，batch sizes 均为 `[12,12,12,20]`；
- normal AMP 的真实 step mask 是 `[true,true,true,true,false,true,true,true]`；
- scaler sequence 是 `[65536,65536,65536,65536,32768,32768,32768,32768]`；
- recursive Source-to-Target semantics/RNG comparison 已存在；
- focused 13 tests、完整 SAPG 43 tests、vendor 37 tests 在当时通过。

这些绿灯只证明当时测试覆盖的结构；它们不能关闭下一节的 evidence completeness 缺口。

## 6. Code #4 为什么仍被拒绝

核心问题不是 Source 与 Target 已记录字典是否相等，而是两边可能同时少记关键证据。控制审查曾对 Source/Target 对称删除 71/72 个 loss/clip/optimizer events 和全部 epoch normalizer snapshots，现有 recursive validator 仍然通过，七个 `evidence_inventory` 标签也不变。

第二次返工必须关闭的风险：

1. `evidence_inventory` 只是标签，未验证字段、数量以及 case/epoch/batch coverage。
2. prepared `old_values`、`returns`、`advantages` 只在临时 trace，未进入最终 fixture。
3. scaled/unscaled gradient stages、详细 scaler transitions 也只在临时 trace。
4. 缺少每 batch native total loss、selected per-row entropy coefficient、autocast dtype、GradScaler growth tracker、normal AMP skipped batch 参数不变和 scheduler 后 optimizer LR。
5. mini-epoch 数、batch 数和 `value_before_dataset` 仍有硬编码结论，没有从 native events 推导。
6. actor/central 两个 mini-epoch 的 normalizer snapshots 不完整，不能证明第一 epoch 更新、第二 epoch 冻结。
7. FP32 numeric comparison 隐式按名字筛选，只覆盖当前 45 arrays 中的 21 个，没有显式 comparison inventory。
8. metadata-success/order 仍可由 bool 声明，缺少能杀死提前 subtraction 的 mutation test。
9. `_capture_update(..., expected_package_root)` 没有真正用该参数 fail closed。
10. generator 缺少 root/ancestor/leaf symlink 与 non-directory output 的近风险安全测试。

完整不变量、RED mutation、instrumentation 限制、fixture 预算和验证命令都在最新 [`simtoolreal_sapg_code4_review_rework2_prompt.md`](simtoolreal_sapg_code4_review_rework2_prompt.md) 中。不要把本节摘要当作可以省略该提示词的替代品。

## 7. Code #4 控制流程

### 7.1 当前 writer 与等待边界

最新 Code #4 第二次返工 prompt 已经下发，现有实现 session 是这五个文件的唯一 writer。新控制 session 不再发送同一 prompt、不启动重复实现 session、不编辑 Code #4 文件，也不在实现完成前写第三份返工 prompt。它只负责理解整体架构并等待完整返工报告。

现有实现 session 应完整读取最新提示词列出的三份 Code #4 文档。它只能修改 section 5 的五个文件，不得修改 vendor、Source、Code #3、生产代码或配置，不得增加第六个文件，不得进入 Code #5。

实现 session 不得执行 `git add`、`git commit`、`git push`、PR、`stash`、`reset`、`clean` 或切换 branch。所有 Python 命令使用 `uv run`，所有手工编辑使用 `apply_patch`。

维护者已经豁免 Code #4 约 900 行的手写规模门槛，但没有豁免五文件边界、native-delegation 要求或 fixture `< 8 MiB` gate。不得通过删除证据满足体积限制。

### 7.2 当前唯一验证矩阵

Oracle capture/replay 只验证：

```text
Python 3.11
Torch 2.7.0+cu128
canonical RTX 4090 / compute capability 8.9
```

不要重复 Python 3.10、3.12、3.13、cu126 或其他 CUDA 矩阵。总计划顶部仍写 Python 3.10–3.13，那是过时的历史口径，不是当前 prompt 要求。

FP32 frozen tensors 使用 `atol=1e-6, rtol=1e-5`；AMP 单独验证 owner path、scaler/overflow 和 semantic relation，不与 FP32 做逐元素相等声明。

### 7.3 控制 session 的独立审查

收到实现 session 的 `# DONE` 不等于接受。控制 session 至少应：

1. 读取五个文件的完整 diff/内容与最终报告，确认范围没有扩大；
2. 核对最终 fixture sizes 和从磁盘重新计算的 NPZ、manifest、payload anchors；
3. 亲自运行 lightweight evidence mutation tests，确认对称删除 Source/Target evidence 也失败；
4. 确认 prepared dataset、normalizer、loss/total loss/entropy coefficient、gradient/clip、真实 optimizer groups/delta/state、autocast/scaler/growth tracker、step mask/parameter relation、scheduler LR 都有逐 case/epoch/batch coverage；
5. 确认 mini-epoch/batch/value-before-dataset 来自 events，不是硬编码标签；
6. 确认 metadata 顺序是 inventory -> 全部 shape/dtype/content -> semantic invariants -> numeric，并有提前 subtraction 的 RED mutation；
7. 确认 explicit FP32 comparison inventory 覆盖所有承诺的 numeric arrays；
8. 确认 namespace guard 和 generator path/symlink tests fail closed；
9. 确认 instrumentation 始终 delegate 给 Source owner，没有复制 loss、AMP 或 optimizer 公式；
10. 确认 Code #3 hashes、72-file vendor identities、Source provenance 和生产代码均未变化。

fresh validation 使用最新返工提示词中的精确命令，最少包括 focused Code #4、完整 SAPG suite、vendor suite、vendor audit、scoped/root Ruff、format check、`git diff --check`。所有 required pytest 必须零 skip。此阶段不运行其他 Python/CUDA 矩阵，也不运行 `make test-all`。

若收到完整返工报告后的审查仍发现问题，先把 findings 和证据汇报 maintainer；只有 maintainer 决定继续返工后，才写新的、证据明确的返工 prompt 文档并提交，再让同一实现 session 继续。不要在控制 session 与实现 session 同时写工作树。

### 7.4 接受与提交

只有所有 Code #4 gate 都关闭后，控制 session 才能精确 stage 这五个路径：

```bash
git add -- \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json
git diff --cached --name-status
git diff --cached --check
git commit -m "test: lock SAPG update and AMP semantics"
```

提交后重新运行 Code #4 的关键 focused/full/audit gate并核对工作树。不要用 `git add .`。Code #4 接受后才可为 Code #5 编写独立 prompt；写好后先向 maintainer 汇报并等待明确 execution approval，不得直接下发。

## 8. 后续批次控制要点

依赖顺序不能调换：

```text
Code 4 update/AMP -> Code 5 checkpoint/player

Code 6 backend contracts/M0-dev
  -> Code 7 assets/task foundations/T0
  -> Code 8 real env composition/T1

Code 5 + Code 8 + Code-9 production-path approval
  -> Code 9 adapter/Hydra/native Runner/tracker/player/CLI
  -> S1 smoke on M0-dev
  -> external M0-release
  -> Code 10 dependency/support promotion
```

每一批仍遵循同一角色分工：上一批接受后，控制 session 起草 prompt并提交 docs；maintainer 明确批准该 batch 后，实现 session 才能按 prompt 写文件并交接；控制 session 独立审查、验证、精确 stage 和 commit。不得让控制或实现 session 因 umbrella 已批准而自动连做下一批。

Code #5 的重点是 Source `.pth` payload、normalizer/optimizer/RNN/env-state 的实际边界、resume 后第一组 action/value/update，以及 6-env canonical player 与 `N != 6` fallback。它仍是 oracle，不是生产 integration。

Code #5、#7、#8 和其他普通 batch 也都需要各自的 execution approval。Code #6 开始新增 backend public contracts，除 batch approval 外还必须获得 maintainer 对新 public surface 的明确确认。Code #7 机械复用 mature donor 已落地的 600 tools/task primitives，但不能复制 donor 的 RSL-RL SAPG。Code #8 才注册并组合真实 MuJoCo env。Code #9 才引入 root `sapg` extra、Hydra owner、adapter 与 native Runner 生产路径，并需要额外 production-path approval。

`make test-all` 在 production execution path 暴露前以及最终 PR 前运行；当前未接受的 Code #4 不运行它。最终 PR 还必须满足根 `AGENTS.md` 的 clean tree、final commit、`make test-all` 和远端 CI gate。

## 9. MuJoCoUni 决策

`mujoco-uni-runtime==0.3.1` 在真实 12-distribution mixed-layout model oracle 上失败：

```text
ValueError: models are not compatible: model[0] and model[7]
```

开发阶段 M0-dev 固定使用：

```text
mujoco-uni-runtime==0.4.0.dev0
Git SHA: 7205e070e983df90d520f0f8593853013e976746
```

版本字符串本身不够；将来 lock/manifest 必须记录完整 Git SHA。若使用 sdist/wheel，还要记录 artifact filename、SHA256 和指回同一提交的 provenance。不得依赖 dirty sibling checkout。

CPU affinity 不是 SimToolReal/SAPG contract。M0-dev owner 必须显式 `env.cpu_ids=null`；这允许吞吐和线程迁移统计变化，不允许改变算法或 task math，也不能宣称 affinity support。

任务、train/play 和 parity 跑通后再执行外部 M0-release：以 0.4 代码线恢复 0.3.1 的 `cpu_ids/worker_cpu_ids` ABI，产出 clean-install artifact，并同时验证 mixed-layout、autoreset 和 affinity。M0-release 阻塞最终依赖/support 晋升，不阻塞算法 oracle 或早期 env smoke。MuJoCoUni owner 仓库的生产修改需要单独 roadmap/授权，本任务不自动授权。

## 10. 当前未完成项与禁止的完成声明

截至本交接：

- Code #4 update/AMP oracle 尚未接受或提交；
- Code #5 checkpoint/player oracle 未开始；
- MuJoCo backend public contracts、600-tool task port、真实 env composition 未开始；
- `sapg` extra、Hydra owner、adapter、native Runner production path、tracker/CLI 未开始；
- 没有真实 MuJoCo train/play smoke；
- M0-release、support promotion、final audit 和 PR 未开始；
- `make test-all` 尚未运行。

因此不能声称“迁移完成”“Source/Target 全面等价”或“只剩 backend 差异”。当前有证据的口径仅是：固定 vendor、network/config oracle 和 rollout/RNG oracle已提交；update/AMP oracle仍在加强证据覆盖。

## 11. 停止并回报 maintainer 的条件

出现以下任一情况应停止当前 batch，不自行扩 scope：

- compatibility 或 adapter 需要改变 SAPG tensor公式、RNG、update、AMP、checkpoint 或 player 语义；
- 需要在脚本中长期翻译 Source算法配置；
- adapter 需要调用 backend 私有方法、解析热路径 asset/XML 或启动新 collector thread；
- Code #4 超过五文件或 8 MiB gate、需要复制算法公式，或 canonical CUDA 无法真实执行；
- oracle 出现无法解释的 Source/Target mismatch；
- 任一下一 code batch 尚未得到 maintainer 的明确 execution approval，或 Code #6/#9/#10 尚无对应 public-contract、production-path 或 support-promotion 授权；
- M0-dev 只能解析到 dirty/unversioned MuJoCoUni，或 provenance 不能固定；
- exact resume 需要新增公共 env snapshot contract；
- 基础路径必须依赖 async、distributed、Motrix、PPO、sim2sim 或 export 才能工作；
- 发现其他 session 的重叠改动，无法在不覆盖它们的情况下继续。

## 12. 新控制 session 的首轮产出

新控制 session 不需要重做已经提交的 Code #1–#3，也不要重新运行无关 Python/CUDA 矩阵。首轮只应：

1. 完整阅读根 `AGENTS.md`、总体计划、本文及三份 Code #4 prompt，理解 Source owner、adapter、MuJoCoUni 和十个 code batch 的边界；
2. 只读核对 section 1 和 section 5 的 Git 实况，确认现有实现 session 仍是唯一 writer；
3. 不下发重复 prompt、不编辑工作树、不从中间文件形成审查结论，等待现有实现 session 的完整返工报告；
4. 收到报告后独立审查 Code #4；有问题先向 maintainer 汇报 findings，由 maintainer 决定是否再返工，无问题则只提交五个 Code #4 文件；
5. Code #4 提交并验证后，向 maintainer 汇报 evidence 与 commit SHA，再起草 Code #5 prompt；必须等 maintainer 明确批准 Code #5 execution 后才能下发。

这份文档是控制状态交接，不是对 Code #4 或整条迁移已完成的声明。
