# SimToolReal SAPG RL-Games 迁移控制交接

> 本文面向接管分支的控制/审查 session，记录 2026-08-20 Code #4 clean restart 后的
> Git、授权和依赖实况。当前实现规格只有
> [simtoolreal_sapg_code4_clean_execution_prompt.md](simtoolreal_sapg_code4_clean_execution_prompt.md)；
> 历史 Code #4 规格和隔离产物都不得下发、恢复或作为 oracle 证据。

## 1. 当前真实状态

唯一工作仓库：

~~~text
/home/user/ws/lemon/rlgame-unilab/UniLab
~~~

固定分支：

~~~text
feat/simtoolreal-sapg-rlgames
~~~

固定 Git 身份和本轮 correction authoring 起点：

~~~text
lineage base:          ba16f5b490c2fcf1bf3bd81a03314b3f57d19770
correction base/HEAD:  910a4309918b1dd2fadc60c43f4250d03d84153a
worktree:              clean
staged:                empty
Code #4 implementation files: absent
~~~

correction base 是 lineage base 的 exact one-commit child；其 diff 恰为新增 clean execution
prompt 和修改本 handoff。本轮 authoritative semantic correction 从上述 clean 状态开始，
correction writer 只允许把这两个既有 docs paths 留为 unstaged modifications：

~~~text
 M docs/simtoolreal_sapg_code4_clean_execution_prompt.md
 M docs/simtoolreal_sapg_rlgames_control_handoff.md
~~~

控制 session 接收 correction 交接后先通过 section 1.1 pre-commit review gate，再独立审查、
精确 stage 这两个 M paths，并创建一笔 correction commit。Code #4 dispatch/resume HEAD
必须通过 section 1.2 post-commit dispatch gate，成为 correction base 的 exact
single-parent one-commit child，且工作树重新干净。任何第三个 changed/staged path 都不是
本交接授权范围，必须先调查，不能用 stash、reset、clean、checkout 或删除来掩盖。

Code #4 已完全 reset，仓库中没有以下实现文件：

~~~text
scripts/generate_simtoolreal_sapg_update_fixture.py
tests/algos/rlgames_sapg/source_update_harness.py
tests/algos/rlgames_sapg/test_update_golden.py
tests/fixtures/simtoolreal_sapg/source_update_fp32.npz
tests/fixtures/simtoolreal_sapg/source_update_manifest.json
~~~

Git 历史用两笔可审计 revert 删除了两份已拒绝文档：

~~~text
f1f43909e6ffaac311a589c787e35c0764e98ae6
ba16f5b490c2fcf1bf3bd81a03314b3f57d19770
~~~

旧五文件位于：

~~~text
/tmp/unilab-code4-rework-sMlAmO
~~~

该目录只证明 reset 可恢复，不是实现输入、fixture seed、artifact anchor 或 Source evidence。
不要检查其实现内容，不要复制、恢复、hash-rebaseline 或向新 agent 下发其中任何文件。

控制 session 是唯一 Git history、staging、commit 与 branch owner。correction writer 只
交回上述两份未暂存 docs；控制 session 完成本轮 exact correction commit 和 section 1.2
post-commit dispatch gate 后，才直接派出一个新的内部实现 agent。该 agent 活跃期间，它是
五个 Code #4 路径的唯一 writer，控制 session 不同时编辑。

Code #5 未批准，不能写其 prompt、派 agent 或开始实现。

### 1.1 correction commit 前 review gate

控制 session 接收 writer 交接时先运行本 gate；这是合法的两份 unstaged M 状态，不能要求
`910a4309..HEAD` count=1，也不能要求 worktree clean：

~~~bash
(
set -e
set -o pipefail
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git rev-parse 910a4309918b1dd2fadc60c43f4250d03d84153a^
git rev-list --count \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..910a4309918b1dd2fadc60c43f4250d03d84153a
git diff --name-status \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..910a4309918b1dd2fadc60c43f4250d03d84153a
git diff --name-status
git diff --name-only
git diff --cached --name-only
git status --short --branch
git status --porcelain=v1 --untracked-files=all
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
test "$(git rev-parse HEAD)" = \
  910a4309918b1dd2fadc60c43f4250d03d84153a
test "$(git rev-parse 910a4309918b1dd2fadc60c43f4250d03d84153a^)" = \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770
test "$(git rev-list --count \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..910a4309918b1dd2fadc60c43f4250d03d84153a)" \
  -eq 1
test "$(git diff --name-status \
  ba16f5b490c2fcf1bf3bd81a03314b3f57d19770..910a4309918b1dd2fadc60c43f4250d03d84153a)" = \
  "$(printf 'A\tdocs/simtoolreal_sapg_code4_clean_execution_prompt.md\nM\tdocs/simtoolreal_sapg_rlgames_control_handoff.md')"
test "$(git diff --name-status)" = \
  "$(printf 'M\tdocs/simtoolreal_sapg_code4_clean_execution_prompt.md\nM\tdocs/simtoolreal_sapg_rlgames_control_handoff.md')"
test "$(git diff --name-only)" = \
  "$(printf 'docs/simtoolreal_sapg_code4_clean_execution_prompt.md\ndocs/simtoolreal_sapg_rlgames_control_handoff.md')"
code4_staged_paths=$(git diff --cached --name-only)
test -z "$code4_staged_paths"
test "$(git status --porcelain=v1 --untracked-files=all)" = \
  "$(printf ' M docs/simtoolreal_sapg_code4_clean_execution_prompt.md\n M docs/simtoolreal_sapg_rlgames_control_handoff.md')"
)
~~~

以上 gate 必须同时证明 HEAD exactly 是 correction base、分支正确、base lineage/count/diff
正确、worktree diff exactly 是两份 unstaged M、unstaged name inventory exactly 是两份 docs、
porcelain v1（`XY PATH` 之间是单个空格）exactly 只有上述两行且没有第三个 tracked/untracked
path、staging 为空且五个 implementation files absent。任何一项不符立即 # BLOCKED；不得
用 post-commit gate 错误拒绝这一本来合法的 review state。

### 1.2 correction commit 后 dispatch gate

控制 session 精确提交两份 docs 后，才运行本 gate：

~~~bash
(
set -e
set -o pipefail
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
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
git log --oneline --decorate -16
)
~~~

post-commit dispatch gate 必须同时证明 correction base 是 HEAD ancestor、HEAD 是 correction
base 的 exact single-parent one-commit child、lineage/correction/cumulative 三层 count 与
diff 全部精确、分支正确、worktree/staging clean 且五个 implementation files absent。仅满足
ancestor、某一个 count、branch 或 clean tree 不足以授权 dispatch；任一项不精确时立即
# BLOCKED。

## 2. 接管阅读顺序与唯一执行规格

依次完整阅读：

1. 根目录 [AGENTS.md](../AGENTS.md)；
2. 总体迁移计划
   [simtoolreal_sapg_source_fidelity_migration_plan.md](simtoolreal_sapg_source_fidelity_migration_plan.md)；
3. clean 重实现计划
   [simtoolreal_sapg_code4_clean_implementation_plan.md](simtoolreal_sapg_code4_clean_implementation_plan.md)；
4. Code #4–#10 控制设计
   [simtoolreal_sapg_code4_10_autonomous_control_design.md](simtoolreal_sapg_code4_10_autonomous_control_design.md)；
5. 本文；
6. 当前唯一 Code #4 执行规格
   [simtoolreal_sapg_code4_clean_execution_prompt.md](simtoolreal_sapg_code4_clean_execution_prompt.md)。

clean implementation plan 只保留为已经完成的 reset/revert/control-planning provenance；
其中 checkbox checklist 不是接管 agent 的待执行清单，不得重做隔离、revert、初版 docs commit
或按其后续 task 直接实现。Code #4 的实现细节、命令、停止条件和交接格式只按 clean
execution prompt。

总体计划和控制设计提供架构、十批依赖与长期治理；若其中历史“当前状态”与 section 1
冲突，以 section 1 和只读 Git 实况为准。clean execution prompt 自包含五文件实现与
验收要求，是唯一可直接下发的 Code #4 prompt；不得把其他历史材料拼成增补指令。

## 3. 已批准的总体架构

最终目标是在 UniLab MuJoCo SimToolReal 任务上运行固定 Source fork 的 RL-Games SAPG。
算法层以固定 Source runtime 为唯一 oracle；UniLab 只拥有 backend/task/resource/config
边界与同步 adapter。

不可重议的 owner 分工：

- 固定 vendored Source RL-Games 是唯一 SAPG 算法 owner；普通 pip RL-Games 不是 oracle。
- 原生 Runner -> A2CAgent 唯一拥有 env step/reset 时机、rollout、storage、
  augmentation、PPODataset、central/actor update、AMP、checkpoint 和 player lifecycle。
- UniLab 只提供 Hydra owner、同步 NpEnv/IVecEnv adapter、MuJoCo task/backend、
  run directory 和 tracker bridge。
- adapter 只转换 observation、action、done 和 info；不重写算法，不新增 collector/learner
  lifecycle，不接 UniLab async runner。
- 不继续迁移或完善 RSL-RL SAPG 仿写。
- 本路线只做 MuJoCo SimToolReal SAPG；不做 PPO、Motrix、sim2sim、distributed、
  torch.compile、export 或通用 RL-Games support。

~~~text
Hydra SAPG owner
  ├─ UniLab task/backend/reward -> registry.make() -> NpEnv
  └─ native Source params.{algo,model,network,config}
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
~~~

“只剩 backend 差异”不能改变给定同一 frozen tensor 时的网络、RNG、rollout、loss、
update、AMP、checkpoint 或 player。24576/4096 与 12288/2048 是显式 resource profiles，
二者都保持六个 blocks；不能用不同规模的训练曲线代替算法 oracle。

## 4. 固定仓库与 provenance

| 角色 | 路径 | 用途 |
|---|---|---|
| Clean target | /home/user/ws/lemon/rlgame-unilab/UniLab | 本路线唯一写入仓库 |
| Mature donor | /home/user/ws/lemon/UniLab | 只参考成熟 600-tool task/assets/backend |
| Source oracle | /home/user/ws/lemon/simtoolreal | 固定 IsaacSim task owner 与定制 RL-Games |

固定 Source identity：

~~~text
Source HEAD:       2a9917533bfea70419ed2667a511d7238e5b3abc
RL-Games tree:     7a6a0bb090998d00565aaefa6ab9f2b3d356ace2
train owner blob:  f363d05d4a24b190b7837703b93270d8f3fe9a9c
task owner blob:   6469d46867081b70edaa589dcb31c7090b64d45e
runtime selection: 72 Python blobs, nested MIT license
~~~

Code #3 frozen input anchors：

~~~text
rollout NPZ:
3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8

rollout manifest:
785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f
~~~

Source package 的 SAPG 语义横跨 ExperienceBuffer、PPODataset、model、central value、
player、checkpoint 和 runner；因此保留完整 72-file Python runtime，不从几个同名文件
重写算法。122 个 Source 示例 YAML 没有进入 vendor。

third_party/simtoolreal_rl_games 下的 README、UPSTREAM、PATCHES 和 source_manifest 是
当前 vendor provenance 与兼容补丁记录。任何新 patch/identity 变化必须重新审计，不能
静默 rebaseline。Source/donor 可能有其他 session 的既存文件；不得修改、删除、stash、
reset 或 clean。

## 5. Code #1–#4 实况与 baseline

| Code # | 状态 | Commit/目标 |
|---:|---|---|
| 1 | 已提交 | ed9c0ae5 vendor: pin SimToolReal RL-Games runtime |
| 2 | 已提交 | 1adb159e fix: make RL-Games compatible and lock network fidelity |
| 3 | 已提交 | 3a712a97 test: lock SAPG rollout and RNG semantics |
| 4 | reset，尚无实现文件或代码 commit | test: lock SAPG update and AMP semantics |

Code #1–#3 的已记录 baseline：

~~~text
required SAPG oracle suite: 30 passed
vendor suite:               37 passed
vendor audit:               passed；72 selected Python blobs verified
~~~

这些数字是 clean Code #4 开始前的已提交 baseline，不是 Code #4 的预期 test count。
新实现不得把它们硬编码成 Code #4 完成条件；Code #4 报告必须使用本次 fresh commands
得到的实际 counts，并要求 required tests 0 skip。

Code #4 的 clean fixture 必须重新从固定 Source capture。最终 update NPZ、manifest file、
canonical payload hashes 和 normal AMP step/scaler facts 当前都未知；不得从任何旧产物
继承、猜测或写入 handoff。

截至本交接还没有：

- update/AMP oracle code commit；
- checkpoint/player oracle；
- MuJoCo backend public contracts 或真实 600-tool env；
- sapg optional extra、Hydra owner、adapter 或 native Runner production path；
- 真实 train/play smoke；
- M0-release、support promotion、final PR；
- 当前路线的 make test-all 最终 gate。

因此不能宣称迁移完成、Source/Target 全面等价或只剩物理差异。

## 6. Code #4 clean 控制流程

### 6.1 docs contract

控制 session 从 clean correction base 接收本 handoff 与 clean execution prompt 的两份
authoritative semantic corrections，先完整通过 section 1.1 pre-commit review gate；独立
审阅后只 stage 这两个已跟踪 docs paths，运行 staged diff gate并提交 exact 一笔 correction
commit。correction writer 不拥有 Git，也不提交。提交后必须完整通过 section 1.2
post-commit dispatch gate 的 topology、single-parent、三层 diff、clean/staging 和五文件
absence assertions，才能 dispatch/resume Code #4。

### 6.2 新实现 agent

correction commit 完成且工作树干净后，控制 session 直接把 clean execution prompt 原文交给
一个新的内部实现 agent。该 agent：

- 只新建 prompt 列出的五个 regular files；
- 所有手工编辑和 mutation恢复使用 apply_patch；
- 所有 Python 命令使用 uv run；
- Source 和 Target 使用独立进程与 namespace；
- 不执行 git add/commit/push/PR、stash/reset/clean/checkout/branch；
- 不运行 make test-all；
- 不进入 Code #5。

实现 agent 的 # DONE 只代表停止写入和交回证据，不代表接受。

### 6.3 独立审查

收到新 agent 的完整交接后，控制 session 至少：

1. 读取三个文本文件完整内容，检查五文件 scope；
2. 从磁盘重新计算 update NPZ/manifest/payload anchors 和总 bytes；
3. 核对固定 Source Git objects、loaded modules、Target distribution 和 Code #3 hashes；
4. 确认 native Runner/A2CAgent/PPODataset/CentralValueTrain/loss/optimizer/GradScaler 是
   唯一 owner，所有 instrumentation delegate 且 finally恢复；
5. 确认 Source 和 Target 分别先通过 evidence invariants，对称删除也会失败；
6. 确认 56 rows、14 sequences、identity shuffle、central-before-actor、actor/central
   两 epoch 与 [12,12,12,20] 都由 events推导；
7. 确认 prepared fields，以及四 RMS 的 role→object mapping、process-local alias proof、
   owner-specific training transitions/forward-update events、mean/var/count snapshots；特别
   拒绝 central input epoch 1 伪 freeze、actor input epoch 1 伪继续 update、伪造 central
   value alias，或给 actor-model value RMS 编造 update；同时确认每 batch native
   loss/reference、gradient、actual optimizer groups/state/delta 和 scheduler 后 LR 全覆盖；
8. 确认 normal_fp32、normal_amp、overflow_amp 是本次 canonical Source capture 的事实，
   AMP step mask/scaler 没有从旧结果硬编码；
9. 确认完整 NumPy/Torch CPU/CUDA RNG、explicit FP32 comparison inventory、
   metadata-before-numeric token、wrong namespace 和 generator path safety；
10. 逐项复核全部 mutation RED，并 fresh 运行 prompt 中的 focused/full SAPG、vendor、
    audit、Ruff、format 和 git diff gates；required pytest 必须 0 skip。

绿色测试不能替代 evidence completeness 审查。若有 finding，控制 session 只把具体、
有证据的修正交回同一个新实现 agent；同一时刻仍只有一个 writer。未关闭 finding 时
不得 stage。

### 6.4 接受与提交

所有 Code #4 gates 关闭后，只有控制 session 可以：

~~~bash
git add -- \
  scripts/generate_simtoolreal_sapg_update_fixture.py \
  tests/algos/rlgames_sapg/source_update_harness.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  tests/fixtures/simtoolreal_sapg/source_update_fp32.npz \
  tests/fixtures/simtoolreal_sapg/source_update_manifest.json
git diff --cached --name-status
git diff --cached --check
git commit -m "test: lock SAPG update and AMP semantics"
~~~

必须看到 exactly five added paths。提交后控制 session 复跑 Code #4 key gates并确认干净
工作树。本批不运行 make test-all。Code #4 接受不自动批准 Code #5。

## 7. 十批路线与不可调换依赖

当前代码路线仍是十个主要结果：

| Code # | 主要结果 | 当前授权状态 |
|---:|---|---|
| 1 | 固定 72-file Source runtime | 完成 |
| 2 | compatibility、dual-hash、network/config oracle | 完成 |
| 3 | rollout/GAE/augmentation/shuffle/RNG oracle | 完成 |
| 4 | update/AMP oracle | clean restart 当前批 |
| 5 | checkpoint/resume/player oracle | 未批准、未开始 |
| 6 | M0-dev、source-model、wrench、autoreset public contracts | 未开始；执行前单独确认 |
| 7 | 600-tool assets、task foundations、T0 | 未开始；执行前单独确认 |
| 8 | real MuJoCo env composition、NpEnv contract、T1 | 未开始；执行前单独确认 |
| 9 | sapg extra、Hydra、adapter、native Runner/tracker/CLI | 未开始；执行前单独确认 |
| 10 | S1、M0-release、dependency/support promotion | 未开始；执行前单独确认 |

依赖：

~~~text
Code 2 compatibility/network
  -> Code 3 rollout/RNG
  -> Code 4 update/AMP
  -> Code 5 checkpoint/player

Code 6 M0-dev/backend contracts
  -> Code 7 assets/task foundations/T0
  -> Code 8 real env composition/T1

Code 5 + Code 8 + Code-9 execution approval
  -> Code 9 adapter/Hydra/native Runner/tracker/player/CLI
  -> S1 small smoke -> 12288/2048 profile on M0-dev
  -> external M0-release
  -> Code 10 dependency/support promotion
  -> final make test-all / PR current-head CI
~~~

Code #3/#4/#5 三个算法 oracle 不能 squash。每个 batch 只有一个主要结果、独立 prompt、
实现交接、控制审查、fresh gates 和代码 commit。Roadmap、autonomous control design 或
“约十个 commits”不等于下一 batch execution approval。

Code #6 public contracts、Code #9 production path 和 Code #10 support promotion 的产品
方向已在总体设计中确认，但具体 batch 仍必须在开始前用普通中文说明只做什么、不做什么、
规模和永久成本，并得到明确 execution approval；新增未列明 public surface/owner/scope
仍须再次确认。Code #5 当前连普通 batch approval 都没有。

## 8. MuJoCoUni M0-dev 与 M0-release

registry mujoco-uni-runtime==0.3.1 在真实 12-distribution mixed-layout oracle 上失败：

~~~text
ValueError: models are not compatible: model[0] and model[7]
~~~

开发阶段 M0-dev 固定：

~~~text
mujoco-uni-runtime==0.4.0.dev0
Git SHA: 7205e070e983df90d520f0f8593853013e976746
~~~

版本字符串本身不能证明身份。未来 lock/manifest 必须记录完整 Git SHA；若使用
sdist/wheel，还必须记录 artifact filename、SHA256 和指回同一 source commit 的 provenance。
不得依赖 dirty sibling checkout。

M0-dev 必须提供 mixed-data-layout allocation 和真实 per-env autoreset surface。CPU
affinity 不是 SimToolReal/SAPG contract；M0-dev SAPG owner 必须显式 env.cpu_ids=null。
它允许吞吐、线程迁移和性能方差变化，不允许改变 frozen algorithm/task math，也不能
宣称 affinity support。

真实 task、train/play 与 parity 跑通后，外部 M0-release 才在 0.4 代码线上恢复
cpu_ids/worker_cpu_ids ABI，产出 clean-install artifact，并组合验证 mixed-layout、
autoreset 和 affinity。M0-release 阻塞最终 dependency/support promotion，不阻塞算法
oracles 或早期 M0-dev smoke。MuJoCoUni owner 仓库的生产修改仍需要单独普通中文 roadmap
与明确授权；本 UniLab 路线不自动授权外部源码改动。

## 9. 后续治理、验证与 PR gate

每批严格串行：

1. 控制 session 写并审查 batch 规格；
2. maintainer 给出该 batch execution approval；
3. 一个实现 agent 独占声明路径，先 RED 后 GREEN；
4. 实现 agent 停止写入并交回完整证据；
5. 控制 session 独立 scope/spec/quality/provenance review 和 fresh validation；
6. finding 回到同一 agent，关闭后重跑 gates；
7. 控制 session 精确 stage、commit、post-commit验证；
8. 回到下一个人工批准点。

子 agent 可做互不依赖的只读审查，但不能代替控制 session，也不能与 writer 并发修改共享
工作树。所有 Python 命令使用 uv run；fixture 只从固定 Source 显式生成，ordinary pytest
不 rebaseline；required tests 0 skip。

production path 暴露前和最终 PR 前运行 make test-all。最终 PR 还必须满足根 AGENTS.md：

- 最终 commit 已完成；
- 工作树干净；
- make test-all 通过或有 maintainer 明确 override；
- PR body 如实记录 Validation；
- 创建/更新 PR 后，按当前 head SHA 等全部远端 CI 完成并通过；
- old-head success、pending、in_progress 或挂起 job 都不算完成。

测试、AI review 和 gate 不能代替 maintainer 的产品判断。只有 Code #4–#10 全部接受、
M0-release clean-install artifact 身份固定、真实 train/play/profile 与 make test-all
通过、support claim 获明确批准、final-head CI 全绿后，才能报告整条路线完成。

## 10. 停止并回报 maintainer

出现任一情况停止当前 batch，不自行扩大 scope：

- 需要改变 SAPG tensor formula、RNG、update、AMP、checkpoint 或 player 语义；
- 需要在 script 长期翻译算法配置、绕开 native Runner 或新增 collector protocol；
- adapter/env 需要调用 backend 私有能力或在热路径解析 asset/XML；
- 当前 batch 超出批准 files/fixture/owner 边界或引入未批准 public contract；
- Source/Target 或 T0/T1 出现无法解释 mismatch；
- Code #4 canonical CUDA、后续真实 MuJoCo 或 M0-release 不能实际执行；
- required test 有 skip/failure、provenance 不完整或 evidence coverage不足；
- M0-dev 只能来自 dirty/unversioned sibling；
- exact resume 需要新 public env snapshot contract；
- Code #5 或其他下一 batch 尚无明确 execution approval；
- 发现 writer overlap，无法在不覆盖他人改动的情况下继续。

本 handoff 只授权控制 session 完成两份 clean docs 的审查/提交，并按唯一 clean prompt
调度 Code #4。它不批准 Code #5，也不是 Code #4 或整条迁移已经完成的声明。
