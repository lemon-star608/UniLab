# SimToolReal SAPG Code #10 执行 Prompt

> 本文件交给一个新的实现 session。用户明确说“看
> docs/simtoolreal_sapg_code10_prompt.md，按文档执行 Code #10 的 child batches”时，
> 表示当前 session 获得了本文件所列 Target 仓库范围的 execution approval。实现 session
> 必须直接执行已批准的顺序，不派 subagent、不重新规划、不进入 Code #11。完成后保留
> 改动未暂存、未提交，交回控制/审查 session。

## 1. 本批唯一结果

把 Code #9 已验证的 M0-dev native RL-Games SAPG vertical slice 晋升为可复验、可维护的
M0-dev provisional 路径。当前 Code #10 不升 MuJoCoUni 版本、不制作正式 sdist，也不把
dev identity 宣称为正式 release；M0-release 作为后续独立 release 任务保留。
完成当前 dev candidate 必须同时满足：

1. 当前 M0-dev runtime 上有超过 Code #9 tiny slice 的有限多 epoch S1 train/play 证据；
2. 12288/2048/6 的真实 native profile 至少完成一个有限 epoch、checkpoint 和 cleanup；
3. UniLab 的 mujoco extra 和 uv.lock 指向已核验的 rebased M0-dev Git SHA，不解析
   dirty sibling checkout；
4. mixed-layout、per-env autoreset 和 CPU affinity 组合近风险回归通过；
5. focused/full local gates、M0-dev 文档和控制 session handoff 证据齐全；
6. 报告明确写出正式 M0-release、artifact promotion 和 maintainer support judgment
   尚未完成。

唯一允许的生产数据流仍然是：

~~~text
Hydra rlgames_sapg owner
  -> registry.make("SimToolReal", "mujoco")
  -> RlGamesNpEnvAdapter
  -> vendored Runner/A2CAgent
  -> native .pth
  -> vendored PpoPlayerContinuous
  -> UniLab MuJoCo playback/video shell
~~~

本批不得改变 Code #1-#9 已锁定的 SAPG tensor、RNG、rollout/update、AMP、checkpoint、
player、task formula、reset、reward、observation、action 或 shared sim2sim 语义。

实现 session 不执行 git add、git commit、git push、PR、stash、reset、clean、
checkout 或切分支。控制 session 在审查后提交代码，运行提交后的验证，更新总指导并
处理 post-commit local/remote CI；正式 release/artifact promotion/support judgment 由后续
release 任务决定。

## 2. 普通中文范围、规模和 child batches

### 2.1 只做什么

- 用当前 M0-dev identity 运行真实 S1 train/play，证明 native path 可连续有限运行；
- 运行真实 mujoco_12k owner，证明资源 profile、六 blocks、checkpoint 和 cleanup；
- 对当前 M0-dev Git SHA 做 provenance、安装态和 public ABI 检查；
- 保持版本为 0.4.0.dev0，不做正式 artifact promotion；
- 组合验证 mixed model layout、autoreset latch 和 CPU affinity；
- 更新 M0-dev 安装说明和 provisional support 记录，不把它写成正式 support；
- 运行仓库规定的完整 local gates，留下可供控制 session 提交后复核的证据。

### 2.2 明确不做什么

- 不修改 third_party/simtoolreal_rl_games/**、Source checkout、donor、Code #1-#8
  fixtures/harnesses、SimToolReal task formula 或 shared sim2sim owner；
- 不重新实现 rollout、collector、learner、GAE、augmentation、PPO loss、AMP、optimizer、
  checkpoint schema、player forward 或 action formula；
- 不把 0.4.0.dev0 重新命名为 release，不伪造 artifact、hash、provenance 或 ABI；
- 不修改外部 /home/user/ws/lemon/mujoco_uni 仓库，不改版本、不打 tag、不发布 PyPI、
  不清理其用户已有的 dirty files；
- 不用 /home/user/ws/lemon/mujoco_uni sibling checkout 作为安装依赖或测试替身；
- 不通过 pytest.skip、importorskip、mock backend、放宽 tolerance、降低 profile、
  关闭 wrench/autoreset 或删除失败证据来绕过 gate；当前 M0-dev 不支持的能力必须明确
  记录为 unsupported，不得伪装成通过；
- 不接 Motrix、IsaacSim、async、distributed、multi-GPU、ROCm、export、torch.compile、
  通用 RL-Games support 或新的 public backend contract；
- 不运行 PR 流程；没有当前提交的 remote CI 时，不声称正式 support 已完成。

### 2.3 规模和永久维护成本

Code #10 是一个 M0-dev support-preparation umbrella，按以下四个 child 顺序执行。每个 child 必须保持
约 15 个以内 touched paths、约 800 行以内净手写改动，并在下一个 child 前单独验证。
测试/manifest 可以新增，生产 owner 只能在真实失败证明需要时做最小 owner-level 修正。

永久成本限制为：

- 一个固定 rebased M0-dev dependency 和一份 source/provenance manifest；
- M0-dev contract tests 和组合 affinity regression；
- 安装/support 文档中的一条 provisional SAPG entry；
- 不新增第二套训练 runtime、release wrapper 或 backend abstraction。

### 2.4 Child batches

必须按顺序执行以下 child；前一个 child 的 blocker 不得被下一个 child 绕过：

~~~text
10A  rebased M0-dev remote provenance、dependency pin、lock 和安装态审计
10B  M0-dev S1 finite multi-epoch train/play 与真实 12288/2048 profile
10C  mixed-layout/autoreset/affinity 组合近风险回归
10D  dev-only docs、完整 local gates 和 control handoff
~~~

10A 必须先让 Target 从远端 Git URL 安装当前 rebased dev identity；不得让后续 train/test
继续使用旧 `7205e070…` 安装态。10B/10C 才在新 pin 上运行；affinity ABI 必须真实通过，
不能 skip。10A 不要求 sdist 或版本号变化。
正式 M0-release、PyPI/artifact promotion 和跨平台 support 仍明确延期，不属于本批。

## 3. 必读内容、起点和固定身份

### 3.1 开始前完整阅读

~~~text
AGENTS.md
docs/simtoolreal_sapg_source_fidelity_migration_plan.md
docs/simtoolreal_sapg_code10_prompt.md
docs/simtoolreal_sapg_code9_prompt.md
pyproject.toml
uv.lock
Makefile
conf/rlgames_sapg/config.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco_12k.yaml
scripts/train_rlgames_sapg.py
src/unilab/algos/torch/rlgames_sapg/
src/unilab/base/backend/mujoco/backend.py
src/unilab/base/backend/mujoco/playback.py
tests/base/backend/test_mujoco_uni_runtime_contract.py
tests/base/backend/test_mujoco_cpu_affinity_wiring.py
tests/envs/manipulation/simtoolreal/test_env_integration.py
tests/algos/rlgames_sapg/test_real_vertical_slice.py
tests/algos/rlgames_sapg/test_dependency.py
tests/scripts/test_support_matrix.py
src/unilab/utils/support_matrix.py
scripts/generate_support_matrix.py
docs/sphinx/source/zh_CN/1-getting_started/2-installation.md
docs/sphinx/source/en/1-getting_started/2-installation.md
docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md
docs/sphinx/source/en/5-reference/5-support_matrix.md
~~~

外部仓库只读核查以下内容，不把它当作 Target 写入范围：

~~~text
/home/user/ws/lemon/mujoco_uni/pyproject.toml
/home/user/ws/lemon/mujoco_uni/.github/workflows/release.yml
/home/user/ws/lemon/mujoco_uni/src/mujoco_uni/metadata.py
/home/user/ws/lemon/mujoco_uni/src/mujoco_uni/runtime/batch.py
/home/user/ws/lemon/mujoco_uni/tests/test_cpu_affinity.py
/home/user/ws/lemon/mujoco_uni/tests/test_batch_env.py
~~~

### 3.2 Target lineage preflight

本 prompt 的父 docs baseline 是：

~~~text
e007c3c036e14a464acef470a099a178ee4cf4c8
docs: align SAPG Code 10 with rebased M0-dev runtime
~~~

开始时运行以下只读检查。预期本次改口径 commit 是该 baseline 的一个直接 docs-only
child；若条件不成立，返回 # BLOCKED，不要清理或覆盖现有改动。

~~~bash
set -e
set -o pipefail
CODE10_BASE=e007c3c036e14a464acef470a099a178ee4cf4c8
test "$(git rev-parse --abbrev-ref HEAD)" = "feat/simtoolreal-sapg-rlgames"
git merge-base --is-ancestor "$CODE10_BASE" HEAD
test "$(git rev-list --count "$CODE10_BASE"..HEAD)" -eq 1
test "$(git diff --name-status "$CODE10_BASE"..HEAD)" = \
  $'M\tdocs/simtoolreal_sapg_code10_prompt.md\nM\tdocs/simtoolreal_sapg_source_fidelity_migration_plan.md'
test -z "$(git status --short)"
test -z "$(git diff --cached --name-only)"
git log -3 --oneline
git status --short --branch
~~~

记录外部仓库的 branch、HEAD 和 dirty files 仅用于 provenance 报告。外部任何 untracked
文件（包括出现时的 MUJOCO_LOG.TXT 和 docs/unilab_extension_status.md）均属于用户，禁止
清理、reset、checkout 或改写。

### 3.3 固定 algorithm/runtime identity

~~~text
Source reference commit:
  2a9917533bfea70419ed2667a511d7238e5b3abc
RL-Games parent tree:
  7a6a0bb090998d00565aaefa6ab9f2b3d356ace2
vendored distribution:
  unilab-simtoolreal-rl-games==1.6.1+simtoolreal.2a991753.compat2
vendor manifest SHA256:
  4f1170b222e4ba008b34070fad7aeaba4cf790cc6ae1917417ee40ef35573ac9
vendor selection SHA256:
  f0517fb198dbbf9dcc456ab6de4a5cf6e0c4b03cdc90e84f12e52f74a70fe0ca
canonical Torch:
  torch==2.7.0+cu128
  CUDA build 12.8
  cuDNN 90701
canonical SAPG validation:
  Python 3.11.15
  Linux x86_64 / NVIDIA RTX 4090

The M0-dev manifest's `build_runtime.python=3.13.15` is external MuJoCoUni
build/test provenance only. It is not the Target SAPG oracle platform and must
not replace the Python 3.11.15 validation identity above.
~~~

Code #10 不重新生成 Code #1-#5 fixtures。UNILAB_REQUIRE_SAPG=1 仍必须显式设置；
错误 distribution、错误 import path、vendor hash drift 或 canonical platform drift
必须 fail closed。

### 3.4 M0-dev identity

10A 开始前核验并记录：

~~~text
mujoco-uni-runtime==0.4.0.dev0
URL: https://github.com/lemon-star608/mujoco_uni.git
source SHA: 54a2197be5b0cd65e9d71ff884d8415191925136
source tree: 771de554330b698bc12e5110682af1d8de433ee2
BatchEnvPool.was_autoreset: real property
BatchEnvPool.__init__: accepts cpu_ids
BatchEnvPool.worker_cpu_ids(): public method
~~~

该 SHA 是 rebase 后的 M0-dev source identity，版本仍保持 0.4.0.dev0。它包含 affinity
基线提交 cf0b759、7d888ed，以及 rebase 后的 geom/autoreset/mixed-layout/mocap commits
614f26f、996004a、194ada6、54a2197。旧的 7205e070 SHA 不再是当前 lock 目标，不能把
“旧 SHA 的安装态”当作新 ABI 证据。

当前 M0-dev candidate 要求 mixed-layout、per-env autoreset 和 CPU affinity 都真实通过；
不允许用 null-only path 或 skip 代替 affinity。正式 M0-release 仍然不在本 Code 完成。

控制 session 已在本地 tracked-clean HEAD 上只读核验 targeted affinity/batch tests 为
26 passed、完整 MuJoCoUni suite 为 50 passed。实现 session 必须重新核验外部 local HEAD，
并确认新 SHA 已从正式 HTTPS Git URL 可获取。建议固定检查 intended branch ref：

~~~bash
set -e
set -o pipefail
M0_SHA=54a2197be5b0cd65e9d71ff884d8415191925136
M0_URL=https://github.com/lemon-star608/mujoco_uni.git
M0_REF=refs/heads/feat/geom-size-pos-per-env-fields
test "$(git -C /home/user/ws/lemon/mujoco_uni rev-parse HEAD)" = "$M0_SHA"
test -z "$(git -C /home/user/ws/lemon/mujoco_uni status --short --untracked-files=no)"
test "$(git ls-remote "$M0_URL" "$M0_REF" | cut -f1)" = "$M0_SHA"
uv run --project /home/user/ws/lemon/mujoco_uni pytest \
  /home/user/ws/lemon/mujoco_uni/tests/test_cpu_affinity.py \
  /home/user/ws/lemon/mujoco_uni/tests/test_batch_env.py -q
uv run --project /home/user/ws/lemon/mujoco_uni pytest \
  /home/user/ws/lemon/mujoco_uni/tests -q
~~~

若 remote ref 还不是该 SHA，立即 # BLOCKED 并请 maintainer 发布 rebased commit；不得改用
local path、editable sibling 或旧 remote SHA。外部 untracked 用户文件不影响 tracked-clean
判定，也不得清理。

### 3.5 Deferred M0-release identity

以下是未来独立 release 任务的要求，不是当前 M0-dev candidate 的 blocker。正式 artifact
仍必须由 MuJoCoUni maintainer 在独立、干净的 0.4 release 工作流中提供；实现 session
不得修改或发布外部代码：

- distribution 名为 mujoco-uni-runtime；
- release version 为明确的 0.4.0，不是 .dev0、local version 或 editable install；
- artifact 是 release workflow 允许的 sdist（当前 workflow 不发布 wheel）；
- 提供 artifact filename、SHA256、构建用 MuJoCo 版本、Python/platform、构建 source
  commit 和 source tree/provenance；
- 构建 source commit 必须有独立 provenance，并明确包含 mixed-layout、per-env autoreset、
  cpu_ids constructor ABI 和
  worker_cpu_ids() public method；
- 安装后的 BatchEnvPool.was_autoreset 是真实 property，不是动态 fallback；
- clean-install 不解析或依赖 /home/user/ws/lemon/mujoco_uni sibling checkout。

未来 release 任务才通过环境变量传入 artifact：

~~~bash
test -n "$MUJOCO_UNI_RELEASE_ARTIFACT"
test -f "$MUJOCO_UNI_RELEASE_ARTIFACT"
sha256sum "$MUJOCO_UNI_RELEASE_ARTIFACT"
~~~

当前 Code #10 不要求这些 release 环境变量；正式 release 任务中变量缺失、文件不存在、
hash/provenance/ABI 任一项无法核验时，才应 # BLOCKED。

## 4. 允许写入的 Target 路径

实现 session 只能写 Target 仓库。以下是每个 child 的预批准路径边界；新建文件必须
落在列出的目录，并在报告中列出精确路径和原因。

### 4.1 10A

~~~text
pyproject.toml
uv.lock
tests/algos/rlgames_sapg/test_dependency.py
tests/base/backend/test_mujoco_uni_runtime_contract.py
tests/fixtures/simtoolreal_sapg/m0_dev_manifest.json
~~~

只允许把 M0-dev dependency 从旧的 7205e070 SHA 更新到
54a2197be5b0cd65e9d71ff884d8415191925136，并记录 remote/source/lock/安装态 provenance。
禁止提交 local path、editable source、临时 index、未审查的 URL 或 sibling checkout。

### 4.2 10B

~~~text
tests/algos/rlgames_sapg/**
tests/fixtures/simtoolreal_sapg/m0_dev_*.json
~~~

优先复用既有真实 CLI 和 vertical-slice harness，只在需要保存/校验 S1 与 12k evidence
时新增窄测试或 manifest 字段；不改 production owner、vendor 或 Code #9 semantics。

### 4.3 10C

~~~text
tests/base/backend/test_mujoco_cpu_affinity_wiring.py
tests/envs/manipulation/simtoolreal/test_m0_dev_matrix.py
tests/fixtures/simtoolreal_sapg/m0_dev_*.json
~~~

允许为近风险测试增加一个组合 harness，并把既有 affinity test 从旧 ABI 下的 module skip
转为新 pin 下真实收集/GREEN；不得修改 backend public contract 或把 cpu_ids 变成默认
生产配置。

### 4.4 10D

~~~text
src/unilab/utils/support_matrix.py
tests/scripts/test_support_matrix.py
scripts/generate_support_matrix.py
docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md
docs/sphinx/source/en/5-reference/5-support_matrix.md
docs/sphinx/source/zh_CN/1-getting_started/2-installation.md
docs/sphinx/source/en/1-getting_started/2-installation.md
tests/base/backend/**
tests/algos/rlgames_sapg/**
tests/envs/manipulation/simtoolreal/**
~~~

10D 不修改总指导文档；控制 session 审查后单独更新
docs/simtoolreal_sapg_source_fidelity_migration_plan.md。安装文档可以记录
rlgames_sapg / SimToolReal / mujoco 的 M0-dev provisional 状态，但 support matrix 不得
把它标为正式 Recommended、Benchmarked 或跨平台 release support。

## 5. Child 10A：rebased M0-dev dependency pin 和安装态审计

### 5.1 目标

先把 Target 的 mujoco extra 和 uv.lock 从旧 `7205e070…` 更新到当前 rebased M0-dev
`54a2197…`。版本必须继续是 `0.4.0.dev0`；本 child 不制作 sdist、不改版本号、不修改
外部仓库。第 3.4 节的 remote ref gate 未通过时不得写 Target dependency。

### 5.2 Target dependency update

1. 将 pyproject.toml 中的 Git revision 更新为
   `54a2197be5b0cd65e9d71ff884d8415191925136`，保留 exact dev version；
2. 用 `uv lock` 重新解析 lock，只接受同一 HTTPS URL 和完整 SHA，不引入无关升级；
3. 用 `uv sync --extra mujoco --extra rlgames-sapg` 构建并安装 Target 环境；
4. 更新 dependency/runtime contract tests 的 expected source SHA；
5. 写入 reviewed、deterministic 的
   `tests/fixtures/simtoolreal_sapg/m0_dev_manifest.json`，至少包含：

~~~text
schema
distribution/version
source URL/ref/commit/tree
rebase mapping and affinity commits
build/runtime mujoco version
python/platform
external focused/full test evidence
mixed-layout and was_autoreset evidence
cpu_ids constructor and worker_cpu_ids evidence
target pyproject/uv.lock/direct_url identity
formal_release = deferred
~~~

manifest 不写临时目录、运行时间或 sibling path；普通 pytest 只校验它，不重写它。

### 5.3 安装态 gate

至少运行：

~~~bash
set -e
set -o pipefail
uv lock --check
uv sync --extra mujoco --extra rlgames-sapg
uv run --extra mujoco --extra rlgames-sapg python - <<'PY'
import importlib.metadata as metadata
import inspect
import json

from mujoco_uni.batch_env import BatchEnvPool

dist = metadata.distribution("mujoco-uni-runtime")
direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
assert dist.version == "0.4.0.dev0"
assert "cpu_ids" in inspect.signature(BatchEnvPool).parameters
assert callable(BatchEnvPool.worker_cpu_ids)
assert isinstance(BatchEnvPool.was_autoreset, property)
print(json.dumps(direct_url, sort_keys=True))
PY
uv run --extra mujoco --extra rlgames-sapg pytest \
  tests/base/backend/test_mujoco_uni_runtime_contract.py \
  tests/algos/rlgames_sapg/test_dependency.py -q
~~~

检查 stdout 中 `direct_url.json` 的 URL、requested revision 和 commit id 均是固定新 SHA，
且不含 `/home/user/ws/lemon/mujoco_uni`。`uv.lock` 只允许该 dependency 的预期 source 变化。
如果 remote 不可获取、安装态仍是旧 SHA、任一 ABI 缺失或必须使用 sibling checkout，返回
`# BLOCKED`；不要退回旧 pin，也不要把版本号改成 0.4.0。

## 6. Child 10B：M0-dev S1 和真实 12k profile

### 6.1 S1 finite multi-epoch train/play

证明新 pin 上的 native Runner/A2CAgent path 能连续完成有限多 epoch、保存 native `.pth`、
exact-once 关闭 env/tracker/writer，再由 native player 生成视频。S1 只用 Hydra overrides，
不新增 production YAML：

~~~text
algo.num_envs=6
rl_games.params.config.expl_coef_block_size=1
rl_games.params.config.horizon_length=4
rl_games.params.config.seq_length=4
rl_games.params.config.minibatch_size=12
rl_games.params.config.central_value_config.minibatch_size=12
rl_games.params.config.mini_epochs=1
rl_games.params.config.central_value_config.mini_epochs=1
rl_games.params.config.max_epochs=2
rl_games.params.config.save_frequency=1
training.play_env_num=6
training.play_steps=8
training.device=cuda:0
+env.cpu_ids=null
~~~

在 repo 外创建 log root，训练和 eval 都使用公开 CLI：

~~~bash
set -e
set -o pipefail
S1_ROOT="$(mktemp -d)"
export UNILAB_REQUIRE_SAPG=1
uv run --extra mujoco --extra rlgames-sapg train \
  --algo rlgames_sapg --task simtoolreal --sim mujoco \
  algo.num_envs=6 \
  rl_games.params.config.expl_coef_block_size=1 \
  rl_games.params.config.horizon_length=4 \
  rl_games.params.config.seq_length=4 \
  rl_games.params.config.minibatch_size=12 \
  rl_games.params.config.central_value_config.minibatch_size=12 \
  rl_games.params.config.mini_epochs=1 \
  rl_games.params.config.central_value_config.mini_epochs=1 \
  rl_games.params.config.max_epochs=2 \
  rl_games.params.config.save_frequency=1 \
  training.device=cuda:0 training.no_play=true training.log_root="$S1_ROOT" \
  +env.cpu_ids=null
S1_TASK_ROOT="$S1_ROOT/SimToolReal"
S1_RUN_PATH="$(find "$S1_TASK_ROOT" -mindepth 1 -maxdepth 1 -type d -name '0_*' -print | sort | tail -n 1)"
test -n "$S1_RUN_PATH"
S1_RUN="$(basename "$S1_RUN_PATH")"
uv run --extra mujoco --extra rlgames-sapg eval \
  --algo rlgames_sapg --task simtoolreal --sim mujoco --render-mode record \
  --load-run "$S1_RUN" \
  training.device=cuda:0 training.play_env_num=6 training.play_steps=8 \
  training.log_root="$S1_ROOT" +env.cpu_ids=null
~~~

CLI task slug 是 `simtoolreal`，runtime `training.task_name` 仍是 `SimToolReal`；cpu_ids 不在
基础 owner 时使用 `+env.cpu_ids=null`。第二次使用独立 root 重复 S1 train/play；不要求两次
physics trajectory bit-exact。

S1 必须证明 epoch/frame 超过 Code #9 单 epoch基线。checkpoint 验收遵循 Code #9/native
schema：actor `optimizer` 的 state 非空；actor model、central model (`assymetric_vf_nets`)
及其各自 RMS/normalizer fields 非空；`rnn_states` 非空且 `env_state=None`。原生 schema 不
落盘 central optimizer state，因此不得要求或伪造该字段。每次最终只有 train/eval 两个
run directories，MP4 非空，native scratch、materialized roots、writer、renderer 和 GPU
process 均清理。

### 6.2 真实 12288/2048/6 profile

必须通过 `--profile 12k` compose 已有 `mujoco_12k.yaml`，固定验证：

~~~text
algo.num_envs=12288
rl_games.params.config.expl_coef_block_size=2048
num_blocks=6
horizon_length=16
seq_length=16
action/actor/critic=29/140/162
training.sim_backend=mujoco
+env.cpu_ids=null
~~~

运行一次真实有限 native train：

~~~bash
PROFILE_ROOT="$(mktemp -d)"
UNILAB_REQUIRE_SAPG=1 uv run --extra mujoco --extra rlgames-sapg train \
  --algo rlgames_sapg --task simtoolreal --sim mujoco --profile 12k \
  rl_games.params.config.max_epochs=1 \
  rl_games.params.config.save_frequency=1 \
  training.device=cuda:0 training.no_play=true training.log_root="$PROFILE_ROOT" \
  +env.cpu_ids=null
~~~

检查真实 profile、checkpoint、actor optimizer state、actor/central model 与 RMS/normalizer
state、RNN state、`env_state=None`、run metadata、pool/native scratch 和显存/进程 cleanup。
central optimizer state 不属于 native checkpoint contract。不得用 S1 overrides 或缩小 block
size 代替；OOM、native crash、missing CUDA 或 native schema 中已有字段缺失均为 blocker。

## 7. Child 10C：mixed-layout/autoreset/affinity 组合近风险

新增一个 focused combination test file，真实构造
含 catalog indexes 0、1、7 的 mixed model assignments，并验证：

1. 三种 topology 的 compiled model/visual mapping 与 Code #8/#9 anchors 一致；
2. `was_autoreset` baseline、selected row、多 substep OR-latch 和下一步 clear exact；
3. selected-row reset/cache/terminal observation 不污染其他 rows；
4. 用当前进程可用 CPU IDs 经 Target public config/backend wiring 创建 pool，
   `worker_cpu_ids()` 返回 exact mapping，并能 step、autoreset 和 close；
5. `cpu_ids=None` 保持 OS scheduling，不能冒充 affinity support。

先运行既有 affinity test，再运行组合测试和邻近 runtime/task gate：

~~~bash
set -e
set -o pipefail
uv run --extra mujoco --extra rlgames-sapg pytest \
  tests/base/backend/test_mujoco_cpu_affinity_wiring.py -q
uv run --extra mujoco --extra rlgames-sapg pytest \
  tests/envs/manipulation/simtoolreal/test_m0_dev_matrix.py -q
~~~

与 `cpu_ids`、`worker_cpu_ids()`、`was_autoreset` 或 mixed-layout 相关的测试必须真实收集
且 GREEN，0 skip；不得 monkeypatch runtime、探测 backend private capability 或恢复旧
SHA 的 module-level skip。

## 8. Child 10D：M0-dev 回归、docs、完整 gates 和 handoff

只有 10C 已验证并且 Target lock 已切换到 rebased M0-dev SHA，才执行本节。当前
candidate 不要求正式 sdist、0.4.0 版本或 release artifact。

### 8.1 Required local gates

按近风险到全局的顺序运行，记录真实退出码和 pass/skip/fail：

~~~bash
set -e
set -o pipefail
export UNILAB_REQUIRE_SAPG=1
uv run --extra mujoco --extra rlgames-sapg pytest tests/algos/rlgames_sapg -q
uv run --extra mujoco --extra rlgames-sapg pytest \
  tests/envs/manipulation/simtoolreal -q
uv run --extra mujoco --extra rlgames-sapg pytest \
  tests/base/backend/test_mujoco_uni_runtime_contract.py \
  tests/base/backend/test_mujoco_cpu_affinity_wiring.py -q
uv run --extra mujoco --extra rlgames-sapg pytest tests/scripts/test_support_matrix.py -q
uv lock --check
uv run ruff check src/unilab scripts tests
uv run ruff format --check src/unilab scripts tests
uv run --extra mujoco --extra rlgames-sapg mypy src/unilab
uv run --extra mujoco --extra rlgames-sapg pyright
git diff --check
make test-all
~~~

M0-dev gate 中与 cpu_ids、worker_cpu_ids、mixed-layout 或 per-env autoreset 相关的测试
不得 skip；这些能力必须 GREEN。其他既有 optional Motrix skip 必须逐项记录原因，不能把
skip 计为 pass。make test-all 失败时不删测试、不改全局 tolerance、不宣称 candidate 完成。

### 8.2 Support docs/matrix

在 M0-dev evidence 已落地并通过 focused gates 后：

1. 用 uv run scripts/generate_support_matrix.py --write 刷新 generated block，不手工
   编辑 generated table；
2. 若当前 generator 尚未表达 SAPG entrypoint，增加一个窄的
   rlgames_sapg / SimToolReal / mujoco evidence owner 和对应 test；当前最高只能是
   `Tested`，不能把其他算法或 backend 一并提升；
3. Chinese/English installation docs 只增加当前 M0-dev Git pin、安装态、ABI 和 canonical
   platform 说明；明确 Linux/aarch64、Motrix、无 affinity ABI 和正式 M0-release 仍是
   unsupported/deferred；
4. 不标记 Recommended 或 Benchmarked，除非已有明确 metadata 和提交的 benchmark
   manifest；
5. 文档必须把 M0-dev provisional 与未来 M0-release 分开写，不能把 dev identity 写成
   正式 support，也不能暗示两个版本可互换。

### 8.3 Control-session handoff

实现 session 不提交、不 push，因此不能自己产生 final-current-head remote CI。完成本节
local gates 后交回：

- exact changed-path list 和每个 child 的 RED→GREEN；
- M0-dev source SHA、manifest/hash、安装态和 public ABI provenance；正式 M0-release
  artifact 标记为 deferred，不得伪造 filename/SHA256；
- target pyproject.toml/uv.lock identity；
- S1、12k、mixed-layout、autoreset、affinity 的命令和实际结果；
- complete local gate 输出；
- docs/support matrix diff；
- 当前工作树未暂存、未提交且无残留日志/进程的证据；
- 明确标记 remote CI: pending control-session commit、maintainer support judgment:
  pending，以及 formal M0-release/artifact promotion: deferred。

控制 session 提交后必须以新的 current HEAD 重跑必要 gates，并等待该 HEAD 的全部 remote
CI 完成；旧 HEAD 的绿色结果、pending/in-progress job 或未运行的 job 都不算通过。

## 9. RED→GREEN、审计和证据规则

每个新增 focused test 必须先在缺失 owner/ABI/manifest 的状态建立可解释 RED，再做最小
GREEN。不得伪造“先失败”日志。报告至少包含：

- 失败命令、首个异常和根因；
- 修改的唯一 owner/path；
- 通过命令及 passed/skipped/failed/warnings；
- M0-dev lock/direct_url/source hash/provenance 和安装态 ABI；
- cold-path、backend-private、Source/sibling access、cleanup audit；
- 未解决的 blocker 或为何没有 blocker。

普通 pytest 只能读取已提交 fixture/manifest，不能生成或重 baseline Code #1-#9 fixture。
任何新 manifest 必须由固定、可复现的 generator 或 artifact attestation 产生，并报告
外层 SHA256。

## 10. 停止条件

遇到以下任一项，立刻停止当前 child 并回报 # BLOCKED：

- Target 起点、branch、工作树或 writer ownership 不符合第 3.2 节；
- M0-dev runtime identity、vendor identity、canonical Torch 或 Source provenance 漂移；
- S1/12k 只能用 fake backend、CPU、缩小 profile、关闭关键 DR/autoreset/wrench 或 skip
  才能通过；
- 12k 真实 profile OOM、native crash、checkpoint 不完整或 cleanup 泄漏；
- M0-dev source SHA、lock、direct_url、manifest 或安装态 ABI 不一致，或只能来自 dirty
  sibling checkout；正式 artifact 缺失或版本仍为 .dev0 不属于当前 candidate blocker，
  但必须在报告中标记为 deferred；
- cpu_ids、worker_cpu_ids 或 was_autoreset property ABI 缺失/不真实；
- 需要修改外部 MuJoCoUni、Source、vendor、Code #1-#9 owner、shared sim2sim 或新公共
  contract；
- mixed-layout、autoreset、affinity、T0/T1/SAPG oracle 出现无法解释的 mismatch；
- M0-dev dependency 造成无关 lock upgrade、平台 marker 漂移或 direct_url 指向 sibling；
- required test failure、未解释 warning、M0-dev gate skip 或 make test-all failure；
- 文档/support matrix 会把未经 maintainer 批准的组合写成 Recommended、Benchmarked
  或跨 backend support；
- 需要 push/PR/remote CI 才能继续，而控制 session 尚未接管。

禁止通过删除证据、改变固定 profile、放宽 tolerance、修改 fixture hash、静默 rebaseline、
把 blocker 改写成 warning 或扩大白名单来绕过停止条件。

## 11. 实现 session 最终报告格式

所有 child 的 local gates 和 M0-dev 证据齐全时，报告：

~~~text
# DONE (local Code #10 M0-dev candidate)

起止 branch/HEAD：
实际修改路径与每个 child 的 scope：
M0-dev identity：
M0-release artifact/version/SHA256/source provenance：deferred；本批不制作、不伪造
S1 train/play 结果：
12288/2048 结果：
mixed-layout/autoreset/affinity 结果：
dependency/lock/direct_url 结果：
support docs/matrix 结果：
focused/full local gates：
cleanup/audit：
remote CI：pending control-session commit
maintainer support judgment：pending；formal M0-release/artifact promotion：deferred
~~~

如果任一停止条件触发，报告：

~~~text
# BLOCKED

触发的 child/停止条件：
失败命令和首个异常：
已核验的固定 identity：
当前工作树/staging：
已完成的前置 child：
需要 maintainer 提供的唯一外部事实或授权：
未执行的后续 child：
~~~

只有控制 session 在提交后以当前 HEAD 完成 local/remote CI，才能把总指导中的 Code #10
改为“M0-dev candidate 已完成”。得到 maintainer 明确 support judgment 并完成独立的
M0-release artifact promotion 后，才可以使用正式 support wording；这两者不是当前
candidate 的隐含结果。
