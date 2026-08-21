# SimToolReal SAPG Code #10 执行 Prompt

> 本文件交给一个新的实现 session。用户明确说“看
> docs/simtoolreal_sapg_code10_prompt.md，按文档执行 Code #10 的 child batches”时，
> 表示当前 session 获得了本文件所列 Target 仓库范围的 execution approval。实现 session
> 必须直接执行已批准的顺序，不派 subagent、不重新规划、不进入 Code #11。完成后保留
> 改动未暂存、未提交，交回控制/审查 session。

## 1. 本批唯一结果

把 Code #9 已验证的 M0-dev native RL-Games SAPG vertical slice 晋升为可以审查和声明
支持的正式路径。Code #10 是 release、安装、组合回归和证据整理批次，不是算法迁移批次。
完成候选必须同时满足：

1. 当前 M0-dev runtime 上有超过 Code #9 tiny slice 的有限多 epoch S1 train/play 证据；
2. 12288/2048/6 的真实 native profile 至少完成一个有限 epoch、checkpoint 和 cleanup；
3. 维护者提供的 M0-release artifact 能在没有 sibling checkout 的 clean environment 中
   安装，并且 mixed-layout、was_autoreset、cpu_ids 和 worker_cpu_ids public ABI
   均可验证；
4. UniLab 的 mujoco extra 和 uv.lock 不再依赖 0.4.0.dev0 或 dirty Git checkout，
   而是指向已核验的正式 artifact；
5. mixed-layout、per-env autoreset、CPU affinity 的组合近风险回归通过，release gate
   没有被 skip；
6. focused/full local gates、support docs、support matrix、make test-all 和最终
   current-head CI 证据齐全。

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
处理 remote CI 和 maintainer support judgment。

## 2. 普通中文范围、规模和 child batches

### 2.1 只做什么

- 用当前 M0-dev identity 运行真实 S1 train/play，证明 native path 可连续有限运行；
- 运行真实 mujoco_12k owner，证明资源 profile、六 blocks、checkpoint 和 cleanup；
- 对 M0-release artifact 做 provenance、clean-install、public ABI 和 target integration
  检查；
- 在 release 可用后，把 Target dependency 从 dev identity 晋升为正式 artifact；
- 组合验证 mixed model layout、autoreset latch 和 CPU affinity；
- 更新安装说明、支持矩阵和 release evidence manifest；
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
  关闭 wrench/autoreset/affinity 或删除失败证据来绕过 gate；
- 不接 Motrix、IsaacSim、async、distributed、multi-GPU、ROCm、export、torch.compile、
  通用 RL-Games support 或新的 public backend contract；
- 不运行 PR 流程；没有当前提交的 remote CI 时，不声称正式 support 已完成。

### 2.3 规模和永久维护成本

Code #10 是一个 release umbrella，按以下四个 child 顺序执行。每个 child 必须保持
约 15 个以内 touched paths、约 800 行以内净手写改动，并在下一个 child 前单独验证。
测试/manifest 可以新增，生产 owner 只能在真实失败证明需要时做最小 owner-level 修正。

永久成本限制为：

- 一个正式 MuJoCoUni dependency 和一份 artifact/provenance manifest；
- M0-dev/M0-release contract tests 和组合 affinity regression；
- 安装/support 文档中的一条 SAPG support entry；
- 不新增第二套训练 runtime、release wrapper 或 backend abstraction。

### 2.4 Child batches

必须按顺序执行以下 child；前一个 child 的 blocker 不得被下一个 child 绕过：

~~~text
10A  M0-dev S1 finite multi-epoch train/play 与 release preflight
10B  真实 12288/2048 profile、mixed-layout/autoreset 组合测试、affinity ABI preflight
10C  外部提供的 M0-release artifact clean-install、provenance 和 dependency promotion
10D  release 组合回归、audit/docs/support matrix、make test-all 和 control handoff
~~~

10A/10B 可以在当前 dev identity 上进行，但只能把 cpu_ids=null 作为开发 smoke，
不能写成 affinity support。10C 若没有维护者提供的正式 artifact，必须立即 # BLOCKED；
实现 session 不得自行修改外部仓库来制造 artifact。10D 只有在 10C 的 artifact 和
target lock 均已验证后才可执行。

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
/home/user/ws/lemon/mujoco_uni/src/mujoco_uni/batch_env.py
/home/user/ws/lemon/mujoco_uni/tests/test_batch_env.py
~~~

### 3.2 Target lineage preflight

本 prompt 的父 docs baseline 是：

~~~text
0a1912490cd6dbb6a952aa93d52ebc06b62dc9b6
docs: record SAPG Code 9 completion
~~~

开始时运行以下只读检查。预期 prompt docs commit 是该 baseline 的一个直接 docs-only
child；若条件不成立，返回 # BLOCKED，不要清理或覆盖现有改动。

~~~bash
set -e
set -o pipefail
CODE10_BASE=0a1912490cd6dbb6a952aa93d52ebc06b62dc9b6
test "$(git rev-parse --abbrev-ref HEAD)" = "feat/simtoolreal-sapg-rlgames"
git merge-base --is-ancestor "$CODE10_BASE" HEAD
test "$(git rev-list --count "$CODE10_BASE"..HEAD)" -eq 1
test "$(git diff --name-status "$CODE10_BASE"..HEAD)" = \
  $'A\tdocs/simtoolreal_sapg_code10_prompt.md\nM\tdocs/simtoolreal_sapg_source_fidelity_migration_plan.md'
test -z "$(git status --short)"
test -z "$(git diff --cached --name-only)"
git log -3 --oneline
git status --short --branch
~~~

记录外部仓库的 branch、HEAD 和 dirty files 仅用于 provenance 报告。外部当前已有的
MUJOCO_LOG.TXT 与 docs/unilab_extension_status.md 属于用户文件，禁止清理、reset、
checkout 或改写。

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
~~~

Code #10 不重新生成 Code #1-#5 fixtures。UNILAB_REQUIRE_SAPG=1 仍必须显式设置；
错误 distribution、错误 import path、vendor hash drift 或 canonical platform drift
必须 fail closed。

### 3.4 M0-dev identity

10A/10B 开始前核验并记录：

~~~text
mujoco-uni-runtime==0.4.0.dev0
URL: https://github.com/lemon-star608/mujoco_uni.git
source SHA: 7205e070e983df90d520f0f8593853013e976746
BatchEnvPool.was_autoreset: real property
cpu_ids/worker_cpu_ids: not a support claim; null-only development path
~~~

M0-dev 只证明 mixed-data-layout 和 per-env autoreset；不证明 CPU affinity，不得把
existing optional affinity test 的 skip 统计为通过。

### 3.5 M0-release acceptance identity

正式 artifact 必须由 MuJoCoUni maintainer 在独立、干净的 0.4 release 工作流中提供。
实现 session 只消费 artifact，不修改或发布外部代码。artifact 必须满足：

- distribution 名为 mujoco-uni-runtime；
- release version 为明确的 0.4.0，不是 .dev0、local version 或 editable install；
- artifact 是 release workflow 允许的 sdist（当前 workflow 不发布 wheel）；
- 提供 artifact filename、SHA256、构建用 MuJoCo 版本、Python/platform、构建 source
  commit 和 source tree/provenance；
- 构建 source commit 是 7205e070e983df90d520f0f8593853013e976746 的 descendant，并
  明确包含 mixed-layout、per-env autoreset、cpu_ids constructor ABI 和
  worker_cpu_ids() public method；
- 安装后的 BatchEnvPool.was_autoreset 是真实 property，不是动态 fallback；
- clean-install 不解析或依赖 /home/user/ws/lemon/mujoco_uni sibling checkout。

artifact 通过环境变量传入，不得猜测路径：

~~~bash
test -n "$MUJOCO_UNI_RELEASE_ARTIFACT"
test -f "$MUJOCO_UNI_RELEASE_ARTIFACT"
sha256sum "$MUJOCO_UNI_RELEASE_ARTIFACT"
~~~

变量缺失、文件不存在、hash/provenance/ABI 任一项无法核验时，10C 立即 # BLOCKED。

## 4. 允许写入的 Target 路径

实现 session 只能写 Target 仓库。以下是每个 child 的预批准路径边界；新建文件必须
落在列出的目录，并在报告中列出精确路径和原因。

### 4.1 10A

~~~text
tests/algos/rlgames_sapg/**
tests/fixtures/simtoolreal_sapg/m0_dev_*.json
~~~

优先只新增 focused test/harness/manifest，不改 production owner、vendor、dependency
或现有 Code #9 semantics。若必须改已有 test，只能是同目录中与 M0-dev identity 直接
相关的最小断言。

### 4.2 10B

~~~text
tests/algos/rlgames_sapg/**
tests/base/backend/test_mujoco_m0_release_matrix.py
tests/envs/manipulation/simtoolreal/test_m0_release_matrix.py
~~~

允许为近风险测试增加一个组合 harness；不得修改 backend public contract 或把
cpu_ids 变成默认生产配置。

### 4.3 10C

~~~text
pyproject.toml
uv.lock
tests/algos/rlgames_sapg/test_dependency.py
tests/base/backend/test_mujoco_uni_runtime_contract.py
tests/fixtures/simtoolreal_sapg/m0_release_manifest.json
~~~

只有在 artifact 已通过 3.5 全部核验后，才允许触碰这组路径。禁止提交 local path、
editable source、临时 index、未审查的 URL 或无 hash 的 release dependency。

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
docs/simtoolreal_sapg_source_fidelity_migration_plan.md。support matrix 只能在所有
release evidence 已提交后把 rlgames_sapg / SimToolReal / mujoco 提升到真实证据等级；
不得自动标为 Recommended 或 Benchmarked，除非仓库中已有对应 maintainer metadata。

## 5. Child 10A：M0-dev S1 finite multi-epoch train/play

### 5.1 目标

证明当前 dev runtime 上不是只跑 Code #9 的单 epoch N=6 vertical slice，而是可以用
同一个 native Runner/A2CAgent path 连续完成有限多 epoch、保存 native .pth、关闭
env/tracker/writer，再用 native player 生成视频。S1 是 test-only invocation，不新增
一个 production YAML profile。

### 5.2 固定 S1 invocation

保持六 blocks 不变，使用以下最小覆盖值；只允许通过 Hydra override 注入，不得把这些
值写回 mujoco.yaml：

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
env.cpu_ids=null
~~~

S1 必须使用当前 mujoco-uni-runtime==0.4.0.dev0，并显式设置
UNILAB_REQUIRE_SAPG=1。缺 CUDA、Torch canonical identity 或真实 MuJoCo pool 时
返回 blocker，不改成 fake/CPU 通过。

### 5.3 必须运行的真实命令

在 repo 外用 mktemp -d 创建 log root，训练和 eval 均通过公开 CLI；命令中的
S1_ROOT、S1_RUN 由 session 实际捕获并记录：

~~~bash
set -e
set -o pipefail
S1_ROOT="$(mktemp -d)"
export UNILAB_REQUIRE_SAPG=1
uv run train --algo rlgames_sapg --task SimToolReal --sim mujoco \
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
  env.cpu_ids=null
S1_TASK_ROOT="$S1_ROOT/SimToolReal"
S1_RUN_PATH="$(find "$S1_TASK_ROOT" -mindepth 1 -maxdepth 1 -type d -name '0_*' -print | sort | tail -n 1)"
test -n "$S1_RUN_PATH"
S1_RUN="$(basename "$S1_RUN_PATH")"
uv run eval --algo rlgames_sapg --task SimToolReal --sim mujoco --render-mode record \
  --load-run "$S1_RUN" \
  training.device=cuda:0 training.play_env_num=6 training.play_steps=8 \
  training.log_root="$S1_ROOT" env.cpu_ids=null
~~~

如果 CLI 不能直接导出 run name，必须从实际 native run directory 读取并再次运行 eval；
不得改脚本来猜测路径。可用 scripts/train_rlgames_sapg.py 的已有 API 作为 near-risk
补充，但不能代替公开 CLI 证据。

### 5.4 S1 acceptance

必须保存以下事实：

- train run 真实经过 Runner.load -> set_vec_env -> run_train -> A2CAgent.train；
- epoch/frame 大于 Code #9 单 epoch基线，native .pth payload、optimizer/RMS/RNN state
  非空且 env_state=None；
- train root 只有一个 run directory；eval 后只有 train/eval 两个 directory；
- eval video 是非空 MP4，使用 assigned visual model，close 后所有 materialized temp
  roots 消失；
- run metadata 记录 dev version、source SHA、Torch/CUDA/cuDNN、cpu_ids=null 和
  command overrides；
- 第二次独立 S1 运行至少复用同一 contract 并完成 cleanup；不要求 physics trajectory
  bit-exact。

## 6. Child 10B：真实 12k profile 和组合近风险

### 6.1 12k profile

必须 compose 已有 task=simtoolreal/mujoco_12k，不能临时修改 base owner。固定检查：

~~~text
algo.num_envs=12288
rl_games.params.config.expl_coef_block_size=2048
num_blocks=6
horizon_length=16
seq_length=16
action/actor/critic=29/140/162
training.sim_backend=mujoco
env.cpu_ids=null for the dev run
~~~

执行一次真实有限 native train（建议 max_epochs=1、no_play=true、save_frequency=1）
并检查 checkpoint、非空 optimizer state、run metadata、pool cleanup 和显存/进程 cleanup。
不得把 6-env S1 override 或缩小后的 block size 当作 12k 证据。OOM、native crash、
missing CUDA 或 profile preflight failure 都是 blocker，不得降低到另一个 profile 后宣称
12k 通过。

### 6.2 mixed-layout/autoreset/affinity 组合

新增一个 focused combination test（或在批准目录中最小扩展现有 near-risk test），真实
构造至少包含 catalog indexes 0、1、7 的 mixed model assignments，并在同一 pool 生命周期
内验证：

1. 三种 topology 的 compiled model/visual mapping 与 Code #8/9 anchors 一致；
2. per-env was_autoreset 在 baseline、selected row 和多 substep OR-latch 上 exact；
3. 选定 rows 的 reset/cache/terminal observation 不污染其他 rows；
4. 正式 runtime 提供 BatchEnvPool.__init__(cpu_ids=...) 和 public
   worker_cpu_ids() 时，用当前可用 CPU IDs 真实创建 pool，检查 worker mapping、step、
   autoreset 和 close；
5. cpu_ids=None 仍保持 OS scheduling，不能把它记录为 affinity support。

若安装态没有 cpu_ids 参数、worker_cpu_ids()、真实 was_autoreset property，在
10B 末尾立即报告缺失 ABI 并停止；不要 skip、monkeypatch、私有 fallback 或修改
Target backend 猜测兼容。

## 7. Child 10C：M0-release clean-install 和 dependency promotion

### 7.1 外部 artifact gate

先执行 3.5 的环境变量、filename/hash、version、provenance 和 public ABI 检查。可用
isolated uv invocation 验证安装态，示例：

~~~bash
set -e
set -o pipefail
RELEASE_ROOT="$(mktemp -d)"
uv run --isolated \
  --with "$MUJOCO_UNI_RELEASE_ARTIFACT" \
  --with "mujoco==$MUJOCO_M0_RELEASE_MUJOCO_VERSION" \
  python -c "import importlib.metadata as m, mujoco_uni; print(m.version('mujoco-uni-runtime')); print(mujoco_uni.__version__)"
~~~

先把 artifact manifest 中的真实 MuJoCo version 写入环境变量
MUJOCO_M0_RELEASE_MUJOCO_VERSION；该变量不是可猜测值。把完整 stdout、installed metadata、
direct_url.json、
BatchEnvPool signature、was_autoreset descriptor 和 worker_cpu_ids() 结果写入报告。
如果 isolated install 仍解析 sibling checkout、.dev0、editable source 或错误 hash，
立即 blocker。

### 7.2 Target dependency promotion

artifact 通过 7.1 后才可修改 Target：

1. 将 pyproject.toml 的 mujoco extra 从 0.4.0.dev0 Git revision 改为
   maintainer 批准的正式 mujoco-uni-runtime==0.4.0 source；
2. 重新生成 uv.lock，保证 lock source、version、hash、marker 与 pyproject 一致，
   不引入无关升级；
3. 更新 dependency/runtime contract tests，使它们同时保留 M0-dev 历史说明和新的
   installed release identity；
4. 写入 tests/fixtures/simtoolreal_sapg/m0_release_manifest.json，至少包含：

~~~text
schema
distribution/version
artifact filename/sha256
source commit/tree/provenance
build mujoco version
python/platform matrix
mixed-layout evidence
was_autoreset property evidence
cpu_ids constructor evidence
worker_cpu_ids result
clean-install command and timestamp
target pyproject/uv.lock source
~~~

5. 在一个不含 sibling checkout 的隔离 uv environment 中运行 target focused gate，并
   验证 importlib.metadata.version("mujoco-uni-runtime")、direct_url.json 和 lock
   source 均指向正式 artifact；
6. 检查 root MUJOCO_LOG.TXT、vendor __pycache__、native scratch 和临时 materialized
   roots，不把可复现的临时日志提交进仓库。

若正式 artifact 只能通过修改 /home/user/ws/lemon/mujoco_uni、使用 dirty checkout、
本地 editable path、无 provenance wheel 或临时 uncommitted patch 得到，返回 # BLOCKED
并保留 Target 依赖为已知 dev identity；不要部分晋升。

## 8. Child 10D：release 回归、docs、完整 gates 和 handoff

只有 10C 已验证并且 Target lock 已切换到正式 artifact，才执行本节。

### 8.1 Required local gates

按近风险到全局的顺序运行，记录真实退出码和 pass/skip/fail：

~~~bash
set -e
set -o pipefail
export UNILAB_REQUIRE_SAPG=1
uv run --extra mujoco pytest tests/algos/rlgames_sapg -q
uv run --extra mujoco pytest tests/envs/manipulation/simtoolreal -q
uv run --extra mujoco pytest tests/base/backend/test_mujoco_uni_runtime_contract.py \
  tests/base/backend/test_mujoco_cpu_affinity_wiring.py \
  tests/base/backend/test_mujoco_m0_release_matrix.py -q
uv run --extra mujoco pytest tests/scripts/test_support_matrix.py -q
uv lock --check
uv run ruff check src/unilab scripts tests
uv run ruff format --check src/unilab scripts tests
uv run --extra mujoco mypy src/unilab
uv run --extra mujoco pyright
git diff --check
make test-all
~~~

release gate 中与 cpu_ids、worker_cpu_ids、mixed-layout 或 M0-release identity 相关的
测试不得 skip。其他既有 optional Motrix skip 必须逐项记录原因，不能把 skip 计为 pass。
make test-all 失败时不删测试、不改全局 tolerance、不宣称 support。

### 8.2 Support docs/matrix

在 release evidence 已提交后：

1. 用 uv run scripts/generate_support_matrix.py --write 刷新 generated block，不手工
   编辑 generated table；
2. 若当前 generator 尚未表达 SAPG entrypoint，增加一个窄的
   rlgames_sapg / SimToolReal / mujoco evidence owner 和对应 test，不能把其他算法或
   backend 一并提升；
3. Chinese/English installation docs 只增加正式 artifact、clean-install 和 canonical
   platform 说明；明确 Linux/aarch64、Motrix、无 artifact、无 affinity ABI 仍 unsupported；
4. 不标记 Recommended 或 Benchmarked，除非已有明确 metadata 和提交的 benchmark
   manifest；
5. 文档必须同时写出 M0-dev（历史 development identity）和 M0-release（当前 support
   identity），避免读者误以为两个版本可互换。

### 8.3 Control-session handoff

实现 session 不提交、不 push，因此不能自己产生 final-current-head remote CI。完成本节
local gates 后交回：

- exact changed-path list 和每个 child 的 RED→GREEN；
- M0-dev/M0-release artifact filename、SHA256、source provenance；
- target pyproject.toml/uv.lock identity；
- S1、12k、mixed-layout、autoreset、affinity 的命令和实际结果；
- complete local gate 输出；
- docs/support matrix diff；
- 当前工作树未暂存、未提交且无残留日志/进程的证据；
- 明确标记 remote CI: pending control-session commit 和
  maintainer support judgment: pending。

控制 session 提交后必须以新的 current HEAD 重跑必要 gates，并等待该 HEAD 的全部 remote
CI 完成；旧 HEAD 的绿色结果、pending/in-progress job 或未运行的 job 都不算通过。

## 9. RED→GREEN、审计和证据规则

每个新增 focused test 必须先在缺失 owner/ABI/manifest 的状态建立可解释 RED，再做最小
GREEN。不得伪造“先失败”日志。报告至少包含：

- 失败命令、首个异常和根因；
- 修改的唯一 owner/path；
- 通过命令及 passed/skipped/failed/warnings；
- artifact/lock/direct_url/hash/provenance；
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
- artifact 缺失、版本为 .dev0、hash/provenance 不全、不是 clean-install sdist，或
  只能来自 dirty sibling checkout；
- cpu_ids、worker_cpu_ids 或 was_autoreset property ABI 缺失/不真实；
- 需要修改外部 MuJoCoUni、Source、vendor、Code #1-#9 owner、shared sim2sim 或新公共
  contract；
- mixed-layout、autoreset、affinity、T0/T1/SAPG oracle 出现无法解释的 mismatch；
- release dependency 造成无关 lock upgrade、平台 marker 漂移或 direct_url 指向 sibling；
- required test failure、未解释 warning、release gate skip 或 make test-all failure；
- 文档/support matrix 会把未经 maintainer 批准的组合写成 Recommended、Benchmarked
  或跨 backend support；
- 需要 push/PR/remote CI 才能继续，而控制 session 尚未接管。

禁止通过删除证据、改变固定 profile、放宽 tolerance、修改 fixture hash、静默 rebaseline、
把 blocker 改写成 warning 或扩大白名单来绕过停止条件。

## 11. 实现 session 最终报告格式

所有 child 的 local gates 和 artifact 证据齐全时，报告：

~~~text
# DONE (local Code #10 candidate)

起止 branch/HEAD：
实际修改路径与每个 child 的 scope：
M0-dev identity：
M0-release artifact/version/SHA256/source provenance：
S1 train/play 结果：
12288/2048 结果：
mixed-layout/autoreset/affinity 结果：
dependency/lock/direct_url 结果：
support docs/matrix 结果：
focused/full local gates：
cleanup/audit：
remote CI：pending control-session commit
maintainer support judgment：pending
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

只有控制 session 在提交后以当前 HEAD 完成 local/remote CI，并得到 maintainer 明确
support judgment 后，才能把总指导中的 Code #10 改为“已完成”并使用正式 support wording。
