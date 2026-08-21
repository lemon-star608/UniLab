# SimToolReal SAPG Code #6 执行 Prompt

> 本文件交给一个新的实现 session。用户明确说“阅读本文件并执行”即表示 Code #6 已获得
> execution approval。实现 session 必须直接完成任务，不派 subagent，不重新规划，不进入
> Code #7。完成后保留全部改动未暂存、未提交，交回控制/审查 session。

## 1. 本批唯一结果

本批只让 SimToolReal 后续所需的 MuJoCo runtime 能力成为可复验的 public backend
contracts：

1. 将开发依赖固定到可复现的 MuJoCoUni M0-dev Git identity；
2. 允许 `ModelVariantSpec` 指向完整 source model，并在 MuJoCo 冷路径直接编译异构
   model variants；
3. 在 `SimBackend` 声明 public `apply_body_wrench`，由 MuJoCo 实现逐 env、逐 body 的
   world-frame force/torque staging；
4. 在 `SimBackend` 声明 public `get_step_autoreset_mask`，由 MuJoCo 报告本次 control
   step 内发生 engine autoreset 的 env，并正确 OR-latch 多 substep 结果。

目标代码提交标题是：

~~~text
feat(backend): add SimToolReal MuJoCo runtime contracts
~~~

但实现 session 不执行 `git add`、`git commit`、`git push`、PR、`stash`、`reset`、
`clean`、`checkout` 或切分支。最终审查、暂存和提交只由控制 session 执行。

## 2. 普通中文范围说明

只做：

- M0-dev dependency identity、lock 和安装态验证；
- `source_model_file` 完整模型冷路径编译；
- public body-wrench contract；
- public step-autoreset mask；
- 12-distribution synthetic mixed-layout、row-isolated wrench、真实 autoreset 近风险测试；
- 保持现有 pre-step-control 测试替身符合新增 public runtime contract。

不做：

- 不迁移 assets、XML、task primitives、真实 SimToolReal env 或 600-tool catalog；
- 不进入 T0/T1、Hydra owner、RL-Games adapter、Runner、tracker、checkpoint、player 或 CLI；
- 不修改 Source、vendored RL-Games 或 Code #1-#5 fixtures/tests；
- 不修改 MuJoCoUni owner 仓库，不依赖本地 sibling checkout；
- 不恢复 M0-release 的 `cpu_ids/worker_cpu_ids` ABI，也不声称 CPU affinity support；
- 不迁移 donor 的 per-env geom/body side tables，不修改 Motrix、MJWarp 或 Drake；
- 不增加本文件未列出的 public backend surface。

预计实现范围为 10 个 paths、约 500-800 行净手写实现/测试，加 generated `uv.lock` diff。
永久维护成本是 3 个 public backend contracts（完整 source model、body wrench、autoreset
mask）以及 M0-dev/M0-release dependency provenance。若需要第 11 个实现 path 或明显超过
800 行净手写改动，先停止并回报原因，不要自行扩张。

## 3. 必读内容与固定身份

开始前完整阅读：

~~~text
AGENTS.md
docs/simtoolreal_sapg_source_fidelity_migration_plan.md
src/unilab/dr/types.py
src/unilab/base/backend/base.py
src/unilab/base/backend/mujoco/backend.py
tests/base/test_backend_pre_step_control.py
~~~

唯一写入仓库：

~~~text
/home/user/ws/lemon/rlgame-unilab/UniLab
~~~

预期分支：

~~~text
feat/simtoolreal-sapg-rlgames
~~~

本 prompt 之前的固定基线：

~~~text
45ff5f88bb40d262bdb59b104928e99a1dce895f
docs: consolidate SAPG migration guidance
~~~

实现 session 的 dispatch HEAD 应是上述基线的单个 docs child，且该 child 只新增本 prompt。
开始时运行：

~~~bash
set -e
set -o pipefail
SAPG_CODE6_BASE=45ff5f88bb40d262bdb59b104928e99a1dce895f
test "$(git rev-parse --abbrev-ref HEAD)" = "feat/simtoolreal-sapg-rlgames"
git merge-base --is-ancestor "$SAPG_CODE6_BASE" HEAD
test "$(git rev-list --count "$SAPG_CODE6_BASE"..HEAD)" -eq 1
test "$(git diff --name-status "$SAPG_CODE6_BASE"..HEAD)" = \
  $'A\tdocs/simtoolreal_sapg_code6_prompt.md'
test -z "$(git status --short)"
test -z "$(git diff --cached --name-only)"
git log -2 --oneline
git status --short --branch
~~~

任一条件不成立就返回 `# BLOCKED`，不要清理或覆盖现有改动。

只读参考身份：

~~~text
Mature donor: /home/user/ws/lemon/UniLab
B1/B3 donor commit:
  8a4f5ccca665d4b85a4cb687fc877f9a9479d7da
B2 donor commit:
  648b0bd40c7044b91b92b85a805fce8a37db05dc

MuJoCoUni M0-dev remote:
  https://github.com/lemon-star608/mujoco_uni.git
MuJoCoUni version:
  0.4.0.dev0
MuJoCoUni source commit:
  7205e070e983df90d520f0f8593853013e976746
~~~

Donor commits只用于理解最小 hunk，不能 cherry-pick：target 已有更新的 CPU affinity、
terrain、render、MjSpec、XML 和 backend 行为。`/home/user/ws/lemon/mujoco_uni` 即使存在也
只是只读参考，不能作为 dependency source，不能修改，也不能相信其当前 dirty/HEAD 状态。

## 4. 唯一允许修改的实现路径

~~~text
pyproject.toml
uv.lock
src/unilab/dr/types.py
src/unilab/base/backend/base.py
src/unilab/base/backend/mujoco/backend.py
tests/base/test_backend_pre_step_control.py
tests/base/backend/test_mujoco_uni_runtime_contract.py
tests/base/backend/test_mujoco_model_source_variants.py
tests/base/backend/test_mujoco_body_wrench.py
tests/base/backend/test_mujoco_autoreset_real_pool.py
~~~

前 6 个为 modify，后 4 个为 create。不要修改总指导文档：控制 session 在 Code #6 代码
commit 形成后，再用完整 commit SHA 和实际验证结果更新总文档。

所有 Python 命令都必须通过 `uv run`。手工文件编辑只使用 `apply_patch`；`uv.lock` 只能由
`uv lock` 生成，不能手改。

## 5. Contract 设计

### 5.1 M0-dev dependency

`pyproject.toml` 的 MuJoCo extra 必须把 runtime 版本改为：

~~~toml
"mujoco-uni-runtime==0.4.0.dev0"
~~~

并在现有 `[tool.uv.sources]` 中增加固定 Git source：

~~~toml
mujoco-uni-runtime = { git = "https://github.com/lemon-star608/mujoco_uni.git", rev = "7205e070e983df90d520f0f8593853013e976746" }
~~~

然后只用 `uv lock` 更新 lock。`uv.lock` 必须解析到同一个完整 source commit；不得出现
`../mujoco_uni`、本地 absolute path、branch-only pin 或未固定的 `0.4.0.dev0`。

本批选择 Git source，不使用 wheel/sdist artifact，因此不需要虚构 artifact filename 或
SHA256。若 Git source 无法从 clean environment 解析和构建，停止；不得回退到 dirty
sibling。M0-release 与正式 artifact 留到 Code #10。

**tests/base/backend/test_mujoco_uni_runtime_contract.py** 必须 fail closed 验证：

- `pyproject.toml` 的 version、HTTPS Git URL 和完整 `rev`；
- `uv.lock` 中 package version 和 resolved Git source 的 fragment 是同一完整 SHA；
- 配置和 lock 都不含 sibling path；
- 安装态 `importlib.metadata.version("mujoco-uni-runtime") == "0.4.0.dev0"`；
- `BatchEnvPool.was_autoreset` 是当前安装态真实存在的 public property。

### 5.2 完整 source model variant

`ModelVariantSpec` 只增加：

~~~python
source_model_file: str | None = None
~~~

`is_empty()` 必须同时考虑 `source_model_file` 与现有 `geom_size_overrides`。不要引入 donor 的
`PerEnvGeomOverride`、`PerEnvBodyOverride` 或其他 side-table types。

MuJoCo `_compile_model_variants` 的规则：

- 所有 XML/model path 处理只发生在 init/materialization 冷路径；
- 若本批 variants 中有 `source_model_file`，逐 variant 选择该完整 source file；字段为
  `None` 时回退 backend base model；
- 现有 `geom_size_overrides` 应施加到该 variant 自己的 source model；
- 复用现有 prepare/compile/MJB cleanup/configuration owner，不新建第二套 XML pipeline；
- 保持现有纯 geom-size variants 的并行编译和行为不变；
- 保持 precompiled materialized scene 的现有 fail-closed 行为；
- step/reset 热路径不得读取 XML、asset 或 model metadata。

**tests/base/backend/test_mujoco_model_source_variants.py** 至少证明：

1. `source_model_file` 使两个 topology 不同但 state/control ABI 相同的 XML 被各自直接编译，
   而不是继续复制 base model；
2. synthetic 12-distribution case 创建 12 个 source files，index 7 使用 upstream M0-dev
   regression 中的 dominant two-geom layout，其余使用 compatible one-geom layout；
3. `model_assignments=np.arange(12)` 后真实 `materialize()`、`step()` 成功，variant 7 的
   `ngeom/nbvh/nC/nbuffer` 至少在相关维度支配 small variant；
4. 原有 geom-size-only path 仍由现有回归测试覆盖。

测试 XML 放在 `tmp_path`，不能提交 Code #7 assets，也不能运行时读取 donor 文件。

### 5.3 Public body wrench

`SimBackend` 增加非 abstract 的 public method：

~~~python
def apply_body_wrench(
    self,
    body_ids: np.ndarray,
    force: np.ndarray,
    torque: np.ndarray,
) -> None:
    ...
~~~

默认实现抛 `NotImplementedError`。MuJoCo 实现必须：

- 将 inputs 转成 `body_ids:int32`、force/torque `float64`；
- 同时要求 force 和 torque shape 为
  `(num_envs, len(body_ids), 3)`；
- 在 `_pending_xfrc_applied` 中按 body 的 6D block 写入
  `[fx, fy, fz, tx, ty, tz]`；
- 使用 `+=`，保持与现有 `apply_body_force` 相同的 accumulation/staging 语义；
- 不在此方法中 step physics，不增加 env/task special case。

**tests/base/backend/test_mujoco_body_wrench.py** 必须调用真实
`MuJoCoBackend.apply_body_wrench` 实现，不能像旧 donor test 那样在测试里复制一份公式。
用最小 backend object 或真实小模型证明：

- 不同 env row、不同 body 的 force/torque 只落入对应 6D slices；
- 未选 env/body slots 保持不变；
- 两次调用正确累加；
- force/torque shape mismatch 分别 fail closed；
- `SimBackend` 默认 public contract 对 unsupported backend surface 抛 `NotImplementedError`。

### 5.4 Public step-autoreset mask

`SimBackend` 增加：

~~~python
def get_step_autoreset_mask(self) -> np.ndarray | None:
    return None
~~~

`None` 表示 backend 不具备报告能力，不等于“本步没有 autoreset”。MuJoCo 在 M0-dev 上必须
直接使用 public `BatchEnvPool.was_autoreset`，不能在 step 热路径通过 `getattr/hasattr`
探测能力，也不能为旧 0.3.1 伪造 mask。

MuJoCo lifecycle：

- backend 初始化一个 shape `(num_envs,)` 的 bool latch；
- 每次 public `step()` 入口先清零；
- 默认 batched `pool.step(nstep=nsteps)` 返回后 OR 一次真实 mask；
- pre-step-control path 每个 `pool.step(nstep=1)` 后都 OR，不能覆盖先前 substep 的 true；
- `get_step_autoreset_mask()` 在 pool 未 materialize 时返回 `None`，materialize 后返回本次
  control step 的 latch；
- 下一次 clean step 必须清掉旧 true；
- 整条热路径不解析 asset/XML/model metadata。

**tests/base/backend/test_mujoco_autoreset_real_pool.py** 使用最小 free-body XML 和真实
`BatchEnvPool`，通过一个大于 MuJoCo divergence threshold 的 qvel 触发 silent autoreset，
至少证明：

- settled baseline mask 全 false；
- 只 diverge env 1 时 exact mask 是 `[False, True, False, False]`；
- 注册 passthrough pre-step control 后，env 2 在 4 个 substeps 的第一个被 reset，最终 mask
  仍是 `[False, False, True, False]`；
- 随后的 clean step 清除 latch；
- default `SimBackend.get_step_autoreset_mask()` 返回 `None`。

用 `monkeypatch.chdir(tmp_path)` 把 MuJoCo warning log 留在 pytest 临时目录，仓库根目录不得
产生 `MUJOCO_LOG.TXT`。

更新 `tests/base/test_backend_pre_step_control.py` 的 fake MuJoCo backend/pool：给 fake pool
一个可由测试设置的 `was_autoreset` bool array，并给 fake backend 初始化 latch。新增一个
小测试证明 fake substep masks 也按 OR-latch 工作；不要让 production code为了测试替身
增加 optional capability probe。

## 6. 执行顺序：严格 RED → GREEN

### Phase A：起点和 M0-dev

1. 运行第 3 节起点检查。
2. 先创建 `test_mujoco_uni_runtime_contract.py`，要求固定 version/URL/SHA/installed property。
3. 运行下列命令，记录它在当前 `0.3.1` 上因 identity/property 不符而 RED：

~~~bash
uv run --extra mujoco pytest \
  tests/base/backend/test_mujoco_uni_runtime_contract.py -q
~~~

4. 用 `apply_patch` 修改 `pyproject.toml`，再运行：

~~~bash
uv lock
uv sync --extra mujoco --reinstall-package mujoco-uni-runtime
uv run --extra mujoco pytest \
  tests/base/backend/test_mujoco_uni_runtime_contract.py -q
~~~

5. 检查 `uv.lock` 的 diff 只来自 runtime source/version 解析；若无关 packages 大面积升级，
   停止并回报，不能把 lock churn 一起带入。

### Phase B：B1 source-model + mixed layout

1. 先创建 source-model test，运行并确认旧 `ModelVariantSpec` 因不接受
   `source_model_file` 而 RED。
2. 最小修改 `types.py` 和 MuJoCo compile owner。
3. 运行：

~~~bash
uv run --extra mujoco pytest \
  tests/base/backend/test_mujoco_model_source_variants.py -q
uv run --extra mujoco pytest \
  tests/base/test_sim_backend.py::TestMuJoCoBasic::test_apply_init_randomization_sets_variants_before_materialization \
  tests/base/test_sim_backend.py::TestMuJoCoBasic::test_get_playback_model_returns_env_specific_variant \
  -m slow -q
~~~

两个 existing tests 带 `slow` marker，必须用显式 node id 和 `-m slow` 覆盖仓库默认的
`-m "not slow"`，不能让 marker 过滤把它们 deselect。记录真实 pass/skip 数，required
commands 必须 0 skip。

### Phase C：B2 body wrench

1. 先创建 body-wrench test，确认 public method/implementation 缺失导致 RED。
2. 在 `SimBackend` 和 MuJoCo owner 做最小实现。
3. 运行：

~~~bash
uv run --extra mujoco pytest \
  tests/base/backend/test_mujoco_body_wrench.py -q
~~~

### Phase D：B3 autoreset

1. 先创建 real-pool autoreset tests，并扩充现有 pre-step fake；确认 contract/latch 缺失导致
   RED。
2. 在 `SimBackend` 和 MuJoCo step owner 做最小实现。
3. 运行：

~~~bash
uv run --extra mujoco pytest \
  tests/base/backend/test_mujoco_autoreset_real_pool.py \
  tests/base/test_backend_pre_step_control.py -q
~~~

required tests 必须真实触发 autoreset、0 skip，并且仓库根目录没有 `MUJOCO_LOG.TXT`。

## 7. 最终验证

先运行 Code #6 focused gate：

~~~bash
uv run --extra mujoco pytest \
  tests/base/backend/test_mujoco_uni_runtime_contract.py \
  tests/base/backend/test_mujoco_model_source_variants.py \
  tests/base/backend/test_mujoco_body_wrench.py \
  tests/base/backend/test_mujoco_autoreset_real_pool.py \
  tests/base/test_backend_pre_step_control.py -q
~~~

再运行邻近 backend regressions：

~~~bash
uv run --extra mujoco pytest \
  tests/base/test_sim_backend_smoke.py \
  tests/base/test_mujoco_batch_env_randomization.py \
  tests/base/backend/test_mujoco_chunk_size_wiring.py \
  tests/base/backend/test_mujoco_chunk_tuner.py -q
~~~

运行 dependency、style 和 type gates：

~~~bash
uv lock --check
uv run --extra mujoco ruff check \
  src/unilab/dr/types.py \
  src/unilab/base/backend/base.py \
  src/unilab/base/backend/mujoco/backend.py \
  tests/base/test_backend_pre_step_control.py \
  tests/base/backend/test_mujoco_uni_runtime_contract.py \
  tests/base/backend/test_mujoco_model_source_variants.py \
  tests/base/backend/test_mujoco_body_wrench.py \
  tests/base/backend/test_mujoco_autoreset_real_pool.py
uv run --extra mujoco ruff format --check \
  src/unilab/dr/types.py \
  src/unilab/base/backend/base.py \
  src/unilab/base/backend/mujoco/backend.py \
  tests/base/test_backend_pre_step_control.py \
  tests/base/backend/test_mujoco_uni_runtime_contract.py \
  tests/base/backend/test_mujoco_model_source_variants.py \
  tests/base/backend/test_mujoco_body_wrench.py \
  tests/base/backend/test_mujoco_autoreset_real_pool.py
uv run --extra mujoco mypy src/unilab
uv run --extra mujoco pyright
~~~

最后核对 scope 和工作树：

~~~bash
test ! -e MUJOCO_LOG.TXT
git diff --check
git diff --cached --name-only
git status --short
git diff --stat
git diff --name-only
~~~

实现 session 不运行 `make test-all`，不创建/更新 PR。控制 session 接收交接后会独立阅读
完整 diff、复跑近风险 gate、决定是否需要更广验证，然后精确 stage 和 commit。

## 8. 停止条件

出现任一情况立即停止写入并返回 `# BLOCKED`：

1. 起点 branch、lineage、clean tree 或 empty staging 不符合第 3 节；
2. M0-dev 不能从固定 HTTPS Git URL + 完整 SHA clean resolve/build，只能使用 local sibling；
3. 需要修改 MuJoCoUni owner 仓库、制作临时 wheel，或无法证明 dependency identity；
4. 需要新增本 prompt 未列出的 public contract、owner 或第 11 个实现 path；
5. 需要迁移 per-env geom/body side tables、修改 Motrix/MJWarp/Drake 或破坏现有 target
   CPU-affinity/terrain/render/XML 行为；
6. source-model direct compile 必须进入 step/reset 热路径读取 asset/XML；
7. body wrench 需要 env/task special case，或测试只能复制 production 公式才能通过；
8. autoreset 只能靠 `getattr/hasattr` 热路径探测、伪造 mask 或兼容 0.3.1 才能工作；
9. synthetic 12-distribution mixed layout 或真实 autoreset 不能在 M0-dev 实际运行；
10. required test 有 failure、skip 或无法解释的 warning，邻近 regression 被破坏；
11. lock 出现无关大面积升级，手写改动明显超过 800 行，或发现 writer overlap；
12. 任务需要进入 assets/task/env/T0/T1/Runner/Code #7 才能完成。

不要通过放宽断言、删除 12-distribution/real-pool case、把真实测试改成 mock、隐藏 skip、
关闭 wrench/autoreset 或扩大 scope 绕过停止条件。

## 9. 实现 session 交接格式

成功时只以 `# DONE` 开头，并依次报告：

1. 起始/结束 branch 和 HEAD；
2. 10 个允许路径中实际修改的文件，以及确认无范围外改动；
3. `git status --short`、`git diff --stat`、staging 为空；
4. M0-dev version、remote URL、完整 source SHA、lock resolved identity，以及未使用 sibling/
   artifact；
5. Phase A-D 每个初始 RED 的命令、失败原因和最终 GREEN；
6. 12-distribution model assignment、dominant layout 证据和真实 step 结果；
7. body-wrench row/body isolation、force/torque slices、accumulation 和 shape rejection；
8. real-pool baseline/exact autoreset mask、4-substep OR-latch、next-step clear；
9. focused、backend regressions、lock、Ruff、format、mypy、pyright 每条实际命令、exit status、
   pass/skip 数和 warnings；
10. `MUJOCO_LOG.TXT` absence、`git diff --check` 结果；
11. 明确确认没有执行 Git 写操作、没有修改 Source/vendor/MuJoCoUni、没有运行
    `make test-all`、没有进入 Code #7。

阻塞时只以 `# BLOCKED` 开头，给出停止条件编号、最后一个成功 gate、失败命令和关键输出、
当前工作树状态及已创建文件。不要自行清理。

无论 `# DONE` 或 `# BLOCKED`，报告后停止，等待控制 session 审查。
