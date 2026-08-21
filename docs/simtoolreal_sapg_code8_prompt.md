# SimToolReal SAPG Code #8 执行 Prompt

> 本文件交给一个新的实现 session。用户明确说“看
> `docs/simtoolreal_sapg_code8_prompt.md`，按文档执行 Code #8 的全部 child batches，
> 不进入 Code #9”即表示 8A-8D 都已获得 execution approval。实现 session 必须直接完成
> 已批准的全部 child batches，不派 subagent、不重新规划、不进入 Code #9。完成后保留
> 全部改动未暂存、未提交，交回控制/审查 session。

## 1. 本批唯一结果

在 Target 内组合并注册真实 MuJoCo `SimToolRealEnv`，让 Code #7 已固定的 assets、600-tool
catalog 和 backend-neutral task primitives 经 Code #6 public backend contracts 接入完整
`NpEnv` lifecycle，并建立 Target-only 的真实 MuJoCo T1 integration oracle。

完成时必须同时具备：

1. `registry.ensure_registries()` 和 `registry.make("SimToolReal", "mujoco")` 可创建真实 env；
2. 600 个 tool source models 在 init/materialization 冷路径真实编译，固定 env-to-tool
   assignment 生效；
3. `init_state/reset/step` 返回正确的 dict obs、raw reward、termination、timeout、autoreset、
   final observation 和 info；
4. action/delay、observation/reward、goal、reset、wrench 和 episode lifecycle 按 Code #7
   primitives 的既定顺序接入；
5. N=6 的 Target-only T1 fixture 可在固定 runtime 下独立逐字节再生成，并由普通 pytest
   重新运行真实 MuJoCo env 后 replay；
6. production step/reset 热路径不读取 XML/asset metadata，不调用 backend 私有字段或方法。

T1 不是 Source IsaacSim 与 Target MuJoCo 的轨迹对拍。它不访问 Source checkout，不比较
接触、随机序列、physics state 或 reward curve 的跨 simulator 等价性。Code #7 T0 继续负责
给定同一 primitive inputs 时的 task math；Code #8 T1 只锁定这些 production primitives
在真实 Target backend、tool pool 和 `NpEnv` lifecycle 中确实接通。

实现 session 不执行 `git add`、`git commit`、`git push`、PR、stash、reset、clean、
checkout 或切分支。控制 session 审查后提交 Code #8 代码，并另行更新总指导文档的完成
状态。

## 2. 普通中文范围、规模和 child batches

### 2.1 只做什么

- 从固定 donor env owner 移植真实 env composition，但继续使用 Target Code #7 已审查的
  config/task modules，不把 donor 的 RSL-RL owner/scaling 带回来；
- 给 `SimToolRealCfg` 和 `SimToolRealEnv` 增加 UniLab registry owner，并加入 manipulation
  registry bootstrap；
- 在 env 构造期完成 catalog、tool XML materialization、backend creation、model variants、
  immutable indices/limits/body IDs/geometry/tool caches 和 DR manager 初始化；
- 在 env hooks 中组合 action、wrench、physics step、task state、raw reward、obs、goal、
  termination、timeout 和 autoreset；
- 用真实 MuJoCo/MuJoCoUni 覆盖 600-tool compile、partial reset、finite steps、forced timeout、
  engine autoreset 和 wrench handoff；
- 生成一次 Target-only T1，普通 pytest 只读取已提交 fixture 并重新执行同一个真实 capture。

### 2.2 明确不做什么

- 不新增任何 Hydra conf/YAML；Code #9 才拥有 native RL-Games SAPG production owner；
- 不接 Runner、`RlGamesNpEnvAdapter`、tracker、W&B、pth resolver、player、train/eval CLI；
- 不增加 RL-Games/RSL-RL 算法参数，不预缩 reward，不建立第二个 rollout loop；
- 不做 IsaacSim/MuJoCo trajectory、contact、randomness 或 training curve 对拍；
- 不重新生成、修改或放宽 T0；
- 不新增 Motrix、sim2sim、viewer/video、async/distributed/export 或 support claim；
- 不修改 Source、donor、vendored RL-Games、MuJoCoUni、Code #1-#6 backend owner、
  `pyproject.toml` 或 `uv.lock`；
- 不运行 `make test-all`，不创建或更新 PR，不进入 Code #9/#10。

### 2.3 规模和永久维护成本

生产面预计是 4 个必要路径加 1 个只在真实 RED 证明需要时才可调整的 reset owner：

- 一个少于 800 physical lines 的 `env.py`，主要来自固定 donor 的成熟 owner；
- config/package/manipulation 三处窄注册改动；
- 最多一个 `dr_provider.py` real-composition correction，不得借机改 task math。

测试面预计是 4 个 text owners、一个 generator 和两个 fixture，约 1.2k-2k test/harness
LOC。手写 adaptation 应集中在 Target raw-reward boundary、registry、public backend surface、
T1 capture 和当前 `NpEnv` lifecycle，预计约 200-400 行；不得重新发明 env 算法。

永久维护成本只有真实 env/registration、real integration tests 和一组 Target T1 fixture。
没有 Hydra/Runner/checkpoint/player 的永久成本；那些属于 Code #9。

### 2.4 Child batches

Code #8 是 roadmap umbrella，用户本次明确批准顺序执行以下全部 child batches：

~~~text
8A  registry、env construction、immutable runtime caches
8B  reset/action/step/obs/reward/termination/timeout/info lifecycle
8C  real 600-tool compile、wrench、engine-autoreset near-risk integration
8D  Target-only real MuJoCo T1 generator、fixture、manifest、replay
~~~

每个 child 必须保持一个可独立验证的结果、约 15 个以内的 touched paths 和约 800 行以内的
净手写 adaptation。`env.py` 的 donor mechanical port 不伪装成新算法设计；仍必须低于 800
physical lines。若需要新 public backend contract、新 production helper path、Hydra owner 或
明显超出上述规模，停止回报，不得顺手扩张。

## 3. 必读内容、起点和固定身份

### 3.1 开始前完整阅读

~~~text
AGENTS.md
docs/simtoolreal_sapg_source_fidelity_migration_plan.md
docs/simtoolreal_sapg_code8_prompt.md
src/unilab/base/base.py
src/unilab/base/np_env.py
src/unilab/base/registry.py
src/unilab/base/backend/base.py
src/unilab/dr/manager.py
src/unilab/dr/types.py
src/unilab/envs/manipulation/__init__.py
src/unilab/envs/manipulation/simtoolreal/config.py
src/unilab/envs/manipulation/simtoolreal/dr_provider.py
src/unilab/envs/manipulation/simtoolreal/observations.py
src/unilab/envs/manipulation/simtoolreal/rewards.py
src/unilab/envs/manipulation/simtoolreal/episode_lifecycle.py
src/unilab/envs/manipulation/simtoolreal/tool_assets.py
src/unilab/envs/manipulation/simtoolreal/tool_catalog.py
tests/envs/manipulation/simtoolreal/test_t0_golden.py
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
eb19779ecef69bbfb495abee9e7e2c4d5988f3ac
docs: record SAPG Code 7 completion
~~~

实现 session 的 dispatch HEAD 应是上述基线的单个 docs child。该 docs child 只新增本
prompt，并把总指导文档中的 Code #8 状态更新为“已规划，待实现”。开始时运行：

~~~bash
set -e
set -o pipefail
SAPG_CODE8_BASE=eb19779ecef69bbfb495abee9e7e2c4d5988f3ac
test "$(git rev-parse --abbrev-ref HEAD)" = "feat/simtoolreal-sapg-rlgames"
git merge-base --is-ancestor "$SAPG_CODE8_BASE" HEAD
test "$(git rev-list --count "$SAPG_CODE8_BASE"..HEAD)" -eq 1
test "$(git diff --name-status "$SAPG_CODE8_BASE"..HEAD)" = \
  $'A\tdocs/simtoolreal_sapg_code8_prompt.md\nM\tdocs/simtoolreal_sapg_source_fidelity_migration_plan.md'
test -z "$(git status --short)"
test -z "$(git diff --cached --name-only)"
git log -2 --oneline
git status --short --branch
~~~

任一条件不成立就返回 `# BLOCKED`，不要清理或覆盖现有改动。

### 3.2 固定 Source reference

Code #8 不运行 Source，也不从 Source 生成 T1。只保留路线 provenance：

~~~text
Source checkout reference:
  /home/user/ws/lemon/simtoolreal
Source HEAD:
  2a9917533bfea70419ed2667a511d7238e5b3abc
task owner:
  isaacsimenvs/cfg/task/SimToolReal.yaml
task owner blob:
  6469d46867081b70edaa589dcb31c7090b64d45e
~~~

普通测试、T1 generator 和 production runtime 都不得访问这个 checkout。T0 fixture 已在
Code #7 固定，Code #8 只运行其普通 Target replay 测试。

### 3.3 固定 mature donor snapshot

~~~text
Donor repository:
  /home/user/ws/lemon/UniLab
Donor commit:
  74075b3238e3176650a9440984a74be3629ff93f
Donor subject:
  revert(simtoolreal): disable mocap table reset
~~~

只通过固定 Git objects 读取 donor，禁止从 donor 当前 dirty working tree 复制，禁止
cherry-pick。Code #8 重点参考 blobs：

~~~text
src/unilab/envs/manipulation/simtoolreal/env.py
  103089c1e8192e25326a58de46be516808853013
src/unilab/envs/manipulation/simtoolreal/__init__.py
  2a2536f77d7ca7f9031c1907047b7a82afb2f47e
src/unilab/envs/manipulation/simtoolreal/config.py
  8e3615cea8769d3c07149dfedb87ee9d18c1bf9f
src/unilab/envs/manipulation/__init__.py
  d6c3c35eaff3a99ac0fcdbaaa52846e9281bd4d9
tests/envs/manipulation/simtoolreal/test_integration.py
  19475702ff90e13312a1a9a6a1e7c73d8c8dac3c
tests/envs/manipulation/simtoolreal/test_tool_pool_integration.py
  f8a05f2dd3aec8e8906b5c858b973d50a82568a9
tests/envs/manipulation/simtoolreal/test_reset_observations.py
  b2b1a20ad4e7517be10a0af8297bf40c859551f3
tests/simtoolreal/test_autoreset_shipped_path.py
  e24a10c1c8f5f55df2931715233faefbb77cff5f
tests/simtoolreal/test_compiled_tool_pool_shipped_path.py
  b004234f2ac9ee24c2e6383407c23d36d3689d5e
~~~

Donor 是成熟参考，不是可盲拷贝的最终 Target owner：

- Target Code #7 的 raw reward defaults/return 与 critic reward feature ×0.01 已经审查，
  不能用 donor RSL-RL scaling 覆盖；
- donor Hydra/RSL-RL configs、training helpers 和后续 debug tests 不属于本批；
- donor tests 中 backend-private inspection 只能提炼为 test-only near-risk assertion，不能
  进入 production env；
- Target 当前 Code #6 public `source_model_file`、`apply_body_wrench` 和
  `get_step_autoreset_mask` contracts 是唯一 backend 接线面。

### 3.4 固定 Target runtime identity

Code #8 使用 Code #6 已锁定的开发 runtime：

~~~text
mujoco-uni-runtime version: 0.4.0.dev0
Git URL: https://github.com/lemon-star608/mujoco_uni.git
source SHA: 7205e070e983df90d520f0f8593853013e976746
cpu_ids: null / None
~~~

开始前必须用 `uv run` 验证安装态 version、`direct_url.json` identity 和真实
`BatchEnvPool.was_autoreset` property；不允许从 sibling checkout 偷换依赖。身份不符就
`# BLOCKED`，本批不修改 dependency lock。

~~~bash
uv run --extra mujoco - <<'PY'
import importlib.metadata
import json

from mujoco_uni.batch_env import BatchEnvPool

package = "mujoco-uni-runtime"
expected_url = "https://github.com/lemon-star608/mujoco_uni.git"
expected_rev = "7205e070e983df90d520f0f8593853013e976746"
distribution = importlib.metadata.distribution(package)
direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
assert importlib.metadata.version(package) == "0.4.0.dev0"
assert direct_url["url"] == expected_url
assert direct_url["vcs_info"]["commit_id"] == expected_rev
assert direct_url["vcs_info"]["requested_revision"] == expected_rev
assert isinstance(BatchEnvPool.was_autoreset, property)
PY
~~~

## 4. 唯一允许新增或修改的路径

### 4.1 Production paths

~~~text
src/unilab/envs/manipulation/__init__.py
src/unilab/envs/manipulation/simtoolreal/__init__.py
src/unilab/envs/manipulation/simtoolreal/config.py
src/unilab/envs/manipulation/simtoolreal/env.py
src/unilab/envs/manipulation/simtoolreal/dr_provider.py
~~~

`dr_provider.py` 只有在新的 real integration test 先证明当前 reset-composition contract
失败时才允许最小修正。若它不需要修改，保持原 bytes；不得以“顺便清理”为由改写。

### 4.2 Tests、T1 harness/generator 和 fixtures

~~~text
tests/envs/manipulation/simtoolreal/test_env_registration.py
tests/envs/manipulation/simtoolreal/test_env_integration.py
tests/envs/manipulation/simtoolreal/target_t1_harness.py
tests/envs/manipulation/simtoolreal/test_t1_golden.py
scripts/generate_simtoolreal_task_t1_fixture.py
tests/fixtures/simtoolreal_task/target_t1_fp32.npz
tests/fixtures/simtoolreal_task/target_t1_manifest.json
~~~

不得修改本 prompt、总指导文档、Code #7 unit tests/T0 files、assets、backend、conf、vendor、
dependency files 或任何其他路径。若一个真实 RED 只能在范围外 owner 修复，停止并报告需要
新增的精确 path/contract，不要自行扩大 whitelist。

所有 Python 命令必须通过 `uv run`。手工文本编辑只使用 `apply_patch`。NPZ/manifest 只能
由本 prompt 规定的 generator 写入；不得用 ad-hoc Python、`cat` 或测试运行静默 rebaseline。

## 5. 8A：registry、construction 和 immutable caches

### 5.1 Registry composition

注册顺序必须明确且无隐式 fallback：

1. `SimToolRealCfg` 使用 `@registry.envcfg("SimToolReal")` 注册唯一 config owner；
2. `SimToolRealEnv` 使用 `@registry.env("SimToolReal", sim_backend="mujoco")` 注册唯一 env
   owner；
3. package `__init__.py` 先暴露 config，再 import env 触发 registration，并导出
   `SimToolRealEnv`；
4. manipulation `__unilab_registry_modules__` 加入
   `unilab.envs.manipulation.simtoolreal`；
5. clean subprocess 中调用 `registry.ensure_registries()` 后，`list_registered_envs()` 必须
   恰有 `SimToolReal -> mujoco`；
6. `registry.make("SimToolReal", sim_backend="mujoco", num_envs=6)` 必须返回真实
   `SimToolRealEnv`，不经过 training script 或 Hydra。

重复 import 依赖 Python module identity 保持幂等；不要在 decorators 外增加
`if not registry.contains(...)` 的静默 duplicate workaround。配置或 env 重复注册应继续由
registry fail closed。

### 5.2 Cold construction order

以 donor env blob 为成熟基线，Target env 构造顺序必须保持：

1. `cfg.validate()`；
2. local deterministic catalog 12×50=600；
3. fixed env assignment `arange(num_envs) % 600`；
4. `materialize_tool_scenes` 一次生成 600 complete source XML；
5. 使用第一个 complete model 创建 MuJoCo backend；
6. `super().__init__` 建立 `NpEnv` carrier；
7. 从 public backend methods 缓存 joint permutations/limits、default pose、body IDs、
   qpos/qvel layout、geometry constants、object scales/masses、delay/wrench/step buffers；
8. `SimToolRealDRProvider` 经 `NpEnv._init_domain_randomization` 提交 600
   `ModelVariantSpec(source_model_file=...)` 和 env assignments；
9. backend `materialize()` 完成后才允许 reset/step。

所有 variant 的 robot/body/joint layout 必须一致；env 在冷路径对 29 robot hinges + one
object free joint fail closed：`nq=36`、`nv=35`、`nu=29`。canonical/backend permutation
必须从 public joint-index contracts 构建，不假定 identity；arm 是 canonical 0:7，hand 是
7:29。

### 5.3 600-tool owner boundary

- production default `object_pool_enabled=True`；
- catalog 必须仍是 Code #7 固定的 600 specs 和 250/300/50 topology census；
- init plan 必须包含恰好 600 个 non-empty `source_model_file` variants；
- assignments 是 length N 的 int32 fixed mapping，不在 reset 时重新抽取；
- object scale、mass、keypoint offsets 全在 init 冷路径按 assignment 缓存并设为 immutable
  where appropriate；
- reset/step 不得读取 `_tool_catalog`、XML、model metadata 或 geom metadata来重建值；
- `object_pool_enabled=False` 仍使用 catalog index 0 的真实 compiled source model，不允许
  placeholder cube 或 side-table override；
- variant count/assignment 越界必须 fail closed，不能 remap 到 tool 0。

test-only near-risk code可以只读检查 backend compiled variants/assignments，以证明 600 个
真实 models 已进入 pool；production env 绝不能访问 `_pool`、`_model_variants`、
`_model_assignments`、`_pending_per_env_fields` 或其他 backend 私有字段。

### 5.4 Resource cleanup

`close()` 必须先让 base/backend owner 释放自己的 scene resources，再释放
`MaterializedToolScenes.cleanup`，且重复 close/异常路径不能遗留 generated XML。不得从 env
直接调用 backend private pool close。构造中途失败时也必须尽可能清理 tool tempdir；若
donor owner在当前 base lifecycle下存在明确泄漏，必须在本 path内最小修正并有测试，不能
新增 backend contract。

## 6. 8B：真实 NpEnv lifecycle

### 6.1 Public spaces 和 state

真实 env 必须满足：

~~~text
action_space: Box(-1, 1, shape=(29,), dtype=float32)
obs_groups_spec: {"obs": 140, "critic": 162}
NpEnvState.obs: dict[str, ndarray]
reset(env_ids): (obs_dict, info_dict)
step(actions): NpEnvState
~~~

`init_state()` 第一次全量 reset 后，两个 obs groups 都必须是 finite、non-zero、float32，不能
返回 zero stub。info 至少包含 steps、action targets/actions、goal、success/lift/d-star、
object state/scale、raw reward、log 和 base final-observation compatibility fields，shape 按
N 或 reset rows 明确。

### 6.2 Action 和 pre-physics order

`apply_action` 必须按 Source/Code #7 已固定顺序：

1. policy action clamp；
2. canonical→backend permutation；
3. action delay；
4. arm delta/double clamp、hand absolute mapping、EMA；
5. state.info target/action bookkeeping；
6. previous-step lifted latch驱动 wrench DR；
7. public `SimBackend.apply_body_wrench`；
8. 返回 backend-order position targets 给 base `backend.step(..., sim_substeps=2)`。

不能在 env 重新写一份 action/delay/wrench 公式。shape/dtype错误必须由既有 task owner
fail closed，不得 silently reshape/broadcast。

### 6.3 Post-physics order 和 reward boundary

`update_state` 必须按以下顺序组合既有 owners：

1. public `get_step_autoreset_mask()` 并处理 engine-reset cache invalidation；
2. tolerance curriculum；
3. body/joint/object/keypoint/fingertip intermediate values；
4. success gate和 same-episode goal advance；
5. 七项 raw reward和 raw total；
6. `info["reward"]` 写 raw total；
7. actor 140 / critic 162 observations；
8. task terminated/truncated masks；
9. previous-step lifted latch snapshot和 scalar log。

env/state reward保持 Source raw scale 200/20/300/50/1000/0.03/0.003 的总和；不得在 env
乘 0.01。critic observation 的 reward feature 仍由 Code #7 observation owner单独取
`raw_reward * 0.01`。Code #9 native RL-Games reward shaper 才会对 learner reward再做一次
0.01；本批不得提前实现或测试那个 learner boundary。

### 6.4 Success、termination、timeout 和 autoreset

- success只推进 goal、清 goal-local trackers并把 steps归零，不结束 episode；
- object drop、hand-far、max successes 和 backend engine-autoreset进入 `terminated`；
- 600 policy steps由 base `NpEnv._compute_truncated`进入 `truncated`；
- `done = terminated | truncated` 后由 base autoreset selected rows；
- returned done mask保持 terminal step，returned `state.obs` 是 reset后的 policy obs；
- terminal obs必须在 `state.final_observation` 和 compatibility info buffers按 exact row保存；
- partial reset只修改 selected rows，其他 obs/info/delay/wrench histories不变；
- engine-autoreset exact row必须清 stale targets、object-init-z、d-star和 lifted cache，再触发
  正常 env reset；unknown mask (`None`) 不能被当成 all-false capability claim。

Env 只能调用 `SimBackend` 已声明 public methods。不得用 `getattr/hasattr` 猜
`BatchEnvPool.was_autoreset`、不得直接 import MuJoCoUni、不得读取 backend subclass fields。

## 7. 8C：real near-risk integration tests

`test_env_registration.py` 至少覆盖：

1. clean subprocess registry bootstrap；
2. config/env owner和 only-mujoco backend identity；
3. package exports；
4. 不依赖 Hydra、Source checkout或 training modules。

`test_env_integration.py` 使用 module-scoped real MuJoCo fixture amortize 600-model compile，
默认 N=6、`cpu_ids=None`，所有 required tests不得 skip。至少覆盖：

1. 真实 `registry.make`、600 catalog、600 materialized XML、600 compiled model variants和
   fixed assignments；
2. 全 600 compiled models 的 object mass/COM/inertia、handle topology和 250/300/50 census
   与 immutable `ToolSpec` 一致；
3. 全部 compiled models共享 `nq/nv/nu/nmesh=36/35/29/40` layout，完整 inventory同时包含
   box-box、capsule-box和box-only三种 topology；
4. first reset、`init_state`、partial reset的 non-zero 140/162 obs和 row isolation；
5. deterministic small-action/zero-action至少 64 real steps保持 obs/reward/terms finite，
   raw total等于 reward terms直接和；
6. controlled success真实推进 goal而不 terminated/truncated；
7. controlled row在 step前置 `steps=599` 后 exact timeout、terminal observation和 selected
   autoreset，其他 rows不变；
8. force/torque probability=1且 previous-step lifted latch=true时，env经 public
   `apply_body_wrench` stage non-zero wrench；未 lifted rows保持0；
9. 通过 public backend state/set_state制造一个 env的 divergent qvel，真实
   `get_step_autoreset_mask`和 env `_autoreset_envs`只标记该 row，cache重置且下一正常 step
   清 latch；
10. close后 materialized tool temp root不存在。

Module-scoped fixture必须在创建前固定 NumPy seed；每个测试负责 reset所需 rows、恢复临时
config/cache mutation，不能依赖 pytest执行顺序。`raw total`只与七个 component terms求和，
不能把 `total_reward` 自己再加一次。

真实 engine-autoreset helper可读取 public `backend.model.nq/nv` 和
`backend.get_physics_state()` 的 documented FULLPHYSICS layout，再调用 public
`backend.set_state()`；不得直接写 pool arrays。测试若为验证“确有600个 compiled variants”
而只读 backend private inventory，必须局限于测试并写明这是 near-risk inspection，不能把
相同 access复制到 production。

不要复制 donor 700-line debug suites、Hydra/RSL-RL config tests、collision research或性能
probe。一个 focused integration owner覆盖上述真实边界即可。

## 8. 8D：Target-only T1 contract

### 8.1 固定 capture case

`target_t1_harness.py` 是测试侧唯一 capture owner。固定：

~~~text
generation_mode: target-real-mujoco-only
N: 6
dtype: CPU FP32 policy/task arrays
seed: 20260821
policy steps H: 8
pool: production default 12x50=600, shuffle seed 42
backend: mujoco + pinned mujoco-uni-runtime 0.4.0.dev0
~~~

Capture 必须通过 registry composition创建 env并在 `finally` close。事件顺序固定：

1. 在 env construction前设置 global NumPy seed；
2. 调用 `init_state()`，记录第一次真实 reset；
3. 使用同时写入 fixture的 8×6×29 float32 actions；固定公式为
   `0.25 * sin(0.37*t + 0.11*env + 0.07*joint)`，先按 NumPy默认浮点计算再统一
   `astype(float32)`；
4. 执行前4个真实 steps；
5. 把 env rows `[1, 4]` 标为 done并走 `NpEnv._reset_done_envs()` 的真实 selected-row
   scatter path，记录 reset后的 obs/info和 terminal compatibility buffers；
6. 再执行3个真实 steps；
7. 把 row 2 的 `info["steps"]`置为599，执行最后一个 action，记录 exact timeout、
   final observation和 autoreset后的 obs。

默认 action/obs/object-state delay和 observation noise保持开启；依靠固定 NumPy seed锁定
Target capture，不为了容易比较关掉 production branches。T1 不故意制造 MuJoCo divergence，
也不强制 lifted wrench；这两个高风险事件由第7节独立真实测试覆盖，避免把异常 physics
混进 golden trajectory。

### 8.2 必须保存的 inputs/outputs

NPZ 至少包含：

- scripted actions、partial-reset ids、timeout row；
- tool indices、object scales，以及六个 public `get_playback_model(env_index)`可见的
  `nq/nv/nu/nmesh/ngeom` signatures；
- initial obs/critic和关键 reset info；
- selected-row reset后的 obs/critic、goal/object/action trackers和 exact reset mask；
- 每步 obs、critic、raw reward、terminated、truncated、steps；
- successes、near-goal、lifted、d-star、goal pose；
- prev/cur targets、object pose/quaternion；
- 固定顺序的八个 reward entries：`fingertip_delta_rew`、`lifting_rew`、
  `lift_bonus_rew`、`keypoint_rew`、`kuka_actions_penalty`、
  `hand_actions_penalty`、`bonus_rew`、`total_reward`；
- backend autoreset latch；
- timeout step的 final obs/critic和 exact terminal mask。

不要保存 wall-clock timing、临时 XML path、object arrays、Python dict pickle、renderer state或
backend-private object identity。所有 float必须 finite；bool/int/mask/index dtypes明确。

### 8.3 Generator 和 manifest

~~~text
scripts/generate_simtoolreal_task_t1_fixture.py
tests/fixtures/simtoolreal_task/target_t1_fp32.npz
tests/fixtures/simtoolreal_task/target_t1_manifest.json
~~~

Generator 必须：

1. `#!/usr/bin/env python3`，但始终由 `uv run --extra mujoco`调用；
2. 只 import Target harness/production和已安装 runtime，不访问 Source/donor；
3. 要求显式 `--output`目录和 `--target-only`，普通 pytest绝不调用它；
4. deterministic key order、fixed ZIP metadata、`allow_pickle=False`兼容 NPZ；
5. sorted/UTF-8/newline-stable JSON；
6. 构造失败或 capture mismatch时清理 env/temp assets，不留下半个正式 fixture。

Manifest 至少记录：

- schema version、`generation_mode="target-real-mujoco-only"`、
  `ordinary_pytest_regenerates=false`、`source_accessed=false`；
- Code #8 base、Source reference identity、donor commit/env blob；
- Target production/harness/generator files的 sha256；
- asset provenance外层 sha256；
- Python/NumPy/MuJoCo/mujoco-uni-runtime versions和 M0-dev direct source SHA；
- N/H/seed、event script、config values、reward-term order和 allowed mapping说明；
- NPZ array exact inventory：name/shape/dtype/sha256；
- fixture filename、NPZ sha256、canonical payload sha256和 canonical command；
- discrete exact inventory；
- float tolerance `rtol=1e-5, atol=1e-6`；
- 明确声明这是 Target regression，不是 Source/Target physical parity。

Manifest 不记录自己的 hash，避免 self-hash循环。`test_t1_golden.py` 固定 NPZ和manifest的
外层 SHA256 anchors；不要把 anchors写入 manifest会 hash的 harness/generator。

### 8.4 Ordinary replay

普通 `test_t1_golden.py` 必须：

- 不访问 `/home/user/ws/lemon/simtoolreal` 或 donor；
- 不 import IsaacSim/IsaacLab；
- 不运行 generator、不更新 fixture；
- 先验证两个外层 hashes、manifest schema/runtime/source-access声明和 array inventory；
- 再调用 harness重新创建真实 MuJoCo env并 capture；
- bool/int/mask/index exact equality；
- float按 manifest tolerance比较；
- mismatch报告 array name、max abs/rel error和首个 bad index；
- required dependency缺失直接 fail，不使用 `pytest.skip`/`importorskip`。

同一环境不能逐字节再生成 NPZ/manifest，或 replay只能靠扩大 tolerance/关掉 delay/noise，
立即停止。先判断是 capture ordering、RNG污染、dtype、backend identity还是 production defect；
不得静默 rebaseline。

## 9. 严格 RED → GREEN 顺序

### Phase 0：起点与只读 census

1. 运行第3.1节全部检查；
2. 验证 donor blobs和 M0-dev安装身份；
3. 运行当前 Code #7 focused gate，确认基线仍 GREEN；
4. 记录当前没有 `env.py`、没有 SimToolReal registry owner、没有 T1 files。

### Phase 8A：registry 和 env construction

1. 先创建 `test_env_registration.py`，断言 clean bootstrap/config/env owner；
2. 运行并记录它因 SimToolReal尚未注册而真实 RED：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal/test_env_registration.py -q
~~~

3. 最小增加 config/env decorators和两个 package bootstrap改动；
4. 从 fixed donor env Git object移植 `env.py`，适配当前 Target public contracts和 raw reward
   boundary；
5. 运行 registration test和一个真实 `registry.make` construction/reset smoke，要求 GREEN、
   0 skip；
6. 确认 `env.py < 800` physical lines且无 backend private access。

### Phase 8B：NpEnv lifecycle

1. 先创建 `test_env_integration.py` 的 reset/step/success/timeout assertions；
2. 在 production env尚不满足时记录真实 behavior RED，再做最小 owner-layer修正；
3. 若 donor mechanical port使某组测试首次即 GREEN，如实记录“first-run GREEN”，不要制造
   假 bug 或改错 expected只为凑 RED；
4. 完成 partial reset、64 finite steps、raw reward、success和timeout/final-observation gate；
5. 只有真实 reset-composition RED需要时才修改 `dr_provider.py`。

### Phase 8C：600 pool、wrench、autoreset

1. 在同一 focused integration owner中先加 600 compiled variants、wrench和engine-autoreset
   tests；
2. 先运行并记录首次结果；不能只靠 Code #6 fake/small-model tests代替真实 task path；
3. 使用 Code #6 public contracts完成最小 wiring；
4. 要求 exact row isolation、真实 600-model census和 cleanup GREEN、0 skip。

### Phase 8D：T1

1. 先创建 `test_t1_golden.py`，确认 fixture/harness缺失导致真实 RED；
2. 创建 harness和generator，首次输出到 `mktemp -d`审查 array inventory/runtime identity；
3. 显式生成 reviewed fixtures：

~~~bash
uv run --extra mujoco scripts/generate_simtoolreal_task_t1_fixture.py \
  --output tests/fixtures/simtoolreal_task \
  --target-only
~~~

4. 运行 ordinary real replay并要求 GREEN、0 skip；
5. 再生成到新 `mktemp -d`，NPZ和manifest都必须逐字节 `cmp`成功。

## 10. 最终验证

先运行 Code #7 + Code #8 focused gate：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal -q
~~~

required tests必须 0 skip；记录实际 passed/skipped/warnings和耗时。Code #7 T0 replay必须仍
通过，但不运行 Source T0 generator。

运行 registry和 Code #6邻近 backend regressions：

~~~bash
uv run --extra mujoco pytest \
  tests/base/test_registry.py \
  tests/base/backend/test_mujoco_uni_runtime_contract.py \
  tests/base/backend/test_mujoco_model_source_variants.py \
  tests/base/backend/test_mujoco_autoreset_real_pool.py \
  tests/base/test_mujoco_batch_env_randomization.py -q
~~~

验证 T1 deterministic reproduction：

~~~bash
SAPG_CODE8_T1_REGEN=$(mktemp -d)
uv run --extra mujoco scripts/generate_simtoolreal_task_t1_fixture.py \
  --output "$SAPG_CODE8_T1_REGEN" \
  --target-only
cmp tests/fixtures/simtoolreal_task/target_t1_fp32.npz \
  "$SAPG_CODE8_T1_REGEN/target_t1_fp32.npz"
cmp tests/fixtures/simtoolreal_task/target_t1_manifest.json \
  "$SAPG_CODE8_T1_REGEN/target_t1_manifest.json"
sha256sum \
  tests/fixtures/simtoolreal_task/target_t1_fp32.npz \
  tests/fixtures/simtoolreal_task/target_t1_manifest.json
~~~

运行 style/type/lock gates：

~~~bash
uv lock --check
uv run --extra mujoco ruff check \
  src/unilab/envs/manipulation/__init__.py \
  src/unilab/envs/manipulation/simtoolreal \
  tests/envs/manipulation/simtoolreal \
  scripts/generate_simtoolreal_task_t1_fixture.py
uv run --extra mujoco ruff format --check \
  src/unilab/envs/manipulation/__init__.py \
  src/unilab/envs/manipulation/simtoolreal \
  tests/envs/manipulation/simtoolreal \
  scripts/generate_simtoolreal_task_t1_fixture.py
uv run --extra mujoco mypy src/unilab
uv run --extra mujoco pyright
~~~

最后审计 cold path、backend isolation、scope和工作树：

~~~bash
test ! -e MUJOCO_LOG.TXT
test -z "$(git diff --cached --name-only)"
git diff --check
git status --short
git diff --stat
git diff --numstat
git diff --name-only
git ls-files --others --exclude-standard
test "$(wc -l < src/unilab/envs/manipulation/simtoolreal/env.py)" -lt 800
if rg -n "_backend\._|backend\._|getattr\([^\n]*backend|hasattr\([^\n]*backend" \
  src/unilab/envs/manipulation/simtoolreal/env.py; then
  exit 1
fi
if rg -n "read_text|read_bytes|open\(|from_xml" \
  src/unilab/envs/manipulation/simtoolreal/env.py; then
  exit 1
fi
rg -n "read_text|read_bytes|open\(|from_xml" \
  src/unilab/envs/manipulation/simtoolreal || true
rg -n "/home/user/ws/lemon/(simtoolreal|UniLab)|isaacsim|isaaclab" \
  src/unilab/envs/manipulation/simtoolreal \
  tests/envs/manipulation/simtoolreal/target_t1_harness.py \
  scripts/generate_simtoolreal_task_t1_fixture.py || true
~~~

审计解释：

- production asset/XML reads只能保留 Code #7 已接受的 `tool_assets.py`显式 materialization
  和 `dr_provider.py` fixed-trajectory init/cache；`env.py`必须 0 hit；
- Source/Isaac名称可出现在 docstring/provenance，不得是 runtime filesystem access/import；
- production env必须 0 backend-private hit；test-only compiled inventory inspection单独人工审查；
- 工作树只能有第4节 whitelist，staging为空；
- 所有 warnings必须解释，required tests不得 skip/fail。

实现 session 不运行 `make test`、`make test-all`，不创建或更新 PR。控制 session会完整阅读
diff、复跑近风险 gates、核对 T1 hashes/runtime identity和工作树，再决定提交。

## 11. 停止条件

出现任一情况立即停止写入并返回 `# BLOCKED`：

1. branch、lineage、single-docs-child、clean tree或empty staging不符合第3.1节；
2. fixed donor Git objects或 M0-dev安装 identity不符；
3. 只能从 dirty donor/Source working tree读取实现，不能从固定 Git object取证；
4. registry/env接线需要 Hydra、training script或 Code #9 owner才能创建 env；
5. 真实 600 source models不能在 init冷路径 compile/materialize，或只能退回 placeholder/
   per-geom side table；
6. env需要调用 backend private field/method、直接 import MuJoCoUni或用 `getattr/hasattr`
   capability probe才能工作；
7. step/reset/DR热路径需要读取 XML/asset/model metadata；
8. 需要修改 backend public contract、MuJoCoUni、dependency lock、Source、vendor或 conf；
9. real lifecycle要求改变 Code #7 task formula、raw reward scale、obs ordering、success/
   termination/reset semantics；
10. T1出现无法解释的 nondeterminism/mismatch，或只能靠扩大 tolerance、关 delay/noise、删
    event branch才能通过；
11. required test有 failure、skip或无法解释 warning；
12. `env.py`达到800行、任一 child明显超过约800行净手写 adaptation或需要新增未批准
    production path/public owner；
13. T1普通 pytest访问 Source/donor、运行 generator或静默更新 fixture；
14. cleanup后残留 generated XML、`MUJOCO_LOG.TXT`或其他 artifact；
15. 出现 writer overlap、范围外工作树改动或 staging不再为空。

不得通过隐藏 skip、mock真实 backend、减少600 pool、预缩 reward、删除 autoreset/wrench/
timeout coverage、使用 backend private hot path、静默 rebaseline或进入 Code #9 绕过停止条件。

## 12. 实现 session 交接格式

成功时只以 `# DONE` 开头，并依次报告：

1. 起始/结束 branch和HEAD；
2. 8A-8D每个 child实际修改路径、行数和确认无范围外改动；
3. `git status --short`、tracked/untracked inventory、staging为空；
4. donor commit/env blob、Source reference和 M0-dev安装 identity；
5. registry bootstrap、config/env owner和 `registry.make`结果；
6. env construction顺序、29/140/162/600-step contracts、`env.py`行数；
7. 600 catalog/materialized/compiled variant census、assignments和 model 36/35/29/40结果；
8. reset/step/raw reward/success/timeout/final-observation/partial-row integration证据；
9. wrench和真实 engine-autoreset exact-row/cache/next-step-clear证据；
10. Phase 8A-8D首次 RED或诚实 first-run GREEN、失败原因和最终 GREEN；
11. T1 N/H/seed/event/array inventory、fixture两个 SHA256、runtime manifest、deterministic
    cmp和 ordinary real replay结果；
12. focused、registry/backend neighbors、Ruff、format、mypy、pyright、lock每条命令的
    exit status、pass/skip/warning数和耗时；
13. cold-path/private-leakage/source-access/cleanup审计、`MUJOCO_LOG.TXT` absence和
    `git diff --check`；
14. 明确确认没有执行Git写操作、没有修改Source/donor/vendor/MuJoCoUni/backend/conf/
    dependency、没有运行 `make test-all`、没有进入 Code #9。

阻塞时只以 `# BLOCKED` 开头，给出停止条件编号、最后一个成功 child/gate、失败命令和关键
输出、当前工作树状态及已创建文件。不要自行清理。

无论 `# DONE` 或 `# BLOCKED`，报告后停止，等待控制 session 审查。
