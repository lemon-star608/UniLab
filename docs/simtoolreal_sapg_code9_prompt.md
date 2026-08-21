# SimToolReal SAPG Code #9 执行 Prompt

> 本文件交给一个新的实现 session。用户明确说“看
> `docs/simtoolreal_sapg_code9_prompt.md`，按文档执行 Code #9 的全部 child batches，
> 不进入 Code #10”即表示 9A-9D 都已获得 execution approval。实现 session 必须直接完成
> 已批准的全部 child batches，不派 subagent、不重新规划、不进入 Code #10。完成后保留
> 全部改动未暂存、未提交，交回控制/审查 session。

## 1. 本批唯一结果

在 Target 内建立一条真实可运行的 SimToolReal native RL-Games SAPG production vertical
slice：Hydra 和 UniLab registry 创建 Code #8 的真实 MuJoCo env，经一个同步
`RlGamesNpEnvAdapter` 注入 vendored `Runner.set_vec_env()`，由 native `A2CAgent` 完整拥有
rollout/update/checkpoint，并由 native `PpoPlayerContinuous` 驱动 UniLab 原有的 MuJoCo
play/video 外壳。

完成时必须同时具备：

1. root `rlgames-sapg` optional extra 安装且 fail-closed 指向仓库内固定的
   `unilab-simtoolreal-rl-games==1.6.1+simtoolreal.2a991753.compat2`；
2. `conf/rlgames_sapg` 独立 owner 可 compose Source-native `params`，不经过 Python 算法参数
   翻译；
3. adapter 严格实现 `NpEnv -> RL-Games IVecEnv` ABI，第一次 reset 复用 `init_state()`，每步
   action 只有一次 Torch-to-NumPy 转换；
4. production train path 只能是 `Runner.load(...) -> Runner.set_vec_env(...) ->
   Runner.run_train(...) -> A2CAgent.train()`，没有第二个 rollout/collector/update loop；
5. UniLab `ExperimentTracker` 是唯一 W&B lifecycle owner，native writer 只写同一 run 下的
   TensorBoard，observer 只桥接 env metrics；
6. native `.pth` 支持新训练、native state resume、weights-load 和 play；不转换 `.pt`，不
   声称 env/RNG/trajectory bit-exact resume；
7. play 使用 vendored `PpoPlayerContinuous` 的 model、normalizer、RNN state、done reset、
   action clamp/rescale 和 stochastic/deterministic decision，同时继续使用 UniLab MuJoCo
   camera/video shell；
8. `train --algo rlgames_sapg` 和 `eval --algo rlgames_sapg` 经现有 CLI/completion 可发现且
   只路由 SimToolReal MuJoCo owner；
9. 一个 N=6、minimal epoch 的真实 MuJoCo train -> `.pth` -> load -> native player -> finite
   steps/video vertical slice 通过；
10. video cold path 真实证明每个 env 使用 assigned tool 的完整 40-mesh visual model，而不
    是把 Code #8 的 19-mesh physics variant 当作 visual parity。

唯一 production data flow：

~~~text
Hydra rlgames_sapg owner
  |-- reward/env/backend fields --> registry.make("SimToolReal", mujoco) --> NpEnv
  `-- rl_games.params -----------> vendored Runner.load()
                                             |
                                  Runner.set_vec_env(adapter)
                                             |
                         native A2CAgent rollout/update/.pth
                                             |
                         native PpoPlayerContinuous action
                                             |
                              UniLab MuJoCo playback/video
~~~

Code #9 的 tiny real vertical slice 是结构接线测试，不是 Code #10 的稳定性或 support
promotion。它不证明 `12288/2048` 持续训练、CPU affinity、release artifact、clean install、
性能、训练曲线或完整 CI。

实现 session 不执行 `git add`、`git commit`、`git push`、PR、stash、reset、clean、
checkout 或切分支。控制 session 审查后提交 Code #9 代码，并另行更新总指导文档的完成
状态。

## 2. 普通中文范围、规模和 child batches

### 2.1 只做什么

- 把已经通过 Code #1-#5 oracle 的 vendored Source runtime 晋升为 root optional extra；
- 修正Code #7增加KUKA license whitespace rule后，Code #2 vendor audit仍硬编码旧单行
  `.gitattributes`的既有compatibility drift；不改`.gitattributes`本身；
- 新增独立 Hydra root/task/profile，保持 native `params.{env,algo,model,network,config}`；
- 新增 production dependency guard、config preflight、同步 env adapter、observer、checkpoint
  resolver、Runner executor 和 native player bridge；
- 让脚本只负责 compose、preflight、registry env creation、tracker lifecycle、train/play
  orchestration 和 cleanup；
- 增加 CLI/completion route；
- 以 fake env 锁定 ABI/lifecycle，以真实 N=6 MuJoCo 锁定最小 train/play vertical slice；
- 在不启动 GL 的 near-risk test 中检查 assigned-tool visual models，再运行短 MP4 record
  验证完整 playback shell。

### 2.2 明确不做什么

- 不修改 `third_party/simtoolreal_rl_games/**` 的任何 byte，不增加第 8 个 compatibility
  patch，不 rebaseline Code #1-#5；
- 不重写 rollout、storage、GAE、augmentation、shuffle、dataset、PPO loss、central value、
  AMP、optimizer、GradScaler、scheduler、checkpoint payload 或 player action formula；
- 不从 production import `tests/**`、fixture harness、audit script、Source checkout 或 donor；
- 不把 RSL-RL SAPG、APPO、PPO、async runner、distributed、multi-GPU、ROCm、
  `torch.compile`、export 或 Motrix 接进本路线；
- 不做 `.pt -> .pth` rename/conversion，不上传 checkpoint artifact；
- 不添加 arbitrary external pickle checkpoint 支持，不从 URL 下载 `.pth`；
- 不承诺 env state、Python/NumPy/Torch RNG 或物理 trajectory 的 bit-exact resume；
- 不修改 `src/unilab/training/sim2sim.py` 的共享 contract；本批只证明同一 MuJoCo owner play，
  复用现有 resolver 和 dimension guard；
- 不运行 `12288/2048` 真实训练，不做 M0-release、support matrix、benchmark、
  `make test-all`、push/PR；
- 不进入 Code #10。

### 2.3 规模和永久维护成本

Code #9 是 umbrella，不得作为一个无边界 implementation issue 一次铺开。预计总范围约
32 paths、3k-4k production+test LOC 加 generated `uv.lock`；每个 child 必须保持约
15 paths以内、约800行以内的净手写 adaptation，并在进入下一个 child 前独立 GREEN。

永久维护成本应限制为：

- 8 个以内、职责单一的 `rlgames_sapg` integration modules；
- 3 个 Hydra YAML owners；
- 一个薄 training script；
- CLI/completion 的一条 route；
- dependency/config/adapter/runtime/tracker/checkpoint/player/real-smoke tests。

不得为“以后也许支持通用 RL-Games”提前增加 registry、plugin、export、distributed 或
backend abstraction。

### 2.4 Child batches

用户本次明确批准顺序执行以下全部 child batches：

~~~text
9A  vendor-audit compatibility、root optional extra、dependency guard、Hydra owner、R2 preflight
9B  RlGamesNpEnvAdapter、native Runner executor、fake-env ABI/lifecycle
9C  ExperimentTracker observer、run directory、trusted .pth resolver、train/resume/weights
9D  native PpoPlayerContinuous bridge、CLI/completion、visual mapping、tiny real vertical slice
~~~

9A 不 import production Runner；9B 不实现 checkpoint/player；9C 不接 CLI/video；9D 只组合
已完成 owners，并在已有 MuJoCo cold-path playback materializer 内修复已证实的 topology
同步缺口。除第4.1节精确批准的 playback helper 外，若任一 child 需要修改 vendor、其他
backend文件、Code #7/#8 task formula、共享sim2sim owner或新增公共 playback contract，立即
停止，不得把 scope 偷渡到下一个 child。

## 3. 必读内容、起点和固定身份

### 3.1 开始前完整阅读

~~~text
AGENTS.md
docs/simtoolreal_sapg_source_fidelity_migration_plan.md
docs/simtoolreal_sapg_code9_prompt.md
pyproject.toml
third_party/simtoolreal_rl_games/README.md
third_party/simtoolreal_rl_games/UPSTREAM.md
third_party/simtoolreal_rl_games/PATCHES.md
third_party/simtoolreal_rl_games/source_manifest.json
third_party/simtoolreal_rl_games/rl_games/torch_runner.py
third_party/simtoolreal_rl_games/rl_games/common/a2c_common.py
third_party/simtoolreal_rl_games/rl_games/common/algo_observer.py
third_party/simtoolreal_rl_games/rl_games/common/player.py
third_party/simtoolreal_rl_games/rl_games/algos_torch/a2c_continuous.py
third_party/simtoolreal_rl_games/rl_games/algos_torch/players.py
third_party/simtoolreal_rl_games/rl_games/algos_torch/torch_ext.py
tests/algos/rlgames_sapg/_runtime_requirement.py
tests/algos/rlgames_sapg/source_checkpoint_harness.py
tests/algos/rlgames_sapg/test_checkpoint_golden.py
tests/fixtures/simtoolreal_sapg/source_network_manifest.json
src/unilab/base/np_env.py
src/unilab/envs/manipulation/simtoolreal/env.py
src/unilab/training/common.py
src/unilab/training/backend_adapter.py
src/unilab/training/experiment.py
src/unilab/training/run.py
src/unilab/training/sim2sim.py
src/unilab/visualization/playback.py
src/unilab/base/backend/mujoco/playback.py
src/unilab/cli.py
src/unilab/tools/completion.py
scripts/train_rsl_rl.py
scripts/train_appo.py
tests/envs/manipulation/simtoolreal/target_t1_harness.py
~~~

`docs/simtoolreal_rlgames_sapg_runtime_design.md` 只存在于 donor 历史 commit
`7935f5afd74bfb5d34efa25bf8e378444d2bc191`。只允许用
`git -C /home/user/ws/lemon/UniLab show 7935f5af:docs/simtoolreal_rlgames_sapg_runtime_design.md`
只读参考其P5/P6边界，但它不是当前权威：其中保留RSL comparison、拒绝resume、export、旧
conf path、旧distribution name和timeout vendor patch等内容已经被Code #1-#8和总指导文档
取代。

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
03c2141437ab558dccad0342cac38e9ae6ae7572
docs: record SAPG Code 8 completion
~~~

实现 session 的 dispatch HEAD 应是上述基线的单个 docs child。该 docs child 只新增本
prompt，并把总指导文档中的 Code #9 状态更新为“已规划，待实现”。开始时运行：

~~~bash
set -e
set -o pipefail
SAPG_CODE9_BASE=03c2141437ab558dccad0342cac38e9ae6ae7572
test "$(git rev-parse --abbrev-ref HEAD)" = "feat/simtoolreal-sapg-rlgames"
git merge-base --is-ancestor "$SAPG_CODE9_BASE" HEAD
test "$(git rev-list --count "$SAPG_CODE9_BASE"..HEAD)" -eq 1
test "$(git diff --name-status "$SAPG_CODE9_BASE"..HEAD)" = \
  $'A\tdocs/simtoolreal_sapg_code9_prompt.md\nM\tdocs/simtoolreal_sapg_source_fidelity_migration_plan.md'
test -z "$(git status --short)"
test -z "$(git diff --cached --name-only)"
git log -2 --oneline
git status --short --branch
~~~

任一条件不成立就返回 `# BLOCKED`，不要清理或覆盖现有改动。

### 3.2 固定 Source、vendor 和 oracle identity

~~~text
Source reference checkout: /home/user/ws/lemon/simtoolreal
Source reference commit:   2a9917533bfea70419ed2667a511d7238e5b3abc
RL-Games parent tree:      7a6a0bb090998d00565aaefa6ab9f2b3d356ace2
train owner path:          isaacsimenvs/cfg/train/SimToolRealSAPG.yaml
train owner blob:          f363d05d4a24b190b7837703b93270d8f3fe9a9c
train owner SHA256:        04f30820094b062412541764b3feeb1492097e75afe5ad0df3fd0e2853496d34

vendored distribution:    unilab-simtoolreal-rl-games
vendored version:         1.6.1+simtoolreal.2a991753.compat2
Python inventory:          72 files
compatibility allowlist:   7 patches
selection SHA256:          f0517fb198dbbf9dcc456ab6de4a5cf6e0c4b03cdc90e84f12e52f74a70fe0ca
source_manifest SHA256:    4f1170b222e4ba008b34070fad7aeaba4cf790cc6ae1917417ee40ef35573ac9
~~~

Code #9 ordinary runtime和 tests不得访问 Source checkout。R2 config oracle使用已提交的
`source_network_manifest.json`中`network_spec.runner_params`，不是重新读取 Source YAML。
Source checkout只允许 Phase 0 通过`git show 2a991753...:<path>`/`git rev-parse`对固定 commit
做一次只读 provenance确认；外部 checkout 当前 branch/HEAD和无关dirty files不属于本路线
identity，不得要求它停在固定 commit，也不得 checkout/reset/清理它。之后把该 checkout临时
移走也不应影响production或测试。

Code #5 已固定的 checkpoint/player anchors：

~~~text
source_checkpoint.pth
bbe577dc7efed068bb38ce6f268e849de6a41e8ab6bb4a78fabeed9b0d7b5e02

source_checkpoint_manifest.json
8d55469d09095827587d502758d477913c76f13e8e9cd0baa23cb142d518c946

canonical player counts: N=6, N=5, N=7
native player owner: rl_games.algos_torch.players.PpoPlayerContinuous
native runner owner: rl_games.torch_runner.Runner
native train owner: rl_games.algos_torch.a2c_continuous.A2CAgent
~~~

Code #9 不修改这些 fixtures/harnesses；production tests复用已证明的语义，不重新生成。

### 3.3 固定 Target runtime identity

~~~text
torch:                   2.7.0+cu128 on current linux/x86_64 environment
gymnasium:               1.2.3
numpy:                   2.4.4
mujoco-uni-runtime:      0.4.0.dev0
M0-dev URL:              https://github.com/lemon-star608/mujoco_uni.git
M0-dev source SHA:       7205e070e983df90d520f0f8593853013e976746
BatchEnvPool.was_autoreset: real property
~~~

当前 root environment在 dispatch前预期没有安装 vendored distribution。Phase 9A必须先用
测试记录缺失 dependency的真实 RED，再通过 root optional extra修复。不得靠 `sys.path`
插入 `third_party` 或 `PYTHONPATH` 伪装安装态。

Linux/aarch64 root当前固定 Torch 2.9，而 vendor metadata固定 Torch 2.7。Code #9 不修改
vendor metadata或 root aarch64 Torch。`rlgames-sapg` optional dependency必须用 PEP 508 marker
`sys_platform != "linux" or platform_machine != "aarch64"`排除不相容组合，并由CLI/
production guard明确报unsupported；不能让extra在该平台静默显示为可用。Code #10再决定
是否扩展platform support。

### 3.4 Code #8 real env/T1 identity

~~~text
registry owner: SimToolReal / mujoco only
action/actor/critic: 29 / 140 / 162
episode steps: 600
catalog: 600 = box_box 250 + capsule_box 300 + box_only 50
complete source XML nmesh: 40
discardvisual physics variant nmesh: 19
T1 NPZ SHA256: 228b704e0a5b8e94269ce4b4da29cff4e51bb57338390d79453fe0d921cfb760
T1 manifest SHA256: 6b87220134e2711939bad47d8ae64c0fa8820e5731b887c751d7646a061a5fdb
~~~

T1 harness当前调用 `env.get_playback_model(index)`并记录19-mesh physics signatures。这只
证明真实 physics assignment，不证明完整 visual playback。Code #9不得修改或 rebaseline T1
来隐藏 visual mismatch。

## 4. 唯一允许新增或修改的路径

### 4.1 Dependency、config和 production paths

~~~text
pyproject.toml
uv.lock

# 9A only：对齐Code #7后root attributes事实，不修改vendor或.gitattributes
scripts/audit_simtoolreal_rlgames_vendor.py

conf/rlgames_sapg/config.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco_12k.yaml

src/unilab/algos/torch/rlgames_sapg/__init__.py
src/unilab/algos/torch/rlgames_sapg/dependency.py
src/unilab/algos/torch/rlgames_sapg/config.py
src/unilab/algos/torch/rlgames_sapg/env_adapter.py
src/unilab/algos/torch/rlgames_sapg/observer.py
src/unilab/algos/torch/rlgames_sapg/checkpoint.py
src/unilab/algos/torch/rlgames_sapg/runtime.py
src/unilab/algos/torch/rlgames_sapg/player.py

scripts/train_rlgames_sapg.py
src/unilab/cli.py
src/unilab/tools/completion.py

# 9D only：修复现有cold-path visual materializer，不新增public contract
src/unilab/base/backend/mujoco/playback.py
~~~

每个 production module必须单一职责。不要把所有逻辑堆进 training script，也不要为了凑
模块数量而合并hot-path adapter和checkpoint/player lifecycle。

### 4.2 Tests

~~~text
tests/config/test_rlgames_sapg_config.py
tests/vendor/test_simtoolreal_rl_games_vendor.py
tests/algos/rlgames_sapg/test_dependency.py
tests/algos/rlgames_sapg/test_env_adapter.py
tests/algos/rlgames_sapg/test_runtime.py
tests/algos/rlgames_sapg/test_observer.py
tests/algos/rlgames_sapg/test_checkpoint.py
tests/algos/rlgames_sapg/test_player.py
tests/algos/rlgames_sapg/test_real_vertical_slice.py
tests/base/backend/test_mujoco_playback_visual_topology.py
tests/scripts/test_train_rlgames_sapg.py
tests/test_cli.py
tests/test_cli_runtime_requirements.py
tests/test_completion.py
~~~

可以不创建某个测试文件并把相邻小测试合并，但不得新增其他 production/test路径。尤其不
允许修改：

~~~text
third_party/simtoolreal_rl_games/**
tests/fixtures/simtoolreal_sapg/**
tests/algos/rlgames_sapg/source_*_harness.py
tests/algos/rlgames_sapg/test_*_golden.py
src/unilab/base/**（精确白名单中的 `src/unilab/base/backend/mujoco/playback.py` 除外）
src/unilab/envs/manipulation/simtoolreal/**
src/unilab/training/experiment.py
src/unilab/training/run.py
src/unilab/training/sim2sim.py
src/unilab/visualization/**
docs/**
~~~

Visual mapping只允许在现有`materialize_visual_playback_model`/
`resolve_render_play_model_files` owner内做第8.4节的cold-path行为修正。若证明还必须修改env、
其他backend/visualization owner或新增公共contract，按停止条件返回精确path和contract，不得
自行扩大whitelist。

所有 Python命令必须通过 `uv run`。手工文本编辑只使用 `apply_patch`。`uv.lock`只能由
`uv lock`生成，不得手工编辑；不得用 ad-hoc脚本改 YAML/Python或 lock。

## 5. 9A：optional extra、dependency guard和Hydra owner

### 5.1 Root optional extra和lock identity

先修复一个已经复现的root audit compatibility RED。当前`.gitattributes`自Code #7起合法包含：

~~~text
third_party/simtoolreal_rl_games/rl_games/**/*.py -whitespace
src/unilab/assets/robots/kuka_sharpa/LICENSE.kuka_iiwa -whitespace
~~~

但`scripts/audit_simtoolreal_rlgames_vendor.py::GIT_ATTRIBUTES_CONTENT`仍只接受第一行，导致
`uv run scripts/audit_simtoolreal_rlgames_vendor.py`和既有
`test_root_git_attributes_are_exact_and_scoped`报
`Git whitespace attribute file content mismatch`。这不是vendor byte drift。

9A必须先原样运行并记录上述RED，然后只做以下owner修正：

- audit constant精确接受当前两行、末尾单个LF；
- `git check-attr` probes同时验证vendor Python和KUKA license为`whitespace: unset`，普通audit
  script仍为`unspecified`；
- vendor test在isolated repo中分别给vendor Python、该license和ordinary file写入trailing
  whitespace，证明前两者被精确豁免、ordinary file仍由`git diff --cached --check`拒绝；
- `.gitattributes`、vendor tree、Source identity和其他audit constants保持不变。

若初始failure不是上述exact mismatch，或修复需要放宽vendor hash/inventory/patch allowlist，
按停止条件返回`# BLOCKED`。

完成该compatibility GREEN后再增加root extra：

`pyproject.toml`只增加一个非基础依赖 extra：

~~~text
extra name: rlgames-sapg
distribution: unilab-simtoolreal-rl-games
version: 1.6.1+simtoolreal.2a991753.compat2
source: checked-in third_party/simtoolreal_rl_games, editable
normal command: uv run --extra mujoco --extra rlgames-sapg ...
~~~

要求：

- 不把 distribution加入 `[project.dependencies]`；
- `[tool.uv.sources]`使用相对 repo path和 `editable=true`，不能用 sibling checkout、wheel、
  public PyPI `rl-games`或裸 file URL；
- dependency requirement锁 exact local version；
- marker精确使用`sys_platform != "linux" or platform_machine != "aarch64"`排除
  Linux/aarch64不相容Torch组合，CLI在该平台给出actionable error；
- `uv.lock`中的 package name/version/source必须指向同一 editable directory；
- base install不应 import/install `rl_games`，选择 extra后必须安装；
- `unilab.algos.torch.rlgames_sapg`、CLI和completion在base install中仍可import；所有
  `rl_games` imports必须位于dependency guard成功后的cold entrypoint/runtime path，不能让
  optional dependency变成UniLab package import-time硬依赖；
- lock不得升级无关 package。若 `uv lock`产生大面积无关升级，先用当前 locked solution重新
  生成并解释；不能接受顺便升级。

### 5.2 Production dependency owner

`dependency.py`是production唯一 runtime identity guard。它不能import tests或
`scripts/audit_simtoolreal_rlgames_vendor.py`。至少验证：

1. distribution metadata name和exact version；
2. editable `direct_url.json`是file URL、`dir_info.editable=true`，URL decode并解析后exact
   等于当前repo的`third_party/simtoolreal_rl_games`；
3. `rl_games` import spec origin位于该vendor package root内；
4. `source_manifest.json`是regular non-symlink file，外层SHA256、schema、Source HEAD、parent
   tree、72-file inventory和7-entry allowlist一致；
5. manifest声明的pristine/current hashes与当前文件一致，不能接受另一个installed
   `rl-games`先占 namespace；
6. 当前platform是本批支持范围，错误信息明确建议
   `uv run --extra mujoco --extra rlgames-sapg ...`；
7. guard只在entrypoint/preflight冷路径运行，不在reset/step/action hot path运行。

`UNILAB_REQUIRE_SAPG=1`是required test mode：在这个mode里 dependency缺失或identity drift
必须fail，不能skip。Production入口本身无论该env var是否存在都必须fail closed；env var只
控制既有oracle tests是否允许在base install中skip。

### 5.3 Hydra owner layout

必须使用以下authoritative paths，不复用 `conf/ppo`：

~~~text
conf/rlgames_sapg/config.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco.yaml
conf/rlgames_sapg/task/simtoolreal/mujoco_12k.yaml
~~~

`config.yaml`包含：

- defaults `_self_`和 `task: simtoolreal/mujoco`；
- UniLab orchestration metadata `algo`、`training`、`env`；
- 完整 `rl_games.params` Source-native schema；
- Hydra run dir保持 `.`、不让Hydra chdir；
- canonical `algo.obs_groups`、`algo.policy.actor_hidden_dims`、
  `algo.policy.critic_hidden_dims`、`algo.empirical_normalization`、
  `algo.obs_normalization`通过YAML interpolation引用native owner字段，供现有
  `ExperimentTracker`/sim2sim snapshot使用；Python不得维护第二份翻译表。

`mujoco.yaml`是唯一base task/backend owner：

- `training.task_name: SimToolReal`；
- `training.sim_backend: mujoco`；
- `algo.num_envs`默认保持Source 24576，native `params.config.num_actors`只用YAML interpolation
  引用它；
- action/actor/critic/episode固定29/140/162/600；
- raw reward必须显式写200/20/300/50/1000/0.03/0.003和其余Code #7 defaults；
- native `reward_shaper.scale_value`仍为0.01；env reward不能预缩，learner也不能重复缩放；
- env owner显式选择600-tool pool、Source-compatible reset/noise/delay/wrench fields；
- backend/resource差异只存在 `training/env/reward` owner，不污染native算法fields。

`mujoco_12k.yaml`继承base owner，只声明Target full-scale candidate：

~~~text
num_actors = 12288
expl_coef_block_size = 2048
num_blocks = 6
~~~

其余Source native字段保持同一owner。Code #9只做compose/preflight测试，不运行这个profile，
不声明它supported；Code #10才做真实`12288/2048`训练、性能和support判断。

### 5.4 R2 field-by-field config gate

`tests/config/test_rlgames_sapg_config.py`普通测试不得读取Source checkout。它从已提交
`source_network_manifest.json`读取固定Source owner `runner_params`，再compose两份Target
owner逐字段比较。

Base owner只允许以下已解释差异：

- Source的跨repo `${....env.scene.num_envs}` interpolation改为Target的
  `${algo.num_envs}`，resolved value仍为24576；
- device fields通过Target `training.device` interpolation，default仍为`cuda:0`；
- dynamic `train_dir`、`full_experiment_name`、`env_info`和`vec_env`尚未注入；
- Target外层 `algo/training/env/reward/hydra` orchestration metadata。

Native `params`的network、central value、RNN、normalization、PPO/SAPG、AMP、scheduler、
reward shaper和player字段不得有其他difference。`mujoco_12k`只允许actor/block两个profile
overlay。测试还必须证明：

- raw env reward与native0.01 boundary只各出现一次；
- six blocks、divisibility、horizon/seq/minibatch约束；
- profile或CLI override产生非6 blocks时在env construction前fail；
- `multi_gpu=true`、distributed env、Motrix、wrong task、wrong dims、wrong native owner、
  `load_checkpoint/load_path`绕过production checkpoint mode都在preflight fail；
- config validation不import/create env，不访问asset/XML或Source。

## 6. 9B：adapter和native Runner executor

### 6.1 Authoritative adapter ABI

Production class命名为 `RlGamesNpEnvAdapter`。公开ABI固定为：

~~~text
reset() -> {"obs": Tensor[N,140], "states": Tensor[N,162]}
step(actions) -> obs_dict, reward, done, info
done = terminated OR truncated
info["time_outs"] = truncated
get_env_info()
get_number_of_agents() -> 1
set_train_info(frame, owner) -> None
get_env_state() -> None
set_env_state(None) -> None
vec_env.env.device -> RL device
~~~

构造参数至少包含已创建的 `NpEnv`、RL device、actor/critic group names和clip bounds；adapter
不创建env、不读取Hydra、不拥有tracker/checkpoint/player。

### 6.2 Reset、observation和space contract

- `NpEnvState.obs`必须恰有Target groups `obs`和`critic`；adapter输出key固定为`obs`和
  `states`；
- 第一次 `reset()` 在 `env.state is None` 时只调用一次 `env.init_state()`并直接复用其结果，
  不能再调用一次全量 `env.reset()`；
- 后续显式reset才允许调用public `env.reset(all_env_ids)`，并保持内部state/info一致；
- actor140和critic162必须float32、finite、shape exact；错group、dtype、shape、non-finite或
  num_env mismatch立即fail；
- observation/state spaces是Gymnasium finite `Box(-10,10)`、action space是env声明的finite
  `Box(-1,1,(29,),float32)`；不能使用legacy Gym spaces；
- `get_env_info()`恰含 `agents=1`、`value_size=1`、observation/state/action spaces；
- 返回tensor直接在RL device构造，不能先GPU再CPU再GPU，也不能额外拼接coefficient ID；
  coefficient embedding属于native runner/player。

### 6.3 Step和conversion boundary

- 接受shape `(N,29)` 的Torch action；先detach，只有一次 `.to("cpu")`/NumPy materialization，
  然后调用一次 `NpEnv.step(np.float32)`；
- 不接受NumPy action作为隐藏fast path，不silent reshape/broadcast/clamp；native player/runner
  已拥有action clamp/rescale；
- actor/critic/reward/done/timeout各只做一次NumPy-to-Torch，device/dtype固定；
- done使用terminal step的 `terminated | truncated`，不能从autoreset后的obs推断；
- `time_outs`必须只等于`truncated`，保持Code #3/#5已锁定的Source timeout语义；不得增加
  vendor final-observation bootstrap patch；
- info至少保留env `log`和terminal compatibility信息，`time_outs`必须是native可消费tensor；
  不要为了logger递归复制所有backend timing arrays；
- adapter不调用 `_backend`、backend subclass、MuJoCoUni，不读XML/assets/model metadata，
  不用 `getattr/hasattr`探测backend能力，不启动thread/queue/collector。

### 6.4 Checkpoint env-state boundary

`get_env_state()`固定返回`None`。`set_env_state(None)`是documented no-op；任何non-None值
明确raise。Native checkpoint仍保存`env_state: None`。这表示native算法state可恢复，不表示
MuJoCo env/DR/RNG/trajectory恢复。

不要删除native checkpoint中的 `obs/rnn_states/dones/current_*`。Resume继续按Source
`A2CAgent.set_full_state_weights()`执行；Code #9报告中必须把这个边界写清楚。

### 6.5 Native Runner executor

`runtime.py`只组合vendored owner：

1. production dependency guard；
2. deep-copy resolved `cfg.rl_games`，不修改Hydra cfg；
3. 只注入runtime handles：`train_dir`、以`0_`开头的`full_experiment_name`、device、
   device_name、play时的num_actors/env_info，以及observer；
4. `runner = rl_games.torch_runner.Runner(algo_observer=observer)`；
5. `runner.load({"params": params})`；
6. `runner.set_vec_env(adapter)`；
7. train时只调用native `runner.run_train(args)`；
8. `args`只描述checkpoint orchestration：`checkpoint`和
   `checkpoint_load_mode in {resume,weights}`；无checkpoint时不伪造空restore；
9. 返回native result和run/checkpoint metadata，不包装或重算loss/update。

必须测试实际owner class path是 `Runner`和`A2CAgent`，并用spy证明调用顺序和次数。Fake
runner tests不能代替至少一次真实vendored Runner construction；real MuJoCo留到9D。

### 6.6 Fake-env near-risk tests

`test_env_adapter.py`至少覆盖：

1. first reset只调用一次init，140/162 routing和finite spaces；
2. non-identity row values保持，不错误拼group；
3. one action transfer/one env step，reward/done/timeout exact；
4. terminated-only、truncated-only和两者同时为true；
5. selected autoreset返回reset obs但done仍是terminal mask；
6. RL device，float32/bool dtype和info log；
7. `set_train_info` no-op以及None/non-None env-state contract；
8. wrong shape/group/dtype/non-finite/action/device fail closed；
9. source AST/audit证明adapter无asset read、backend private probe或thread/queue。

`test_runtime.py`至少覆盖：

1. native config deep copy，只注入enumerated runtime handles；
2. `Runner.load -> set_vec_env -> run_train` exact order；
3. native `A2CAgent` factory owner，不出现UniLab rollout/update callback；
4. fresh/resume/weights args routing；
5. writer/env cleanup即使native train raise也执行；
6. unsupported mode和config在env创建前由preflight拒绝。

## 7. 9C：tracker、run directory和native `.pth`

### 7.1 Run directory owner

UniLab log root继续使用 `get_log_root(ROOT_DIR,cfg)`，task root固定：

~~~text
logs/rlgames_sapg/SimToolReal/
~~~

新训练run name必须以native可解析的policy prefix开头：

~~~text
0_<YYYY-MM-DD_HH-MM-SS>_mujoco
~~~

Native配置使用task root作为`train_dir`、run name作为`full_experiment_name`，使native
`experiment_dir`正好等于Tracker的`log_dir`，checkpoint留在：

~~~text
<run>/nn/*.pth
<run>/last/model.pth
<run>/best/model.pth
~~~

不得让RL-Games再嵌套第二层`0_simtoolreal_sapg`目录。run collision必须fail或使用明确的
deterministic suffix，不能复用/覆盖已有run。

Play-only不得覆盖source run的 `run_config.json`/`run_summary.json`。若play需要tracker，使用
新eval sibling run并在summary记录source checkpoint/run；after-train play可复用当前training
tracker。

### 7.2 Single tracker/W&B lifecycle

`ExperimentTracker`仍由script创建、`start()`一次、`finish()`一次并包在`try/finally`。
`observer.py`实现native `AlgoObserver` callback surface：

~~~text
before_init
after_init
process_infos
after_steps
after_clear_stats
after_print_stats
~~~

要求：

- env `info["log"]` scalar metrics按frame写入native TensorBoard writer；
- tracker有active W&B run时通过其public `run`桥接同一metrics，不调用`wandb.init/finish`；
- 没有W&B或logger=tensorboard时正常工作；
- summary至少记录native result、epoch/frame、checkpoint、load mode、video和wall time；
- callback不能修改actions/rewards/dones/RNG，不读取backend/asset；
- native writer close和tracker finish各自exact once，exception path也成立；
- 禁止复制RSL-RL monkeypatch或启动第二个W&B lifecycle。

### 7.3 Trusted local `.pth` resolver

`checkpoint.py`只接受本地native `.pth`：

- `algo.load_run=-1`解析task root中最新合法training run；
- named run必须通过现有CLI `RUN_ID_PATTERN`且位于task root直接子目录；
- explicit checkpoint只能是run-relative basename/path，不允许absolute、`..`、URL、symlink
  escape或非`.pth`；
- default selection确定性覆盖`nn/last_*_ep_*`、`nn/last_*_frame_*`、`last/model.pth`和
  explicit filename，并有清晰priority测试；
- resolved run/checkpoint都必须是regular path且`resolve()`仍位于trusted task root；
- legacy native local `.pth`使用vendored `torch_ext.load_checkpoint`的
  `weights_only=False`语义前，必须经过上述explicit trust gate；
- arbitrary external pickle默认永久fail closed；本批不添加“信任任意路径”开关；
- `.pt`、wrong outer payload、missing rank0/model或dimension mismatch给出actionable error，
  不靠改suffix继续；
- load只由native agent/player执行；resolver不重写payload。

### 7.4 Train、resume和weights modes

Production orchestration field固定为：

~~~text
algo.checkpoint_load_mode: none | resume | weights
algo.load_run: -1 | trusted run name
algo.checkpoint: -1 | trusted relative .pth selection
~~~

规则：

- `none`只允许fresh train；
- `resume`调用native `agent.restore()`，恢复Source保存的model、actor optimizer、scaler、
  normalizers、central state、epoch/frame、rollout/RNN fields和`env_state=None`；
- `weights`调用vendored Runner已固定的weights path，只初始化model和可兼容central weights，
  不恢复epoch/optimizer/rollout；
- play使用nativeplayer restore，不把train load mode翻译成RSL概念；
- missing checkpoint或mode冲突在env构造前fail；
- resume的测试只声明native state restore，明确不声明external RNG或env trajectory exact；
- Code #5 oracle继续证明外部重设RNG后的首action/value/update，不在Code #9重写fixture。

`test_checkpoint.py`至少覆盖resolver traversal/symlink/suffix/payload fail-closed、latest priority、
fresh/resume/weights routing、native outer rank layout、env_state None、weights不恢复epoch/optimizer、
resume恢复native state，以及dimension guard把model mismatch变成明确sim2sim诊断。

## 8. 9D：native player、CLI和真实vertical slice

### 8.1 Player bridge owner

`player.py`不得调用native `BasePlayer.run()`，因为UniLab必须保留backend playback/video
lifecycle；也不得手写policy forward。固定流程：

1. 从resolved Hydra native params deep copy；
2. 注入play device、`num_actors=play_env_num`、adapter `env_info`和`vec_env`；
3. `Runner.load()`并`Runner.set_vec_env()`；
4. `player = Runner.create_player()`，assert owner为native
   `PpoPlayerContinuous`；
5. 在`policy_load_dim_guard(env_obs_dim=140,env_action_dim=29,algo_name="rlgames_sapg")`
   内调用native `player.restore(.pth)`；
6. initialize callback使用native `player.env_reset(adapter)`并`player.init_rnn()`；
7. step callback只调用native `player.get_action(obs, player.is_deterministic)`和
   `player.env_step(adapter, action)`；
8. done rows按native `BasePlayer.run()`相同index/rnn-state zeroing语义清零；
9. callback交给`env.run_playback(...)`/现有UniLab visualization shell采集physics state、camera
   和video；
10. close env、native writer和tracker，无论render/encoder异常都cleanup。

不得复制normalizer、coefficient IDs、block selector、action distribution、clamp/rescale或RNN
forward。`player.deterministic`直接由native YAML/Hydra override拥有，Source default保持false。

### 8.2 Player contract tests

`test_player.py`用fake adapter/native vendored player至少覆盖：

- N=6 source row routing；
- N=5和N=7 equality/argmax fallback继续与Code #5 oracle一致；
- deterministic使用mu、stochastic使用sample并具有正确RNG consumption；
- input RMS restore、actor model restore、action bounds；
- recurrent state初始化、exact done rows zero、其他rows不变；
- one action transfer per step；
- bridge确实没有调用`BasePlayer.run()`或自有model forward；
- `.pt`/dimension mismatch/unsafe checkpoint fail closed。

这些tests不得修改Code #5 fixture。若production bridge与已锁定player oracle冲突，修
bridge，不改expected/source runtime。

### 8.3 CLI和completion

CLI public spelling固定：

~~~bash
uv run --extra mujoco --extra rlgames-sapg train \
  --algo rlgames_sapg --task simtoolreal --sim mujoco

uv run --extra mujoco --extra rlgames-sapg eval \
  --algo rlgames_sapg --task simtoolreal --sim mujoco --load-run -1 \
  --render-mode record
~~~

`build_route("rlgames_sapg",...)`固定：

~~~text
script_name: scripts/train_rlgames_sapg.py
config_group: rlgames_sapg
owner: conf/rlgames_sapg/task/<task>/<sim>[_profile].yaml
generated Hydra override: task=<task>/<sim>[_profile]
~~~

要求：

- `--profile 12k`可发现`mujoco_12k.yaml`但不自动运行；
- 只有simtoolreal/mujoco owner存在；Motrix/mjwarp/wrong task因owner缺失或preflight fail；
- missing `mujoco` extra和missing/wrong rlgames distribution分别给出正确安装命令；
- Linux/aarch64明确unsupported，不误报为installed support；
- eval `--load-run`继续只接受run name或`-1`，不允许path traversal；
- completion列出algo/task/sim/profile和`logs/rlgames_sapg/SimToolReal` runs；
- 保持现有PPO/APPO/off-policy CLI outputs bytes/semantics不变。

### 8.4 Assigned-tool visual mapping gate

这是Code #9必须修复并验证、不能overprotect也不能跳过的已知真实风险。

Code #8当前事实：

- `env.get_playback_model(i)`继承backend public method，返回assigned 19-mesh physics model；
- backend另有一个完整40-mesh visual source；
- 现有visual materializer会按geom name把physics尺寸写回visual base；
- 600-tool catalog包含box_box、capsule_box、box_only三种不同object topology。

控制session已用现有public playback path创建N=8 real env，并经
`resolve_render_play_model_files`解析indexes `0`、`1`、`7`，确认当前真实RED：index 0
正确为box_box；index 1应为capsule_box，却仍是tool 0的box handle；index 7应为box_only，
却残留tool 0的`object_head`。三者虽然都是`36/35/29/40`，但40 meshes本身不能证明tool
topology正确。实现session必须先把同一诊断固化为失败测试并复现RED，不能直接相信本段记录。

唯一批准的production修复点是既有
`src/unilab/base/backend/mujoco/playback.py::materialize_visual_playback_model`及其调用者；这是
cold-path materialization行为修正，不增加或改变`SimBackend`/`NpEnv` public method。最小
修复必须从assigned physics model同步named contact geometry，同时保留visual base中的完整
robot/hand visuals：

1. 同名contact geom同步type、size、pos、quat以及render/contact所需属性；
2. playback model中不存在的stale contact geom从visual spec删除；
3. playback model新增的named contact geom在同名body下创建；
4. visual-only geom和40个robot/hand meshes保持不变；
5. 不加入SimToolReal名字分支，不读取`ToolSpec`、asset/XML catalog或env private fields，不把
   task业务规则写进shared helper。

`tests/base/backend/test_mujoco_playback_visual_topology.py`先用小型临时MJCF锁定同一visual base
对box_box、capsule_box、box_only的通用contact-topology同步，再由N=8 real gate验证真实task。
真实test只在cold playback path compile输出model，并分别验证：

1. 每个visual model `nq/nv/nu/nmesh=36/35/29/40`；
2. robot/hand 40-mesh inventory完整；
3. object contact geom names/types/count/size分别exact匹配assigned `ToolSpec`：box_box为box
   handle+box head，capsule_box为capsule handle+box head，box_only只有box handle；
4. env index到catalog assignment没有退回index0；
5. model paths在playback temp cleanup后消失。

不能只assert “输出了三个不同path”或“nmesh=40”；必须检查object topology和尺寸。不能读取
rendered pixel颜色来替代model contract。

上述generic owner修复GREEN后继续video gate。若正确修复还需要以下任一项，立即按停止条件
#14 返回 `# BLOCKED`：

- 修改 `SimBackend`/MuJoCo backend public contract；
- 修改 `NpEnv`/SimToolRealEnv的playback public surface；
- 修改其他backend文件或visualization owner；
- rebaseline T1从19改40；
- production读取private `_tool_variant_files/_tool_index/_backend`。

报告必须给出actual/expected topology、所需精确owner path和最小contract建议。不得把所有env
渲染成tool0、丢掉robot visuals、退回19 meshes或把test改成只看physics assignment。

### 8.5 Tiny real train/play vertical slice

Visual mapping GREEN后，运行一个真实但最小的N=6 CUDA/MuJoCo slice。它必须保留Source完整
1024/1024/512/512 + LSTM1024 network；只允许通过Hydra override缩小workload：

~~~text
num_actors: 6
expl_coef_block_size: 1
six blocks: yes
horizon_length: 4
seq_length: 4
minibatch_size: 12
actor mini_epochs: 1
central minibatch_size: 12
central mini_epochs: 1
max_epochs: 1
save_frequency: 1
play_env_num: 6
play_steps: 4
~~~

不得把network/RNN缩成Code #5 fixture尺寸、关闭central value、关闭SAPG、改成
`use_others_experience=none`、关normalization、关AMP或mock MuJoCo。

真实slice必须证明：

1. CLI/Hydra/registry创建N=6真实600-tool env；
2. native Runner/A2CAgent完成至少一个真实rollout和一次actor+central update；
3. obs/reward/loss/grad/parameters保持finite；
4. native checkpoint落在current run `nn/*.pth`，outer rank0、env_state None、required native
   state keys存在；
5. 新play env由native `PpoPlayerContinuous` restore同一checkpoint并跑4个finite real steps；
6. N=6 coefficient routing、stochastic default和done/RNN reset生效；
7. `MUJOCO_GL=egl` record生成非空MP4，使用assigned完整visual models；
8. `run_config.json`、`run_summary.json`、TensorBoard和唯一tracker lifecycle在同一run边界；
9. env/tool tempdirs、playback model tempdir、writer和renderer全部cleanup；
10. 没有Source/donor runtime filesystem access。

这不要求reward curve变好、episode在4步内自然结束或checkpoint逐字节可重现。Code #10才做
持续S1、`12288/2048`、affinity、性能、release和完整suite。

## 9. 严格 RED -> GREEN 顺序

### Phase 0：起点和只读census

1. 运行第3.1节lineage/clean-tree checks；
2. 运行vendor audit和对应root-attributes test，确认只出现第5.1节已知的旧单行constant RED；
3. 记录root环境缺失vendored distribution，但用ephemeral editable command确认Code #1-#5
   oracle baseline仍GREEN：

~~~bash
uv run scripts/audit_simtoolreal_rlgames_vendor.py
uv run pytest \
  tests/vendor/test_simtoolreal_rl_games_vendor.py::test_root_git_attributes_are_exact_and_scoped \
  -q

UNILAB_REQUIRE_SAPG=1 uv run \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_import.py \
         tests/algos/rlgames_sapg/test_source_config.py \
         tests/algos/rlgames_sapg/test_checkpoint_golden.py -q
~~~

4. 验证M0-dev identity、CUDA和Code #8 focused baseline；
5. 记录当前没有`conf/rlgames_sapg`、production package、script或CLI route。

### Phase 9A：dependency/config

1. 按第5.1节更新root audit owner/test，要求既有RED转为GREEN且vendor bytes未变；
2. 先写dependency和config tests；
3. 在未改root extra/owners前运行，记录missing distribution/config真实RED；
4. 最小修改`pyproject.toml`，运行`uv lock`生成lock；
5. 实现production dependency guard和3个YAML owners；
6. 要求base/profile compose、R2 field-by-field、root extra/install identity GREEN；
7. 再运行完整vendor audit确认root attributes语义和vendor identity同时GREEN。

### Phase 9B：adapter/runtime

1. 先写fake env ABI tests，记录module/behavior RED；
2. 实现最小adapter，再要求reset/step/done/timeout/conversion/state GREEN；
3. 先写Runner lifecycle spies，记录runtime缺失 RED；
4. 实现native executor并验证owner/call order/fresh-resume-weights routing；
5. 不创建checkpoint/player/CLI stub来凑pass。

### Phase 9C：tracker/checkpoint/script

1. 先写observer/tracker tests，记录missing bridge RED；
2. 实现single lifecycle和run dir；
3. 先写trusted resolver和native mode tests，记录missing resolver RED；
4. 实现`.pth` path/trust/mode contract；
5. 写thin script fake integration test，验证preflight-before-env和finally cleanup；
6. required tests全部GREEN后才进入9D。

### Phase 9D：player/CLI/real

1. 先写native player bridge tests，记录missing bridge RED；
2. 实现player并要求Code #5 N=6/5/7 semantics GREEN；
3. 先写CLI/completion tests，再最小增加route；
4. 先复现第8.4节已知visual topology RED，再在唯一批准的playback helper内最小修复；若需要
   范围外contract立即BLOCKED；
5. visual GREEN后写/运行real vertical slice test；
6. 最后用真实CLI train/eval命令再跑一次，不用test-only harness替代。

如果机械接线使某个首次test已经GREEN，如实报告first-run GREEN，不制造假bug。RED必须来自
缺失owner或真实behavior mismatch，不能通过先写错误expected再修正。

## 10. 最终验证

### 10.1 Dependency、config和Code #1-#5 oracle

~~~bash
uv lock --check
uv run --extra rlgames-sapg - <<'PY'
import importlib.metadata
import importlib.util
from pathlib import Path

dist = importlib.metadata.distribution("unilab-simtoolreal-rl-games")
assert dist.version == "1.6.1+simtoolreal.2a991753.compat2"
spec = importlib.util.find_spec("rl_games")
assert spec is not None and spec.origin is not None
assert Path(spec.origin).resolve().is_relative_to(
    Path("third_party/simtoolreal_rl_games/rl_games").resolve()
)
PY

uv run scripts/audit_simtoolreal_rlgames_vendor.py
uv run pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q
UNILAB_REQUIRE_SAPG=1 uv run --extra rlgames-sapg pytest \
  tests/algos/rlgames_sapg/test_import.py \
  tests/algos/rlgames_sapg/test_source_config.py \
  tests/algos/rlgames_sapg/test_network_golden.py \
  tests/algos/rlgames_sapg/test_rollout_golden.py \
  tests/algos/rlgames_sapg/test_update_golden.py \
  tests/algos/rlgames_sapg/test_checkpoint_golden.py -q
~~~

Required SAPG oracle必须0 skip。记录actual pass/warning/time，不沿用旧数字。

### 10.2 Code #9 focused tests

~~~bash
UNILAB_REQUIRE_SAPG=1 uv run --extra mujoco --extra rlgames-sapg pytest \
  tests/config/test_rlgames_sapg_config.py \
  tests/algos/rlgames_sapg/test_dependency.py \
  tests/algos/rlgames_sapg/test_env_adapter.py \
  tests/algos/rlgames_sapg/test_runtime.py \
  tests/algos/rlgames_sapg/test_observer.py \
  tests/algos/rlgames_sapg/test_checkpoint.py \
  tests/algos/rlgames_sapg/test_player.py \
  tests/base/backend/test_mujoco_playback_visual_topology.py \
  tests/scripts/test_train_rlgames_sapg.py \
  tests/test_cli.py \
  tests/test_cli_runtime_requirements.py \
  tests/test_completion.py -q

UNILAB_REQUIRE_SAPG=1 MUJOCO_GL=egl uv run \
  --extra mujoco --extra rlgames-sapg pytest \
  tests/algos/rlgames_sapg/test_real_vertical_slice.py -m slow -q
~~~

两组required tests必须0 skip。Real test不能mock registry/backend/Runner/agent/player/checkpoint或
visual model resolver。

### 10.3 真实CLI vertical slice

使用一个新的临时log root，所有算法缩放只通过Hydra override：

~~~bash
SAPG_CODE9_LOG_ROOT=$(mktemp -d)
UNILAB_REQUIRE_SAPG=1 MUJOCO_GL=egl uv run \
  --extra mujoco --extra rlgames-sapg train \
  --algo rlgames_sapg --task simtoolreal --sim mujoco \
  training.log_root="$SAPG_CODE9_LOG_ROOT" \
  training.no_play=true \
  algo.num_envs=6 \
  rl_games.params.config.expl_coef_block_size=1 \
  rl_games.params.config.horizon_length=4 \
  rl_games.params.config.seq_length=4 \
  rl_games.params.config.minibatch_size=12 \
  rl_games.params.config.mini_epochs=1 \
  rl_games.params.config.central_value_config.minibatch_size=12 \
  rl_games.params.config.central_value_config.mini_epochs=1 \
  rl_games.params.config.max_epochs=1 \
  rl_games.params.config.save_frequency=1

SAPG_CODE9_TASK_ROOT="$SAPG_CODE9_LOG_ROOT/SimToolReal"
test -d "$SAPG_CODE9_TASK_ROOT"
test "$(find "$SAPG_CODE9_TASK_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1
SAPG_CODE9_RUN_DIR=$(find "$SAPG_CODE9_TASK_ROOT" -mindepth 1 -maxdepth 1 -type d)
SAPG_CODE9_RUN_NAME=$(basename "$SAPG_CODE9_RUN_DIR")
test "$(find "$SAPG_CODE9_RUN_DIR/nn" -maxdepth 1 -type f -name '*.pth' | wc -l)" -ge 1
test -f "$SAPG_CODE9_RUN_DIR/run_config.json"
test -f "$SAPG_CODE9_RUN_DIR/run_summary.json"

UNILAB_REQUIRE_SAPG=1 MUJOCO_GL=egl uv run \
  --extra mujoco --extra rlgames-sapg eval \
  --algo rlgames_sapg --task simtoolreal --sim mujoco \
  --load-run "$SAPG_CODE9_RUN_NAME" --render-mode record \
  training.log_root="$SAPG_CODE9_LOG_ROOT" \
  training.play_env_num=6 \
  training.play_steps=4
~~~

检查产生的MP4存在且非空、summary引用正确source checkpoint，并记录checkpoint path/size、
native outer keys、epoch/frame和video path。不要删除临时root；在handoff中报告路径供控制
session抽查。

### 10.4 Code #8和邻近regressions

~~~bash
uv run --extra mujoco pytest tests/envs/manipulation/simtoolreal -q
uv run --extra mujoco pytest \
  tests/base/backend/test_mujoco_uni_runtime_contract.py \
  tests/base/backend/test_mujoco_model_source_variants.py \
  tests/base/backend/test_mujoco_autoreset_real_pool.py \
  tests/config/test_locomotion_params.py \
  tests/envs/locomotion/go2_arm/test_manip_loco_contract.py -q
~~~

T0/T1 hashes保持第3.4节值，普通tests不运行generator、不访问Source。

### 10.5 Style、types、scope和cold-path audit

~~~bash
uv run --extra mujoco --extra rlgames-sapg ruff check \
  src/unilab/algos/torch/rlgames_sapg \
  scripts/audit_simtoolreal_rlgames_vendor.py \
  scripts/train_rlgames_sapg.py \
  src/unilab/cli.py \
  src/unilab/tools/completion.py \
  tests/config/test_rlgames_sapg_config.py \
  tests/algos/rlgames_sapg \
  tests/base/backend/test_mujoco_playback_visual_topology.py \
  tests/vendor/test_simtoolreal_rl_games_vendor.py \
  tests/scripts/test_train_rlgames_sapg.py \
  tests/test_cli.py tests/test_cli_runtime_requirements.py tests/test_completion.py
uv run --extra mujoco --extra rlgames-sapg ruff format --check \
  src/unilab/algos/torch/rlgames_sapg \
  scripts/audit_simtoolreal_rlgames_vendor.py \
  scripts/train_rlgames_sapg.py \
  src/unilab/cli.py \
  src/unilab/tools/completion.py \
  tests/config/test_rlgames_sapg_config.py \
  tests/algos/rlgames_sapg \
  tests/base/backend/test_mujoco_playback_visual_topology.py \
  tests/vendor/test_simtoolreal_rl_games_vendor.py \
  tests/scripts/test_train_rlgames_sapg.py \
  tests/test_cli.py tests/test_cli_runtime_requirements.py tests/test_completion.py
uv run --extra mujoco --extra rlgames-sapg mypy src/unilab
uv run --extra mujoco --extra rlgames-sapg pyright

test ! -e MUJOCO_LOG.TXT
test -z "$(git diff --cached --name-only)"
git diff --check
git status --short
git diff --stat
git diff --numstat
git diff --name-only
git ls-files --others --exclude-standard

if rg -n "_backend|backend\._|getattr\([^\n]*backend|hasattr\([^\n]*backend|read_text|read_bytes|open\(" \
  src/unilab/algos/torch/rlgames_sapg/env_adapter.py; then
  exit 1
fi
if rg -n "tests\.|source_.*_harness|/home/user/ws/lemon/(simtoolreal|UniLab)|sys\.path|PYTHONPATH" \
  src/unilab/algos/torch/rlgames_sapg scripts/train_rlgames_sapg.py; then
  exit 1
fi
rg -n "wandb\.init|wandb\.finish|BasePlayer\.run|run_play\(" \
  src/unilab/algos/torch/rlgames_sapg scripts/train_rlgames_sapg.py || true
~~~

最后人工确认：

- production只有dependency guard cold path读取vendor manifest；adapter/runtime/player hot path
  不读asset/XML/vendor metadata；
- production没有backend-private access和Source/donor filesystem access；
- `wandb.init/finish`只来自`ExperimentTracker`内部既有owner，新增code为0 hit；
- train只进入native `run_train/A2CAgent.train`，player bridge不进入native `BasePlayer.run`；
- vendor tree和Code #1-#8 fixtures/owners bytes未变；
- 每个child path/handwritten LOC符合第2.3节；
- 工作树只有第4节whitelist、staging为空；
- required tests无skip/fail，warnings逐条解释。

实现 session不运行`make test`或`make test-all`，不创建或更新PR。控制 session会完整阅读
diff、复跑近风险gates、核对native owner/path/visual mapping和真实CLI artifacts后决定提交。

## 11. 停止条件

出现任一情况立即停止写入并返回 `# BLOCKED`：

1. branch、lineage、single-docs-child、clean tree或empty staging不符合第3.1节；
2. 固定 Source commit/blob不存在或hash不符，vendor/oracle/M0-dev identity不符，或vendored
   distribution不能从checked-in path exact安装；第5.1节已知root-attributes audit RED不是
   vendor identity drift，外部Source checkout的当前HEAD前进本身也不触发本条件；
3. root `uv lock`只有修改vendor metadata、放宽Source runtime identity、升级无关dependencies
   或虚假支持Linux/aarch64才能resolve；
4. R2 config除第5.4节allowlist外出现native field drift，或只能靠Python翻译表修复；
5. production接线需要修改vendor、Source、Code #1-#5 fixture/expected或新增compatibility patch；
6. adapter需要backend private access、XML/asset/model metadata、hot-path capability probe、
   多次action transfer或额外collector/thread；
7. first reset无法复用`init_state()`，或必须二次reset/改变Code #8 lifecycle才能工作；
8. train不能通过`Runner.set_vec_env`进入native `A2CAgent`，或需要UniLab自写rollout/update；
9. tracker bridge需要第二个`wandb.init/finish`、修改native算法callback/RNG或覆盖source run
   metadata；
10. `.pth`需要schema conversion、suffix伪装、arbitrary unsafe path、payload rewrite或修改native
    checkpoint owner；
11. resume只有声称env/RNG/trajectory bit-exact、删除native rollout state或新增env snapshot
    public contract才能成立；
12. player需要重写normalizer/RNN/action/coefficient routing、修改Source deterministic语义或
    调用RSL policy；
13. N=6/5/7 player语义与Code #5冲突且不能在production bridge内解释；
14. assigned-tool visual mapping在批准的cold-path playback helper修正后仍不正确，或修复还
    需要修改env、其他backend/visualization owner、新公共playback contract、private
    production access或T1 rebaseline；
15. tiny real slice出现non-finite、没有actor/central update、没有native `.pth`、player不能
    restore/step/video，或只能缩network/关SAPG/关central来通过；
16. required test出现skip/failure或无法解释warning；
17. 任一child超过约15 paths/800行净手写adaptation，任一production file达到800行，或需要
    新增未批准owner/path；
18. cleanup后残留generated XML、playback temp models、`MUJOCO_LOG.TXT`、writer/renderer
    process，或出现writer overlap、范围外working-tree改动、staging不为空。

不得通过隐藏skip、mock真实backend/Runner/player、减少600 pool、预缩env reward、关闭AMP/
central/SAPG、加载public PyPI RL-Games、`sys.path`注入、放宽checkpoint trust、渲染tool0、
退回19-mesh visual、静默rebaseline或进入Code #10绕过停止条件。

## 12. 实现 session 交接格式

成功时只以 `# DONE` 开头，并依次报告：

1. 起始/结束branch和HEAD；
2. 9A-9D每个child实际修改paths、行数、净手写规模和确认无范围外改动；
3. `git status --short`、tracked/untracked inventory、staging为空；
4. Source reference commit/train owner、vendor distribution/version/path/72+7/manifest
   hashes和M0-dev identity；
5. root extra/uv.lock exact source、base install absence、selected-extra install和Linux/aarch64
   fail-closed证据；
6. base/12k Hydra compose、R2 field-by-field allowlist、raw reward × native0.01 boundary和
   preflight failures；
7. adapter reset/step/obs/action/done/timeout/device/one-transfer/env-state ABI证据；
8. native Runner/A2CAgent owner、load/set_vec_env/run_train call order和没有第二rollout loop；
9. tracker/run dir/TensorBoard/W&B exact-once lifecycle和exception cleanup；
10. `.pth` resolver trust边界、fresh/resume/weights行为、native payload/env_state None和明确的
    non-bit-exact resume声明；
11. native PpoPlayerContinuous N=6/5/7 routing、deterministic/stochastic、normalizer/RNN/done
    reset/action bounds；
12. CLI/completion exact commands、wrong owner/platform/dependency fail-closed；
13. assigned visual mapping中indexes0/1/7的40-mesh、object topology/size、assignment和cleanup
    evidence；
14. tiny real N=6 workload、真实rollout/update/checkpoint/player/4 steps/video、artifact paths/
    sizes和finite evidence；
15. Phase 9A-9D首次RED或诚实first-run GREEN、失败原因和最终GREEN；
16. full SAPG oracle、Code #9 focused/real、Code #8/neighbor、Ruff、format、mypy、pyright、
    lock/vendor audit每条命令的exit/pass/skip/warning/time；
17. cold-path/private/source-access/single-W&B/no-second-loop/scope/cleanup审计、
    `MUJOCO_LOG.TXT` absence和`git diff --check`；
18. 明确确认没有Git写操作、没有修改Source/donor/vendor/backend/env/task formula/fixtures/
    shared sim2sim、没有运行`make test-all`、没有进入Code #10。

阻塞时只以 `# BLOCKED` 开头，给出停止条件编号、最后一个成功child/gate、失败命令和关键
输出、actual/expected、所需最小owner/path/contract、当前工作树状态及已创建文件。不要自行
清理。

无论 `# DONE` 或 `# BLOCKED`，报告后停止，等待控制 session审查。
