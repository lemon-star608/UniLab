# SimToolReal SAPG Code #7 执行 Prompt

> 本文件交给一个新的实现 session。用户明确说“看
> docs/simtoolreal_sapg_code7_prompt.md，按文档执行 Code #7 的全部 child batches，
> 不进入 Code #8”即表示 7A-7D 都已获得 execution approval。实现 session 必须直接完成
> 已批准的全部 child batches，不派 subagent，不重新规划，不进入 Code #8。完成后保留
> 全部改动未暂存、未提交，交回控制/审查 session。

## 1. 本批唯一结果

把旧 UniLab 真正用于 SimToolReal 训练的最小资源闭包和 backend-neutral task math 放入
target，并形成一个不依赖真实 MuJoCo/IsaacSim 轨迹的 Source T0 oracle。

完成时必须同时具备：

1. 旧 UniLab 生产 XML 实际引用的 40 个 mesh、2 个生产 XML 和完整 license/provenance；
2. 不注册 env 的 SimToolReal config、constants、tool catalog/materializer 和 task
   primitives；
3. action、delay、goal、keypoint、observation、raw reward、episode lifecycle、reset 和
   wrench DR 的 focused unit tests；
4. 固定 Source native task utilities 生成的 T0 fixture、manifest 和 Target replay；
5. 所有 asset/XML 读取只发生在 init、materialization 或 cache 冷路径。

本批不要求 Source IsaacSim 与 Target MuJoCo 的真实 trajectory、接触、随机序列、reward
curve 或训练曲线相同。T0 只比较给定同一组 backend-neutral primitive inputs 和显式随机
draws 时的 task math。

实现 session 不执行 git add、git commit、git push、PR、stash、reset、clean、checkout
或切分支。控制 session 审查后决定按 7A-7D 形成一个或多个代码提交，并在代码提交后更新
总指导文档。

## 2. 普通中文范围与 child batches

### 2.1 只做什么

- 只迁移旧 UniLab 的 kuka_sharpa.xml 和 scene.xml 实际引用的 mesh 闭包；
- 只迁移生产训练会用到的 robot visual/collision mesh，不复制完整 Source/donor 资产树；
- 固定 Source、donor、XML、license、mesh 的 provenance；
- 从固定 donor snapshot 移植约 14 个成熟 task foundation modules；
- 为 Source-native RL-Games 路径保留 raw task reward，不带入 donor 的 RSL-RL owner
  scaling；
- 以 unit-test env carrier 和显式随机 draws 验证 task primitives；
- 生成一次 Source T0，普通 pytest 只读取 fixture 并运行 Target replay。

### 2.2 明确不做什么

- 不创建 env.py，不注册 SimToolReal，不实例化真实 NpEnv；
- 不新增 registry composition、Hydra production owner 或任何 conf 文件；
- 不 materialize、compile 或 step 真实 600-tool pool；Code #7 只验证 catalog census 和每种
  topology 的代表性冷路径；
- 不做 T1、真实 reset/step、MuJoCo rollout、IsaacSim/MuJoCo 轨迹对拍；
- 不接 Runner、RlGamesNpEnvAdapter、tracker、pth resolver、player 或 CLI；
- 不迁移 RSL-RL SAPG、collision research、primitive tuning、viewer、DexToolBench、
  Menagerie preview、Sharpa Wave、外部评估工具或 Motrix；
- 不迁移 StudentObsCfg、student camera、foundation stereo、depth preprocessing 或
  distillation-only observation path；
- 不修改 Source、donor、vendored RL-Games、MuJoCoUni 或 Code #1-#6 owner；
- 不新增 backend public contract，不调用 backend 子类私有能力；
- 不把 keyframe 放进 robot XML。

### 2.3 为什么拆成 child batches

Code #7 是一个 roadmap umbrella，资产本身就有 40 个 binary files，不能伪装成一个
15-file implementation issue。用户本次明确批准执行全部 child batches；实现时仍按以下
边界顺序推进和独立验证：

~~~text
7A  training asset closure、XML、license/provenance
7B  config/constants、deterministic tool catalog、cold materializer
7C1 action/delay、goal/keypoint primitives
7C2 observation、raw reward、episode lifecycle primitives
7C3 reset provider、wrench DR primitives
7D  fixed Source T0 generator、fixture、manifest、Target replay
~~~

40 个 mesh 是精确、机械的 asset payload；不是 40 个手写 owner files。除 7A 的 binary
payload 外，每个 child 的 production/test text paths 均保持在约 15 个以内。若任一 child
需要独立的新 public contract、execution path 或明显超过约 800 行净手写 adaptation，
停止并回报，不要顺手扩张。

预计永久维护成本是：

- 40 个训练 XML 实际引用的 mesh、2 个 XML、2 个 license 和 1 个 provenance inventory；
- 约 14 个 backend-neutral task modules；
- focused task tests 和 1 个 Source T0 golden。

donor 中约 3.8k 行 task foundations 可机械移植；本批新增判断应主要是 owner 隔离、
Source-native reward boundary、cold-path materialization、provenance 和 T0 harness，而
不是重新发明第二套 task 算法。

## 3. 必读内容、起点和固定身份

### 3.1 开始前完整阅读

~~~text
AGENTS.md
docs/simtoolreal_sapg_source_fidelity_migration_plan.md
src/unilab/base/base.py
src/unilab/base/np_env.py
src/unilab/base/backend/base.py
src/unilab/dr/provider.py
src/unilab/dr/types.py
src/unilab/dr/manager.py
src/unilab/utils/rotation.py
src/unilab/dtype_config.py
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
a1302dda84880335a08dd300496bfb9082e13e93
docs: record SAPG Code 6 completion
~~~

实现 session 的 dispatch HEAD 应是上述基线的单个 docs child。该 docs child 只新增本
prompt，并把总指导文档中的 42-mesh 旧描述修正为 40-file training closure。开始时运行：

~~~bash
set -e
set -o pipefail
SAPG_CODE7_BASE=a1302dda84880335a08dd300496bfb9082e13e93
test "$(git rev-parse --abbrev-ref HEAD)" = "feat/simtoolreal-sapg-rlgames"
git merge-base --is-ancestor "$SAPG_CODE7_BASE" HEAD
test "$(git rev-list --count "$SAPG_CODE7_BASE"..HEAD)" -eq 1
test "$(git diff --name-status "$SAPG_CODE7_BASE"..HEAD)" = \
  $'A\tdocs/simtoolreal_sapg_code7_prompt.md\nM\tdocs/simtoolreal_sapg_source_fidelity_migration_plan.md'
test -z "$(git status --short)"
test -z "$(git diff --cached --name-only)"
git log -2 --oneline
git status --short --branch
~~~

任一条件不成立就返回 # BLOCKED，不要清理或覆盖现有改动。

### 3.2 固定 Source identity

~~~text
Source checkout:
  /home/user/ws/lemon/simtoolreal
Source HEAD:
  2a9917533bfea70419ed2667a511d7238e5b3abc
task owner:
  isaacsimenvs/cfg/task/SimToolReal.yaml
task owner blob:
  6469d46867081b70edaa589dcb31c7090b64d45e
~~~

T0 相关 Source task owners：

~~~text
isaacsimenvs/tasks/simtoolreal/simtoolreal_env.py
  42c03361249fffde36c9f212b118c416f394cf53
isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py
  892f03c4b65e96427e59a2bbabf9386c09524be6
isaacsimenvs/tasks/simtoolreal/utils/action_utils.py
  17da0abbc42b67447019dfa806b825ee0d7bbc6c
isaacsimenvs/tasks/simtoolreal/utils/goal_sampling.py
  73a091e4a5dbb7c478dd3f5b36a8f942bb6940f7
isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py
  ffd8ff96fc3960af658ac7d76cd43b57552b6f72
isaacsimenvs/tasks/simtoolreal/utils/reset_utils.py
  310bd76004bf21a9a600d66ab1e5a56fa3de4b22
isaacsimenvs/tasks/simtoolreal/utils/reward_utils.py
  f8c0ea5c7d64efa1e702a3b458c7a44294400743
isaacsimenvs/tasks/simtoolreal/utils/termination_utils.py
  4960d133d44a630a13fbe3aeb0d29f5dff5c508f
isaacsimenvs/tasks/simtoolreal/utils/object_size_distributions.py
  015471e5f5ddab3438efe2d203e9ce062466353f
isaacsimenvs/tasks/simtoolreal/utils/generate_objects.py
  73e2129fd21186061f8a69e8370d736d75523547
isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py
  9cef43173237b003c0bb7ee6043ca5e67827fa7d
~~~

manifest 必须记录实际加载的 Source modules；上表是允许的 owner universe，不代表 generator
必须为凑 inventory 强行加载所有文件。generator 不得从 Source 当前未跟踪文件或当前
working tree 偷读第二套实现。

### 3.3 固定 mature donor snapshot

~~~text
Donor repository:
  /home/user/ws/lemon/UniLab
Donor commit:
  74075b3238e3176650a9440984a74be3629ff93f
Donor subject:
  revert(simtoolreal): disable mocap table reset
~~~

选择这个 snapshot 是因为它已包含：

- multiccd="disable"；
- static fixed table；
- corrected goal refresh 和 per-object keypoint geometry；
- real 600-tool foundations；
- object wrench 仍按生产训练语义激活；
- 尚未带入后续 DexToolBench/external-tool 和 target-only lift-confirmation 扩张。

donor 当前 checkout 是 dirty 且 HEAD 已前进。所有 donor 读取必须通过固定 commit 的
git show、git ls-tree、git cat-file 或 git archive；禁止直接 cp 当前 working tree，
禁止 cherry-pick。

固定 donor XML blobs：

~~~text
src/unilab/assets/robots/kuka_sharpa/kuka_sharpa.xml
  389380675eeb4b0ed7a4989eeeb236a1fbd31d8b
src/unilab/assets/robots/kuka_sharpa/scene.xml
  69629b71d1299c1463a200880ab124b6543d3202
~~~

### 3.4 固定 license 和特殊 mesh provenance

~~~text
Source root LICENSE
  license: MIT
  blob: 1dc25f4cf8eef86f7ff2be000492b3e737b91f1c

Source assets/licenses/kukaiiwa-LICENSE.txt
  license: BSD-2-Clause
  blob: 46670489513480eff80b81e3ec780abf29e347bd
~~~

40 个目标 mesh 必须来自 donor commit
74075b3238e3176650a9440984a74be3629ff93f。其中 39 个与固定 Source 对应文件
byte-identical；唯一例外是实际训练使用的：

~~~text
target/donor:
  assets/left_sharpa_meshes/left_hand_C_MC_visual.STL
  donor blob 4eaa0d5d0d57fb42b50e8e66e91bf3904f9a47fa

Source original:
  assets/urdf/kuka_sharpa_description/left_sharpa_meshes/left_hand_C_MC_visual.STL
  Source blob ec9632db49c79c84e25868a28c2796b350121360
~~~

必须保留 donor 的训练版本并在 ASSET_PROVENANCE 中显式记录差异；不要因为“更接近 Source”
而换回 1,254,651-byte Source 文件。

## 4. 唯一允许新增或修改的实现路径

### 4.1 训练资产闭包

~~~text
src/unilab/assets/robots/kuka_sharpa/LICENSE.simtoolreal
src/unilab/assets/robots/kuka_sharpa/LICENSE.kuka_iiwa
src/unilab/assets/robots/kuka_sharpa/ASSET_PROVENANCE
src/unilab/assets/robots/kuka_sharpa/kuka_sharpa.xml
src/unilab/assets/robots/kuka_sharpa/scene.xml
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/collision/link_0.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/collision/link_1.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/collision/link_2.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/collision/link_3.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/collision/link_4.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/collision/link_5.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/collision/link_6.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/collision/link_7.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/visual/link_0.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/visual/link_1.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/visual/link_2.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/visual/link_3.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/visual/link_4.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/visual/link_5.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/visual/link_6.stl
src/unilab/assets/robots/kuka_sharpa/assets/new_iiwa14_meshes/visual/link_7.stl
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_hand_C_MC_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_hand_C_MC.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_thumb_CMC_VL.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_thumb_MC_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_thumb_MC.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_thumb_MCP_VL_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_thumb_PP_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_thumb_PP.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_thumb_DP_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_thumb_DP.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/thumb_elastomer_surface.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/thumb_elastomer.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_MCP_VL_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/MCP_VL.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_PP_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_PP.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_MP_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_MP.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_DP_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_DP.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/elastomer_surface.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/elastomer.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_pinky_MC_visual.STL
src/unilab/assets/robots/kuka_sharpa/assets/left_sharpa_meshes/left_pinky_MC.STL
~~~

最终 kuka_sharpa/assets 下必须恰好是以上 40 个 regular mesh files。

### 4.2 Task foundation modules

~~~text
src/unilab/envs/manipulation/simtoolreal/__init__.py
src/unilab/envs/manipulation/simtoolreal/action_pipeline.py
src/unilab/envs/manipulation/simtoolreal/config.py
src/unilab/envs/manipulation/simtoolreal/constants.py
src/unilab/envs/manipulation/simtoolreal/delay_buffer.py
src/unilab/envs/manipulation/simtoolreal/dr_provider.py
src/unilab/envs/manipulation/simtoolreal/dr_wrench.py
src/unilab/envs/manipulation/simtoolreal/episode_lifecycle.py
src/unilab/envs/manipulation/simtoolreal/goal_sampling.py
src/unilab/envs/manipulation/simtoolreal/keypoints.py
src/unilab/envs/manipulation/simtoolreal/observations.py
src/unilab/envs/manipulation/simtoolreal/rewards.py
src/unilab/envs/manipulation/simtoolreal/tool_assets.py
src/unilab/envs/manipulation/simtoolreal/tool_catalog.py
~~~

不要创建 env.py。若 donor 的 TYPE_CHECKING import 指向不存在的 .env，改为窄 Any/Protocol
seam，使 mypy/pyright 在 Code #7 独立通过；不要为了类型检查创建假的 env owner。

### 4.3 Focused tests、T0 generator 和 fixtures

~~~text
tests/envs/manipulation/simtoolreal/test_assets.py
tests/envs/manipulation/simtoolreal/test_config.py
tests/envs/manipulation/simtoolreal/test_tool_catalog.py
tests/envs/manipulation/simtoolreal/test_tool_assets.py
tests/envs/manipulation/simtoolreal/test_delay_buffer.py
tests/envs/manipulation/simtoolreal/test_action_pipeline.py
tests/envs/manipulation/simtoolreal/test_goal_sampling.py
tests/envs/manipulation/simtoolreal/test_keypoints.py
tests/envs/manipulation/simtoolreal/test_observations.py
tests/envs/manipulation/simtoolreal/test_rewards.py
tests/envs/manipulation/simtoolreal/test_episode_lifecycle.py
tests/envs/manipulation/simtoolreal/test_dr_provider.py
tests/envs/manipulation/simtoolreal/test_dr_wrench.py
tests/envs/manipulation/simtoolreal/source_t0_harness.py
tests/envs/manipulation/simtoolreal/test_t0_golden.py
future generator directory: scripts
future generator filename: generate_simtoolreal_task_t0_fixture.py
tests/fixtures/simtoolreal_task/source_t0_fp32.npz
tests/fixtures/simtoolreal_task/source_t0_manifest.json
~~~

允许合并相邻 focused test 内容，但不允许增加第 16 个 task test path；若确有必要，停止并
说明不能在现有测试 owner 中表达的原因。不要复制 donor 的 integration、real pool、
collision、script、RSL-RL 或 env tests。

实现 session 不修改总指导文档、本 prompt、pyproject.toml、uv.lock、Source、vendor、
backend 或任何 conf 文件。

所有 Python 命令必须通过 uv run。手工文本编辑只使用 apply_patch。40 个 binary meshes
和固定 license/XML 的精确机械导入可先用 git archive 从固定 donor/Source commit 解到
mktemp -d，再按本节 inventory 复制；不得从 dirty working tree 复制，不得把整个目录盲目
导入。生成 NPZ 只能使用本 prompt 规定的 generator。

## 5. 7A：training asset closure contract

### 5.1 精确 40-mesh inventory

kuka_sharpa.xml 的 mesh file attributes 是 inventory owner：

- new_iiwa14_meshes/collision/link_0.stl 到 link_7.stl：8 个；
- new_iiwa14_meshes/visual/link_0.stl 到 link_7.stl：8 个；
- left_sharpa_meshes 下由 XML 直接引用的文件：24 个。

donor 的 left_sharpa_meshes 目录有 26 个文件。以下两个没有被生产 XML 引用，必须排除：

~~~text
left_hand_C_MC_visual_.STL
left_thumb_MC_modified.STL
~~~

也不得复制：

- sharpa_mount.stl；
- Source URDF/USD；
- primitive tuning XML/mesh；
- external-tool/DexToolBench/Menagerie/Sharpa Wave assets；
- preview、viewer、trajectory、evaluation data；
- donor build/debug outputs。

visual meshes 不能因为训练 physics 主要使用 collision mesh 而删除：kuka_sharpa.xml 明确
引用它们，后续 UniLab play 使用同一 production asset。

### 5.2 XML roles

kuka_sharpa.xml 是纯 robot description：

- 29 个 robot joints 和 29 个 position actuators；
- 40 个 mesh assets；
- KUKA + Sharpa body/joint/actuator/contact exclusions；
- 不含 task object、table、goal 或 keyframe。

在安装了 MuJoCo 的测试中直接编译后必须至少满足：

~~~text
nq = 29
nv = 29
nu = 29
nmesh = 40
~~~

scene.xml 是 production task scene template：

- include kuka_sharpa.xml；
- sim timestep 1/120、gravity -9.81；
- multiccd="disable"；
- fixed world-welded narrow table，surface z=0.53；
- task-level home keyframe；
- object body 由 tool_assets.py 在冷路径插入。

scene.xml 在 object 插入前因为 home keyframe 已包含 object qpos 而不是独立可编译模型；
测试不能把这个预期 template 形态误报成错误。代表性 tool materialization 后必须真实
MuJoCo compile，并得到：

~~~text
nq = 36
nv = 35
nu = 29
nmesh = 40
~~~

keyframe 必须只在 scene.xml，不得移进 kuka_sharpa.xml。删除 scene.xml 中对未迁移
build_simtoolreal_assets.py 的“继续编辑 generator”误导性注释可以作为 comment-only
adaptation；不得改变 XML 数值/结构语义。

### 5.3 License 和 ASSET_PROVENANCE

LICENSE.simtoolreal 必须是固定 Source root LICENSE 的原始 bytes；
LICENSE.kuka_iiwa 必须是固定 kukaiiwa-LICENSE.txt 的原始 bytes。

ASSET_PROVENANCE 使用 deterministic UTF-8 strict JSON，即使文件名没有 .json。至少包含：

- schema_version；
- Source path、HEAD；
- donor path、commit；
- 两个 XML 的 donor path/blob/sha256；
- 两个 license 的 Source path/blob/sha256 和目标文件名；
- 40 个 mesh 的 target relative path、donor path/blob/sha256、Source 对应 path/blob
  或明确的 donor-only adaptation；
- 39-byte-identical + 1-different census；
- 两个明确排除的 donor mesh；
- generated/ordinary tests 不会更新 provenance 的声明。

XML entry 要分别记录 pristine donor sha256 和 target sha256；若 scene.xml 只改了第5.2节
允许的注释，这个差异也必须明确。ASSET_PROVENANCE 不记录自己的 hash，避免 self-hash
循环；test_assets.py 固定它的外层 sha256 anchor。

test_assets.py 必须 fail closed 验证：

1. XML references 恰好 40、去重后仍为 40；
2. 每个 reference 都是目标目录内 regular file，不越界、不 symlink；
3. asset 目录没有第 41 个 mesh；
4. XML/mesh/license/provenance inventory 与记录的 sha256 一致；
5. 特殊 left_hand_C_MC_visual.STL 使用 donor blob 对应 bytes，而不是 Source original；
6. 两个排除文件不存在；
7. robot XML 无 keyframe，scene XML 有且只有 task-level keyframe；
8. robot XML 真实编译得到 29/29/29/40。

普通测试不访问 donor 或 Source checkout；所有固定 hash facts 来自已 review 的
ASSET_PROVENANCE 和 test anchors。

## 6. 7B-7C：task foundation contracts

固定 donor commit 是成熟参考，不是可盲拷贝的最终 owner。必须保留 task math，同时做本节
列出的 Code #7 isolation/adaptation。

### 6.1 Package 和 config isolation

__init__.py：

- 可以导出 config classes 和 SimToolRealDRProvider；
- 不能 import .env，不能触发 registry registration；
- import unilab.envs.manipulation.simtoolreal 必须在 env.py 不存在时成功。

config.py：

- 可定义 AssetsCfg、ObsCfg、ActionCfg、RewardCfg、GoalCfg、ResetCfg、TerminationCfg、
  DomainRandomizationCfg 和 SimToolRealCfg；
- SimToolRealCfg 可继承 EnvCfg，并把 scene 指向 shipped scene.xml；
- 不 import registry，不使用 registry.envcfg decorator；
- actor obs=140、critic state=162、action=29；
- sim_dt=1/120、ctrl_dt=1/60、600 policy steps/10 seconds；
- reward term defaults 保持 Source raw scales：
  200、20、300、50、1000、0.03、0.003；
- production reset default 使用 source_random/full SO(3)，固定 MuJoCo table mapping 的
  object_spawn_z_reference_range 为 0.0；
- reduced pool 是 12 distributions × 50 = 600，local seed 42；
- force/torque wrench DR 保持开启语义，不新增 donor 后续的 default-off switch；
- validate 对 shape/range/frequency/episode/asset-owned values fail closed。

不要保留指向本批未迁移 Source URDF/table URDF 的 dead runtime config path。原始 asset
来源由 ASSET_PROVENANCE 持有，不由 AssetsCfg 中一个部署时必然不存在的路径冒充。

不要把 Code #9 的 RL-Games params 或 Hydra owner 塞进 config.py。算法超参数最终直接由
Code #9 YAML compose owner 持有。

### 6.2 Constants、catalog 和 cold materializer

constants.py 保留：

- 29/7/22 joint counts 和 canonical joint order；
- palm/body/fingertip names；
- joint gains/default pose/offsets；
- 4 keypoint corners；
- OBS_FIELD_SIZES 和 compute_obs_dim；
- 所有值从固定 Source owner逐值迁移，不重新圆整。

tool_catalog.py：

- 12 个 Source object-size distributions；
- 每个 distribution 50 samples，合计 600；
- 使用 local np.random.RandomState(seed)，不能污染 global NumPy RNG；
- draw 顺序保持 handle densities、head densities、handle scales、head scales；
- shuffle lockstep；
- exact topology census：box_box=250、capsule_box=300、box_only=50；
- ToolSpec 只保存编译前 immutable geometry/mass/COM/inertia/scale metadata；
- 不读取 XML/assets，不 materialize backend。

tool_assets.py：

- 只在显式 materialize_tool_scenes 调用中读取 scene XML；
- 在编译前把一个 immutable ToolSpec 写成完整 source model；
- 使用系统临时目录或调用者显式 temp root，不在 installed package asset directory 下写；
- 解析 include/mesh path 后仍能从只读 asset parent materialize；
- cleanup owner 明确，异常时清理；
- Code #7 只真实 compile box_box、capsule_box、box_only 各一个代表，不编译 600 个。

### 6.3 Action 和 delay

delay_buffer.py 与 action_pipeline.py 必须锁定 Source 顺序：

1. policy action clamp 到 [-1,1]；
2. canonical order 转 backend order；
3. action queue roll/flush 和 per-env delay index；
4. arm velocity-delta accumulator，dt=1/60、speed scale=1.5；
5. arm raw clamp、EMA=0.1、第二次 clamp；
6. hand absolute [-1,1] 映射、EMA=0.1、最终 clamp；
7. cur_targets 写回并 copy 到 prev_targets；
8. raw action bookkeeping 保持 canonical order。

fresh episode 只有 steps==0 且 successes==0；同 episode goal advance 不得误清 delay queue。
测试必须覆盖 per-row delay indices、queue roll、partial-row reset、no alias、shape rejection 和
canonical/backend permutation。

### 6.4 Goal 和 keypoints

goal_sampling.py 必须保留：

- absolute goal uniform workspace；
- workspace center scaling；
- delta position clamp；
- random axis/angle quaternion perturbation；
- wxyz internal convention。

keypoints.py 必须保留：

- fixed 4-corner ordering；
- object-local offsets到 world；
- max keypoint distance；
- object scale phi × 0.04 metre conversion，避免 25x trap；
- fixed-size reward geometry 与 per-object observation geometry 的明确分离。

所有 random draws 在 production 仍由 NumPy owner产生；focused tests/T0 可 monkeypatch
明确 draws/indices，但不能把 test-only fixture path写入 production config。

### 6.5 Observations

observations.py 返回：

~~~text
{"obs": float32[N,140], "critic": float32[N,162]}
~~~

必须锁定：

- actor/critic field list、顺序和 width；
- canonical joint pos normalize、joint vel、previous targets；
- palm center/fingertip offsets；
- wxyz internal，stack boundary 对 actor/critic 都转 xyzw；
- actor object pose/velocity noise、object-state delay、joint-velocity noise 和 actor obs delay；
- critic clean/no-delay privileged fields；
- progress=log(steps/10+1)，successes=log(successes+1)；
- clamp 到 [-10,10]；
- reset rows 只更新选中 rows 的 queues/outputs；
- asset/XML/model metadata 不进入本函数。

Source-native reward boundary必须特别修正 donor 的 RSL-RL-specific behavior：

- compute_rewards 返回 raw total；
- info["reward"] 在 observation build 时也是 raw total；
- critic 的 reward feature 必须是 raw total × 0.01，匹配固定 Source
  obs_utils.py 的 env.reward_buf * 0.01；
- Code #9 的 native RL-Games reward_shaper 再把 learner 收到的 raw env reward ×0.01。

critic feature scaling 和 learner reward scaling 是两个不同消费者，不是对同一 tensor 连乘。
不得把 RewardCfg defaults 预缩成 2/0.2/3/0.5/10，也不得保留 donor 为 RSL-RL owner做的
“直接消费已经 env-scaled reward”注释/测试。

### 6.6 Raw reward 和 episode lifecycle

rewards.py 必须返回七项 raw reward 的直接和：

~~~text
fingertip_delta_rew
lifting_rew
lift_bonus_rew
keypoint_rew
kuka_actions_penalty
hand_actions_penalty
bonus_rew
total_reward
~~~

锁定：

- z_lift = 0.05 + object_z - object_init_z；
- lifted latch 和 one-shot lift bonus；
- lift 前 fingertip d-star，lift 后 keypoint d-star；
- 两个所谓 action penalty 实际都是 joint-velocity L1 penalty；
- default reach bonus 按 success_steps=10 amortize；
- no ctrl_dt scaling、no 0.01 global scaling；
- tracker updates 与返回 reward dtype/shape。

episode_lifecycle.py 锁定：

- success threshold=current tolerance × keypoint_scale；
- cumulative/default near-goal counter 与 optional consecutive mode；
- success 只推进 goal，不结束 episode；
- goal advance 清 d-star/lift/near-goal trackers 并避免 timeout 同步触发；
- drop、hand-far、max-success、timeout masks；
- tolerance curriculum；
- Code #6 engine-autoreset mask作为明确输入/已缓存 state 使用，不探测 backend 私有能力。

### 6.7 Reset provider 和 wrench DR

SimToolRealDRProvider 只拥有 task reset plan：

- canonical robot default + arm/finger position noise；
- joint velocity Uniform(-0.5,0.5)；
- source_random object x/y/z 和 full SO(3)；
- fixed MuJoCo table reference且 z-reference jitter=0；
- first goal absolute sampling；
- qpos/qvel/index mapping；
- success/lift/d-star/reward trackers；
- delay queues、wrench cache和 object-scale multiplier reset；
- per-reset log-uniform force/torque trigger probabilities；
- optional trajectory file只在 init/cache 冷路径读取。

provider.validate 不得使用 getattr/hasattr 探测 env._backend.set_state 或任何 backend private
surface。reset manager已经通过 SimBackend public set_state contract工作；若真实 owner
接线需要变化，留给 Code #8，不在 Code #7 发明 capability shim。

dr_wrench.py 锁定：

- force/torque decay；
- per-env Bernoulli fire；
- randn × object mass × force_scale/torque_scale；
- previous-step lifted latch gates；
- 调用 SimBackend public apply_body_wrench；
- force/torque shape为 (N,1,3)；
- 不 step physics，不读 XML/assets，不调用 MuJoCo private field。

test_dr_provider.py 和 test_dr_wrench.py 使用窄 fake env/public fake backend验证公式、row
isolation、reset cache 和 call arguments；不实例化 env.py，不复制 production 公式作为
expected implementation。

## 7. 7D：Source T0 contract

### 7.1 T0 的边界

T0 使用固定 N=6 的 CPU FP32 synthetic cases，至少覆盖：

- joint/object/palm/fingertip/tool state；
- raw policy action、previous targets、canonical/backend permutation；
- action/obs/object-state queues和显式 per-env delay indices；
- goal pose、object scale、keypoint geometry；
- steps、successes、near-goal、lift和 d-star trackers；
-显式 uniform、normal、Bernoulli、orientation draws；
-必要的命名 simulator-query inputs，例如 body pose/velocity和 fingertip distance；
- source_random reset和 wrench DR inputs。

T0 明确不比较 Torch 与 NumPy 的 RNG algorithm/state。uniform、normal、Bernoulli、
delay-index 和 orientation draws 是 fixture primitive inputs，分别注入 Source native
utility 和 Target production function；T0 比较这些 draws 之后的 task transform。manifest
必须把这个边界写清楚，不能声明随机序列 parity。

允许的 backend/resource mapping 只有：

- Source movable table-height sample映射为 Target fixed table reference，Target
  object_spawn_z_reference_range 为 0.0；不比较 table pose trajectory；
- Source 12×100 catalog映射为 Target 12×50 catalog；distribution定义和单个 ToolSpec math
  比较，完整 pool assignment留给 Code #8 T1；
- IsaacLab tensor carrier映射为 NumPy/public fake-backend query inputs。
- Source observation key "policy" 映射为 Target NpEnv observation key "obs"；"critic" 保持
  "critic"。

除此之外，action/obs/reward/success/termination/reset/DR formula不允许以 backend差异为由
改变。

Source native task utility functions必须产生并保存：

- 29D backend-order action target；
- actor observation (6,140)；
- critic state (6,162)；
-七个 raw reward terms和 raw total；
- updated lifted/d-star/near-goal/success trackers；
- terminated、truncated和 goal/reset masks；
- sampled qpos/qvel、goal pose、wrench probability/force/torque和 scale multiplier；
-所有字段的顺序、shape、dtype和单位。

fixture 同时保存公式所需 primitive inputs、queues和 explicit draws，不能只保存最终
obs/reward。Target replay 必须用这些 primitive inputs调用本批 production functions；
测试中不得手写一份 Target formula然后与 Source output比较。

### 7.2 Source-native execution

generator 必须：

1. 使用 #!/usr/bin/env python3 shebang，并始终通过 uv run调用；
2. 验证 Source checkout path和固定 HEAD；
3. 从固定 Git objects materialize实际需要的 Source task files到隔离临时 namespace，或
   对每个 working-tree loaded byte先与固定 blob逐字节验证；
4. 实际调用 action_utils、goal_sampling、obs_utils、reward_utils、termination_utils 和
   reset_utils 中本 case需要的 native functions；
5. 记录实际 loaded Source modules的 path/blob/sha256；
6. 不 import target task functions来生成 expected outputs；
7. 不需要 IsaacSim process或真实 physics。

Source utility import需要的最小 isaaclab.utils.math surface可以在
source_t0_harness.py 中建立轻量 stub，仅限：

~~~text
convert_quat
quat_apply
quat_from_angle_axis
quat_mul
random_orientation
~~~

以及 reset_utils import所需的 constants/env carrier。stub 必须：

- 不包含 action/observation/reward/reset/termination task formula；
- 有独立 quaternion known-vector tests；
- random_orientation 对 T0 返回显式注入的 primitive orientation draws，不在 stub 中发明
  另一套随机 sampler；
- 在 manifest 中记录 stub symbol inventory和文件 sha256；
- 在 Source module import前安装，在 capture后移除隔离 namespace；
- 不伪装成仓库安装了 IsaacLab。

若某一输出只能通过在 harness 中重新手抄完整 Source task formula才能产生，停止并返回
# BLOCKED；不能把翻译副本称为 Source-native oracle。

### 7.3 Fixture 和 manifest

~~~text
tests/fixtures/simtoolreal_task/source_t0_fp32.npz
tests/fixtures/simtoolreal_task/source_t0_manifest.json
~~~

manifest 至少包含：

- schema_version、generation_mode="source-only"；
- ordinary_pytest_regenerates=false；
- Source HEAD、task owner path/blob/sha256；
- loaded Source module inventory；
- stub inventory和限制；
- N=6、FP32、CPU、seed和 case names；
- explicit config values和允许的 MuJoCo table/tool-count mapping；
- NPZ array inventory：name、shape、dtype、sha256；
- fixture filename、NPZ sha256、canonical payload sha256；
- canonical generation command；
- float tolerance：rtol=1e-5、atol=1e-6；
-离散 exact字段 inventory。

test_t0_golden.py 必须固定 NPZ 和 manifest 外层 SHA256 anchors。manifest 自身不能在
自身内容中形成循环 hash；不能把 anchors 放进会被 manifest 记录 sha256 的
source_t0_harness.py，否则会形成 harness → manifest → test anchor 的循环。

NPZ：

- allow_pickle=False；
- key inventory exact；
-不保存 object arrays；
-所有 float finite；
- bool/int/index/mask dtype明确；
- deterministic key order和 deterministic zip metadata，保证同环境再生成 bytes一致。

普通 pytest：

- 不访问 /home/user/ws/lemon/simtoolreal；
- 不 import IsaacLab；
- 不运行 generator；
- 先验证两个 fixture hashes、manifest schema和 array inventory；
- 再执行 Target task foundations replay；
- bool/int/mask/index exact equality；
- float按 rtol=1e-5、atol=1e-6；
-任一 mismatch必须报告 array name、max abs/rel error和首个 bad index。

T0 mismatch不能通过扩大 tolerance、删 case、把随机 branch关掉或把 Source output直接喂给
Target output slot来绕过。先定位是 carrier mapping、dtype/order还是 task公式；无法解释就
停止。

## 8. 严格 RED → GREEN 执行顺序

### Phase 0：起点与只读 census

1. 运行第 3.1 节起点检查。
2. 用 fixed Git object命令核对 Source/donor identities和 blobs。
3. 从 donor fixed XML解析 mesh refs，记录 40；核对 donor mesh directory是42且恰好排除
   本 prompt列出的两个文件。
4. 核对 donor 当前 working tree即使 dirty也没有被当作复制来源。

### Phase 7A：assets、XML、license/provenance

1. 先创建 test_assets.py，断言目标 inventory、hash、XML和 compile contract。
2. 运行并记录它因 target asset root不存在而 RED：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal/test_assets.py -q
~~~

3. 用 mktemp -d + fixed git archive机械提取 donor assets/XML和 Source licenses，只写入第4.1
   节精确目标。
4. 用 apply_patch创建 ASSET_PROVENANCE和必要的 comment-only XML adaptation。
5. 重跑同一测试，必须 GREEN、0 skip。
6. 用 find/sha256sum/git diff --numstat独立确认 40 mesh、2 XML、2 licenses、1 provenance，
   没有额外 asset。

### Phase 7B：config、constants、catalog、materializer

1. 先创建 test_config.py、test_tool_catalog.py、test_tool_assets.py。
2. 运行并记录因 package/modules缺失而 RED：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal/test_config.py \
  tests/envs/manipulation/simtoolreal/test_tool_catalog.py \
  tests/envs/manipulation/simtoolreal/test_tool_assets.py -q
~~~

3. 从 fixed donor snapshot移植最小 modules，完成第6.1-6.2节 adaptations。
4. 代表性 materialized tool scene必须使用真实 shipped XML/mesh和真实 MuJoCo compile；
   不得 mock XML compiler。
5. 重跑并要求 GREEN、0 skip。

### Phase 7C1：action/delay、goal/keypoints

1. 先建立四个 focused tests并确认 module/behavior缺失导致 RED。
2. 最小移植 delay_buffer.py、action_pipeline.py、goal_sampling.py、keypoints.py。
3. 运行：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal/test_delay_buffer.py \
  tests/envs/manipulation/simtoolreal/test_action_pipeline.py \
  tests/envs/manipulation/simtoolreal/test_goal_sampling.py \
  tests/envs/manipulation/simtoolreal/test_keypoints.py -q
~~~

required tests必须覆盖非 identity permutation和非零 delay indices，0 skip。

### Phase 7C2：observations、raw rewards、lifecycle

1. 先建立三个 focused tests并确认 RED。
2. 最小移植 observations.py、rewards.py、episode_lifecycle.py。
3. 明确修正 donor RSL-RL scaling boundary：raw reward defaults/return保持 raw，critic reward
   feature单独 ×0.01。
4. 运行：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal/test_observations.py \
  tests/envs/manipulation/simtoolreal/test_rewards.py \
  tests/envs/manipulation/simtoolreal/test_episode_lifecycle.py -q
~~~

必须真实得到 actor 140、critic 162、七项 raw reward和 exact masks，0 skip。

### Phase 7C3：reset provider、wrench DR

1. 先建立 test_dr_provider.py和 test_dr_wrench.py并确认 RED。
2. 最小移植 dr_provider.py和 dr_wrench.py，移除 backend private capability probe。
3. 运行：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal/test_dr_provider.py \
  tests/envs/manipulation/simtoolreal/test_dr_wrench.py -q
~~~

必须覆盖 source_random full SO(3)、fixed table z mapping、row-restricted reset、cache clear、
log-uniform ranges、lift gate和 public apply_body_wrench arguments，0 skip。

### Phase 7D：T0

1. 先创建 test_t0_golden.py，确认 fixture/harness缺失导致 RED。
2. 创建 source_t0_harness.py和 generator；先只写到明确的临时 output并检查 manifest/
   inventory，不覆盖任何既有 fixture。
3. 在 fixed Source identity上显式生成首次 reviewed fixture：

~~~bash
CODE7_T0_GENERATOR_DIR=scripts
CODE7_T0_GENERATOR="$CODE7_T0_GENERATOR_DIR/generate_simtoolreal_task_t0_fixture.py"
uv run "$CODE7_T0_GENERATOR" \
  --source /home/user/ws/lemon/simtoolreal \
  --output tests/fixtures/simtoolreal_task \
  --source-only
~~~

4. 运行 Target replay：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal/test_t0_golden.py -q
~~~

5. 再生成到 mktemp -d，逐字节 cmp NPZ和manifest，证明 deterministic reproduction。若同一
   environment不能 byte-reproduce，停止；不得只比较松散的最终 values。

## 9. 最终验证

先运行 Code #7 focused gate：

~~~bash
uv run --extra mujoco pytest \
  tests/envs/manipulation/simtoolreal -q
~~~

required tests必须 0 skip。记录实际 passed、skipped、warnings。

再运行最邻近的 Code #6 source-model regression，证明 representative complete model仍走
public source_model_file contract：

~~~bash
uv run --extra mujoco pytest \
  tests/base/backend/test_mujoco_model_source_variants.py -q
~~~

再验证 T0 deterministic reproduction：

~~~bash
CODE7_T0_REGEN=$(mktemp -d)
CODE7_T0_GENERATOR_DIR=scripts
CODE7_T0_GENERATOR="$CODE7_T0_GENERATOR_DIR/generate_simtoolreal_task_t0_fixture.py"
uv run "$CODE7_T0_GENERATOR" \
  --source /home/user/ws/lemon/simtoolreal \
  --output "$CODE7_T0_REGEN" \
  --source-only
cmp tests/fixtures/simtoolreal_task/source_t0_fp32.npz \
  "$CODE7_T0_REGEN/source_t0_fp32.npz"
cmp tests/fixtures/simtoolreal_task/source_t0_manifest.json \
  "$CODE7_T0_REGEN/source_t0_manifest.json"
sha256sum \
  tests/fixtures/simtoolreal_task/source_t0_fp32.npz \
  tests/fixtures/simtoolreal_task/source_t0_manifest.json
~~~

运行 style/type gates：

~~~bash
CODE7_T0_GENERATOR_DIR=scripts
CODE7_T0_GENERATOR="$CODE7_T0_GENERATOR_DIR/generate_simtoolreal_task_t0_fixture.py"
uv run --extra mujoco ruff check \
  src/unilab/envs/manipulation/simtoolreal \
  tests/envs/manipulation/simtoolreal \
  "$CODE7_T0_GENERATOR"
uv run --extra mujoco ruff format --check \
  src/unilab/envs/manipulation/simtoolreal \
  tests/envs/manipulation/simtoolreal \
  "$CODE7_T0_GENERATOR"
uv run --extra mujoco mypy src/unilab
uv run --extra mujoco pyright
~~~

最后核对隔离、scope和工作树：

~~~bash
test ! -e src/unilab/envs/manipulation/simtoolreal/env.py
test ! -e MUJOCO_LOG.TXT
test -z "$(git diff --cached --name-only)"
git diff --check
git status --short
git diff --stat
git diff --numstat
git diff --name-only
git ls-files --others --exclude-standard
find src/unilab/assets/robots/kuka_sharpa/assets -type f | sort
if rg -n "registry\.envcfg|from \. import env|from \.env import|import \.env" \
  src/unilab/envs/manipulation/simtoolreal; then
  exit 1
fi
rg -n "read_text|read_bytes|open\(|from_xml" \
  src/unilab/envs/manipulation/simtoolreal || true
if rg -n "getattr\(.*backend|hasattr\(.*backend" \
  src/unilab/envs/manipulation/simtoolreal; then
  exit 1
fi
~~~

最后三个 rg 是人工审计命令：

- env/registry pattern必须没有 production hit；
- asset reads只允许 tool_assets materialization和 dr_provider fixed-trajectory cache冷路径；
- backend capability probe必须没有 hit。

实现 session 不运行 make test、make test-all，不创建或更新 PR。控制 session会独立阅读
完整 diff、复跑近风险 gates、核对 fixture/provenance，再决定提交。

## 10. 停止条件

出现任一情况立即停止写入并返回 # BLOCKED：

1. branch、lineage、single-docs-child、clean tree或empty staging不符合第3.1节；
2. fixed Source/donor Git objects、task owner、XML或license blob不可用；
3. 只能从 dirty donor/Source working tree复制，不能从固定 Git object取证；
4. 40-file XML closure不能成立，必须复制完整42-file目录或更多外部资产才能继续；
5. license/provenance无法明确，或特殊 visual mesh identity无法解释；
6. 需要创建 env.py、registry/config owner、真实env/T1、Runner或Code #8/#9路径；
7. 需要修改 Code #6 backend、MuJoCoUni、vendor、pyproject或uv.lock；
8. task hot path必须读取 XML/asset/model metadata；
9. env/task只能通过 getattr/hasattr探测 backend私有能力才能工作；
10. representative topology不能用 shipped asset真实 materialize/compile；
11. T0只能通过手抄第二套 Source task formula、运行真实 IsaacSim或把 expected output直接
    当 Target input才能建立；
12. Source/Target T0出现无法解释 mismatch；
13. required test有failure、skip或无法解释 warning；
14. 任一 text child明显超过约800行净手写 adaptation、需要第16个 task test path或新增
    未批准 public contract；
15. 出现 writer overlap、范围外工作树改动或staging不再为空。

不要通过扩大 tolerance、改 raw reward scales、关掉 delay/noise/wrench、删掉 branch
coverage、把真实 compile改成 mock、隐藏 skip、复制全部 donor assets或进入 Code #8 来绕过
停止条件。

## 11. 实现 session 交接格式

成功时只以 # DONE 开头，并依次报告：

1. 起始/结束 branch和HEAD；
2. 7A-7D每个 child实际修改路径及确认无范围外改动；
3. git status --short、tracked diff、untracked inventory、staging为空；
4. Source HEAD/task owner、donor commit、XML/license blobs；
5. 40-mesh census、16 KUKA + 24 Sharpa分组、两个明确排除文件；
6. 特殊 left_hand_C_MC_visual.STL 的 donor/Source双 identity；
7. XML compile结果：robot 29/29/29/40和 representative tool 36/35/29/40；
8. license/provenance schema、inventory/hash验证；
9. config isolation、无 env.py/registry import、raw reward/critic feature scaling边界；
10. action/delay、goal/keypoint、obs 140/state 162、raw reward/lifecycle、reset/wrench focused
    证据；
11. Phase 7A-7D每个初始 RED命令、失败原因和最终 GREEN；
12. T0 native Source modules/stubs、case inventory、fixture两个SHA256、deterministic cmp和
    Target replay结果；
13. focused、Code #6 neighbor、Ruff、format、mypy、pyright每条命令的exit status、
    pass/skip数和warnings；
14. cold-path audit、backend capability audit、MUJOCO_LOG.TXT absence和git diff --check；
15. 明确确认没有执行Git写操作、没有修改Source/donor/vendor/MuJoCoUni、没有运行
    make test-all、没有创建env.py、没有进入Code #8。

阻塞时只以 # BLOCKED 开头，给出停止条件编号、最后一个成功child/gate、失败命令和关键
输出、当前工作树状态及已创建文件。不要自行清理。

无论 # DONE 或 # BLOCKED，报告后停止，等待控制 session 审查。
