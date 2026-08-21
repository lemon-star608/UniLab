# SimToolReal Sharpa Menagerie 碰撞迁移设计

## 目标

把正式 SimToolReal MuJoCo 场景的 Sharpa 碰撞几何从当前 Mesh Sharpa 切换为
Menagerie Sharpa collision geometry，并固定为 `reference + tip_stiff` 接触 profile。
迁移只改变 Sharpa collision geoms 和正式 scene 的 MuJoCo 接触求解选项。

## 现状与 owner

- `scene.xml` 通过 `<include file="kuka_sharpa.xml"/>` 加载纯机器人 XML；floor、table、
  keyframe 属于 task-level scene owner。
- `kuka_sharpa.xml` 是 body/joint/inertial/actuator/visual、KUKA arm geometry 和 50 个
  contact exclude pair 的 canonical source。
- `tool_assets.py` 只在 cold path 复制 scene 并插入 object body；工具池不写入机器人 XML。
- 正式训练时间尺度是 `sim_dt=1/120`、`ctrl_dt=1/60`；simtoolreal 对比脚本的 1/600
  benchmark 不属于本 owner。

## 方案

保留现有 robot XML 的 body hierarchy、transform、inertia、joint/actuator 顺序和 visual
mesh。只替换 Sharpa collision geoms：

- hand palm 使用 32 个 Menagerie `palm000.obj`–`palm031.obj` mesh；
- thumb/index/middle/ring/pinky 的 MCP/VL、PP、MP 使用 donor 的 cylinder/capsule fit；
- DP/fingertip 使用 donor capsule collision；
- elastomer 只保留 visual-only geom；
- collision 统一 `contype=1 conaffinity=1 group=3 density=0 margin=0 gap=0`，visual 统一
  `contype=0 conaffinity=0 group=2`；KUKA arm collision/visual 保持现状。

唯一的正式接触参数 owner 是 robot XML 的 Sharpa collision geom：

| role | condim | friction | solimp | solref |
| --- | --- | --- | --- | --- |
| palm | 3 | `1.0 0.005 0.0001` | `0.9 0.95 0.001 0.5 2.0` | `0.01 1.0` |
| regular finger segment | 3 | `1.0 0.005 0.0001` | same | `0.02 1.0` |
| fingertip collision | 3 | `1.0 0.005 0.0001` | same | `0.04 1.0` |

`scene.xml` 是全局 MuJoCo option 的唯一 owner，声明 timestep `0.008333333333333333`,
`integrator=implicitfast`, `solver=Newton`, `cone=elliptic`, `impratio=10`,
`iterations=100`, `ls_iterations=50`, `contact=enable`, `multiccd=enable`。不在 robot
include 中重复声明冲突 option。

## 资产与 provenance

把 Menagerie 的 32 个 palm OBJ、Apache `LICENSE` 和 `SOURCE.md` 复制到
`src/unilab/assets/robots/kuka_sharpa/assets/menagerie_sharpa_wave/`。旧 Sharpa collision
STL 若不再被 XML 引用则删除；保留 visual STL 和仍被引用的 pinky MC mesh。更新
`ASSET_PROVENANCE`、mesh inventory 和每个文件 SHA-256。所有正式 mesh 都是普通文件、相对路径，
不使用 symlink。

## 验证

contract tests 将覆盖：robot/scene/tool XML 编译和 missing mesh、hand body/joint/actuator
顺序及 29 action、Sharpa collision 接触参数、compiled model option、50 excludes、visual
非碰撞、robot XML 无 scene/task 元素、reset/step finite、600-tool pool materialization。
现有 reward、reset、goal、domain randomization、object pool、table 尺寸与 friction、tool
生成逻辑均不改；中文文档明确记录这些 non-goals 以及 floor/table/tool friction 未变。
