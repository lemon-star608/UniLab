# SimToolReal MuJoCo Viser 验证设计

## 1. 目标与范围

本设计为 UniLab 增加一条基于 MuJoCo 后端的 SimToolReal 交互式 Viser 验证链路。第一版需要和原 SimToolReal 仓库的 Isaac Sim Viser 保持工具身份、轨迹身份和页面选择语义一致：

- 6 个工具类别；
- 12 个真实工具实例；
- 24 个 DexToolBench 轨迹任务；
- 每次启动只加载一个 checkpoint；
- 页面选择 `category/object/task`，不在运行中切换策略网络；
- worker 使用 UniLab 已有 MuJoCo env/backend contract；
- 评估采用原交互验证的确定性设置。

本设计只覆盖资产导入、MJCF 物化、MuJoCo worker、Viser 状态流和验证门槛。不新增训练 runner、不改变 checkpoint 格式、不支持 Isaac Sim 后端、不在第一版实现页面内的多权重对比。

## 2. 已确认的物理语义

真实工具的碰撞几何和训练 primitive 的碰撞几何不同，不能把两者称作几何完全一致。这里的“一致”定义为：

- 机器人碰撞几何和物理结构完全沿用 UniLab 训练时的 KUKA+Sharpa MJCF；
- 真实工具视觉几何使用原仓库视觉 OBJ；
- 真实工具动态碰撞几何使用原仓库离线生成的 CoACD 凸分解片；
- 工具质量、COM、惯量使用真实工具的权威元数据，并采用 UniLab 的单 body 显式 inertial 处理规则；
- 摩擦、接触参数、控制 dt、substeps、solver 和碰撞过滤规则由 UniLab MuJoCo owner 配置负责；
- 任务环境使用原仓库对应任务的视觉/碰撞几何，但套用 UniLab 的物理参数 profile。

因此，MuJoCo Viser 展示的是原仓库真实工具形状，运行的是 UniLab 训练参数下的 MuJoCo 物理环境，不声称复现 PhysX 的数值轨迹。

## 3. 资产归属与数据流

运行时不依赖 `/home/user/ws/lemon/simtoolreal` 或任何 sibling checkout。离线导入阶段从原仓库复制必要资产到 UniLab 版本库，并保存来源哈希和导入版本。

数据流如下：

```text
原仓库 DexToolBench 资产、任务 URDF、轨迹 JSON
                    |
                    v
         离线导入、路径校验、SHA-256 哈希
                    |
                    v
        UniLab DexToolBench manifest（稳定 ID）
                    |
                    v
       离线 URDF/Mesh -> MJCF materializer
                    |
                    v
      24 个自包含 category/object/task 场景
                    |
                    v
      Viser parent <-> MuJoCo worker subprocess
                    |
                    v
          deterministic policy rollout + 状态流
```

manifest 是 MuJoCo Viser 和原仓库选择语义的唯一目录。UI 只传稳定 ID，不接受任意路径。

### 3.1 manifest 最小字段

每个工具记录：

- `category`、`object`；
- 视觉 mesh 相对路径；
- 凸分解 collision mesh 相对路径列表；
- policy 使用的 `object_scale`；
- `mass`、`com`、完整惯量；
- 原始 URDF 和每个 mesh 的来源哈希。

每个任务记录：

- `category/object/task` 稳定 ID；
- 任务环境 visual/collision mesh 列表；
- 原始 table/task URDF 来源哈希；
- `start_pose`（原始 xyzw）；
- `goals[]`（原始 xyzw）；
- 物化 MJCF 路径和生成器版本/输入哈希。

导入命令必须校验 6 类、12 工具、24 任务、24 条轨迹全部存在，检查引用文件和轨迹格式，并拒绝缺失或重复 ID。

## 4. MJCF 物化结构

每个任务生成一个自包含的 `scene_<category>_<object>_<task>.xml`，其中 mesh 路径只指向 UniLab 资产包。

```text
scene_<category>_<object>_<task>.xml
├── include: UniLab KUKA + Sharpa robot
├── include: task environment fragment
├── asset: tool visual/collision meshes and task meshes
├── worldbody
│   ├── robot
│   ├── fixed table/task props
│   ├── object (one freejoint, one inertial)
│   └── goal visualization (visual-only copy)
└── keyframe (task-level)
```

### 4.1 机器人

机器人 XML 不改写。必须保留训练时的关节顺序、关节限制、执行器增益、armature、joint friction、碰撞过滤、qpos/qvel 布局和控制接口。新增场景只能 include 现有机器人资源。

### 4.2 工具

真实工具的所有视觉 mesh geom 设置为 `contype="0" conaffinity="0" density="0"`，不参与物理接触且不贡献质量。

每个 CoACD 凸分解片成为一个 collision mesh geom，挂在同一个 `object` body 下。collision geoms 使用 UniLab 物理 profile 的 contype/conaffinity/friction/contact 参数。

工具 body 只拥有一个 freejoint 和一个显式 `<inertial>`。视觉与碰撞共享同一 body 坐标和缩放；不得根据视觉网格再次自动推导质量。

### 4.3 任务环境和 goal

每个任务保留原任务 URDF 中的视觉和碰撞物件。视觉 geoms 不参与碰撞，碰撞 geoms 使用固定 body 或 task fragment 中声明的静态 body。goal 是同一视觉资产的无碰撞副本，不创建第二个动态物体。

`<keyframe>` 必须位于 task-level scene XML，不能写入 `robot.xml`。

## 5. 参数 owner 与训练配置

```text
机器人结构和碰撞过滤       -> UniLab robot MJCF
真实工具 visual/collision   -> 导入的 DexToolBench 资产
工具质量/COM/惯量           -> manifest + 单 body 显式 inertial
摩擦与接触参数              -> UniLab MuJoCo owner 配置/materializer
控制 dt/substeps/solver     -> UniLab MuJoCo owner 配置
任务 start pose/goals       -> manifest（原仓库轨迹）
```

物化器在冷路径把参数写入 XML；Viser worker 不在启动后修改 geom 属性，也不在 step/reset 中解析 XML 或读取资产元数据。默认采用当前 UniLab 训练 profile，包括训练时 robot/object/table friction 及 scene option。若 profile 发生变化，必须重新物化并更新输入哈希，不能通过运行时 override 静默改变已生成资产。

## 6. 轨迹和评估契约

worker 必须使用与原交互验证一致的固定设置：

- 读取原始 `trajectory.json` 的全部 goals，不下采样；
- `start_pose.z += 0.03`；
- 固定初始物体姿态；
- `startArmHigher=True`；
- `table_reset_z=0.38`；
- 关闭 observation/action/object-state delay、观测噪声、外力和力矩；
- 成功阈值 `0.01`，`success_steps=1`；
- 轨迹文件使用 xyzw，MuJoCo 内部使用 wxyz；
- 首次策略动作前执行与原 evaluator 一致的零动作 physics tick。

Viser 状态消息至少包含机器人关节、工具 pose、goal pose、当前 goal 索引、总 goal 数、step 和 episode 结果。状态坐标采用 env-local 坐标，页面显示的工具和 goal 使用同一 visual asset。

## 7. Viser 与 worker 边界

parent 进程负责 Viser UI、场景节点和用户命令；MuJoCo worker 为独立子进程，负责构造 env、加载 checkpoint 和 rollout。建议命令为：

```text
uv run <viser-entry> --config <policy-config> --checkpoint <normalized-checkpoint>
```

启动参数只固定一份 config/checkpoint。页面操作包括加载环境、运行、暂停、继续、停止和重置；切换 `category/object/task` 时终止旧 worker，再按 manifest 生成/加载对应场景。

worker 不复制训练脚本中的奖励或观测逻辑，而是调用 UniLab 注册的 SimToolReal env。跨后端字段校验和 checkpoint 维度 guard 在创建 env、加载策略前执行。

## 8. checkpoint NumPy 兼容

当前 checkpoint 可能由 NumPy 2.x 保存，而 Isaac Sim/其他运行环境使用 NumPy 1.26。长期方案是冷路径 normalize/export：

```text
原始 model.pth -> normalize/export -> Viser checkpoint
```

规范化只转换 checkpoint 中的 NumPy 标量/数组元数据，不改变模型 state dict、网络结构和张量值。Viser 启动时只做一次预检；发现不兼容时报告明确原因和 normalize 命令，禁止 runtime monkey patch、隐式重试或修改原始 checkpoint。

## 9. 错误处理

以下错误必须在 cold path 或启动前显式失败，并包含稳定 ID 和修复提示：

- manifest 缺少工具、任务、轨迹或引用文件；
- 来源/物化哈希不匹配；
- XML 无法编译、mesh 路径失效或 body/freejoint/inertial 数量错误；
- robot qpos/qvel/nu 或 actor observation 维度与 checkpoint 不兼容；
- checkpoint NumPy 序列化版本不兼容；
- worker 启动、加载或 rollout 异常。

worker 异常通过结构化 error 消息返回 parent，页面显示失败原因并允许重新加载；不得留下仍在运行的孤儿 worker。

## 10. 验证与验收

提交前测试应靠近风险边界：

1. manifest 审计：数量、ID 唯一性、路径、来源哈希和轨迹格式；
2. MJCF 物化测试：24 个 XML 全部可被 MuJoCo 编译；
3. 结构测试：机器人布局与训练 contract 一致，工具一个 freejoint/一个 inertial，visual/collision geom 标志正确；
4. 参数测试：friction/contact profile 与 UniLab owner 配置一致；
5. checkpoint 测试：normalize 后可加载，actor 输入维度匹配；
6. worker smoke test：至少一个任务完成 reset、确定性 rollout 和状态消息流；
7. Viser 生命周期测试：load/stop/reload 不产生孤儿进程；
8. 验收运行：24 个任务各执行确定性 rollout，记录完成率、耗时和失败原因。

实现应控制为一个主要结果、一个实现 commit，避免顺手扩展到多 checkpoint 对比、Isaac Sim worker 或训练流程改造。

