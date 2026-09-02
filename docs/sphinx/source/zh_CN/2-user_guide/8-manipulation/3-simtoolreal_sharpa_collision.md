# SimToolReal Sharpa 碰撞 profile

正式 SimToolReal MuJoCo 场景使用 Menagerie Sharpa collision geometry：palm 使用
palm000.obj 到
palm031.obj，普通 MCP/PP/MP 指节使用 Menagerie cylinder/capsule fit，指尖使用
capsule collision。Menagerie 资产位于
HF 数据集 `unilabsim/unilab-robots` 的
`robots/kuka_sharpa/meshes/menagerie_sharpa_wave/`，并随附 Apache LICENSE 与 SOURCE.md。

## 唯一 owner

- kuka_sharpa.xml 是 Sharpa collision geometry 和接触参数的唯一 owner；它仍然只描述
  机器人。body、joint、inertial、actuator、visual pose、KUKA arm collision geometry 和
  50 个 contact exclude pair 以原 UniLab XML 为 canonical source。
- scene.xml 通过 include file="kuka_sharpa.xml" 加载机器人，并拥有 floor、table、
  keyframe 和全局 MuJoCo option。工具不固定写入 robot XML。
- tool_assets.py 仍在 cold path materialize 动态工具场景；六类工具各 50 个、pool seed、
  shuffle 和 object pool materialization 均未改变。
- `ASSET_NOTICES.md` 指向 HF 上的完整许可证和 `ASSET_PROVENANCE`。正式 XML 只使用相对
  路径，资产下载到本地后不是 symlink。

## 接触 profile

本轮统一采用 reference 接触刚度：

| Sharpa collision role | condim | friction | solimp | solref |
| --- | ---: | --- | --- | --- |
| palm | 3 | 1.0 0.005 0.0001 | 0.9 0.95 0.001 0.5 2.0 | 0.02 1.0 |
| regular finger segment | 3 | 1.0 0.005 0.0001 | 0.9 0.95 0.001 0.5 2.0 | 0.02 1.0 |
| fingertip collision | 3 | 1.0 0.005 0.0001 | 0.9 0.95 0.001 0.5 2.0 | 0.02 1.0 |

所有 Sharpa collision geom 显式使用 contype=1、conaffinity=1、group=3、density=0、
margin=0、gap=0；visual geom 使用 contype=0、conaffinity=0、group=2。当前 profile 不
使用 tip_soft=[0.10, 0.8]。elastomer 继续是 visual-only。

工具、floor 和 table 的 friction 统一为 `1.0 0.005 0.0001`；KUKA arm
collision 继续使用 `0.5 0.005 0.0001`。

全局 compiled MjModel.opt 由 scene.xml 唯一声明：

```text
timestep=0.008333333333333333  (1/120)
integrator=implicitfast
solver=Newton
cone=elliptic
impratio=10
iterations=100
ls_iterations=50
contact=enable
multiccd=enable
```

正式 UniLab 训练仍使用 sim_dt=1/120、ctrl_dt=1/60（两个 physics substeps 一个
control step）。simtoolreal 对比脚本中的 1/600 physics benchmark 是实验设置，不是
正式训练 owner 的 timestep。

scene.xml 是可独立编译的机器人模板，因此 home keyframe 只声明 29 个 actuator 的 ctrl；
动态工具 materialization 后 MuJoCo 会按 materialized freejoint 产生完整 qpos。SimToolReal
reset owner 仍显式写入机器人与 object qpos，训练行为和 home joint targets 不变。

## 未修改范围

该 profile 不修改 reward、reset、goal sampling、domain randomization、episode length、
object_name=handle_head_primitives、策略 observation/action layout、29 action 维度、
工具池生成逻辑、table 尺寸/位置或 object pool seed/shuffle。KUKA arm collision 的
friction 仍为 0.5 0.005 0.0001；工具、floor 和 table 的 friction 已统一调整为
1.0 0.005 0.0001，并同步更新 AssetsCfg 与 tool_assets.py owner。

## Contract tests

以下测试是本 profile 的维护入口：

- tests/envs/manipulation/simtoolreal/test_assets.py：HF resolver、robot XML 纯度、29
  joint/actuator 顺序、50 excludes、Sharpa 接触属性和 visual bits。
- tests/envs/manipulation/simtoolreal/test_tool_assets.py：多种工具 topology 的
  materialized XML 编译、全局 option、finite reset/step。
- tests/envs/manipulation/simtoolreal/test_env_integration.py：29 action、600-tool pool、
  compiled collision mesh inventory、reset/step 后 finite state。

运行方式统一使用 uv run pytest。
