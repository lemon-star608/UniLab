# DexToolBench MuJoCo Viser 在线验证

`scripts/play_dexbench_mujoco_viser.py` 是一个长期入口，用于在网页 Viser 中验证
SimToolReal 的 native RL-Games SAPG 权重。它复用原仓库 DexToolBench 的六类工具、十二个
具体工具、任务表面和 trajectory，并保留原 viewer 的三级选择器以及
`Load Environment`、`Run Episode`、`Pause/Resume`、`Stop Episode` 生命周期。

与原仓库的差别只有两项：环境由 UniLab 注册的 MuJoCo 后端运行，机器人使用 UniLab 正式
Sharpa XML（包括 Menagerie collision geometry 和 `reference + tip_stiff` 接触参数）。DexBench
二进制资产托管在公开 HF 数据集，首次加载任务时自动下载并缓存；不会写入正式机器人 XML，
也不会生成视频。

## 启动

先安装这三个可选依赖（`uv sync` 需要一次性声明，否则会卸载未声明的 extra）：

```bash
uv sync --extra mujoco --extra viser --extra rlgames-sapg
```

使用训练权重启动：

```bash
uv run scripts/play_dexbench_mujoco_viser.py \
  task=simtoolreal/mujoco_12k \
  algo.load_run=0_2026-08-27_22-38-06_mujoco \
  algo.checkpoint=nn/0_simtoolreal_sapg.pth \
  dexbench.port=8083
```

浏览器打开 `http://localhost:8083`，选择 category、object 和 task 后点击 Load。运行时只读
HF 缓存中的 manifest，不依赖 sibling checkout。若要在有网络环境中提前准备离线缓存，可执行：

```bash
huggingface-cli download unilabsim/unilab-robots \
  --repo-type dataset \
  --include 'robots/kuka_sharpa/**' 'dexbench/**' \
  --local-dir src/unilab/assets
```

若要从原仓库重新生成 manifest，使用一次性的离线命令：

默认 viewer 以 `dexbench.render_hz=60`（约 60 FPS）轮询 worker；如果机器或浏览器负载较高，
可在启动命令中降低到 `dexbench.render_hz=30`。viewer 会丢弃积压的旧状态，只显示最新物理帧，
因此不会因 worker 短时领先而出现成批跳帧。

```bash
uv run scripts/import_dexbench_assets.py \
  --source-root /path/to/simtoolreal \
  --destination /tmp/unilab-dexbench \
  --common-scene src/unilab/assets/robots/kuka_sharpa/scene.xml
```

导入器会校验 6 类、12 工具、24 任务和来源哈希；生成的目录可按发布流程上传到 HF，正式
viewer 使用 `src/unilab/assets/dexbench/manifest.json`（首次运行时自动拉取）。

## 评估边界

- 正式训练时间尺度不变：`sim_dt=1/120`、`ctrl_dt=1/60`；对比脚本中的 `1/600` physics
  benchmark 不会被带入这里。
- 策略输入输出仍为 actor 140、action 29；权重通过 native SAPG bridge 加载，不重写策略网络。
- Load 时固定 DexBench trajectory 的起始姿态和目标序列，关闭 observation/action delay、观测噪声、
  wrench DR 和 object-scale noise；成功阈值为 `0.01`，每个 trajectory goal 计一次成功。
- MuJoCo 环境和 native SAPG checkpoint 在独立 `spawn` worker 中构造；Viser parent 只处理选择器、
  场景句柄和 `load/run/pause/resume/stop/reset` 命令。切换任务会先终止旧 worker。
- reward、reset/task 逻辑、工具池训练配置、episode 时间尺度和 object pool 不被永久修改。
- 本入口只改变选定评估环境的外部工具/table 资产，不修改正式训练的 floor/table/tool friction；
  外部 DexBench object/table 继续使用 cold-path 资产适配器的显式摩擦值。

## 唯一 owner 与测试

- DexBench 路径、URDF、trajectory 和 xyzw→wxyz 转换：
  `src/unilab/envs/manipulation/simtoolreal/dexbench_assets.py`。
- 单工具注入和 object scale：`SimToolRealEnv._prepare_tool_pool()`。
- Viser UI、策略 step、MuJoCo state 更新：`scripts/play_dexbench_mujoco_viser.py`。
- 通用 MuJoCo geom 更新：`src/unilab/visualization/viser_scene.py`。

定向 contract：

```bash
uv run pytest tests/envs/manipulation/simtoolreal/test_dexbench_assets.py \
  tests/scripts/test_dexbench_viser.py \
  tests/visualization/test_dexbench_mujoco_playback.py \
  tests/algos/rlgames_sapg/test_checkpoint_normalize.py -q
```

它覆盖任务目录和 12 个对象 catalog、临时 scene 编译/29 actuator、评估 override、以及
Load 前不能 Run 的状态机。正式资产和工具池回归仍使用：

```bash
uv run pytest tests/envs/manipulation/simtoolreal/test_assets.py -q
uv run pytest tests/envs/manipulation/simtoolreal/test_tool_assets.py -q
uv run pytest tests/envs/manipulation/simtoolreal/test_env_integration.py -q
```
