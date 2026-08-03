# SimToolReal → UniLab 迁移交接文档

**日期**：2025-01-30  
**状态**：Env 层迁移完成（T0-T7），训练配置待启动（T8-T12）  
**当前分支**：`feat/simtoolreal-t0-env-skeleton`  
**当前 commit**：`0cbd1ca7` ("T7: Integrate all modules into apply_action/update_state")

---

## 项目背景

SimToolReal 是 KUKAiiwa14 + Sharpa HA4 手的工具操纵任务（6-DOF 位姿达到）。原始仓库基于 IsaacLab/IsaacSim，使用 SAPG 算法训练。本次迁移目标是将其移植到 UniLab 的 MuJoCo backend + rsl_rl 训练框架。

**关键数字**：
- Actor 观测：140 维（本体感知 + 物体位姿 keypoints）
- Critic 状态：162 维（actor + 22 维特权信息）
- 动作：29 维（7 arm + 22 hand）
- 训练规模：24576 envs（源码），SAPG 要求 `num_envs % 4096 == 0`

---

## 已完成工作（T0-T7）

### Env 层完整迁移

| 模块 | 文件 | 状态 |
|------|------|------|
| 配置 schema | `config.py` | ✅ 44 个物理参数逐值验证一致 |
| 动作管线 | `action_pipeline.py` | ✅ 延迟队列、双clamp、EMA 顺序完全匹配 |
| 观测装配 | `observations.py` | ✅ 140/162 维字段列表、wxyz→xyzw、clamp 一致 |
| 奖励计算 | `rewards.py` | ✅ 7 项奖励、d* 解析、latch 逻辑完全匹配 |
| Episode 生命周期 | `episode_lifecycle.py` | ✅ 成功判定、goal 推进、容差课程正确 |
| Keypoints | `keypoints.py` | ✅ 固定偏移（reward）与 phi 缩放（obs）双路正确 |
| Wrench DR | `dr_wrench.py` | ✅ 衰减、Bernoulli、质量缩放、lift gate 正确 |
| MuJoCo D6 扩展 | `backend/mujoco/backend.py` | ✅ `apply_body_wrench` 写入 `[0:3]` 力、`[3:6]` 力矩 |
| Goal 采样 | `goal_sampling.py` | ✅ from_trajectory + epsilon-ball 正确 |
| 场景构建 | `scene.py` | ✅ MJCF 编译、keyframe、排列映射正确 |

**测试覆盖**：
- 136/136 单元测试通过（`tests/envs/manipulation/simtoolreal/` + `tests/envs/test_simtoolreal*.py`）
- 100 步 E2E rollout 无 NaN、量级正常
- d* sentinel 在 step 0 正确解析为真实距离（非 -1）

**审计结论**：无 P0/P1 问题，env 层与源码逐行匹配。详见 `AUDIT_REPORT.md`。

---

## 待完成工作（T8-T12）

### 阻塞训练的（立刻需要）

**T8：训练配置 + 首次跑通**
- 创建 `conf/ppo/task/simtoolreal/mujoco.yaml`
- 非对称 Actor-Critic obs 路由验证（actor 读 `obs` 140 维，critic 读 `critic` 162 维）
- 奖励缩放 `scale_value: 0.01`（源码 YAML 有，UniLab 需找等价配置方式）
- 网络规模：MLP `[1024, 1024, 512, 512]`（actor + critic 均用此规模）
- 核心超参：`horizon_length=16`, `mini_epochs=2`, `lr=1e-4`, `gamma=0.99`, `tau=0.95`, `entropy_coef=0.0`, `clip_param=0.1`, `value_loss_coef=4.0`
- 验收标准：100 iter reward 有上升趋势，无 NaN/inf

### 算法保真度（训练后优化）

**T9：LSTM Actor**
- 源码：`LSTM(units=1024, before_mlp=True) + seq_length=16`
- rsl_rl 默认 MLP，需要添加 RNN 支持
- **可以先用 MLP 跑通 T8，LSTM 是后续优化**

**T10：SAPG 算法**
- 源码关键差异：`use_others_experience=lf`, `expl_type=mixed_expl_learn_param`, `expl_coef_block_size=4096`
- rsl_rl 无此机制，需要算法层扩展
- **可以先用标准 PPO 验证训练流程，SAPG 是样本效率优化**

### Env 完整性（可并行）

**T11：1200 随机工具**
- 当前：1 个固定锤子（`scene.py:52`）
- 需要：12 类 × 100 变体的程序化 MJCF 生成 + per-reset 切换
- 源码参考：`scene_utils.py:generate_objects()`（注意：源码是 IsaacSim USD，需重写为 MuJoCo MJCF）

### 部署闭环（训练完成后）

**T12：部署适配**
- rsl_rl checkpoint → 实机 ROS 节点适配
- FoundationPose 集成（外部 repo：https://github.com/kushal2000/FoundationPose）
- 位姿估计器 → `/robot_frame/current_object_pose` topic → `rl_policy_node.py`

---

## 已知限制（无法简单修复）

1. **桌面高度随机化缺失**：MuJoCo backend 白名单不暴露 `set_body_pose`，无法 per-env 随机化桌面高度。源码有此 DR，UniLab 因架构约束无法实现。

2. **物理参数 DR 未实现**：源码 v1 本身就不包含 gravity/DOF damping/mass 等物理参数 DR（`simtoolreal_env_cfg.py:453-458` 明确说明），UniLab 同样未实现，两边一致。

3. **深度图像观测未实现**：源码有完整的 5 阶段深度图噪声 pipeline（`StudentObsCfg`），但训练主线路不使用，仅用于实验性的蒸馏扩展（未完成）。UniLab 当前不实现深度图，不影响训练和实机部署（实机部署依赖外部 FoundationPose 提供物体位姿，不需要 student policy）。

---

## 源码真相索引（权威参考）

**训练配置**：
```
~/code/simtoolreal/simtoolreal/isaacsimenvs/cfg/train/SimToolRealSAPG.yaml
```
关键字段：`reward_shaper.scale_value: 0.01`, `minibatch_size: 98304`, `horizon_length: 16`, `mini_epochs: 2`, `learning_rate: 1e-4`, MLP `[1024,1024,512,512]`, LSTM `units: 1024`, `entropy_coef: 0.0`, `e_clip: 0.1`, `critic_coef: 4.0`。

**Env 配置**：
```
~/code/simtoolreal/simtoolreal/isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py
```
物理参数、奖励权重、episode 长度、DR 参数全在此文件。

**观测/动作/奖励实现**：
```
~/code/simtoolreal/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py
~/code/simtoolreal/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/action_utils.py
~/code/simtoolreal/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/reward_utils.py
```

**部署代码**：
```
~/code/simtoolreal/simtoolreal/deployment/rl_policy_node.py
~/code/simtoolreal/simtoolreal/deployment/rl_player.py
```
显示实机部署使用同一个 140 维 obs 的训练策略 + FoundationPose 位姿估计器。

---

## UniLab 对应文件

**Env 层**：
```
src/unilab/envs/manipulation/simtoolreal/
  ├── env.py                    # 主环境类，apply_action / update_state / obs_groups_spec
  ├── config.py                 # SimToolRealCfg（全部配置 schema）
  ├── action_pipeline.py        # 动作延迟、EMA、clamp
  ├── observations.py           # build_observations（140/162 维装配）
  ├── rewards.py                # compute_rewards（7 项奖励）
  ├── episode_lifecycle.py      # compute_success / update_goals
  ├── keypoints.py              # compute_keypoints / keypoints_max_dist
  ├── goal_sampling.py          # sample_goals_from_trajectory
  ├── dr_wrench.py              # apply_domain_randomization_wrench
  ├── delay_buffer.py           # DelayBuffer（action/obs 延迟队列）
  ├── scene.py                  # build_scene（MJCF 构建）
  └── constants.py              # 物理参数常量（44 个值）
```

**测试**：
```
tests/envs/manipulation/simtoolreal/
  ├── test_action_pipeline.py
  ├── test_observations.py
  ├── test_rewards.py
  ├── test_episode_lifecycle.py
  ├── test_goal_sampling.py
  └── test_integration.py
tests/envs/
  ├── test_simtoolreal.py
  └── test_simtoolreal_keypoints.py
```

**训练侧（待创建）**：
```
conf/ppo/task/simtoolreal/mujoco.yaml    # ← T8 需要创建
scripts/train_rsl_rl.py                   # 训练入口（已存在）
```

---

## 执行原则（第一纪律）

1. **永远对照源码，不信任文档**。任何不确定的地方立刻停下，查源码或上报。
2. **遇到无法解决的问题立刻停止**，提交详细报告（错误信息 + 尝试了什么 + 推测原因），不要继续推进。
3. **每次只完成一个任务**（T8 → T9 → T10），验收通过后再进行下一个。
4. **禁止修改 env 层代码**（除非确认是 env 层 bug 且上报）。训练侧问题在训练侧解决。

---

## T8 Session 提示词（立刻执行）

已在之前的消息中给出完整提示词，核心任务：
1. 理解 rsl_rl 非对称 AC obs 路由机制
2. 创建 `conf/ppo/task/simtoolreal/mujoco.yaml`
3. 验证 10 iter 启动无误
4. 运行 100 iter smoke test，确认 reward 有上升趋势

**验收标准**：
- actor obs 140 维，critic state 162 维（维度正确）
- 100 iter 内 reward 不是 NaN，有上升趋势
- value loss 和 surrogate loss 均有限（不爆炸）
- 提交完整验收报告

---

## 资源配置建议

**T8 测试阶段**：
- `num_envs=64`（10 iter smoke test）
- `num_envs=256`（100 iter 验证）
- GPU：单卡即可

**T8 正式训练（验收通过后）**：
- `num_envs=4096`（初期调试）
- `num_envs=24576`（复现论文，需要 SAPG 时改为 4096 的倍数）
- GPU：多卡或大显存单卡

---

## 交接检查清单

迁移到新机器时确认：

- [ ] UniLab 仓库 clone 完成（包含 `feat/simtoolreal-t0-env-skeleton` 分支）
- [ ] SimToolReal 源码仓库可访问（`~/code/simtoolreal`）
- [ ] Python 环境配置完成（`uv` + 依赖安装）
- [ ] 测试通过：`uv run pytest tests/envs/manipulation/simtoolreal/ -v`（应 136/136 通过）
- [ ] `AUDIT_REPORT.md` 已读（了解 env 层验证结果）
- [ ] `SIMTOOLREAL_HANDOVER.md`（本文档）已读

---

## 联系方式

指挥官：新 session 负责执行  
中间人：用户负责传话验收  
审查原则：源码真相 > 文档 > 推测

---

*交接文档版本：v1.0*  
*最后更新：2025-01-30*
