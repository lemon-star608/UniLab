# SimToolReal → UniLab 迁移审计报告

## 审计范围

- **审计日期**：2026-07-30
- **审计员**：独立审计员（Claude Code, 接管自前任指挥官）
- **源码版本**：SimToolReal（`~/code/simtoolreal`，非 git repo，基于 IsaacLab/IsaacSim）
- **UniLab 版本**：`feat/simtoolreal-t0-env-skeleton` @ `0cbd1ca7` ("T7: Integrate all modules into apply_action/update_state")
- **审计方法**：独立源码对照（不依赖前任文档），逐文件、逐行交叉核对

---

## 已知范围外（不作为 bug）

以下 5 项差异系有意阉割，为 MVP 已知限制，不作为 bug 报告：

1. **物体池简化**：源码 1200 个随机工具（12 种 × 100 变体），UniLab 当前只加载 1 个固定锤子（`scene.xml:52`）。`config.py` 声明了 6 种类型但 scene 构建未实现。标记为 **DEFERRED - Phase-2.5 扩展物体池**。

2. **物理参数 DR 未实现**：源码 `simtoolreal_env_cfg.py:453-458` 明确说明 gravity/DOF damping/stiffness/mass/friction 等物理参数 DR 不在 v1 范围。UniLab `dr_provider.py` 同样声明 `randomization=None`。标记为 **NOT IN SCOPE（源码 v1 也没有）**。

3. **桌面高度随机化缺失**：源码 `reset_utils.py` 有 `table_reset_z_range` 随机化桌面高度，UniLab 因 MuJoCo backend 白名单不暴露 `set_body_pose` 而无法 per-env 随机化（T0 已记录）。标记为 **KNOWN LIMITATION（架构约束）**。

4. **深度图像观测未实现**：源码 `simtoolreal_env_cfg.py:295-380` 有完整5阶段深度图像噪声 pipeline + 相机配置，UniLab 完全未实现。标记为 **DEFERRED - Phase-3 视觉观测扩展**。

5. **Critic 观测特权物理参数缺失**：源码 critic 观测不含物体真实质量/摩擦等特权参数（obs_utils.py 中的 state_list 与 UniLab 完全一致）。经独立核对，**无差异**，此条目已排除。

---

## 审计发现

### P0 问题（立即修复）

**无。**

### P1 问题（训练前修复）

**无。**

### P2 问题（记录后续处理）

**无。** 所有可疑差异均经确认为设计决策（`GoalCfg` 字段重组）或范围外阉割，未发现任何数值或逻辑偏差。

### 已验证通过的模块

| 模块 | 源码文件 | UniLab 文件 | 审计结论 |
|------|---------|------------|---------|
| 配置 schema | `simtoolreal_env_cfg.py` | `config.py` | ✅ 字段值完全一致；`keypoint_scale` / `success_steps` / `success_tolerance` 经设计重组到 `GoalCfg`，值不变 |
| 动作管线 | `action_utils.py:18-75` | `action_pipeline.py` | ✅ canonical→backend 排列、延迟队列、arm 速度增量（双 clamp）、hand 绝对映射（单 clamp）、EMA 顺序与公式逐行匹配 |
| 延迟队列 | `obs_utils.py:119-133` | `delay_buffer.py` | ✅ flush-before-roll 逻辑完全一致 |
| 观测装配 | `obs_utils.py:231-349` | `observations.py` | ✅ actor(140 dim) / critic(162 dim) 字段列表、wxyz→xyzw 转换、clamp 均一致 |
| 奖励 | `reward_utils.py` | `rewards.py` | ✅ 7 项奖励、无全局缩放、d* in-place 更新、latch 逻辑完全匹配 |
| Episode 生命周期 | `termination_utils.py` + `obs_utils.py:122-202` | `episode_lifecycle.py` | ✅ 成功判定（×keypoint_scale 阈值）、goal 推进、steps 清零、容差课程均正确 |
| Keypoints | `obs_utils.py:28-33` + `reset_utils.py:74-83` | `keypoints.py` | ✅ 固定偏移（reward/success 路径）与 phi 缩放（obs 路径）双路正确 |
| 物理参数 | `scene_utils.py:59-120` | `constants.py` | ✅ 全部 44 个数值（ARM 7×刚度/阻尼，HAND 22×刚度/阻尼/armature/friction）逐值完全一致 |
| wrench DR | `action_utils.py:77-130` | `dr_wrench.py` | ✅ 衰减、Bernoulli 触发、质量缩放、lift gate、backend 调用全部正确 |
| backend wrench | Isaac Lab `set_external_force_and_torque` | `mujoco/backend.py:1030-1063` | ✅ `apply_body_wrench` 将 force 写入 `_pending_xfrc_applied[0:3]`，torque 写入 `[3:6]`，world frame |

---

## 前任声称修复的问题验证结果

### 1. d* sentinel 解析
**✅ 已正确实现**

源码 `obs_utils.py:183-190` 中每步检查 `_closest_keypoint_max_dist < 0`，若是则填入当前距离。

UniLab `env.py:_compute_intermediate_values`（第 542-545 行）使用：
```python
kp_star = info["closest_keypoint_max_dist"]
np.copyto(kp_star, self._keypoints_max_dist, where=kp_star < 0.0)
ft_star = info["closest_fingertip_dist"]
np.copyto(ft_star, self._curr_fingertip_distances, where=ft_star < 0.0)
```

E2E 验证：step 0 的 `closest_keypoint_max_dist` = `[0.338, 0.415, 0.341, 0.432]`，确认为真实距离，**非 -1**。

### 2. 奖励配置路径

**✅ 已正确实现**

`rewards.py:282-283` 注释明确说明："success_steps lives on GoalCfg, not TerminationCfg — 源码保留在 TerminationCfg，但 contract 将目标侧字段重组到 GoalCfg (§5.0)。" 代码正确读取 `goal_cfg.success_steps`（值 = 10，与源码一致）。

`keypoint_scale` 同理：源码在 `RewardCfg`，UniLab 在 `GoalCfg`，值均为 1.5。

### 3. env._near_goal / env._is_success 发布

**✅ 已正确实现**

`episode_lifecycle.py:134-135`：
```python
env._near_goal = near_goal
env._is_success = is_success
```

在 `update_state` 流程中，`compute_success` 在 `compute_rewards` 之前调用，保证 `env._near_goal`/`env._is_success` 在奖励计算时已经可用。

### 4. backend.apply_body_wrench 签名（D6）

**✅ 已正确实现**

`base/backend/base.py:347-368`：`apply_body_wrench(body_ids, force, torque)` 已声明于 `SimBackend` 抽象接口。

`mujoco/backend.py:1060-1063`：
```python
for body_offset, body_id in enumerate(body_ids_np):
    start = 6 * int(body_id)
    self._pending_xfrc_applied[:, start : start + 3] += force_np[:, body_offset, :]     # force
    self._pending_xfrc_applied[:, start + 3 : start + 6] += torque_np[:, body_offset, :]  # torque
```

MuJoCo `xfrc_applied` 约定：`[0:3]` = 力，`[3:6]` = 力矩，world frame。**正确**。

### 5. 奖励归一化

**✅ 已正确实现**

`rewards.py:18-19` 注释："No global scaling. Line :141 returns the direct sum." 与源码 `reward_utils.py:141` 一致。

`obs_utils.py:326` 的 `×0.01` 是 critic 观测中的特征归一化（`reward_feat = info["reward"] * 0.01`），**不是**奖励信号缩放。UniLab `observations.py:468` 同样处理。

### 6. 物理参数

**✅ 全部 44 个数值完全一致**

逐值核对 `scene_utils.py:59-120` vs `constants.py:95-209`：

- ARM 刚度 (7)：`[600, 600, 500, 400, 200, 200, 200]` ✅
- ARM 阻尼 (7)：`[27.027026473513512, 27.027026473513512, 24.672186769721083, 22.067474708266914, 9.752538131173853, 9.147747263670984, 9.147747263670984]` ✅
- HAND 刚度 (22)：所有值完全一致（含 `left_5_pinky_CMC: 1.38`，最易出错项）✅
- HAND 阻尼 (22)：所有浮点值完全一致（含 `left_3_middle_MCP_FE: 0.2085923` 末位截断）✅
- HAND Armature (22)：`left_5_pinky_CMC: 0.00012`（最小值，易漏）✅
- HAND Friction (22)：`left_index_DIP: 0.00378738`（高精度值，逐位一致）✅

---

## 端到端验证结果

```
=== E2E Rollout (100 steps, num_envs=4) ===
100 步无崩溃、无 NaN   ✅
actor obs shape  (4, 140)     ✅
critic obs shape (4, 162)     ✅
Step 0 closest_keypoint_max_dist = [0.338, 0.415, 0.341, 0.432]  ✅（非 -1）
奖励不全为 0（step 0: mean=0.944）  ✅
obs 范围在 [-3.12, 6.16]（在 [-10, 10] clamp 内）  ✅

=== 单元测试（136 tests）===
136 passed in 2.53s  ✅
（含 test_simtoolreal.py, test_simtoolreal_keypoints.py,
 test_action_pipeline.py, test_rewards.py, test_observations.py,
 test_episode_lifecycle.py, test_goal_sampling.py, test_integration.py）
```

---

## 总体结论

**✅ 可以进入 T8 训练**

全部 P0/P1 风险点经独立源码对照和 E2E 运行验证，**无任何问题**：

- 所有前任声称修复的 6 项问题已确认修复正确
- 物理参数、观测维度、奖励公式与源码逐行匹配
- 136 个单元测试全部通过
- 100 步端到端 rollout 无崩溃、无 NaN、量级合理

本次迁移（T0-T7）在 MVP 范围内完整且正确。剩余工作（物体池扩展、视觉观测）属于已知 Deferred 范围，不影响 T8 训练启动。

---

*审计完成时间：2026-07-30*  
*审计员：独立审计员（不与前任交互，只对照源码）*
