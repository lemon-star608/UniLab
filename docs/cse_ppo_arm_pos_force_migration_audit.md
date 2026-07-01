# CSE-PPO + Arm Pos-Force 迁移差异核对与失稳归因

> 工作文档（非对外 sphinx 文档）。记录把 CSE-PPO 算法 + 四足臂力位控制任务从原仓库 **UniFP**（`~/code/UniFP`，legged_gym / IsaacGym）迁移到 **UniLab** 后的实现差异，及对"收敛后在非平稳力课程下失稳"的归因。

| 项 | 值 |
|---|---|
| 日期 | 2026-06-30 |
| 分支 | `feat/arm-pos-force-cse-ppo` |
| Ground truth | UniFP `~/code/UniFP/legged_gym/`（B2 四足 + Z1 臂，17 actions） |
| 迁移目标 | UniLab `A2ArmPosForce` / `Go2ArmPosForce`（A2/Go2 + Airbot，18 actions） |
| 症状 | 相同配置多次重跑,只有部分 seed 收敛,其余在后段崩溃 |

---

## 1. 核对方法与证据等级

- **算法核心**（`ppo_cse_pf/*` vs `cse_ppo/*`）与**最关键的算法-任务接口**（estimator target 取数、力施加、obs 构造）：逐行读两边源码确认。
- **reward 函数体、domain randomization 范围、PD 增益**等机型相关项：交叉核对。
- 所有 load-bearing 数字均回溯到源码 `file:line`。

> 提示:UniLab 代码注释里多处提到 "UniFP" 作为来源参考,**只能当线索**;本文以 UniFP 实际源码为准。已发现至少一处注释与代码不符(见 §2)和一处换算错误(见 §3)。

---

## 2. 结论速览(后段失稳首要嫌疑,按可疑度)

1. **UniFP 的 base 外力被注释关闭,UniLab 全开。** 最大语义偏差,失稳头号嫌疑。
2. **力课程整体不是 UniFP 那一套**:起始偏早(iter 6000 vs 8000)、gripper 力 3–4× 偏弱、base 力凭空多出来、触发更稀疏。
3. **batch 缩小 4×(1024 vs 4096)+ a2 仍用 entropy 0.01 + std 顶到 0.7**:边缘稳定配置,力课程开启把部分 seed 推过临界点 → 解释"只有部分收敛"。

**算法的机理/公式/接口本身迁移是干净的**(见 §6),问题集中在**力课程语义**和**为小 batch 做的再调参**。

---

## 3. 算法层差异

| 维度 | UniFP(ground truth) | UniLab | 偏差? | 影响 |
|---|---|---|---|---|
| latent 是否 detach 喂 actor | **不 detach**,encoder 被策略梯度 co-train(`ppo_cse_pf/actor_critic.py:144-145`) | **不 detach**(`cse_ppo/actor_critic.py:90-93`)。⚠️ 顶部 docstring `:4` 写"(detached)"是**错误注释** | 一致(注释误导) | 无;建议删掉误导注释 |
| estimator 监督目标 | env 独立 `obs_pred` buffer(`legged_robot_b2z1_pos_force.py:405-412`)= `[lin_vel·s, ee_sphe·s, force_ee·s, force_base·s]` | `critic_obs[:,0:12]`(`cse_ppo/estimator.py:177-184`);critic history **newest-first** 把当前帧放 offset 0(`pos_force.py:1448-1452`) | **等价**(逐字段+逐缩放核对:`cse_targets` `pos_force.py:1347-1356` == UniFP obs_pred) | 无。设计正确,且对 `num_critic_history` 鲁棒 |
| estimation loss 公式 | per-group `mse(pred_g·w, tgt_g·w)` 求和,w=[.2,.2,1,1] → **有效 w² 加权**(`ppo_cse_pf/ppo.py:187-196`) | 同(`cse_ppo/estimator.py:152-158`,显式注释对齐 UniFP) | 一致 | 无 |
| 更新时机 | 每 minibatch:先 PPO step,后 estimator step(`ppo.py:167-200`) | 同(`algorithm.py:267-284`) | 一致 | 无 |
| 优化器结构 | 主 Adam(全参,自适应 LR)+ adaptation Adam(全参,固定 1e-5);encoder 被两者更新 | 主 Adam(全参含 estimator)+ estimator 自带 Adam(enc+dec,固定 1e-5)(`estimator.py:127`) | 等价 | 无 |
| action std 参数化 | 裸 `nn.Parameter`,直接当 std,**无任何钳制**(`actor_critic.py:113,146`) | 同参数化,但每步 `clamp_(min=1e-2, max=0.7)`(`algorithm.py:196`) | **偏差** | init_noise_std=1.0 首步即被砍到 0.7;0.7 上限封顶探索(见 §6 归因) |
| KL 自适应 LR 上下限 | 硬编码 floor **1e-5** / ceil 1e-2,factor 1.5(`ppo.py:138-141`) | 可配 `min/max_learning_rate`(`algorithm.py:176-181`),a2 设 floor **5e-5** / ceil 1e-2 | **偏差(floor 5×)** | 高扰动期 LR 压不到 UniFP 那么低,残留更新噪声更大 |
| estimator 梯度裁剪 | **不裁剪** | `clip_grad_norm_(., 10.0)`(`estimator.py:196`) | 偏差(小) | 10.0 很松,影响很小 |
| timeout bootstrap | `r += γ·V(s_t)·timeout`(`ppo.py:94-95`) | 若 env 提供 `time_out_bootstrap_obs` 用真实 next-obs 的 V,否则 fallback 同 UniFP(`algorithm.py:133-156`) | 偏差(更正确方向) | 轻微改变 return target;非失稳因素 |
| advantage 归一化 / GAE / surrogate / value-clip / entropy | 全局归一化、标准 clipped PPO | 全部一致(`storage.py:130-133`,`algorithm.py:244-265`) | 一致 | 无 |
| bf16 AMP | 无 | 有但**关闭**(`config.yaml:64`,注释说明开启会破坏 on-policy 精度一致性) | 一致(保持关) | 无;**勿开启** |

---

## 4. 力扰动课程差异(核心)

UniFP `step()` 确认(`legged_robot_b2z1_pos_force.py:125-130`):`_push_gripper` 启用,**`_push_robot_base` 整行注释掉**;gate=`global_steps > force_start_step·24`,`global_steps` 每控制步 +1(`:141`);reset 把 base force buffer 全清零且从不写入(`:253-257`)。**所以 UniFP 全程只有 gripper 力,base 力恒为 0**,`tracking_lin_vel_force_world` 退化为纯速度跟踪,estimator target 的 force_base 三维恒 0。

| 参数 | UniFP(实际行为) | UniLab A2(dataclass+yaml 生效值) | 偏差? | 影响 |
|---|---|---|---|---|
| **base 外力/命令力** | **从不施加**(注释关闭) | **施加 ±30/±30**(`pos_force.py:1118-1131`;A2Cfg ±60/±50 `:1765-1766` 被 `a2/mujoco.yaml:93-94` 覆盖为 ±30) | **重大偏差** | 收敛后 base 突遭物理扰动;速度 setpoint 被 `F/base_force_kd` 偏移;obs/estimator 的 force_base 由"死"变"活"。**头号失稳源** |
| gripper 力 cmd/ext | ±60/±60(`b2z1_pos_force_config.py:153-154`) | ±18/±15(`pos_force.py:325-326`,未被任何处覆盖) | **偏差(3.3–4× 偏弱)** | UniFP 唯一真实扰动被大幅削弱 |
| force_start | iter **8000**(=192000 控制步)(`config:178`×24) | iter **6000**(=144000 控制步)(`a2/mujoco.yaml:90`) | **偏差(早 2000 iter)** | yaml 注释"aligned to UniFP"是**错误换算**;策略更不成熟时受冲击 |
| 触发概率 cmd/ext | 0.8 / 0.8(`config:144,147,162,165`) | 0.4 / 0.3(`pos_force.py:341-344`) | 偏差(更稀疏) | 力 episode 更稀疏 |
| settling(峰值保持) | gripper 1.0s / base 3.0s(`config:156,176`) | gripper 0.5s / base 1.0s(`pos_force.py:339-340`) | 偏差(更短) | 峰值停留更短 |
| interval / duration | gripper [3.5,9]/[3.5,9]s,ramp [1,3]s(`config:142-146`) | gripper [5,10]/[6,12]s,ramp [0.5,1.5]s(`pos_force.py:332-337`) | 偏差(更稀、ramp 更陡) | 同上 |
| gripper_force_kp / base_force_kd | 200 / 200(`config:149,168`) | 200 / 200(`pos_force.py:347-348`) | 一致 | — |
| force_z_base_ext_scale | 0.1(未用)(`config:174`) | 0.05(`a2/mujoco.yaml:92`) | 偏差(UniFP 未用) | — |
| 速度脉冲 push max_vel | b2z1 override `_push_robots` max_vel 0.8(`legged_robot_b2z1_pos_force.py:1025-1034`;触发条件未完全追) | 0.3(`pos_force.py:409`) | 偏差(偏弱) | 次要 |

**一句话**:UniLab 的力课程在**作用部位(多了 base)、强度(gripper 砍 4×)、时机(早 2000 iter)、频率(0.4/0.3 vs 0.8)**四个维度上都不是 UniFP 那一套。UniFP 的稳定配方是"iter 8000 起,仅 gripper ±60"。

---

## 5. 配置数值差异(算法超参)

| 参数 | UniFP | UniLab A2 | 偏差? | 影响 |
|---|---|---|---|---|
| num_envs | 4096(`legged_robot_config.py`) | **1024**(`a2/mujoco.yaml:12`) | **偏差 4×** | rollout batch 小 4×,advantage/std 信号更噪 —— config 注释自指的根因 |
| learning_rate(初始) | 1e-3 | 5e-4(`a2:24`) | 偏差 | 自适应吸收,但起点不同 |
| min_learning_rate(KL floor) | 1e-5(硬编码 `ppo.py:139`) | 5e-5(`a2:28`) | **偏差 5×** | 高扰动期 LR 压不下去 |
| entropy_coef | 0.01 | **0.01**(`a2:25`,故意用 UniFP 值) | 一致**但危险** | `config.yaml:45` 默认已降到 0.003 防 std-runaway,a2 改回 0.01 靠 std 上限兜底(见 §6) |
| max_policy_std | 无 | 0.7(`config.yaml:54`) | 偏差 | 见 §3 / §6 |
| num_learning_epochs | 5 | 5 | 一致 | — |
| desired_kl / clip / γ / λ / mini_batches / value_coef / grad_norm | 0.01 / 0.2 / 0.99 / 0.95 / 4 / 1.0 / 1.0 | 全一致 | 一致 | — |
| estimator latent/enc/dec/groups/weights/lr | 64 / [512,256,128] / [128,64] / [3,3,3,3] / [.2,.2,1,1] / 1e-5 | 经 `a2:32-36` 覆盖后**全一致** | 一致 | a2 正确覆盖了 estimator 默认值(默认 latent=19/enc[256,128] 会错,但已覆盖) |
| 网络/激活/init_noise_std | [512,256,128] / elu / 1.0 | 一致 | 一致 | — |

> 机型相关差异(A2+Airbot ≠ B2+Z1)如 PD 增益、质量、num_actions(18 vs 17)、single-obs(76 vs 73)、base_height、cycle_time、DR 范围(UniLab 普遍更窄)、臂 action_scale(0.12 vs 0.25)等,**不算迁移 bug**,但意味着 UniFP 调好的算法常数从未在新机型动力学上重新验证。

---

## 6. 后段失稳归因

把上面几条串成因果链(与 `config.yaml:41-44` 作者自指诊断一致):

1. **batch 1024(非 4096)** → advantage 对 std 的"精度下拉"减弱;
2. **a2 保留 entropy 0.01** → 持续把 std 上推;小 batch 下上推赢过下拉,std 趋向上限;
3. **std 顶到 0.7** → 动作噪声长期偏高,策略处于边缘稳定;
4. **iter 6000 力课程开启**(早于 UniFP 的 8000)→ 引入**非平稳扰动**,尤其是 **UniFP 从未训练过的 base 外力 ±30** → advantage 信号进一步变噪,精度下拉再减弱;
5. → 部分 seed 的 std 维持高位 + base 扰动 → 已收敛的干净 locomotion 被推离 → **后段崩溃**;另一部分 seed 侥幸扛过 → **"只有部分成功收敛"**(高方差、seed 依赖,正是边缘稳定 + 非平稳扰动的特征)。

0.7 上限、5e-4/5e-5 的 LR 再调参是**止血**,没消除根因。base 力是 UniFP 配方里**根本不存在**的新应力,且落在最关键的 base 上。

---

## 7. 已验证等价(非问题根源,给信心)

- estimator target 取数(`critic_obs[:,0:12]` 等价 UniFP `obs_pred`,逐字段+逐缩放、newest-first 帧序正确);
- estimation loss 公式(per-group w² 求和)、更新顺序(PPO→est)、双优化器结构、latent 不 detach(encoder co-train)、KL 公式、GAE/advantage 全局归一化、clipped value/surrogate/entropy;
- reward 项集合与权重/sigma(力跟踪 exp 核 σ_ee=1.0、速度跟踪 σ=0.25 等)逐条一致(机型阈值放大除外);
- obs 结构、clip ±100、estimator 网络结构(经 yaml 覆盖)。

---

## 8. 改动决策清单(待勾选)

> 按"最可能修复失稳"排序。每条给出落点。**先不动代码**,待确认勾选哪些。

**A. 力课程语义对齐 UniFP(纯迁移偏差,信号最干净 —— 建议先做)**

- [ ] **#1 base 外力**:对齐 UniFP 关掉(纯 gripper 力)。落点:`conf/ppo_cse/task/a2_arm_pos_force/mujoco.yaml` 力幅置 0 或加开关 / `pos_force.py:1118-1131` 跳过 base schedule。
- [ ] **#2 force_start**:`144000 → 192000`(对齐 UniFP iter 8000)。落点:`a2/mujoco.yaml:90`(同步修 `go2` 与 dataclass `pos_force.py:353`、错误注释 `:88-90`)。

**B. 力强度/频率对齐(若想复现 UniFP 主扰动)**

- [ ] **#3a gripper 力** `±18/±15 → ±60`。落点:`pos_force.py:325-326`。
- [ ] **#3b 触发概率** `0.4/0.3 → 0.8`。落点:`pos_force.py:341-344`。
- [ ] **#3c settling/interval/duration** 对齐 UniFP(gripper settling 0.5→1.0s 等)。落点:`pos_force.py:332-340`。

**C. 边缘稳定性根因(若 A 之后仍高方差崩溃再动)**

- [ ] **#4a entropy_coef** a2 `0.01 → 0.003`(对齐 config.yaml 作者对小 batch 的诊断)。落点:`a2/mujoco.yaml:25`。
- [ ] **#4b num_envs** `1024 → 更大`(条件允许时,治本)。落点:`a2/mujoco.yaml:12`。
- [ ] **#5 min_learning_rate** `5e-5 → 1e-5`(对齐 UniFP floor)。落点:`a2/mujoco.yaml:28`。

**D. 清理(无训练影响)**

- [ ] 删除 `cse_ppo/actor_critic.py:4` 顶部 docstring 里"(detached)"的错误描述。

---

## 9. 建议的实验顺序

1. **先只改 #1 + #2**(把非平稳扰动的语义对齐 UniFP),固定 seed 跑一轮 —— 这两条是纯迁移偏差,信号最干净。
2. 若仍出现高方差崩溃,再动 **C 组(#4)** 这条边缘稳定性根因。
3. #3 组决定是否要复现 UniFP 的主扰动强度,属"想不想要更强鲁棒性"的设计选择,可后置。

> 复核位点速查 — UniFP:`legged_robot_b2z1_pos_force.py:125,128-129,141,405-412`、`b2z1_pos_force_config.py:140-178`;UniLab:`pos_force.py:325-353,1118-1131,1347-1356,1448-1452`、`conf/ppo_cse/{config.yaml,task/a2_arm_pos_force/mujoco.yaml}`、`algorithm.py:176-196`、`estimator.py:152-184`。
