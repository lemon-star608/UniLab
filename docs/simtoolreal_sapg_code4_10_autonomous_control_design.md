# SimToolReal SAPG Code #4–#10 自主控制设计

## 目标

在 `feat/simtoolreal-sapg-rlgames` 分支中撤销尚未接受的 Code #4
实现与两轮返修提示词，随后由同一控制 session 串行完成 Code #4–#10 的规格、
实现调度、独立审查、验证和提交。维护者不再人工转发 prompt；控制 session 直接把
已提交的 batch 规格交给实现 agent。

最终结果仍以现有
`docs/simtoolreal_sapg_source_fidelity_migration_plan.md` 的 Source-fidelity 架构为准：
固定 Source RL-Games runtime 是 SAPG 的唯一算法 owner，UniLab 只拥有 Hydra owner、
同步 `NpEnv`/`IVecEnv` adapter、MuJoCo task/backend、run directory 和 tracker bridge。

## 当前基线

- 当前分支 HEAD 为 `78c8bd979547ef304dcd94635d61453ac1352c5c`。
- Code #1–#3 已提交；Code #4 尚未接受或提交。
- 工作树只有五个未跟踪 Code #4 文件，tracked diff 和 staging area 为空。
- `dbe5bf3a66055218ea109aae67f6736d87f3e4e3` 与
  `5b08333397f436feba1ad3f2376ddd96b9d2ee02` 分别新增第一、第二轮返修提示词。
- 后续 handoff commits `5af99216`、`78c8bd97` 引用了这两份提示词，清理后必须同步
  改写 handoff，不能留下断链或旧 writer 状态。

## 历史与工作树清理

采用保留历史、可审计的撤销方式，不重写分支历史：

1. 再次只读确认分支、HEAD、tracked/staged diff 和五文件精确边界。
2. 确认旧实现 session 不再是 writer；若五文件在清理期间继续变化，停止并调查并发。
3. 把五个未跟踪文件移动到仓库外、带时间戳的可恢复隔离目录；不使用 `rm`、
   `git clean`、`reset` 或 `checkout`。
4. 按从新到旧的顺序执行 `git revert`，分别撤销 `5b083333` 和 `dbe5bf3a`。
5. 修改 control handoff，使它记录 Code #4 已重置、控制 session 是唯一 Git owner、
   后续 prompt 由控制 session 直接交给内部 agent。
6. 精确 stage 和提交 handoff 修订；确认工作树恢复干净，并报告隔离目录和恢复方式。

不采用历史重写，因为后续 handoff commits 已依赖返修文档，且旧实现 session 可能仍持有
旧 HEAD。也不保留返修文档后仅用新文档覆盖，因为这不满足维护者要求的“撤销两笔返修
提交”。

## 批次控制模型

Code #4–#10 严格串行；前一批未接受、验证和提交时不得开始下一批。每批执行相同流程：

1. 控制 session 根据总体计划和上一批证据写一份独立、普通中文可审查的 batch 规格。
2. 规格明确主要结果、非目标、文件边界、预计规模、永久维护成本、RED/GREEN、
   provenance、mutation、验证命令和停止条件。
3. 控制 session 自审规格并精确提交 docs-only commit。
4. 在该 batch 的 execution approval 到位后，控制 session 直接派一个实现 agent；
   用户不转发 prompt。
5. 实现 agent 是声明文件范围的唯一 writer，只用 `apply_patch` 编辑、只用 `uv run`
   执行 Python，不执行任何 staging、commit、push、branch、stash、reset、clean 或 PR 操作。
6. 实现 agent 先建立真实 RED，再实现 GREEN，完成后停止写入并提交完整 handoff 报告。
7. 控制 session 阅读完整 diff 和生成物，执行 scope/spec review、代码质量审查和近风险
   fresh validation。可增加只读 reviewer agent，但 reviewer 不代替控制 session 的责任。
8. 有 findings 时，只把证据明确的修正任务交回同一实现 agent；关闭所有 findings 后重新
   fresh validation。
9. 只有控制 session 精确 stage 声明路径、运行 staged diff gate 并创建代码 commit。
10. 提交后复跑该批关键 gate，核对 commit 内容和干净工作树，再进入下一批准点。

同一共享工作树任一时刻只能有一个 writer。控制 session 在实现 agent 工作期间不编辑；
实现完成后的审查阶段，agent 不再写入。并行 agent 只允许做互不依赖的只读分析或审查。

## Code #4–#10 边界

### Code #4：update 与 AMP oracle

主要结果是重新建立干净的 O1b Source→Target oracle。新规格合并原始提示词中仍有效的
要求和两轮审查暴露的 evidence-completeness 风险，但不恢复两份返修文档本身。实现仍限
五个文件和小于 8 MiB fixture，必须调用 native Runner/A2CAgent/PPODataset/
CentralValueTrain/loss/optimizer/GradScaler owner，不复制算法公式。

验收包括 frozen 56-row handoff、central-before-actor、两个 mini-epoch、loss branches、
normalizer、KL/reference、scheduler、gradient/clip、optimizer、FP32/AMP/overflow、RNG、
完整 evidence invariants、metadata-before-numeric 和 mutation tests。只验证固定
Python 3.11、Torch 2.7.0+cu128、RTX 4090 canonical 平台。本批不运行
`make test-all`，不进入 checkpoint/player。

### Code #5：checkpoint 与 player oracle

主要结果是 O1c：固定 Source `.pth` payload、实际可恢复字段、未保存 RNG 和
`env_state=None` 边界、外部恢复 RNG 后的首个 action/value/update，以及 canonical
6-env 和 `N != 6` player routing。仍是算法 oracle，不接生产训练入口。目标五文件，
fixture 总计小于 2 MiB。

### Code #6：MuJoCo runtime public contracts

主要结果是 M0-dev、source-model variant、public body-wrench 和 public step-autoreset
mask。只修改 backend owner 层及近风险测试，不把 backend-specific 能力泄漏到 env 或
脚本。M0-dev provenance 必须固定完整 Git SHA；如使用 artifact，同时固定 filename、
SHA256 和 source provenance。维护者已在本设计批准点明确批准这组 backend public
contracts，但任何新增 public surface 仍需停下重新确认。

### Code #7：assets、task foundations 与 T0

主要结果是机械迁移 600-tool assets、两个生产 XML、许可证/provenance、backend-neutral
task primitives，并从固定 Source 生成 T0 oracle。本批不注册真实 env，不迁移 donor 的
RSL-RL SAPG，也不把 keyframe 放进 robot XML。资产和 XML 只在冷路径访问。

### Code #8：真实 MuJoCo env composition 与 T1

主要结果是组合、注册真实 SimToolReal MuJoCo env，并锁定 `NpEnv` contract、600-tool
pool、reset/action/observation/reward/termination/DR 公式。允许差异仅限已 manifest 的
MuJoCo/table/tool/resource mapping；不把物理轨迹差异宣称为逐步等价。

### Code #9：Source RL-Games SAPG production path

主要结果是 `sapg` optional extra、Hydra owner、同步 adapter、native Runner、tracker、
checkpoint/player 和 CLI train/eval vertical slice。原生 Runner 唯一拥有 rollout、env
step/reset、update、checkpoint 和 player lifecycle；禁止新 collector、async 协议或算法
配置翻译。维护者已在本设计批准点明确批准该 production execution path。

### Code #10：release 与 support promotion

先在 M0-dev 上完成小规模 S1 和 12288/2048 profile 的真实 train/play smoke，再完成
外部 M0-release，最后晋升正式依赖、lock、audit、docs 和 support claim。维护者已批准在
全部 release gate 真实通过后晋升 SAPG support，但测试不能代替最终产品判断。

MuJoCoUni owner 仓库中的生产修改仍需单独的普通中文 roadmap 和明确授权；当前 UniLab
设计不预先授权未知的外部源码改动。没有 clean-install M0-release artifact 时，Code #10
保持未完成，不能用 M0-dev 或 dirty sibling checkout 替代。

## 人工批准点

控制 session 承担 prompt 编写和传递，但以下 maintainer 判断不能自动化：

1. 本设计文件提交后的书面 spec 审阅；通过后 Code #4 可以开始。
2. Code #5–#10 每批开始前，对该批普通中文范围、非目标、规模和永久成本的 execution
   approval。该批准不要求用户转发 prompt。
3. MuJoCoUni M0-release 的独立 roadmap 和外部仓库修改授权。
4. 最终真实 smoke、依赖 provenance、完整测试和 CI 证据到齐后的 support claim 产品判断。

Code #6 public-contract、Code #9 production-path 和 Code #10 support-promotion 的产品方向
已在本设计批准时明确确认；若具体规格新增了当前计划未列出的公共 surface、owner 或
support scope，仍须重新确认。

## 验证与完成条件

每批优先验证最接近风险的边界。Oracle fixtures 必须由固定 Source checkout 显式生成，
普通 pytest 不得 rebaseline；Source 与 Target namespace 分进程，loaded module provenance
必须 fail closed。所有 required pytest 必须零 skip。

Code #4 使用其 canonical CUDA gate；后续批次使用各自规格中的精确命令。production path
暴露前和最终 PR 前运行 `make test-all`。最终提交必须完成、工作树干净，且 PR body 如实
记录 validation。创建或更新 PR 后，按最终 head SHA 等待所有远程 CI 完成并通过；旧 head
结果、pending、in-progress 或挂起 job 均不算完成。

只有以下事实全部有当前证据时才能宣称目标完成：

- Code #4–#10 各自代码 commit 已存在且范围正确；
- Source provenance、vendor identities 和已接受 oracle 未发生未声明漂移；
- M0-release 是可 clean-install、身份固定的正式 artifact；
- 真实 MuJoCo train/play smoke、12288/2048 profile、`make test-all` 均通过；
- 最终 support claim 已获 maintainer 产品判断；
- 最终 PR 的当前 head CI 全部通过。

## 停止条件

出现以下任一情况，停止当前 batch 并向维护者报告证据，不通过扩 scope 或放宽 gate继续：

- 需要改变 SAPG tensor 公式、RNG、update、AMP、checkpoint 或 player 语义；
- 需要在脚本长期翻译算法配置、绕开 native Runner lifecycle 或新增 collector 协议；
- env 需要调用 backend 私有方法或在热路径解析 asset/XML；
- 当前 batch 超出批准文件/fixture边界、引入新 public owner 或无法满足 provenance；
- Source→Target 或 T0→T1 出现无法解释的 mismatch；
- canonical CUDA、真实 MuJoCo 或 M0-release gate 无法真实执行；
- 其他 session 与当前 writer 重叠，无法在不覆盖其改动的情况下继续；
- required tests 有 skip、失败、未解释 warning 或证据覆盖不足。
