# SimToolReal SAPG Code #4–#5 输入输出保真设计

## 1. 决策与目标

2026-08-20，maintainer 明确批准将 Code #4 和 Code #5 的验收标准收缩为：在固定
Source、固定 canonical 平台和代表性关键分支下，只要 vendored Target 对同一完整输入
产生相同的完整可观察输出状态，即判定算法迁移保真。

本文件从提交之日起是 Code #4 与 Code #5 的唯一当前验收设计。此前 Code #4 clean
execution prompt 中关于逐 primitive、逐 batch forensic ledger、private capability token、
大量 mutation、8 MiB 硬预算和穷举文件系统事务矩阵的要求不再适用。

这次收缩不放宽以下基础边界：

- Source identity、vendored Target identity 和兼容补丁清单必须固定；
- Source 与 Target 使用独立进程和独立 `rl_games` namespace；
- 必须调用 native Runner、A2CAgent、PPODataset、optimizer、GradScaler、checkpoint 和
  player owner，不能在测试中重写算法公式或训练循环；
- 同一 case 的输入状态必须相同；
- Code #4/#5 不修改 vendor、Source 或 UniLab production runtime；
- 所有 Python 命令继续使用 `uv run`。

## 1.1 范围、规模与永久成本

Code #4 只新增一个 Source-only generator、一个 Source/Target 共用 update harness、一个
focused pytest 和两个生成物，共五个文件；Code #5 也作为独立 child issue，采用相同的
三份手写测试基础设施加两个生成物边界。两批分别提交，不修改 production、vendor 或
Source。现有 Code #4 中间实现约 5,400 行，主要来自已经废止的 forensic trace；本轮应
大幅删除该部分，只保留构造原生 owner、完整输入/输出快照、序列化和必要 drift gate。
最终规模以 native owner 构造所需代码为准，不再把 800 行或 fixture 字节数当 hard gate，
但不得借此增加新的公共 contract 或运行路径。

永久维护成本是两套固定 Source golden：Source 身份或已审计 vendor patch 改变时，需要在
canonical CUDA 平台显式重新生成并审查；普通 pytest 只做 Target replay，不能自动
rebaseline。维护者长期只需维护有限 case、external anchors 和状态比较 schema，不维护
逐算子 trace 系统。

## 2. 保真判据

训练算法被视为状态转换：

~~~text
(batch/config/model/optimizer/scaler/RMS/LR/RNG)
    -> native Source or Target execution
    -> (native return/final model/optimizer/scaler/RMS/LR/RNG)
~~~

保真并不要求记录每个内部算子。只要输入状态一致、关键分支真实执行，而且完整可观察
输出状态一致，即接受该代表性 case。有限 golden case 不宣称数学上覆盖所有可能输入；
它与固定源码身份及 vendor patch audit 一起构成工程验收证据。

结构、离散状态、RNG bytes、参数/optimizer tensor hashes 和 branch outcome 使用 exact
comparison。需要保留浮点数组时，FP32 默认使用 `atol=1e-6, rtol=1e-5`；canonical
Source/Target 实际 bit-exact 时同时记录 exact hash。AMP 不与 FP32 横向比较，只比较同一
AMP case 的 Source 与 Target 状态转换。

## 3. Code #4：update 与 AMP

### 输入

- Code #3 已冻结并验证 anchor 的 post-shuffle 56-row/14-sequence batch；
- 相同 Runner params 和 Code #4 test-only resource overrides；
- 相同 deterministic parameter fill；
- 相同 actor/central model、optimizer、GradScaler、RMS、LR 和完整 NumPy/Torch/CUDA RNG
  初始状态；
- canonical Python 3.11、Torch 2.7.0+cu128、RTX 4090 execution flags。

### 必须执行的 case

1. `normal_fp32`：native complete update，actor mixed precision disabled；
2. `normal_amp`：native complete update，真实 CUDA autocast 和 enabled GradScaler；
3. `overflow_amp`：从 native prepared actor batch clone，仅将 `advantages[0]` 改为
   `+inf`，调用真实 native scaler/update path。

两个 normal case 仍运行 native prepare、central-before-actor、central/actor 各两个
mini-epoch。identity shuffle freeze 只负责把 Code #3 frozen batch 注入原生 train path，
必须不改变对象、顺序或 RNG。

### 比较的输出

- native prepared actor/central dataset 的字段 inventory、shape、dtype、row order 和内容；
- native train return 中的 actor/value/bounds loss、entropy、KL 和 LR summary；
- final actor/central parameter inventory 与 aggregate/per-parameter content hashes；
- final actor/central optimizer param groups、state keys、step 和 state tensor hashes；
- final GradScaler state及 successful/overflow case 的 step-or-skip outcome；
- final actor input、central input、actor-model value、active central value RMS 的完整
  mean/var/count/mode；
- final scheduler/LR state；
- case 前后完整 NumPy、Torch CPU 和全部 CUDA RNG state；
- overflow case 必须表现为参数和 optimizer state 不变、scaler scale backoff。

允许按 `initial`、`after_prepare`、`final` 保存少量阶段快照用于定位问题；不要求逐 batch
或逐 primitive trace。

### 明确删除的旧要求

- 不再拦截或持久化每个 `torch.exp/clamp/max/neg/pow/mul`；
- 不再要求数千条 start/end event ledger；
- 不再要求每参数每 batch 的 64 sentinel、scaled/unscaled/clip 全链路取证；
- 不再要求 private metadata/evidence/semantic capability token；
- 不再要求十二类 forensic mutation 或 Source/Target 对称删除审计；
- 不再以 8 MiB 作为 hard blocker，只要求 artifact 合理且不保存完整模型副本；
- 不再要求 FIFO/socket/device、pair-transaction rollback 等穷举路径测试。保留普通
  regular-file、symlink、external-anchor、inventory/shape/dtype/content drift gate 即可。

### 完成条件

- source-only generator 在 canonical 平台重新生成 fixture；
- fresh Target replay 的三个 case 全部通过；
- focused Code #4、完整 SAPG oracle、vendor suite、72-file audit 和 scoped Ruff/format
  通过且 required tests 为 0 skip；
- 独立 reviewer 确认相同输入、native path 和完整最终状态比较真实存在；
- 只提交五个 Code #4 文件，commit title 为
  `test: lock SAPG update and AMP semantics`。

## 4. Code #5：checkpoint、resume 与 player

Code #4 接受并提交后，maintainer 已批准自动进入 Code #5，不再另设一次人工批准。Code #5
仍单独写简短 execution plan、单独实现、验证和提交；不得与 Code #4 squash。

Code #5 使用同一输入输出判据：

- Source 通过 native owner 生成固定 checkpoint payload；
- Source 与 Target 在相同初始/外部恢复 RNG 下加载同一 payload；
- 比较 load 后 model、optimizer、RMS、scaler、LR 和 runner-visible state；
- 比较 resume 后首个 native action、value 和 update 的返回与最终状态；
- 比较 canonical 6-env 与 `N != 6` player routing 的可观察输出；
- 明确记录 Source 原本未保存的 RNG 和 `env_state=None` 边界，但不逐字段取证 native save
  implementation，也不复制 checkpoint/player 逻辑。

Code #5 的测试保留 provenance、external anchors、inventory/content drift 和少量负例；不建立
新的 forensic event system。Code #5 仍不是 production train/play 入口。

## 5. 后续边界

本决策只批准 Code #4 与 Code #5。Code #6–#10 的 backend、asset、真实 env、production
path 和 support promotion 范围不因本文件自动开始。Code #5 完成后必须停在 Code #6
边界回报 maintainer。
