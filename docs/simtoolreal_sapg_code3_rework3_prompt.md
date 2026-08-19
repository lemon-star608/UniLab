# SimToolReal SAPG Code #3 Final Test-Hardening Prompt

> 实现 session：完整阅读本文件与根目录 `AGENTS.md` 后直接执行。不要重新规划。本轮只补最后一个 whole-inventory metadata gate 的 mutation-sensitive 测试；保持实现未暂存、未提交，交回控制 session 审查。

## 结论与目标

code #3 当前生产 oracle 实现、canonical replay 和 fixture 都是正确的：

- focused：17 passed；
- 全部SAPG：29 passed；
- 固定Source重生后NPZ/manifest逐字节一致；
- ancestor symlink、raw handoff、content hash、RNN double-delegate、carrier mutation均已被测试锁定。

只剩一个测试证据缺口：

`tests/algos/rlgames_sapg/test_rollout_golden.py::test_target_metadata_is_whole_inventory_gate_before_numeric`

当前只监听 `np.abs`，所以以下违规顺序仍可逃过：

```text
validate(A) -> subtract(A) -> validate(B) -> B shape drift
```

当前测试也没有实际触发 missing/extra inventory gate。

最终目标：让测试直接观察 subtraction，并锁住 sorted missing/extra 诊断。不要改变正确的算法、fixture或Source capture。

## 仓库与范围

工作目录：

`/home/user/ws/lemon/rlgame-unilab/UniLab`

预期分支：

`feat/simtoolreal-sapg-rlgames`

固定父级HEAD：

`1127d2f4e00efb43afc55d53409b8f5c63487d96`

实现 session 开始时记录：

```bash
git rev-parse --abbrev-ref HEAD
SAPG_CODE3_REWORK3_START_HEAD=$(git rev-parse HEAD)
printf '%s\n' "$SAPG_CODE3_REWORK3_START_HEAD"
git merge-base --is-ancestor \
  1127d2f4e00efb43afc55d53409b8f5c63487d96 HEAD
git status --short
git diff --cached --name-only
```

实际HEAD可以在固定父级之上仅包含本prompt的docs commit。开始和结束时，Target必须仍恰好只有以下五个未跟踪文件，暂存区为空：

```text
?? scripts/generate_simtoolreal_sapg_rollout_fixture.py
?? tests/algos/rlgames_sapg/source_rollout_harness.py
?? tests/algos/rlgames_sapg/test_rollout_golden.py
?? tests/fixtures/simtoolreal_sapg/source_rollout_fp32.npz
?? tests/fixtures/simtoolreal_sapg/source_rollout_manifest.json
```

本轮最终应只改变：

`tests/algos/rlgames_sapg/test_rollout_golden.py`

除非测试暴露当前实现确有错误，否则不得改generator、harness或两个fixture。不得新增第六个实现文件，不得修改vendor、Source、生产代码、配置或其他测试。

所有手写编辑使用 `apply_patch`。不得add、commit、push、stash、reset、clean、checkout或切换分支。不得运行`make test-all`。

只使用Python 3.11。不要运行Python 3.10/3.12/3.13或其他CUDA矩阵。最终三个手写文件净LOC上限从999放宽到1020；不得通过异常压缩或移动代码规避。

## 必须增加的两个回归关系

### 1. 直接观察 subtraction

改造现有whole-inventory测试，使数组A能记录 `actual - expected` 是否发生。推荐在同一测试文件定义一个很小的 `np.ndarray` subclass：

```python
class SubtractionProbe(np.ndarray):
    def __new__(cls, values, events):
        result = np.asarray(values).view(cls)
        result.events = events
        return result

    def __array_finalize__(self, source):
        self.events = getattr(source, "events", None)

    def __sub__(self, other):
        self.events.append("subtract")
        return super().__sub__(other)
```

可使用等价的更短实现，但必须直接观测 `__sub__`，不能只观测后续 `np.abs`。

构造：

- A：metadata合法的 `SubtractionProbe`；
- B：shape drift；
- metadata顺序为A、B。

调用现有真实owner：

`rollout_harness._target_array_diagnostics(actual, expected, metadata)`

必须得到B shape drift，并严格断言事件中只有：

```text
validate:a
validate:b
```

不允许出现 `subtract`、`abs`、divide或其他numeric事件。

测试必须对以下mutant敏感：在验证A之后、验证B之前插入：

```python
actual_arrays["a"] - expected_arrays["a"]
```

该mutant必须使测试失败。

### 2. 实际触发inventory gate

在同一测试或一个参数化测试中构造：

- expected names：`a`, `b`
- actual names：`a`, `c`

调用真实 `_target_array_diagnostics`，断言错误消息精确包含排序后的：

```text
missing=['b'], extra=['c']
```

并断言在inventory drift发生前后没有任何metadata validation或numeric事件。

测试必须对删除以下真实gate的mutant敏感：

```python
if missing or extra:
    raise RuntimeError(...)
```

删除gate后，测试必须失败，而不是因另一个宽泛异常被误判为通过。

## Mutation RED / final GREEN

由于当前实现已经正确，普通新增测试会直接GREEN。本轮RED采用临时mutation验证，不伪造初始TDD失败：

1. 用`apply_patch`临时在允许的harness中插入“validate A后subtract A”的mutant。
2. 只运行whole-inventory测试，必须RED。
3. 用`apply_patch`精确恢复原两阶段实现。
4. 用`apply_patch`临时删除/绕过missing-extra gate。
5. 只运行inventory测试，必须RED。
6. 用`apply_patch`精确恢复原gate。
7. 再次检查最终harness与开始时字节完全一致。
8. 运行最终GREEN。

不得使用git checkout/reset恢复。不得把mutant留在最终文件。报告两次mutant的准确失败原因。

轻量命令：

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py \
  -k 'whole_inventory' -q
```

## 必跑验证

轻量全部：

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py \
  -k 'not target_replays_canonical_source_rollout_exactly' -q
```

Canonical focused：

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg/test_rollout_golden.py -q
```

完整SAPG：

```bash
UV_INDEX=https://download.pytorch.org/whl/cu128 \
UNILAB_REQUIRE_SAPG=1 \
uv run --python 3.11 \
  --with-editable ./third_party/simtoolreal_rl_games \
  pytest tests/algos/rlgames_sapg -q
```

Vendor、audit和静态gate：

```bash
env -u UV_INDEX uv run --python 3.11 \
  pytest tests/vendor/test_simtoolreal_rl_games_vendor.py -q
env -u UV_INDEX uv run --python 3.11 \
  scripts/audit_simtoolreal_rlgames_vendor.py
env -u UV_INDEX uv run --python 3.11 ruff check \
  scripts/generate_simtoolreal_sapg_rollout_fixture.py \
  tests/algos/rlgames_sapg/source_rollout_harness.py \
  tests/algos/rlgames_sapg/test_rollout_golden.py
env -u UV_INDEX uv run --python 3.11 ruff format --check \
  scripts/generate_simtoolreal_sapg_rollout_fixture.py \
  tests/algos/rlgames_sapg/source_rollout_harness.py \
  tests/algos/rlgames_sapg/test_rollout_golden.py
env -u UV_INDEX uv run --python 3.11 ruff check .
env -u UV_INDEX uv run --python 3.11 ruff format --check .
git diff --check
git diff --cached --name-only
git status --short
```

本轮不需要重生fixture：最终harness必须与开始时字节一致，fixture hashes必须保持：

- NPZ：`3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8`
- manifest file：`785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f`
- canonical payload：`7d88cb01dce4607391a39d1fb31b21d8366d2bdadae2e0dce6eb02323c06901d`

所有required pytest必须0 skip，canonical replay必须exact，所有FP误差仍为0。

## 停止条件

出现任一情况返回`# BLOCKED`：

- 当前正确的harness必须永久修改才能测试；
- generator或fixture必须变化；
- 需要第六个实现文件或任何越界修改；
- 需要修改Source/vendor/算法数学；
- canonical Python 3.11 + cu128不可用；
- required test skip或出现未解释数值差异；
- 最终净手写LOC超过1020；
- 进入code #4或运行`make test-all`。

## 交接报告

返回`# DONE`并包含：

1. 最终文件状态和净LOC。
2. subtraction mutant的RED命令、失败原因和事件序列。
3. inventory-gate mutant的RED命令、失败原因。
4. 最终合法B-drift事件序列，证明无subtraction/numeric。
5. missing/extra精确诊断与零validation/numeric事件。
6. 最终harness、generator、fixture hashes与开始时一致。
7. 轻量、focused、完整SAPG、vendor、audit、Ruff实际结果。
8. 完整`git status --short`、空暂存区和未变HEAD。
9. 明确确认未进入code #4、未运行`make test-all`、未执行Git破坏操作、未修改Source、未增加版本矩阵。

控制 session 独立复核通过后才会提交code #3并运行提交后的`make test-all`。
