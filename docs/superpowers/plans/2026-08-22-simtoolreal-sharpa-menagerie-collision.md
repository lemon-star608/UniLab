# SimToolReal Sharpa Menagerie Collision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将正式 SimToolReal MuJoCo owner 迁移到 Menagerie Sharpa collision geometry 和 reference + tip_stiff profile，同时保持任务、工具池、I/O 与时间尺度不变。

**Architecture:** `kuka_sharpa.xml` 继续是 canonical robot XML，只替换 Sharpa collision geoms；`scene.xml` 唯一声明全局 MuJoCo option；工具 materialization 和任务代码不变。Menagerie palm meshes 作为静态、可审计资产 vendored 到 robot asset root，provenance 记录每个 mesh/license/source hash。

**Tech Stack:** MuJoCo XML / `mujoco.MjModel`, pytest, NumPy, `uv run`, JSON provenance。

---

### Task 1: Add failing asset and compiled-model contracts

**Files:**
- Modify: `tests/envs/manipulation/simtoolreal/test_assets.py`
- Modify: `tests/envs/manipulation/simtoolreal/test_tool_assets.py`
- Modify: `tests/envs/manipulation/simtoolreal/test_env_integration.py`

- [ ] **Step 1: Extend asset inventory expectations first.** Add the 32 relative palm OBJ paths, the `menagerie_sharpa_wave/LICENSE` and `SOURCE.md` ancillary paths, and change the expected final mesh census to 62 (`61` byte-identical, `1` adapted). Assert every XML mesh path is present and non-symlink, every final asset file is either an expected mesh or the two provenance files, and the ten retired old collision STL files are absent.
- [ ] **Step 2: Run the asset test and confirm the intended RED failure.** Run `uv run pytest tests/envs/manipulation/simtoolreal/test_assets.py -q`; it must fail on the old inventory/hash contract before production assets are changed.
- [ ] **Step 3: Add robot XML structural and contact contract tests.** Parse `kuka_sharpa.xml` and assert exactly 29 joint/actuator names in the pre-migration order, exactly 50 excludes, no `floor/table/object/goal_object/keyframe`, visual geoms use `0/0/group=2`, and every Sharpa collision geom uses the required role-specific `condim`, friction, solref, solimp, margin, gap, bits and density. Compile `scene.xml` with `mujoco.MjModel` and assert `nq=36`, `nv=35`, `nu=29`, finite state after `mj_resetData` plus several `mj_step` calls, `model.opt.timestep == 1/120`, integrator/solver/cone/impratio/iterations/ls_iterations and contact/MultiCCD flags. Use public MuJoCo enum/flag fields available in the installed version and fail with a clear assertion if a required field is missing.
- [ ] **Step 4: Run the new contract tests and confirm RED.** Run the focused test node(s) with `uv run pytest ... -q`; failures must identify old visual groups, old collision meshes/contact values, or old global options rather than test syntax errors.

### Task 2: Vendor Menagerie collision assets and update provenance

**Files:**
- Create: `src/unilab/assets/robots/kuka_sharpa/assets/menagerie_sharpa_wave/palm/palm000.obj` through `palm031.obj`
- Create: `src/unilab/assets/robots/kuka_sharpa/assets/menagerie_sharpa_wave/LICENSE`
- Create: `src/unilab/assets/robots/kuka_sharpa/assets/menagerie_sharpa_wave/SOURCE.md`
- Modify: `src/unilab/assets/robots/kuka_sharpa/ASSET_PROVENANCE`

- [ ] **Step 1: Copy only the pinned donor assets.** Copy the 32 OBJ files from `/home/user/ws/lemon/simtoolreal/assets/mjcf/kuka_sharpa_collision_compare/assets/menagerie_sharpa_wave/palm/` and the donor `LICENSE`/`SOURCE.md` into the exact target subtree. Do not copy donor floor/table/object/keyframe or unrelated meshes.
- [ ] **Step 2: Remove retired collision STL files.** Delete only the old Sharpa collision meshes no longer referenced after the XML migration (`left_hand_C_MC.STL`, `left_thumb_MC.STL`, `left_thumb_PP.STL`, `left_thumb_DP.STL`, `thumb_elastomer.STL`, `MCP_VL.STL`, `left_PP.STL`, `left_MP.STL`, `left_DP.STL`, `elastomer.STL`). Keep all visual meshes and `left_pinky_MC.STL`.
- [ ] **Step 3: Update JSON provenance mechanically from final files.** Preserve existing donor/license records, add Menagerie pinned repository/source commits from `SOURCE.md`, add one SHA-256 entry per final mesh and the Apache license/source hashes, set final mesh census to `total=62, byte_identical=61, different=1`, and update robot/scene XML hashes after XML edits. Keep provenance self-contained and free of absolute runtime paths.
- [ ] **Step 4: Re-run the asset RED test.** Run `uv run pytest tests/envs/manipulation/simtoolreal/test_assets.py -q`; it should now move past missing files and fail only on XML references/hash fields until Task 3 is complete.

### Task 3: Replace Sharpa collision geoms and centralize scene options

**Files:**
- Modify: `src/unilab/assets/robots/kuka_sharpa/kuka_sharpa.xml`
- Modify: `src/unilab/assets/robots/kuka_sharpa/scene.xml`

- [ ] **Step 1: Add Menagerie mesh declarations.** Keep all KUKA and existing visual mesh declarations; remove retired collision mesh declarations; add `menagerie_palm000`–`menagerie_palm031` pointing to `menagerie_sharpa_wave/palm/palm000.obj`–`palm031.obj`.
- [ ] **Step 2: Replace only Sharpa collision geoms.** Preserve every current Sharpa body transform, inertial, joint and visual geom. Add donor palm000–031 mesh geoms under `left_hand_C_MC`, donor cylinder/capsule fit geoms for regular segments, donor fingertip capsules, and pinky MC mesh collision exactly at donor sizes/poses/quaternions. Make elastomer mesh visual-only. Explicitly author the required role-specific contact attributes on every collision geom, with friction `1.0 0.005 0.0001` and no `tip_soft` values.
- [ ] **Step 3: Normalize Sharpa visual geoms and preserve arm contract.** Set all Sharpa visual geoms to `contype=0 conaffinity=0 group=2 density=0`; leave KUKA arm collision geometry/friction and all body transforms untouched. Keep the existing 50 contact excludes byte-for-byte in membership and ordering.
- [ ] **Step 4: Make `scene.xml` the sole global option owner.** Set the exact 1/120 timestep and required `implicitfast/Newton/elliptic/impratio/iterations/ls_iterations/contact/multiccd` values; remove conflicting option declarations from included robot XML. Keep floor/table/keyframe and their dimensions/friction unchanged.
- [ ] **Step 5: Run focused XML contracts.** Run the asset test plus the new compiled-model contract. Expected result: PASS for XML compile, mesh inventory, hand structure, contact parameters, option values, visual bits and excludes.

### Task 4: Preserve tool pool and finite-state integration

**Files:**
- Modify: `tests/envs/manipulation/simtoolreal/test_tool_assets.py`
- Modify: `tests/envs/manipulation/simtoolreal/test_env_integration.py`
- Create or modify: `docs/sphinx/source/zh_CN/2-user_guide/8-manipulation/3-simtoolreal_sharpa_collision.md`
- Modify: `docs/sphinx/source/zh_CN/2-user_guide/8-manipulation/0-index.md`

- [ ] **Step 1: Add materialized-tool assertions.** Keep the existing three-topology compile test and assert materialized XML still includes the robot include, produces `(nq,nv,nu)=(36,35,29)`, and contains an object body without changing `tool_assets.py`.
- [ ] **Step 2: Add finite reset/step coverage at the compiled model boundary.** Extend the real integration fixture assertions to check finite qpos/qvel/ctrl after reset and several MuJoCo steps, action shape `(29,)`, and unchanged 600-tool catalog topology counts.
- [ ] **Step 3: Write the Chinese owner document.** Record old Mesh Sharpa → Menagerie collision migration, reference + tip_stiff values, `sim_dt=1/120`/`ctrl_dt=1/60`, the non-training 1/600 benchmark distinction, unchanged reward/task/tool pool/table/floor/tool friction scope, and the XML/provenance/contract-test owners. Link it from the manipulation index.
- [ ] **Step 4: Run the required three test files.** Run `uv run pytest tests/envs/manipulation/simtoolreal/test_assets.py -q`, `uv run pytest tests/envs/manipulation/simtoolreal/test_tool_assets.py -q`, and `uv run pytest tests/envs/manipulation/simtoolreal/test_env_integration.py -q`.

### Task 5: Full verification and handoff

**Files:**
- Verify all changed files; no additional production scope.

- [ ] **Step 1: Run focused and neighboring tests.** Run all three required files plus `uv run pytest tests/envs/manipulation/simtoolreal -q`.
- [ ] **Step 2: Audit scope and XML ownership.** Use `git diff --stat`, `git diff --check`, XML parsing, and `rg` to confirm no reward/config/tool-generation changes, no absolute mesh paths/symlinks, no robot-level scene elements, no `tip_soft`, and no 1/600 timestep.
- [ ] **Step 3: Review the final diff against the approved design.** Verify all 10 user contract points and document any unrelated pre-existing changes (none expected).
- [ ] **Step 4: Commit the implementation with a focused message.** Commit only the approved Sharpa collision migration, assets, provenance, tests and Chinese documentation after fresh verification.
