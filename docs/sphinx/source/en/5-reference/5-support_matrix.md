# Support Matrix

This matrix is generated conceptually from registry entries, owner YAMLs, and
tests. The generator implementation is `src/unilab/utils/support_matrix.py`; the
write target for the generated block is currently the Chinese reference page
`docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md`.

## Backend Selection Rules

- The default backend is `mujoco`.
- Switch to Motrix with `--sim motrix` on the unified CLI.
- `--algo`, `--task`, and `--sim` jointly select the owner YAML.
- Do not treat `training.sim_backend` as a standalone backend switch.

## Playback Differences

- `mujoco`: `--render-mode auto` exports `play_video.mp4`.
- `motrix`: `--render-mode auto` opens an interactive renderer window; it does
  not record a video and is not bound by `play_steps`.
- `--render-mode record`: both backends record a video only.
- `--render-mode none`: no playback.

## Evidence Grades

| Grade | Repository Evidence |
| --- | --- |
| `Registered` | The env/backend pair appears after `registry.ensure_registries()`. |
| `Configured` | A matching owner YAML exists under `conf/ppo/task`, `conf/appo/task`, `conf/offpolicy/task`, or `conf/rlgames_sapg/task`. |
| `Tested` | Automated tests cover the entrypoint/task-owner/backend combination through config compose or runtime smoke. |
| `Benchmarked` | A checked-in benchmark manifest exists for the combination. |
| `Recommended` | Explicit recommendation metadata exists in the repo. |

The current generator reports no checked-in benchmark manifest and no separate
recommendation metadata, so rows do not auto-promote to `Benchmarked` or
`Recommended`.

`RL-Games SAPG` / `simtoolreal` / MuJoCo is `Tested` only for the Code #10
M0-dev provisional identity `mujoco-uni-runtime==0.4.0.dev0`. It is not a
formal M0 release, benchmark, recommendation, or cross-backend/platform support
claim. Motrix and mjwarp remain unsupported for this entry.

## Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix |
| --- | --- | --- | --- |
| PPO (torch) | `go1_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2_joystick_rough` | Tested | Tested |
| PPO (torch) | `g1_walk_flat` | Tested | Tested |
| PPO (torch) | `g1_motion_tracking` | Tested | Tested |
| PPO (torch) | `g1_flip_tracking` | Tested | Tested |
| PPO (torch) | `g1_wall_flip_tracking` | Tested | Tested |
| PPO (torch) | `allegro_inhand` | Tested | Tested |
| PPO (torch) | `sharpa_inhand` | Tested | Tested |
| PPO (torch) | `sharpa_inhand_grasp` | Tested | Tested |
| PPO (torch) | `allegro_inhand_grasp` | Tested | Tested |
| PPO (torch) | `g1_box_tracking` | Tested | Tested |
| PPO (torch) | `g1_climb_tracking` | Tested | Tested |
| PPO (torch) | `g1_motion_tracking_deploy` | Tested | Registered |
| PPO (torch) | `go1_joystick_rough` | Tested | Tested |
| PPO (torch) | `go2_arm_manip_loco` | Tested | - |
| PPO (torch) | `go2_footstand` | Tested | - |
| PPO (torch) | `go2w_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2w_joystick_rough` | Tested | Tested |
| APPO (torch) | `go1_joystick_flat` | Tested | Registered |
| APPO (torch) | `go2_joystick_flat` | Tested | Registered |
| APPO (torch) | `g1_walk_flat` | Tested | Registered |
| APPO (torch) | `g1_motion_tracking` | Tested | Tested |
| APPO (torch) | `g1_flip_tracking` | Tested | Tested |
| APPO (torch) | `g1_wall_flip_tracking` | Tested | Tested |
| APPO (torch) | `allegro_inhand` | Tested | Tested |
| APPO (torch) | `sharpa_inhand` | Tested | Registered |
| APPO (torch) | `g1_climb_tracking` | Tested | Tested |
| SAC (torch) | `g1_walk_flat` | Tested | Tested |
| SAC (torch) | `g1_walk_rough` | Tested | Tested |
| SAC (torch) | `g1_motion_tracking` | Tested | Tested |
| SAC (torch) | `g1_wbt_obs` | Tested | Registered |
| TD3 (torch) | `go1_joystick_flat` | Registered | Tested |
| TD3 (torch) | `go2_joystick_flat` | Registered | Tested |
| TD3 (torch) | `g1_walk_flat` | Tested | Registered |
| FlashSAC (torch) | `go2_joystick_flat` | Tested | Registered |
| FlashSAC (torch) | `g1_walk_flat` | Tested | Registered |
| RL-Games SAPG | `simtoolreal` (SimToolReal) | Tested | - |

## Source Index

- Registry bootstrap: `src/unilab/envs/**` registrations via
  `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`,
  `conf/offpolicy/task/**`, `conf/rlgames_sapg/task/**`.
- Generic compose coverage:
  `tests/config/test_config_system.py::test_supported_task_composes`.
- Provisional SAPG evidence:
  `tests/fixtures/simtoolreal_sapg/m0_dev_manifest.json`,
  `tests/algos/rlgames_sapg/**`, and
  `tests/envs/manipulation/simtoolreal/test_m0_dev_matrix.py`.
