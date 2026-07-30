"""T0 acceptance tests for the SimToolReal env skeleton.

Covers the task-card criteria that do not need a physics rollout:

* config defaults match the SimToolReal source configclasses,
* the ported physics tables match the source value for value,
* fingertip friction lands only on the five fingertip links,
* the joint permutation round-trips,
* ``state.info`` carries every interface-contract key with the right
  shape and dtype,
* ``env.step(zeros)`` runs without NaNs.

The MJCF/rollout tests are skipped when the generated assets or the MuJoCo
backend are unavailable, so the pure-python checks still run in a bare checkout.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg
from unilab.envs.manipulation.simtoolreal.constants import (
    ARM_JOINT_DAMPING,
    ARM_JOINT_STIFFNESS,
    DEFAULT_JOINT_POS,
    FINGERTIP_LINK_NAMES,
    HAND_JOINT_ARMATURE,
    HAND_JOINT_DAMPING,
    HAND_JOINT_FRICTION,
    HAND_JOINT_STIFFNESS,
    JOINT_NAMES_CANONICAL,
    NUM_JOINTS,
    compute_obs_dim,
)

ROBOT_XML = ASSETS_ROOT_PATH / "robots" / "kuka_sharpa" / "kuka_sharpa.xml"
SCENE_XML = ASSETS_ROOT_PATH / "robots" / "kuka_sharpa" / "scene.xml"

requires_assets = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="run `python -m unilab.tools.build_simtoolreal_assets` to generate the MJCF assets",
)


def test_config_defaults_match_source() -> None:
    """Config defaults reproduce the SimToolReal source configclasses."""
    cfg = SimToolRealCfg()

    # Frequencies (decision D4): decimation=2 at 120 Hz physics / 60 Hz policy.
    assert cfg.sim_dt == pytest.approx(1.0 / 120.0)
    assert cfg.ctrl_dt == pytest.approx(1.0 / 60.0)
    assert cfg.sim_substeps == 2
    assert cfg.max_episode_seconds == pytest.approx(10.0)
    assert cfg.max_episode_steps == cfg.termination.episode_length == 600
    assert cfg.action_space == NUM_JOINTS

    # AssetsCfg (cfg:98-101).
    assert cfg.assets.robot_friction == pytest.approx(0.5)
    assert cfg.assets.finger_tip_friction == pytest.approx(1.5)
    assert cfg.assets.object_friction == pytest.approx(0.5)
    assert cfg.assets.table_friction == pytest.approx(0.5)

    # ActionCfg (cfg:320-322).
    assert cfg.action.arm_moving_average == pytest.approx(0.1)
    assert cfg.action.hand_moving_average == pytest.approx(0.1)
    assert cfg.action.dof_speed_scale == pytest.approx(1.5)

    # ObsCfg (cfg:145).
    assert cfg.obs.clamp_abs_observations == pytest.approx(10.0)

    # RewardCfg (cfg:336-350).
    assert cfg.reward.keypoint_rew_scale == pytest.approx(200.0)
    assert cfg.reward.object_base_size == pytest.approx(0.04)
    assert cfg.reward.fixed_size == (0.141, 0.03025, 0.0271)
    assert cfg.reward.lifting_rew_scale == pytest.approx(20.0)
    assert cfg.reward.lifting_bonus == pytest.approx(300.0)
    assert cfg.reward.lifting_bonus_threshold == pytest.approx(0.15)
    assert cfg.reward.distance_delta_rew_scale == pytest.approx(50.0)
    assert cfg.reward.reach_goal_bonus == pytest.approx(1000.0)
    assert cfg.reward.kuka_actions_penalty_scale == pytest.approx(0.03)
    assert cfg.reward.hand_actions_penalty_scale == pytest.approx(0.003)

    # GoalCfg — regrouped from ResetCfg/TerminationCfg/RewardCfg (contract §5.0).
    assert cfg.goal.goal_sampling_type == "delta"
    assert cfg.goal.delta_goal_distance == pytest.approx(0.1)
    assert cfg.goal.delta_rotation_degrees == pytest.approx(90.0)
    assert cfg.goal.mins == (-0.35, -0.2, 0.6)
    assert cfg.goal.maxs == (0.35, 0.2, 0.95)
    assert cfg.goal.target_volume_region_scale == pytest.approx(1.0)
    assert cfg.goal.success_tolerance == pytest.approx(0.075)
    assert cfg.goal.target_success_tolerance == pytest.approx(0.01)
    assert cfg.goal.success_steps == 10
    assert cfg.goal.keypoint_scale == pytest.approx(1.5)

    # ResetCfg (cfg:363-386).
    assert cfg.reset.reset_position_noise_x == pytest.approx(0.1)
    assert cfg.reset.reset_position_noise_y == pytest.approx(0.1)
    assert cfg.reset.reset_position_noise_z == pytest.approx(0.02)
    assert cfg.reset.reset_dof_pos_random_interval_arm == pytest.approx(0.1)
    assert cfg.reset.reset_dof_pos_random_interval_fingers == pytest.approx(0.1)
    assert cfg.reset.reset_dof_vel_random_interval == pytest.approx(0.5)
    assert cfg.reset.table_reset_z == pytest.approx(0.38)
    assert cfg.reset.table_reset_z_range == pytest.approx(0.01)
    assert cfg.reset.table_object_z_offset == pytest.approx(0.25)

    # TerminationCfg (cfg:431-444).
    assert cfg.termination.max_consecutive_successes == 50
    assert cfg.termination.force_consecutive_near_goal_steps is False
    assert cfg.termination.tolerance_curriculum_increment == pytest.approx(0.9)
    assert cfg.termination.tolerance_curriculum_interval == 3000
    assert cfg.termination.tolerance_curriculum_success_threshold == pytest.approx(3.0)


def test_domain_randomization_defaults_match_source() -> None:
    """DR defaults reproduce the source ``DomainRandomizationCfg`` (cfg:452-507)."""
    dr = SimToolRealCfg().domain_randomization
    assert (dr.use_obs_delay, dr.obs_delay_max) == (True, 3)
    assert (dr.use_action_delay, dr.action_delay_max) == (True, 3)
    assert (dr.use_object_state_delay_noise, dr.object_state_delay_max) == (True, 10)
    assert dr.object_state_xyz_noise_std == pytest.approx(0.01)
    assert dr.object_state_rotation_noise_degrees == pytest.approx(5.0)
    assert dr.object_scale_noise_multiplier_range == (1.0, 1.0)
    assert dr.joint_velocity_obs_noise_std == pytest.approx(0.1)
    assert dr.force_scale == pytest.approx(20.0)
    assert dr.torque_scale == pytest.approx(2.0)
    assert dr.force_prob_range == dr.torque_prob_range == (0.001, 0.1)
    assert dr.force_decay == dr.torque_decay == pytest.approx(0.0)
    assert dr.force_decay_interval == dr.torque_decay_interval == pytest.approx(0.08)
    assert dr.force_only_when_lifted is True
    assert dr.torque_only_when_lifted is True
    assert dr.object_friction_scale_range == (1.0, 1.0)
    assert dr.fingertip_friction_scale_range == (1.0, 1.0)
    assert dr.friction_n_buckets == 16


def test_obs_group_widths_sum_from_config_lists() -> None:
    """Actor/critic widths come from the ObsCfg field lists, not magic numbers."""
    cfg = SimToolRealCfg()
    assert cfg.num_actor_obs == compute_obs_dim(cfg.obs.obs_list) == 140
    assert cfg.num_critic_obs == compute_obs_dim(cfg.obs.state_list) == 162
    # Every actor field is also a critic field (asymmetric actor-critic).
    assert set(cfg.obs.obs_list) <= set(cfg.obs.state_list)


def test_validate_rejects_inconsistent_episode_length() -> None:
    """``episode_length`` must agree with ``max_episode_seconds / ctrl_dt``."""
    cfg = SimToolRealCfg()
    cfg.termination.episode_length = 599
    with pytest.raises(ValueError, match="episode_length"):
        cfg.validate()


def test_validate_rejects_wrong_substep_count() -> None:
    """``sim_substeps`` must stay at the source's ``decimation=2``."""
    cfg = SimToolRealCfg()
    cfg.sim_dt = 1.0 / 60.0
    with pytest.raises(ValueError, match="sim_substeps"):
        cfg.validate()


def _mjcf_joints(xml_path: Path) -> dict[str, ET.Element]:
    """Return a mapping of joint name to element for an MJCF file."""
    root = ET.parse(xml_path).getroot()
    return {
        str(joint.get("name")): joint
        for body in root.iter("body")
        for joint in body.findall("joint")
        if joint.get("name")
    }


def _mjcf_actuators(xml_path: Path) -> list[ET.Element]:
    """Return the position actuator elements of an MJCF file, in file order."""
    root = ET.parse(xml_path).getroot()
    actuator_root = root.find("actuator")
    return [] if actuator_root is None else list(actuator_root)


@requires_assets
def test_mjcf_actuator_gains_match_source_value_for_value() -> None:
    """All 29 actuators carry the source PD gains, in canonical order.

    Source: ``ARM_JOINT_STIFFNESS``/``ARM_JOINT_DAMPING`` (scene_utils.py:59,64)
    and ``HAND_JOINT_STIFFNESS``/``HAND_JOINT_DAMPING`` (:71,83). No sampling —
    every one of the 29 is checked.
    """
    actuators = _mjcf_actuators(ROBOT_XML)
    assert len(actuators) == NUM_JOINTS

    for index, name in enumerate(JOINT_NAMES_CANONICAL):
        actuator = actuators[index]
        assert actuator.tag == "position"
        assert actuator.get("joint") == name, f"actuator {index} drives the wrong joint"
        expected_kp = ARM_JOINT_STIFFNESS.get(name, HAND_JOINT_STIFFNESS.get(name))
        expected_kv = ARM_JOINT_DAMPING.get(name, HAND_JOINT_DAMPING.get(name))
        assert float(str(actuator.get("kp"))) == pytest.approx(expected_kp, rel=0, abs=0)
        assert float(str(actuator.get("kv"))) == pytest.approx(expected_kv, rel=0, abs=0)


@requires_assets
def test_mjcf_hand_armature_and_friction_match_source() -> None:
    """Hand joints carry armature+frictionloss; arm joints carry neither.

    Isaac Gym leaves arm armature unset on purpose ("Not setting armature matches
    real KUKA robot behavior", isaacgymenvs/tasks/simtoolreal/utils.py:100-101)
    and sets hand armature/friction at :217-218.
    """
    joints = _mjcf_joints(ROBOT_XML)

    for name in JOINT_NAMES_CANONICAL[:7]:
        assert float(str(joints[name].get("armature"))) == 0.0
        assert float(str(joints[name].get("frictionloss"))) == 0.0

    for name in JOINT_NAMES_CANONICAL[7:]:
        assert float(str(joints[name].get("armature"))) == pytest.approx(
            HAND_JOINT_ARMATURE[name], rel=0, abs=0
        )
        assert float(str(joints[name].get("frictionloss"))) == pytest.approx(
            HAND_JOINT_FRICTION[name], rel=0, abs=0
        )

    # Damping lives on the actuator kv, not on passive joint damping.
    for name in JOINT_NAMES_CANONICAL:
        assert float(str(joints[name].get("damping"))) == 0.0


@requires_assets
def test_fingertip_friction_applies_only_to_fingertip_links() -> None:
    """Sliding friction is 1.5 on fingertip links only, 0.5 on every other geom.

    Mirrors ``scene_utils.py:1584``, which assigns the fingertip material solely
    to ``FINGERTIP_LINK_NAMES``. The welded elastomer pads count as fingertip
    surfaces because Isaac merges fixed-joint children into their parent link.
    """
    root = ET.parse(ROBOT_XML).getroot()

    def welded_subtree(body: ET.Element) -> list[ET.Element]:
        collected = [body]
        for child in body.findall("body"):
            if child.find("joint") is None and child.find("freejoint") is None:
                collected.extend(welded_subtree(child))
        return collected

    bodies = {body.get("name"): body for body in root.iter("body")}
    fingertip_ids: set[int] = set()
    for link in FINGERTIP_LINK_NAMES:
        assert link in bodies, f"missing fingertip body {link}"
        for member in welded_subtree(bodies[link]):
            fingertip_ids.add(id(member))

    n_fingertip_geoms = 0
    n_other_geoms = 0
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue  # visual-only geom, deliberately untouched
            sliding = float(str(geom.get("friction")).split()[0])
            if id(body) in fingertip_ids:
                assert sliding == pytest.approx(1.5), (
                    f"fingertip geom {geom.get('name')} has friction {sliding}"
                )
                n_fingertip_geoms += 1
            else:
                assert sliding == pytest.approx(0.5), (
                    f"non-fingertip geom {geom.get('name')} on body "
                    f"{body.get('name')} has friction {sliding}"
                )
                n_other_geoms += 1

    assert n_fingertip_geoms >= len(FINGERTIP_LINK_NAMES)
    assert n_other_geoms > 0


@requires_assets
def test_mjcf_root_pose_is_fixed_base_at_source_pose() -> None:
    """Robot root sits at (0, 0.8, 0) with identity quat and no free joint."""
    root = ET.parse(ROBOT_XML).getroot()
    bodies = {body.get("name"): body for body in root.iter("body")}
    root_body = bodies["iiwa14_link_0"]
    assert [float(v) for v in str(root_body.get("pos")).split()] == pytest.approx([0.0, 0.8, 0.0])
    assert [float(v) for v in str(root_body.get("quat")).split()] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )
    assert root_body.find("freejoint") is None
    # disable_gravity=True on the articulation -> gravcomp on every robot body.
    assert root_body.get("gravcomp") == "1"


@requires_assets
def test_scene_compiles_with_expected_state_layout() -> None:
    """The scene compiles to 29 hinges + one object free joint, 29 actuators."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))

    assert model.nu == NUM_JOINTS
    assert model.nq == NUM_JOINTS + 7
    assert model.nv == NUM_JOINTS + 6

    # Robot hinges occupy qpos[0:29]; the object free joint is the tail.
    for name in JOINT_NAMES_CANONICAL:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"scene is missing joint {name}"
        assert int(model.jnt_qposadr[jid]) < NUM_JOINTS

    object_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
    assert int(model.jnt_type[object_jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
    assert int(model.jnt_qposadr[object_jid]) == NUM_JOINTS

    # Joint 0 must not be a free joint, or the backend would treat it as a
    # floating root and offset every dof view (mujoco/backend.py:46-49).
    assert int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE)

    for body_name in ("table", "object", "iiwa14_link_7"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0


@requires_assets
def test_scene_keyframe_holds_default_pose() -> None:
    """The ``home`` keyframe encodes the source default arm pose."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    qpos = np.asarray(model.key_qpos[key_id])

    for name, expected in DEFAULT_JOINT_POS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert qpos[model.jnt_qposadr[jid]] == pytest.approx(expected)


def _make_env(num_envs: int = 2) -> Any:
    """Build a MuJoCo-backed env, skipping the test if the backend is unusable.

    Returned as ``Any`` because ``@registry.env`` is typed
    ``Callable[[Type[ABEnv]], Type[ABEnv]]`` and therefore erases the concrete
    class; this is the same ``cast(Any, ...)`` convention the repo uses in
    ``tests/envs/test_env_configs.py``.

    Args:
        num_envs: Number of vectorized environments.

    Returns:
        A constructed :class:`SimToolRealEnv`.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_uni")
    from unilab.envs.manipulation.simtoolreal.env import SimToolRealEnv

    env_cls = cast(Any, SimToolRealEnv)
    return env_cls(SimToolRealCfg(), num_envs=num_envs, backend_type="mujoco")


@requires_assets
def test_joint_permutation_round_trips() -> None:
    """``perm_backend_to_canon[perm_canon_to_backend] == arange(29)``."""
    env = _make_env(num_envs=1)
    try:
        canon_to_backend = env._perm_canon_to_backend
        backend_to_canon = env._perm_backend_to_canon

        assert canon_to_backend.shape == (NUM_JOINTS,)
        assert backend_to_canon.shape == (NUM_JOINTS,)
        np.testing.assert_array_equal(backend_to_canon[canon_to_backend], np.arange(NUM_JOINTS))
        np.testing.assert_array_equal(canon_to_backend[backend_to_canon], np.arange(NUM_JOINTS))
        # Both are permutations of 0..28.
        np.testing.assert_array_equal(np.sort(canon_to_backend), np.arange(NUM_JOINTS))
    finally:
        env.close()


@requires_assets
def test_joint_limits_and_slices() -> None:
    """Canonical/backend limit views agree and the arm/hand slices are 7/22."""
    env = _make_env(num_envs=1)
    try:
        assert env._joint_lower_canon.shape == (NUM_JOINTS,)
        assert env._joint_upper_canon.shape == (NUM_JOINTS,)
        assert np.all(env._joint_lower_canon < env._joint_upper_canon)

        assert env._arm_lower.shape == env._arm_upper.shape == (7,)
        assert env._hand_lower.shape == env._hand_upper.shape == (22,)
        assert env._arm_slice == slice(0, 7)
        assert env._hand_slice == slice(7, 29)

        # Backend-order limits gathered into canonical order must match the
        # canonical view (they are the same data, two orderings).
        backend_lower = np.concatenate([env._arm_lower, env._hand_lower])
        np.testing.assert_allclose(
            backend_lower[env._perm_backend_to_canon], env._joint_lower_canon
        )

        # The default pose is inside the limits.
        assert np.all(env._default_joint_pos_canon >= env._joint_lower_canon)
        assert np.all(env._default_joint_pos_canon <= env._joint_upper_canon)
    finally:
        env.close()


@requires_assets
def test_state_info_contract_keys_shapes_and_dtypes() -> None:
    """``state.info`` carries every interface-contract §2 key, correctly typed."""
    num_envs = 3
    env = _make_env(num_envs=num_envs)
    try:
        state = env.init_state()
        info = state.info

        float_keys = {
            "prev_targets": (num_envs, NUM_JOINTS),
            "cur_targets": (num_envs, NUM_JOINTS),
            "last_actions": (num_envs, NUM_JOINTS),
            "current_actions": (num_envs, NUM_JOINTS),
            "goal_pos": (num_envs, 3),
            "goal_quat": (num_envs, 4),
            "object_init_z": (num_envs,),
            "closest_keypoint_max_dist": (num_envs,),
            "closest_fingertip_dist": (num_envs, 5),
            "prev_object_pos": (num_envs, 3),
            "prev_object_quat": (num_envs, 4),
            "object_scales": (num_envs, 3),
            "reward": (num_envs,),
        }
        for key, shape in float_keys.items():
            assert key in info, f"missing state.info key {key!r}"
            assert info[key].shape == shape, f"{key} shape {info[key].shape} != {shape}"
            assert info[key].dtype == np.float32, f"{key} dtype {info[key].dtype}"

        for key in ("successes", "near_goal_steps", "prev_episode_successes"):
            assert info[key].shape == (num_envs,)
            assert info[key].dtype == np.int32

        assert info["lifted_object"].shape == (num_envs,)
        assert info["lifted_object"].dtype == np.bool_
        assert isinstance(info["log"], dict)

        # d* sentinel is -1 on a fresh episode (migration guide §5).
        np.testing.assert_allclose(info["closest_keypoint_max_dist"], -1.0)
        np.testing.assert_allclose(info["closest_fingertip_dist"], -1.0)

        # prev_targets seeds from reset joint positions (with noise), not zeros (contract §2.1).
        # After T4 completion, this includes the reset noise from build_reset_plan.
        assert np.any(info["prev_targets"] != 0.0)
        # Check that prev_targets is within joint limits (sanity check).
        # prev_targets is in backend order, so use backend limits.
        lower_backend = env._joint_lower_canon[env._perm_canon_to_backend]
        upper_backend = env._joint_upper_canon[env._perm_canon_to_backend]
        assert np.all(info["prev_targets"] >= lower_backend[None, :])
        assert np.all(info["prev_targets"] <= upper_backend[None, :])

        # Trackers start cleared.
        np.testing.assert_array_equal(info["successes"], np.zeros(num_envs, dtype=np.int32))
        np.testing.assert_array_equal(info["lifted_object"], np.zeros(num_envs, dtype=bool))
    finally:
        env.close()


@requires_assets
def test_instance_attribute_contract() -> None:
    """Contract §3 instance attributes exist with the documented shapes."""
    num_envs = 2
    env = _make_env(num_envs=num_envs)
    try:
        cfg = env.cfg
        assert env._palm_offset.shape == (3,)
        np.testing.assert_allclose(env._palm_offset, [-0.0, -0.02, 0.16])
        assert env._fingertip_offset.shape == (3,)
        np.testing.assert_allclose(env._fingertip_offset, [0.02, 0.002, 0.0])

        assert env._kp_corners.shape == (4, 3)
        assert env._keypoint_offsets_fixed.shape == (4, 3)
        # corners * 0.5 * keypoint_scale * fixed_size (reset_utils.py:80-83).
        expected = np.asarray(env._kp_corners) * (
            0.5 * cfg.goal.keypoint_scale * np.asarray(cfg.reward.fixed_size)
        )
        np.testing.assert_allclose(env._keypoint_offsets_fixed, expected, rtol=1e-6)

        assert env._action_queue.shape == (num_envs, 3, NUM_JOINTS)
        assert env._obs_queue.shape == (num_envs, 3, cfg.num_actor_obs)
        assert env._object_state_queue.shape == (num_envs, 10, 13)
        assert env._object_forces.shape == (num_envs, 3)
        assert env._object_torques.shape == (num_envs, 3)

        assert env._fingertip_body_ids.shape == (5,)
        assert isinstance(env._palm_body_id, int)
        assert isinstance(env._object_body_id, int)

        assert env.obs_groups_spec == {
            "obs": cfg.num_actor_obs,
            "critic": cfg.num_critic_obs,
        }
        assert env.action_space.shape == (NUM_JOINTS,)
    finally:
        env.close()


@requires_assets
def test_step_zeros_runs_without_nan() -> None:
    """``env.step(zeros)`` survives 10 steps with finite obs/reward.

    The T0 stubs mean control is all zeros and ``update_state`` is a pass-through,
    so this only checks that the scene, actuators, and reset path are wired up.
    """
    num_envs = 2
    env = _make_env(num_envs=num_envs)
    try:
        state = env.init_state()
        actions = np.zeros((num_envs, NUM_JOINTS), dtype=np.float32)

        for step_index in range(10):
            state = env.step(actions)
            for group, values in state.obs.items():
                assert np.all(np.isfinite(values)), f"non-finite {group} at step {step_index}"
            assert np.all(np.isfinite(state.reward)), f"non-finite reward at {step_index}"

        # apply_action is a documented zero stub for T0.
        np.testing.assert_array_equal(
            env.apply_action(actions, state), np.zeros((num_envs, NUM_JOINTS))
        )
        # update_state is a documented pass-through for T0.
        assert env.update_state(state) is state
    finally:
        env.close()


@requires_assets
def test_object_scale_is_handle_bbox_normalized() -> None:
    """``phi`` is the handle bbox over ``object_base_size``, head excluded."""
    env = _make_env(num_envs=1)
    try:
        phi = env.resolve_object_scale()
        assert phi.shape == (3,)
        assert np.all(phi > 0.0)

        from unilab.tools.build_simtoolreal_assets import HAMMER_HANDLE_SIZE

        expected = np.asarray(HAMMER_HANDLE_SIZE) / env.cfg.reward.object_base_size
        np.testing.assert_allclose(phi, expected, rtol=1e-5)
    finally:
        env.close()
