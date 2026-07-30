"""T7 acceptance tests: the integrated ``apply_action`` / ``update_state`` chain.

These are the only SimToolReal tests that need a real MuJoCo rollout. They check
that the T1-T6 modules, wired together in the source's call order
(simtoolreal_env.py:58-77), actually behave as the source does:

* a 600-step random-action rollout stays finite, with stable obs widths,
* the ``d*`` sentinel resolves on the first step and then never increases,
* the lifting progress reward is positive while the object is still settling,
* a success advances the goal and zeroes ``steps`` **without** terminating (D2),
* wrench DR fires only once the lifted latch is set, reading the previous step's
  value (action_utils.py:109),
* the reward-term dict and ``info["log"]`` are published every step.

Skipped when the generated MJCF assets or the MuJoCo backend are unavailable.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg
from unilab.envs.manipulation.simtoolreal.constants import (
    NUM_ARM_JOINTS,
    NUM_FINGERTIPS,
    NUM_JOINTS,
)

SCENE_XML = ASSETS_ROOT_PATH / "robots" / "kuka_sharpa" / "scene.xml"

requires_assets = pytest.mark.skipif(
    not SCENE_XML.is_file(),
    reason="run `python -m unilab.tools.build_simtoolreal_assets` to generate the MJCF assets",
)

REWARD_TERM_KEYS = (
    "fingertip_delta_rew",
    "lifting_rew",
    "lift_bonus_rew",
    "keypoint_rew",
    "kuka_actions_penalty",
    "hand_actions_penalty",
    "bonus_rew",
    "total_reward",
)


def _make_env(num_envs: int = 2, cfg: SimToolRealCfg | None = None) -> Any:
    """Build a real MuJoCo-backed env.

    Returned as ``Any`` because ``@registry.env`` erases the concrete class; same
    ``cast(Any, ...)`` convention as ``tests/envs/test_simtoolreal.py``.

    Args:
        num_envs: Number of vectorized environments.
        cfg: Optional config override. Defaults to stock :class:`SimToolRealCfg`.

    Returns:
        A constructed ``SimToolRealEnv``.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_uni")
    from unilab.envs.manipulation.simtoolreal.env import SimToolRealEnv

    env_cls = cast(Any, SimToolRealEnv)
    return env_cls(cfg or SimToolRealCfg(), num_envs=num_envs, backend_type="mujoco")


def _random_actions(rng: np.random.Generator, num_envs: int) -> np.ndarray:
    """Return uniform ``[-1, 1]`` policy actions of shape ``(num_envs, 29)``."""
    return rng.uniform(-1.0, 1.0, size=(num_envs, NUM_JOINTS)).astype(np.float32)


@requires_assets
def test_random_rollout_600_steps_stays_finite() -> None:
    """600 random-action steps: no NaN, stable obs widths, obs inside the clamp."""
    num_envs = 2
    steps = 600
    np.random.seed(0)
    rng = np.random.default_rng(0)

    env = _make_env(num_envs=num_envs)
    try:
        cfg = env.cfg
        clip = cfg.obs.clamp_abs_observations
        expected = {"obs": cfg.num_actor_obs, "critic": cfg.num_critic_obs}
        assert env.obs_groups_spec == expected
        # Widths are summed from the ObsCfg lists (contract §1); the source's
        # field table makes those 140 / 162 for the stock lists.
        assert expected == {"obs": 140, "critic": 162}

        state = env.init_state()
        for step_index in range(steps):
            state = env.step(_random_actions(rng, num_envs))

            for group, width in expected.items():
                values = state.obs[group]
                assert values.shape == (num_envs, width), f"{group} width drifted at {step_index}"
                assert np.all(np.isfinite(values)), f"non-finite {group} at step {step_index}"
                assert np.all(np.abs(values) <= clip + 1e-5), f"{group} exceeds clamp"

            assert np.all(np.isfinite(state.reward)), f"non-finite reward at step {step_index}"
            assert state.reward.dtype == np.float32

            # Every reward term is published, finite, and correctly signed.
            assert set(env._reward_terms) == set(REWARD_TERM_KEYS)
            for name, term in env._reward_terms.items():
                assert term.shape == (num_envs,), f"{name} shape at step {step_index}"
                assert np.all(np.isfinite(term)), f"non-finite {name} at step {step_index}"
            assert np.all(env._reward_terms["kuka_actions_penalty"] <= 0.0)
            assert np.all(env._reward_terms["hand_actions_penalty"] <= 0.0)
            assert np.all(env._reward_terms["fingertip_delta_rew"] >= 0.0)
            assert np.all(env._reward_terms["keypoint_rew"] >= 0.0)

            assert isinstance(state.info["log"], dict)
            assert np.all(np.isfinite(list(state.info["log"].values())))
    finally:
        env.close()


@requires_assets
def test_d_star_sentinel_resolves_then_never_increases() -> None:
    """``d*`` is ``-1`` before the first step, then real and monotone.

    The sentinel resolution lives in ``_compute_intermediate_values``
    (obs_utils.py:182-190). Without it every delta would be negative, clip to
    zero, and both progress rewards would stay dead for the whole run.
    """
    num_envs = 2
    np.random.seed(1)
    rng = np.random.default_rng(1)

    env = _make_env(num_envs=num_envs)
    try:
        state = env.init_state()
        assert np.all(state.info["closest_keypoint_max_dist"] < 0.0)
        assert np.all(state.info["closest_fingertip_dist"] < 0.0)

        state = env.step(_random_actions(rng, num_envs))
        kp_star = state.info["closest_keypoint_max_dist"]
        ft_star = state.info["closest_fingertip_dist"]
        assert kp_star.shape == (num_envs,)
        assert ft_star.shape == (num_envs, NUM_FINGERTIPS)
        assert np.all(kp_star >= 0.0), "keypoint d* sentinel never resolved"
        assert np.all(ft_star >= 0.0), "fingertip d* sentinel never resolved"

        prev_kp = kp_star.copy()
        prev_ft = ft_star.copy()
        for _ in range(50):
            state = env.step(_random_actions(rng, num_envs))
            cur_kp = state.info["closest_keypoint_max_dist"]
            cur_ft = state.info["closest_fingertip_dist"]
            assert np.all(cur_kp <= prev_kp + 1e-6), "d* keypoint increased"
            assert np.all(cur_ft <= prev_ft + 1e-6), "d* fingertip increased"
            prev_kp = cur_kp.copy()
            prev_ft = cur_ft.copy()
    finally:
        env.close()


@requires_assets
def test_lifting_reward_positive_while_object_settles() -> None:
    """``lifting_rew`` is positive until the object drops below ``init_z - 0.05``.

    The object spawns 10 cm above the table surface (``table_reset_z +
    table_object_z_offset`` = 0.63 vs a surface at 0.53) with a random
    orientation, so it free-falls and settles — source behaviour, reset_utils.py
    :287-289. ``z_lift = 0.05 + z - init_z`` (reward_utils.py:17) therefore starts
    at exactly 0.05 and decays to zero over roughly
    ``sqrt(2*0.05/9.81) / ctrl_dt`` = 6 steps, plus settling.
    """
    num_envs = 2
    np.random.seed(0)
    rng = np.random.default_rng(0)

    env = _make_env(num_envs=num_envs)
    try:
        scale = env.cfg.reward.lifting_rew_scale
        state = env.init_state()
        init_z = state.info["object_init_z"].copy()

        state = env.step(_random_actions(rng, num_envs))
        first = env._reward_terms["lifting_rew"]
        # First step is still ~5 cm above the settling threshold: z_lift ~= 0.05,
        # so lift_rew ~= 0.05 * 20 = 1.0.
        assert np.all(first > 0.0), "lifting_rew dead on the first step"
        assert np.all(first <= 0.5 * scale + 1e-4), "lifting_rew above its 0.5 clip"
        assert np.all(~state.info["lifted_object"]), "latched lifted during free-fall"

        positive_steps = int(np.all(first > 0.0))
        for _ in range(29):
            state = env.step(_random_actions(rng, num_envs))
            positive_steps += int(np.any(env._reward_terms["lifting_rew"] > 0.0))

        # The window is short but must exist; random actions never lift 15 cm, so
        # the object stays unlatched and the keypoint reward stays gated off.
        assert 3 <= positive_steps <= 20, f"settling window was {positive_steps} steps"
        assert np.all(env._object_pos[:, 2] < init_z), "object never settled downward"
        assert np.all(env._reward_terms["keypoint_rew"] == 0.0), "keypoint reward not lift-gated"
        assert np.all(env._reward_terms["lift_bonus_rew"] == 0.0)
    finally:
        env.close()


@requires_assets
def test_success_advances_goal_without_terminating() -> None:
    """A success resamples the goal, zeroes ``steps``, and leaves ``terminated`` False.

    This is decision D2 end to end. The gate is
    ``current_success_tolerance * keypoint_scale`` (obs_utils.py:195), so a loose
    tolerance makes ``near_goal`` true immediately and the success fires on the
    tenth consecutive near step (``success_steps=10``, 0-indexed step 9).
    """
    num_envs = 2
    np.random.seed(2)
    rng = np.random.default_rng(2)

    cfg = SimToolRealCfg()
    # Deliberately far looser than the 0.075 default: observed keypoint distances
    # under random actions are ~0.45 m, so the gate has to clear that.
    cfg.goal.success_tolerance = 5.0
    env = _make_env(num_envs=num_envs, cfg=cfg)
    try:
        success_steps = cfg.goal.success_steps
        state = env.init_state()
        goal_before = state.info["goal_pos"].copy()

        for step_index in range(success_steps - 1):
            state = env.step(_random_actions(rng, num_envs))
            assert np.all(env._near_goal), "loose tolerance did not put envs near goal"
            assert np.all(state.info["successes"] == 0), f"early success at {step_index}"
            assert np.allclose(state.info["goal_pos"], goal_before), "goal moved early"
            # Amortized bonus = 1000 / 10 = 100 per near-goal step (:89).
            expected_bonus = cfg.reward.reach_goal_bonus / success_steps
            np.testing.assert_allclose(env._reward_terms["bonus_rew"], expected_bonus, rtol=1e-5)

        state = env.step(_random_actions(rng, num_envs))

        assert np.all(state.info["successes"] == 1), "success did not register"
        assert not np.any(state.terminated), "D2 violated: success terminated the episode"
        assert not np.any(state.truncated), "D2 violated: success truncated the episode"
        # advance_goal_on_success zeroed steps, then the base class incremented it.
        np.testing.assert_array_equal(state.info["steps"], np.ones(num_envs, dtype=np.uint32))
        assert not np.allclose(state.info["goal_pos"], goal_before), "goal was not resampled"
        assert np.all(state.info["near_goal_steps"] == 0), "near_goal_steps not cleared"
        # d* is re-seeded to the sentinel for the new goal, then resolved next step.
        assert np.all(state.info["closest_keypoint_max_dist"] < 0.0)
    finally:
        env.close()


@requires_assets
def test_wrench_dr_gated_on_previous_step_lifted_latch() -> None:
    """Wrench DR stays zero until lifted, and reads the previous step's latch.

    ``apply_action`` runs before ``update_state``, so the gate must read the cache
    snapshotted at the end of the last step — the source makes the same point
    ("``_lifted_object`` is from the previous step because rewards update later",
    action_utils.py:109).
    """
    num_envs = 2
    np.random.seed(3)
    rng = np.random.default_rng(3)

    cfg = SimToolRealCfg()
    cfg.domain_randomization.force_prob_range = (1.0, 1.0)  # always fire
    cfg.domain_randomization.torque_prob_range = (1.0, 1.0)
    env = _make_env(num_envs=num_envs, cfg=cfg)
    try:
        state = env.init_state()
        env.step(_random_actions(rng, num_envs))

        # Random actions never lift the object, so the gate keeps the wrench at 0
        # even though both trigger probabilities are 1.0.
        assert not np.any(state.info["lifted_object"])
        assert np.all(env._object_forces == 0.0), "wrench fired before lift"
        assert np.all(env._object_torques == 0.0), "torque fired before lift"

        # Latch it and re-run only apply_action: the wrench must now be live.
        env._state_cache_lifted_object[:] = True
        env.apply_action(_random_actions(rng, num_envs), state)
        assert np.any(env._object_forces != 0.0), "wrench did not fire once lifted"
        assert np.any(env._object_torques != 0.0), "torque did not fire once lifted"

        # Magnitude scales with object mass (force ~ randn * mass * 20).
        assert env._object_mass.shape == (num_envs,)
        assert np.all(env._object_mass > 0.0)
        sigma = float(env._object_mass[0]) * cfg.domain_randomization.force_scale
        assert np.all(np.abs(env._object_forces) < 12.0 * sigma), "force wildly off scale"
    finally:
        env.close()


@requires_assets
def test_arm_hand_slices_align_with_backend_order() -> None:
    """The arm occupies backend joint slots 0-6, which the action/penalty split assumes.

    ``apply_action_pipeline`` indexes backend-order targets with ``_arm_slice`` /
    ``_hand_slice`` (action_pipeline.py:179-180), and ``action_penalty`` applies
    the same slices to canonical-order joint velocities (rewards.py:311-316).
    Both are only correct if the arm block maps onto the first seven backend
    slots. T7's rollout depends on it, so it is asserted rather than assumed.
    """
    env = _make_env(num_envs=1)
    try:
        arm_backend_slots = np.sort(env._perm_canon_to_backend[:NUM_ARM_JOINTS])
        np.testing.assert_array_equal(arm_backend_slots, np.arange(NUM_ARM_JOINTS))
        assert env._arm_slice == slice(0, NUM_ARM_JOINTS)
        assert env._hand_slice == slice(NUM_ARM_JOINTS, NUM_JOINTS)
    finally:
        env.close()
