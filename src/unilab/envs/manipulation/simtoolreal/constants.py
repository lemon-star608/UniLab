"""SimToolReal embodiment constants, ported verbatim from the Isaac source.

Every number here is copied from the SimToolReal repository. Do not round or
re-derive any of them: the pretrained Isaac Gym checkpoints depend on the exact
values. Source locations, relative to the SimToolReal repo root:

    JOINT_NAMES_CANONICAL   isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:36
    PALM_BODY_NAME          isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:49
    FINGERTIP_LINK_NAMES    isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:52
    ARM_JOINT_STIFFNESS     isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:59
    ARM_JOINT_DAMPING       isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:64
    HAND_JOINT_STIFFNESS    isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:71
    HAND_JOINT_DAMPING      isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:83
    HAND_JOINT_ARMATURE     isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:97
    HAND_JOINT_FRICTION     isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:109
    ARM_DEFAULT_JOINT_POS   isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:127
    ROBOT_ROOT_POS          isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:163
    PALM_CENTER_OFFSET      isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py:22
    FINGERTIP_OFFSET        isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py:25
    KEYPOINT_CORNERS        isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py:28
    OBS_FIELD_SIZES         isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py:34

The arm deliberately gets no armature and no joint friction: Isaac Gym leaves
both unset ("Not setting armature matches real KUKA robot behavior",
isaacgymenvs/tasks/simtoolreal/utils.py:100-101), and only the hand receives
armature/friction (same file, :217-218).
"""

from __future__ import annotations

NUM_JOINTS: int = 29
NUM_ARM_JOINTS: int = 7
NUM_HAND_JOINTS: int = 22
NUM_FINGERTIPS: int = 5
NUM_KEYPOINTS: int = 4

# Policy-facing canonical order. Backend tensors are permuted at the action/obs
# boundary so this order remains stable across simulator implementations.
JOINT_NAMES_CANONICAL: tuple[str, ...] = (
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)
assert len(JOINT_NAMES_CANONICAL) == NUM_JOINTS

ARM_JOINT_NAMES: tuple[str, ...] = JOINT_NAMES_CANONICAL[:NUM_ARM_JOINTS]
HAND_JOINT_NAMES: tuple[str, ...] = JOINT_NAMES_CANONICAL[NUM_ARM_JOINTS:]

PALM_BODY_NAME: str = "iiwa14_link_7"
OBJECT_BODY_NAME: str = "object"
TABLE_BODY_NAME: str = "table"
ROBOT_ROOT_BODY_NAME: str = "iiwa14_link_0"

# Merged fingertip bodies land on the DP links in both sims.
FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
)
assert len(FINGERTIP_LINK_NAMES) == NUM_FINGERTIPS

# Robot root pose: fixed base, identity orientation (scene_utils.py:163-164).
ROBOT_ROOT_POS: tuple[float, float, float] = (0.0, 0.8, 0.0)
ROBOT_ROOT_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Per-joint PD gains and dynamics (verified against the pretrained checkpoint).
ARM_JOINT_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0,
    "iiwa14_joint_2": 600.0,
    "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0,
    "iiwa14_joint_5": 200.0,
    "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}
ARM_JOINT_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512,
    "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083,
    "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853,
    "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}

HAND_JOINT_STIFFNESS: dict[str, float] = {
    "left_1_thumb_CMC_FE": 6.95,
    "left_thumb_CMC_AA": 13.2,
    "left_thumb_MCP_FE": 4.76,
    "left_thumb_MCP_AA": 6.62,
    "left_thumb_IP": 0.9,
    "left_2_index_MCP_FE": 4.76,
    "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9,
    "left_index_DIP": 0.9,
    "left_3_middle_MCP_FE": 4.76,
    "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9,
    "left_middle_DIP": 0.9,
    "left_4_ring_MCP_FE": 4.76,
    "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9,
    "left_ring_DIP": 0.9,
    "left_5_pinky_CMC": 1.38,
    "left_pinky_MCP_FE": 4.76,
    "left_pinky_MCP_AA": 6.62,
    "left_pinky_PIP": 0.9,
    "left_pinky_DIP": 0.9,
}
HAND_JOINT_DAMPING: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.28676845,
    "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083,
    "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    "left_2_index_MCP_FE": 0.20859232,
    "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185,
    "left_index_DIP": 0.03504461,
    "left_3_middle_MCP_FE": 0.2085923,
    "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185,
    "left_middle_DIP": 0.03504461,
    "left_4_ring_MCP_FE": 0.20859226,
    "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183,
    "left_ring_DIP": 0.0350446,
    "left_5_pinky_CMC": 0.02782345,
    "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528,
    "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}
HAND_JOINT_ARMATURE: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.0032,
    "left_thumb_CMC_AA": 0.0032,
    "left_thumb_MCP_FE": 0.00265,
    "left_thumb_MCP_AA": 0.00265,
    "left_thumb_IP": 0.0006,
    "left_2_index_MCP_FE": 0.00265,
    "left_index_MCP_AA": 0.00265,
    "left_index_PIP": 0.0006,
    "left_index_DIP": 0.00042,
    "left_3_middle_MCP_FE": 0.00265,
    "left_middle_MCP_AA": 0.00265,
    "left_middle_PIP": 0.0006,
    "left_middle_DIP": 0.00042,
    "left_4_ring_MCP_FE": 0.00265,
    "left_ring_MCP_AA": 0.00265,
    "left_ring_PIP": 0.0006,
    "left_ring_DIP": 0.00042,
    "left_5_pinky_CMC": 0.00012,
    "left_pinky_MCP_FE": 0.00265,
    "left_pinky_MCP_AA": 0.00265,
    "left_pinky_PIP": 0.0006,
    "left_pinky_DIP": 0.00042,
}
HAND_JOINT_FRICTION: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.132,
    "left_thumb_CMC_AA": 0.132,
    "left_thumb_MCP_FE": 0.07456,
    "left_thumb_MCP_AA": 0.07456,
    "left_thumb_IP": 0.01276,
    "left_2_index_MCP_FE": 0.07456,
    "left_index_MCP_AA": 0.07456,
    "left_index_PIP": 0.01276,
    "left_index_DIP": 0.00378738,
    "left_3_middle_MCP_FE": 0.07456,
    "left_middle_MCP_AA": 0.07456,
    "left_middle_PIP": 0.01276,
    "left_middle_DIP": 0.00378738,
    "left_4_ring_MCP_FE": 0.07456,
    "left_ring_MCP_AA": 0.07456,
    "left_ring_PIP": 0.01276,
    "left_ring_DIP": 0.00378738,
    "left_5_pinky_CMC": 0.012,
    "left_pinky_MCP_FE": 0.07456,
    "left_pinky_MCP_AA": 0.07456,
    "left_pinky_PIP": 0.01276,
    "left_pinky_DIP": 0.00378738,
}

assert len(ARM_JOINT_STIFFNESS) == len(ARM_JOINT_DAMPING) == NUM_ARM_JOINTS
assert len(HAND_JOINT_STIFFNESS) == len(HAND_JOINT_DAMPING) == NUM_HAND_JOINTS
assert len(HAND_JOINT_ARMATURE) == len(HAND_JOINT_FRICTION) == NUM_HAND_JOINTS
# Proven-working default arm pose (scene_utils.py:127). Hand joints default to 0
# (scene_utils.py:167 sets every hand joint to 0.0).
ARM_DEFAULT_JOINT_POS: dict[str, float] = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}
assert len(ARM_DEFAULT_JOINT_POS) == NUM_ARM_JOINTS

DEFAULT_JOINT_POS: dict[str, float] = {
    **ARM_DEFAULT_JOINT_POS,
    **{name: 0.0 for name in HAND_JOINT_NAMES},
}
assert len(DEFAULT_JOINT_POS) == NUM_JOINTS

# Policy was trained against the palm center, not the raw wrist body.
PALM_CENTER_OFFSET: tuple[float, float, float] = (-0.0, -0.02, 0.16)
# Shift fingertip body origins to the approximate pad centers.
FINGERTIP_OFFSET: tuple[float, float, float] = (0.02, 0.002, 0.0)

# Object-frame keypoint corners before scaling.
KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)
assert len(KEYPOINT_CORNERS) == NUM_KEYPOINTS

# Per-field observation widths. N_ACTOR / N_CRITIC are summed from the ObsCfg
# field lists rather than hardcoded (interface contract §1).
OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}


def compute_obs_dim(field_list: tuple[str, ...] | list[str]) -> int:
    """Return the total flat width of an ordered list of observation fields.

    Args:
        field_list: Ordered observation field names, as configured on ``ObsCfg``.

    Returns:
        Sum of each field's width from :data:`OBS_FIELD_SIZES`.
    """
    return sum(OBS_FIELD_SIZES[name] for name in field_list)
