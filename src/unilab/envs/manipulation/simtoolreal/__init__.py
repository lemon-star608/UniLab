"""SimToolReal goal-pose-reaching task (KUKA iiwa14 + Sharpa left hand).

Migrated from the SimToolReal Isaac Sim / Isaac Gym implementation. Task T0
provides the env skeleton, config schema, joint permutations, MJCF assets, and
backend wiring; the action, observation, reward, goal, and DR pipelines land in
later tasks.
"""

from . import env  # registers SimToolReal via @registry decorators
from .config import (
    ActionCfg,
    AssetsCfg,
    DomainRandomizationCfg,
    GoalCfg,
    ObsCfg,
    ResetCfg,
    RewardCfg,
    SimToolRealCfg,
    TerminationCfg,
)
from .dr_provider import SimToolRealDRProvider
from .env import SimToolRealEnv

__all__ = [
    "ActionCfg",
    "AssetsCfg",
    "DomainRandomizationCfg",
    "GoalCfg",
    "ObsCfg",
    "ResetCfg",
    "RewardCfg",
    "SimToolRealCfg",
    "SimToolRealDRProvider",
    "SimToolRealEnv",
    "TerminationCfg",
    "env",
]
