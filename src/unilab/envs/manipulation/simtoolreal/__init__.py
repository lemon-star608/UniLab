"""Registered SimToolReal MuJoCo task and backend-neutral task foundations."""

# Registration order is semantic: the config owner must exist before env import.
# isort: off
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
from . import env  # registers SimToolReal after its config owner
from .env import SimToolRealEnv
# isort: on

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
