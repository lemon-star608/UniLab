"""Backend-neutral SimToolReal task foundations.

This Code #7 package intentionally has no environment owner or registry import.
The real NpEnv composition is a later child batch.
"""

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
    "TerminationCfg",
]
