from __future__ import annotations

import json
import subprocess
import sys

from unilab.base import registry


def test_clean_registry_bootstrap_registers_only_mujoco_owner() -> None:
    script = """
import json
from unilab.base import registry

registry.ensure_registries()
print(json.dumps(registry.list_registered_envs()["SimToolReal"], sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    registered = json.loads(result.stdout)
    assert registered == {
        "available_backends": ["mujoco"],
        "config_class": "SimToolRealCfg",
    }


def test_package_exports_registered_config_and_env_owner() -> None:
    registry.ensure_registries()
    from unilab.envs.manipulation import simtoolreal

    registered = registry.list_registered_envs()["SimToolReal"]
    assert registered["config_class"] == simtoolreal.SimToolRealCfg.__name__
    assert registered["available_backends"] == ["mujoco"]
    assert simtoolreal.SimToolRealEnv.__module__ == ("unilab.envs.manipulation.simtoolreal.env")
