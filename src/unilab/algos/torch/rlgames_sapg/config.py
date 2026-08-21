"""Cold-path validation for the Source-native SAPG Hydra owner."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def _value(cfg: DictConfig, path: str):
    value = OmegaConf.select(cfg, path)
    if value is None:
        raise ValueError(f"missing required RL-Games SAPG config field: {path}")
    return value


def preflight_config(cfg: DictConfig) -> None:
    """Reject unsupported execution/config modes before environment creation."""
    if str(_value(cfg, "training.task_name")) != "SimToolReal":
        raise ValueError("RL-Games SAPG supports only the SimToolReal task")
    if str(_value(cfg, "training.sim_backend")) != "mujoco":
        raise ValueError("RL-Games SAPG Code #9 supports only the MuJoCo backend")
    if str(_value(cfg, "algo.algo")) != "rlgames_sapg":
        raise ValueError("wrong native owner: algo.algo must be rlgames_sapg")
    if int(_value(cfg, "env.action_space")) != 29:
        raise ValueError("SimToolReal action dimension must be 29")

    params = cfg.rl_games.params
    native = params.config
    actors = int(_value(cfg, "algo.num_envs"))
    native_actors = int(native.num_actors)
    block = int(native.expl_coef_block_size)
    if (
        actors != native_actors
        or block <= 0
        or native_actors % block
        or native_actors // block != 6
    ):
        raise ValueError("num_actors must form exactly six exploration blocks")
    horizon = int(native.horizon_length)
    sequence = int(native.seq_length)
    if sequence <= 0 or horizon % sequence:
        raise ValueError("horizon_length must be divisible by seq_length")
    rows = native_actors * horizon
    for path, size in (
        ("minibatch_size", int(native.minibatch_size)),
        ("central_value_config.minibatch_size", int(native.central_value_config.minibatch_size)),
    ):
        if size <= 0 or rows % size:
            raise ValueError(f"{path} must divide num_actors * horizon_length")
    if bool(native.multi_gpu):
        raise ValueError("multi_gpu is outside Code #9")
    if str(native.env_name) != "rlgpu" or str(params.algo.name) != "a2c_continuous":
        raise ValueError("wrong native RL-Games owner")
    if bool(params.load_checkpoint):
        raise ValueError("native load_checkpoint must remain false; use checkpoint_load_mode")
    if str(params.load_path):
        raise ValueError("native load_path must remain empty; use the trusted .pth resolver")
    mode = str(_value(cfg, "algo.checkpoint_load_mode"))
    if mode not in {"none", "resume", "weights"}:
        raise ValueError("checkpoint_load_mode must be none, resume, or weights")
