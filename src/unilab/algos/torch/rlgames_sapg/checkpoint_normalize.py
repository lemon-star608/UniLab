"""Explicit cold-path normalisation for native SAPG checkpoints."""

from __future__ import annotations

import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointReport:
    path: Path
    actor_input_dim: int | None
    action_dim: int | None
    numpy_metadata_found: bool


def _metadata(value: Any) -> tuple[Any, bool]:
    if isinstance(value, np.generic):
        return value.item(), True
    if isinstance(value, np.ndarray):
        return value.tolist(), True
    # OrderedDict is a dict subclass; handle it first so checkpoint key order
    # and the concrete mapping type are preserved byte-for-byte where possible.
    if isinstance(value, OrderedDict):
        converted_ordered: Any = OrderedDict()
        found = False
        for key, item in value.items():
            converted_item, item_found = _metadata(item)
            converted_ordered[key] = converted_item
            found |= item_found
        return converted_ordered, found
    if isinstance(value, dict):
        converted_dict: Any = {}
        found = False
        for key, item in value.items():
            converted_item, item_found = _metadata(item)
            converted_dict[key] = converted_item
            found |= item_found
        return converted_dict, found
    if isinstance(value, list):
        converted_list: Any = []
        found = False
        for item in value:
            converted_item, item_found = _metadata(item)
            converted_list.append(converted_item)
            found |= item_found
        return converted_list, found
    if isinstance(value, tuple):
        converted_tuple: Any = []
        found = False
        for item in value:
            converted_item, item_found = _metadata(item)
            converted_tuple.append(converted_item)
            found |= item_found
        return tuple(converted_tuple), found
    return value, False


def normalize_checkpoint(src: str | Path, dst: str | Path) -> Path:
    """Write a normalised copy without mutating the original checkpoint."""
    source = Path(src).expanduser().resolve()
    target = Path(dst).expanduser().resolve()
    if source == target:
        raise ValueError("checkpoint normalisation requires a different destination")
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"checkpoint is not a regular file: {source}")
    if target.exists():
        raise FileExistsError(f"normalised checkpoint destination already exists: {target}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"checkpoint NumPy serialization is incompatible: {source}; "
            f"export a compatible copy with normalize_checkpoint(...): {exc}"
        ) from exc
    # Model tensors are never traversed or converted; only rank metadata and
    # auxiliary NumPy values are normalised.
    if isinstance(payload, dict):
        converted: dict[Any, Any] = {}
        found = False
        for key, value in payload.items():
            if key == 0 and isinstance(value, dict) and isinstance(value.get("model"), dict):
                state = dict(value)
                extras = {k: v for k, v in state.items() if k != "model"}
                extras, extra_found = _metadata(extras)
                state.update(extras)
                converted[key] = state
                found |= extra_found
            else:
                item, item_found = _metadata(value)
                converted[key] = item
                found |= item_found
        payload = converted
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def _model_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get(0), dict):
        raise ValueError("native checkpoint must contain rank-0 state")
    model = payload[0].get("model")
    if not isinstance(model, dict):
        raise ValueError("native checkpoint rank-0 state must contain model weights")
    return model


def preflight_checkpoint(path: str | Path, *, expected_actor_dim: int = 140) -> CheckpointReport:
    """Inspect actor/observation dimensions before constructing the worker env."""
    candidate = Path(path).expanduser().resolve()
    try:
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"unable to load checkpoint {candidate}; run normalize_checkpoint(src, dst): {exc}"
        ) from exc
    model = _model_mapping(payload)
    _, numpy_found = _metadata(model)
    actor_dim: int | None = None
    for key in ("running_mean_std.running_mean", "obs_rms.mean", "running_mean"):
        value = model.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 1:
            actor_dim = int(value.shape[0])
            break
    # Native RL-Games checkpoints generally do not persist running statistics;
    # infer the actor input width from the first actor MLP linear layer instead.
    if actor_dim is None:
        actor_weights = [
            value
            for key, value in model.items()
            if "critic" not in str(key).lower()
            and "value" not in str(key).lower()
            and "actor" in str(key).lower()
            and str(key).endswith("weight")
            and isinstance(value, torch.Tensor)
            and value.ndim == 2
        ]
        if actor_weights:
            actor_dim = int(actor_weights[0].shape[1])
    action_dim: int | None = None
    for key in ("a2c_network.sigma", "a2c_network.mu.weight", "mu.weight"):
        value = model.get(key)
        if isinstance(value, torch.Tensor):
            # SAPG stores sigma as [exploration_blocks, action_dim], while mu
            # is [action_dim, hidden].  Prefer the dimension matching the
            # declared 29-action contract when available.
            if key.endswith("sigma") and value.ndim == 2:
                action_dim = int(value.shape[-1])
            else:
                action_dim = int(value.shape[0] if value.ndim > 1 else value.shape[-1])
            break
    if actor_dim is not None and actor_dim != expected_actor_dim:
        raise ValueError(
            f"checkpoint actor input dimension mismatch: expected {expected_actor_dim}, got {actor_dim} ({candidate})"
        )
    return CheckpointReport(
        path=candidate,
        actor_input_dim=actor_dim,
        action_dim=action_dim,
        numpy_metadata_found=numpy_found,
    )


__all__ = ["CheckpointReport", "normalize_checkpoint", "preflight_checkpoint"]
