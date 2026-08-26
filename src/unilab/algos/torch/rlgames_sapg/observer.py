"""Native algorithm observer that bridges env metrics to UniLab tracking."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
import torch


class ExperimentTrackerObserver:
    """Bridge scalar ``info['log']`` values without owning a W&B lifecycle."""

    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker
        self.algo: Any | None = None
        self.writer: Any | None = None
        self._metrics: dict[str, float] = {}
        self._writer_closed = False

    def before_init(self, base_name: str, config: dict[str, Any], experiment_name: str) -> None:
        del base_name, config, experiment_name

    def after_init(self, algo: Any) -> None:
        self.algo = algo
        self.writer = algo.writer

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, np.generic) and np.issubdtype(value.dtype, np.number):
            return float(value)
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            return float(value.detach().cpu().item())
        return None

    def process_infos(self, infos: Any, done_indices: Any, **kwargs: Any) -> None:
        del done_indices, kwargs
        if not isinstance(infos, dict) or not isinstance(infos.get("log"), dict):
            return
        metrics: dict[str, float] = {}
        for name, value in infos["log"].items():
            scalar = self._scalar(value)
            if scalar is not None and np.isfinite(scalar):
                metrics[f"env/{name}"] = scalar
        self._metrics = metrics

    def after_steps(self) -> None:
        return None

    def after_clear_stats(self) -> None:
        self._metrics.clear()

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        if self.writer is not None:
            for name, value in self._metrics.items():
                self.writer.add_scalar(name, value, frame)
        run = self.tracker.run
        if run is not None and self._metrics:
            run.log(dict(self._metrics), step=int(frame))
        self.tracker.update_summary(
            {
                "native_frame": int(frame),
                "native_epoch": int(epoch_num),
                "native_total_time_sec": float(total_time),
            }
        )

    def close_writer(self) -> None:
        if self.writer is not None and not self._writer_closed:
            self.writer.close()
            self._writer_closed = True
