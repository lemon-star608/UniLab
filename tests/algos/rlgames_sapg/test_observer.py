from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


class Writer:
    def __init__(self):
        self.scalars = []
        self.close_calls = 0

    def add_scalar(self, key, value, frame):
        self.scalars.append((key, float(value), int(frame)))

    def close(self):
        self.close_calls += 1


class Run:
    def __init__(self):
        self.logs = []

    def log(self, payload, step=None):
        self.logs.append((payload, step))


class Tracker:
    def __init__(self, run=None):
        self.run = run
        self.summaries = []

    def update_summary(self, value):
        self.summaries.append(value)


def test_observer_bridges_only_env_log_scalars_to_native_writer_and_active_run():
    from unilab.algos.torch.rlgames_sapg.observer import ExperimentTrackerObserver

    writer, run = Writer(), Run()
    tracker = Tracker(run)
    observer = ExperimentTrackerObserver(tracker)
    observer.before_init("run", {}, "0_test")
    observer.after_init(SimpleNamespace(writer=writer))
    observer.process_infos(
        {
            "log": {
                "reward": 2.5,
                "count": np.int64(3),
                "scalar": torch.tensor(4.0),
                "rows": np.ones(2),
            }
        },
        torch.tensor([0]),
    )
    observer.after_steps()
    observer.after_print_stats(frame=12, epoch_num=2, total_time=1.5)
    assert writer.scalars == [
        ("env/reward", 2.5, 12),
        ("env/count", 3.0, 12),
        ("env/scalar", 4.0, 12),
    ]
    assert run.logs == [({"env/reward": 2.5, "env/count": 3.0, "env/scalar": 4.0}, 12)]
    assert tracker.summaries[-1] == {
        "native_frame": 12,
        "native_epoch": 2,
        "native_total_time_sec": 1.5,
    }


def test_observer_works_without_wandb_and_closes_native_writer_exactly_once():
    from unilab.algos.torch.rlgames_sapg.observer import ExperimentTrackerObserver

    writer = Writer()
    observer = ExperimentTrackerObserver(Tracker())
    observer.after_init(SimpleNamespace(writer=writer))
    observer.process_infos({"log": {"metric": 1}}, torch.tensor([], dtype=torch.int64))
    observer.after_clear_stats()
    observer.after_print_stats(1, 1, 0.1)
    observer.close_writer()
    observer.close_writer()
    assert writer.close_calls == 1
