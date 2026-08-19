# SimToolReal RL-Games Source Snapshot

This directory contains the V1 pristine Python-runtime selection from the SimToolReal
RL-Games fork. The selected 72 Python files are byte-identical to their Git blobs at the
fixed Source commit recorded in `source_manifest.json`.

V1 intentionally does not include the 122 YAML files in the selected Source parent tree,
Source examples, notebooks, or the Source top-level `rl_games/tests/` test suite. The
runtime selection does retain test-named Python modules inside `rl_games/rl_games`, including
`common/test_utils.py`, `envs/test/**`, and `envs/test_network.py`; these are part of the 72
selected runtime blobs. V1 does not wire this distribution into UniLab, add a root
dependency, or import `rl_games` from a production path.

Run the read-only integrity gate from the repository root:

```bash
uv run scripts/audit_simtoolreal_rlgames_vendor.py
```

Do not format files below `rl_games/`. The root Ruff configuration excludes this directory
so automated formatting cannot rewrite the pristine Source bytes.
