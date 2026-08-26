"""Deterministic procedural SimToolReal tool catalog.

Clean port of the upstream procedural distributions used by the direct-compiled
600-tool reduced pool: 12 ObjectSizeDistributions × 50 samples.
No imports from isaacsimenvs (would pull isaaclab).

Returns ToolSpec tuples with MuJoCo-ready geometry (geom_size half-lengths, not URDF
full-lengths), mass, COM, and diagonal inertia. Sampling is bit-exact with seed=42:
  handle_densities → head_densities → handle_scales → head_scales (lockstep shuffle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------
# ObjectSizeDistribution (inlined from object_size_distributions.py)
# ----------------------------------------------------------------------------

LOW_DENSITY_MIN, LOW_DENSITY_MAX = 300.0, 600.0
HIGH_DENSITY_MIN, HIGH_DENSITY_MAX = 800.0, 2000.0


@dataclass
class _ObjectSizeDistribution:
    """Per-handle-head-type procedural size distribution."""

    type: str
    handle_min_lengths: tuple[float, ...]
    handle_max_lengths: tuple[float, ...]
    head_min_lengths: tuple[float, ...] | None
    head_max_lengths: tuple[float, ...] | None
    handle_min_density: float
    handle_max_density: float
    head_min_density: float | None
    head_max_density: float | None

    @property
    def shape(self) -> str:
        return "cuboid" if len(self.handle_min_lengths) == 3 else "cylinder"

    def sample_handle_densities(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(self.handle_min_density, self.handle_max_density, size=n)

    def sample_head_densities(self, n: int, rng: np.random.RandomState) -> np.ndarray | None:
        if self.head_min_density is None or self.head_max_density is None:
            return None
        return rng.uniform(self.head_min_density, self.head_max_density, size=n)

    def sample_handle_scales(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(
            self.handle_min_lengths,
            self.handle_max_lengths,
            size=(n, len(self.handle_min_lengths)),
        )

    def sample_head_scales(self, n: int, rng: np.random.RandomState) -> np.ndarray | None:
        if self.head_min_lengths is None or self.head_max_lengths is None:
            return None
        return rng.uniform(
            self.head_min_lengths,
            self.head_max_lengths,
            size=(n, len(self.head_min_lengths)),
        )


_OBJECT_SIZE_DISTRIBUTIONS = [
    # Hammer — cuboid handle
    _ObjectSizeDistribution(
        type="hammer",
        handle_min_lengths=(0.15, 0.02, 0.015),
        handle_max_lengths=(0.30, 0.04, 0.03),
        head_min_lengths=(0.02, 0.05, 0.02),
        head_max_lengths=(0.06, 0.12, 0.06),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Hammer — cylinder handle
    _ObjectSizeDistribution(
        type="hammer",
        handle_min_lengths=(0.15, 0.015),
        handle_max_lengths=(0.30, 0.03),
        head_min_lengths=(0.02, 0.05, 0.02),
        head_max_lengths=(0.06, 0.12, 0.06),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Screwdriver — cuboid
    _ObjectSizeDistribution(
        type="screwdriver",
        handle_min_lengths=(0.07, 0.025, 0.025),
        handle_max_lengths=(0.12, 0.04, 0.04),
        head_min_lengths=(0.07, 0.01, 0.01),
        head_max_lengths=(0.15, 0.015, 0.015),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Screwdriver — cylinder
    _ObjectSizeDistribution(
        type="screwdriver",
        handle_min_lengths=(0.07, 0.025),
        handle_max_lengths=(0.12, 0.04),
        head_min_lengths=(0.07, 0.01, 0.01),
        head_max_lengths=(0.15, 0.015, 0.015),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=HIGH_DENSITY_MIN,
        head_max_density=HIGH_DENSITY_MAX,
    ),
    # Marker — cylinder only
    _ObjectSizeDistribution(
        type="marker",
        handle_min_lengths=(0.075, 0.015),
        handle_max_lengths=(0.15, 0.03),
        head_min_lengths=(0.01, 0.005, 0.005),
        head_max_lengths=(0.03, 0.01, 0.01),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Spatula — cuboid
    _ObjectSizeDistribution(
        type="spatula",
        handle_min_lengths=(0.1, 0.0125, 0.006),
        handle_max_lengths=(0.2, 0.025, 0.025),
        head_min_lengths=(0.05, 0.03, 0.01),
        head_max_lengths=(0.15, 0.07, 0.03),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Spatula — cylinder
    _ObjectSizeDistribution(
        type="spatula",
        handle_min_lengths=(0.1, 0.0125),
        handle_max_lengths=(0.2, 0.025),
        head_min_lengths=(0.05, 0.03, 0.01),
        head_max_lengths=(0.15, 0.07, 0.03),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Eraser — cuboid, no head
    _ObjectSizeDistribution(
        type="eraser",
        handle_min_lengths=(0.07, 0.02, 0.02),
        handle_max_lengths=(0.15, 0.07, 0.07),
        head_min_lengths=None,
        head_max_lengths=None,
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=None,
        head_max_density=None,
    ),
    # Brush — cuboid + head v1
    _ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01, 0.01),
        handle_max_lengths=(0.2, 0.04, 0.03),
        head_min_lengths=(0.05, 0.03, 0.03),
        head_max_lengths=(0.12, 0.05, 0.08),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Brush — cylinder + head v1
    _ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01),
        handle_max_lengths=(0.2, 0.03),
        head_min_lengths=(0.05, 0.03, 0.03),
        head_max_lengths=(0.12, 0.05, 0.08),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Brush — cuboid + head v2
    _ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01, 0.01),
        handle_max_lengths=(0.2, 0.04, 0.03),
        head_min_lengths=(0.05, 0.05, 0.02),
        head_max_lengths=(0.12, 0.12, 0.04),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
    # Brush — cylinder + head v2
    _ObjectSizeDistribution(
        type="brush",
        handle_min_lengths=(0.05, 0.01),
        handle_max_lengths=(0.2, 0.03),
        head_min_lengths=(0.05, 0.05, 0.02),
        head_max_lengths=(0.12, 0.12, 0.04),
        handle_min_density=LOW_DENSITY_MIN,
        handle_max_density=LOW_DENSITY_MAX,
        head_min_density=LOW_DENSITY_MIN,
        head_max_density=LOW_DENSITY_MAX,
    ),
]


# ----------------------------------------------------------------------------
# Mass and inertia computation (ported from generate_objects.py)
# ----------------------------------------------------------------------------


def _compute_mass_and_inertia_box(
    lx: float, ly: float, lz: float, density: float
) -> tuple[float, float, float, float]:
    """Exact box mass and inertia."""
    v = lx * ly * lz
    m = v * density
    ixx = (1 / 12) * m * (ly * ly + lz * lz)
    iyy = (1 / 12) * m * (lx * lx + lz * lz)
    izz = (1 / 12) * m * (lx * lx + ly * ly)
    return m, ixx, iyy, izz


def _compute_mass_and_inertia_capsule(
    h: float, d: float, density: float
) -> tuple[float, float, float, float]:
    """Capsule mass and inertia for an authored cylinder (cylinder + 2 hemispheres)."""
    r = d / 2
    # Cylinder mass
    m_c = density * math.pi * r * r * h
    # Hemisphere mass
    m_h = density * (2 / 3) * math.pi * r**3
    m = m_c + 2 * m_h

    # Cylinder inertia about centroid (axis = z)
    i_c_axis = 0.5 * m_c * r * r
    i_c_perp = (1 / 12) * m_c * (3 * r * r + h * h)

    # Hemisphere inertia about its own centroid
    i_h_axis = (2 / 5) * m_h * r * r
    i_h_perp = (83 / 320) * m_h * r * r
    d_com = (h / 2) + (3 * r / 8)

    izz = i_c_axis + 2 * i_h_axis
    ixx = iyy = i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com)
    return m, ixx, iyy, izz


# ----------------------------------------------------------------------------
# ToolSpec and catalog builder
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Per-tool geometry and mass properties for direct source compilation.

    All sizes are MuJoCo half-lengths (not URDF full-lengths).
    """

    type: str
    topology: str  # "box_box" | "capsule_box" | "box_only"
    authored_shape: str
    collision_shape: str
    handle_size: tuple[float, float, float]  # box: (hx, hy, hz); cyl: (radius, half_length, unused)
    head_size: tuple[float, float, float]  # always box: (hx, hy, hz)
    head_pos: tuple[float, float, float]  # head geom position relative to body origin
    mass: float
    com: tuple[float, float, float]  # body COM (ipos)
    diaginertia: tuple[float, float, float]  # (ixx, iyy, izz)
    object_scale: tuple[float, float, float]  # phi = handle_scale_3d / 0.04


def build_tool_catalog(
    types: tuple[str, ...],
    num_per_type: int,
    seed: int,
    shuffle: bool,
) -> tuple[ToolSpec, ...]:
    """Build deterministic reduced catalog with bit-exact sampling (seed=42).

    Sampling order (matches generate_objects.py:351-358):
      handle_densities → head_densities → handle_scales → head_scales

    Returns
    -------
    catalog : tuple[ToolSpec, ...]
        Length = num_per_type × len(matching distributions). Each ToolSpec has MuJoCo-ready
        geometry (half-lengths), mass, COM, and diagonal inertia.

    Notes
    -----
    Uses a **local** ``np.random.RandomState`` rather than seeding the global
    numpy RNG. Seeding globally here silently overwrote ``training.seed`` --
    this function runs during env construction, which happens *after*
    ``scripts/train_rsl_rl.py`` applies the configured seed, so every
    downstream numpy draw (reset poses, orientations, DR noise, goal sampling)
    became seed-42-determined regardless of what the run asked for.

    ``RandomState`` (legacy MT19937) is deliberate, **not** ``default_rng``:
    it reproduces the upstream ``np.random.seed(seed)`` +
    ``np.random.uniform(...)`` draws for the requested pool size.
    ``default_rng`` (PCG64) would silently produce a different reduced-pool
    realization.
    """
    rng = np.random.RandomState(seed)

    type_set = set(types)
    matching = [d for d in _OBJECT_SIZE_DISTRIBUTIONS if d.type in type_set]
    if not matching:
        raise ValueError(
            f"No matching distribution for types={types}. "
            f"Valid: {sorted({d.type for d in _OBJECT_SIZE_DISTRIBUTIONS})}"
        )

    specs: list[ToolSpec] = []

    for dist in matching:
        # Sample order MUST match generate_objects.py:355-358
        handle_densities = dist.sample_handle_densities(num_per_type, rng)
        head_densities = dist.sample_head_densities(num_per_type, rng)
        handle_scales = dist.sample_handle_scales(num_per_type, rng)
        head_scales = dist.sample_head_scales(num_per_type, rng)

        for i in range(num_per_type):
            h_scale_urdf = handle_scales[i]
            h_density = handle_densities[i]
            head_scale_urdf = head_scales[i] if head_scales is not None else None
            head_density = head_densities[i] if head_densities is not None else None

            # Convert URDF full-lengths to MuJoCo half-lengths
            is_cylinder = len(h_scale_urdf) == 2
            if is_cylinder:
                h_len, h_diam = h_scale_urdf
                h_m, h_izz, h_iyy, h_ixx = _compute_mass_and_inertia_capsule(
                    h_len, h_diam, h_density
                )
                # MuJoCo capsule geom_size = (radius, half_length, unused)
                # Rotated by -π/2 about y so axis is along +x: ixx ↔ izz
                handle_size_mj = (h_diam / 2, h_len / 2, 0.0)
                handle_scale_3d = (h_len, h_diam, h_diam)
                topology = "capsule_box" if head_scale_urdf is not None else "box_only"
            else:
                h_lx, h_ly, h_lz = h_scale_urdf
                h_m, h_ixx, h_iyy, h_izz = _compute_mass_and_inertia_box(
                    h_lx, h_ly, h_lz, h_density
                )
                handle_size_mj = (h_lx / 2, h_ly / 2, h_lz / 2)
                handle_scale_3d = (h_lx, h_ly, h_lz)
                topology = "box_box" if head_scale_urdf is not None else "box_only"

            if head_scale_urdf is not None:
                # Head is always a box
                head_lx, head_ly, head_lz = head_scale_urdf
                assert head_density is not None
                head_m, head_ixx, head_iyy, head_izz = _compute_mass_and_inertia_box(
                    head_lx, head_ly, head_lz, head_density
                )
                head_size_mj = (head_lx / 2, head_ly / 2, head_lz / 2)

                # Head offset: handle_len/2 + head_lx/2
                x_offset = handle_scale_3d[0] / 2 + head_lx / 2
                head_pos = (x_offset, 0.0, 0.0)

                # Composite COM and inertia (parallel-axis theorem)
                total_mass = h_m + head_m
                com_x = (h_m * 0.0 + head_m * x_offset) / total_mass
                d_handle = -com_x
                d_head = x_offset - com_x

                ixx = h_ixx + head_ixx
                iyy = (h_iyy + h_m * d_handle * d_handle) + (head_iyy + head_m * d_head * d_head)
                izz = (h_izz + h_m * d_handle * d_handle) + (head_izz + head_m * d_head * d_head)
                com = (com_x, 0.0, 0.0)
            else:
                # Handle-only (eraser)
                head_size_mj = (0.0, 0.0, 0.0)
                head_pos = (0.0, 0.0, 0.0)
                total_mass = h_m
                ixx, iyy, izz = h_ixx, h_iyy, h_izz
                com = (0.0, 0.0, 0.0)

            # Normalize by object_base_size (0.04)
            object_scale = tuple(x / 0.04 for x in handle_scale_3d)

            specs.append(
                ToolSpec(
                    type=dist.type,
                    topology=topology,
                    authored_shape="cylinder" if is_cylinder else "box",
                    collision_shape="capsule" if is_cylinder else "box",
                    handle_size=handle_size_mj,
                    head_size=head_size_mj,
                    head_pos=head_pos,
                    mass=total_mass,
                    com=com,
                    diaginertia=(ixx, iyy, izz),
                    object_scale=object_scale,
                )
            )

    # Shuffle in lockstep (matches generate_objects.py:393-397)
    if shuffle:
        indices = np.arange(len(specs))
        rng.shuffle(indices)
        specs = [specs[i] for i in indices]

    return tuple(specs)


ALL_TYPES: tuple[str, ...] = tuple(dict.fromkeys(d.type for d in _OBJECT_SIZE_DISTRIBUTIONS))

__all__ = ["ALL_TYPES", "ToolSpec", "build_tool_catalog"]
