# Kuka/Sharpa asset notices

The binary meshes used by the SimToolReal MuJoCo task are hosted in the public
Hugging Face dataset [`unilabsim/unilab-robots`](https://huggingface.co/datasets/unilabsim/unilab-robots).
UniLab keeps the task XML (`kuka_sharpa.xml` and `scene.xml`) and downloads the
mesh tree on the cold path when it is not present locally.

The HF asset directory is:

```text
robots/kuka_sharpa/meshes/
```

The HF copy includes the complete source and license metadata:

- `robots/kuka_sharpa/LICENSE.simtoolreal` (MIT; SimToolReal source assets)
- `robots/kuka_sharpa/LICENSE.kuka_iiwa` (BSD-2-Clause; KUKA iiwa source assets)
- `robots/kuka_sharpa/meshes/menagerie_sharpa_wave/LICENSE` (Apache-2.0)
- `robots/kuka_sharpa/SOURCE.md` (Menagerie/Sharpa source revisions)
- `robots/kuka_sharpa/ASSET_PROVENANCE` (file inventory and hashes)

The same HF dataset also contains the complete DexBench tree under
`dexbench/`. DexBench is resolved independently when the Viser playback entry
point loads a task; its manifest and object/task assets are not part of the
robot mesh marker.

The XML and binary assets are derived from the pinned source revisions recorded
in the HF `SOURCE.md` and `ASSET_PROVENANCE` files. Do not remove those notices
when redistributing the downloaded assets.
