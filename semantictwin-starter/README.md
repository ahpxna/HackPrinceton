# SemanticTwin — RGB-only, model-agnostic Semantic 3D Starter

> **Scope**: a reusable, testable software **scaffold** for HackPrinceton 2026. It is **not** a trained 3DGS model, a Matterport or Vexcel API client, and not a proven production-quality 3D reconstruction algorithm.
>
> **Data inputs**: ordinary calibrated overlapping RGB views (aerial or indoor), optionally semantic masks, Nerfstudio transforms, COLMAP-derived poses, depth maps, and point/Gaussian-center exports. No radar or LiDAR required.

## Why this refactor

The earlier `layerscan-hackprinceton.zip` relied on a synthetic room, oracle hidden-wall geometry, LiDAR-like first returns and simulated RF range profiles. This is a **new independent codebase**, with all RF simulation discarded. It is deliberately adaptable to workshop findings rather than choosing Vexcel-only outdoor applications or Matterport-only defurnishing now.

## Quick start (Mac CPU or Linux GPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
semantictwin demo --out runs/demo
# Open runs/demo/result/viewer.html in a browser.
python -m unittest discover -s tests -v
```

Demo creates a small procedural room with 4 calibrated RGB views, oracle semantic masks, and **synthetic Gaussian *seeds***. It projects those seeds into masks and votes on 3D labels, exports inspectable semantic points and an offline interactive 3D viewer. **This does not train Gaussian Splatting or produce a photorealistic model**; it proves that data contracts, visibility, lifting, and viewer outputs work.

Demo outputs:
- `runs/demo/scene/scene.json`: neutral scene manifest
- `runs/demo/scene/images/` and `masks/`: rendered demo views and known labels
- `runs/demo/scene/gaussian_seeds.npz`: point/gaussian-center carrier (not trained splats)
- `runs/demo/result/semantic_points.npz`: 3D semantic labels and confidence
- `runs/demo/result/semantic_preview.ply`: colored point cloud (not native 3DGS PLY)
- `runs/demo/result/viewer.html`: self-contained orbit, zoom, select, class visibility, uncertainty
- `runs/demo/result/report.json`: counts and optional oracle-only quality smoke test

## Real data workflow

### Route A — Existing processed Nerfstudio dataset

```bash
semantictwin from-nerfstudio \
  --transforms /path/to/processed/transforms.json \
  --out runs/imported_scene
semantictwin validate --scene runs/imported_scene/scene.json
```

Assumptions: undistorted **pinhole** images, known camera intrinsics, per-frame camera-to-world poses. Converts Nerfstudio/OpenGL camera axes to internal OpenCV axes. Import marks 3D units as **relative** by default, because SfM often lacks true metric scale. Add `--meters-confirmed` **only after external scale validation**. This import converts metadata and copies images; it **does not run COLMAP**.

### Route B — Known sponsor images/poses

Write a `scene.json` following `examples/scene_manifest.example.json` or implement an `AssetProvider` in `adapters/provider.py`. Ensure all camera matrices refer to **one common local 3D frame**. Do not treat longitude/latitude as X/Y meters. For aerial data, convert geospatial coordinates to a local metric CRS first and handle camera-frame conventions explicitly. For equirectangular panoramas, set camera `projection: equirectangular` and use actual camera orientation; do not pass a panorama through a global pinhole K.

### Step 1 — Obtain/train 3D splats using an external backend

On the RTX 3090 **after** installing official Nerfstudio dependencies and preparing a suitable dataset:

```bash
ns-process-data images --data /path/to/raw_overlapping_images --output-dir /path/to/processed
bash scripts/train_splatfacto.sh /path/to/processed
```

`ns-process-data` with `images` will normally run SfM (COLMAP); run it **only if the workshop data/license and your camera geometry permit**. For supplied poses/photogrammetric cameras, prefer the documented custom data parser or conversions rather than repeating SfM needlessly. The exact training command shown delegates to the official `ns-train splatfacto` backend; the starter **does not bundle or alter Nerfstudio**. Export a PLY containing 3D Gaussian centers/properties using the trained backend's export workflow, or produce a canonical `gaussians.npz` with `xyz` and normalized `rgb`.

### Step 2 — Plug in per-view segmentation

`src/semantictwin/adapters/segmentation.py` has an interface for SAM3, SAM2, Mask2Former, GroundingDINO or another backend. Implement the adapter to save **integer single-channel PNG masks** at the exact view resolution. The class-index map is defined in `scene.json`. Do not assume a SAM instance ID automatically has stable identity across frames; run association where appropriate.

### Step 3 — Fuse 2D labels onto 3D points/gaussian centers

```bash
semantictwin fuse \
  --scene runs/imported_scene/scene.json \
  --points /path/to/exported_gaussians.ply \
  --out runs/semantic
```

Supports `.npz` directly and `.ply` with optional dependency `pip install -e '.[ply]'`. For `.npz`, use `xyz: (N,3)`, `rgb: (N,3) float [0,1]`, optionally `scale: (N,3)` and `opacity: (N,)`. Standard 3DGS PLY fields `f_dc_{0,1,2}`, `scale_{0,1,2}`, `opacity` are recognized on **import**; actual interoperability must still be tested on the exporter in use. The semantic fusion baseline projects *point centers*, checks approximate depth visibility, votes over masks and marks low-confidence points unknown. It does not perform differentiable splat-level feature learning, fully resolve occlusion, or modify 3DGS renderer parameters.

### Step 4 — Explore outputs

Open `runs/semantic/viewer.html` locally; no external network, web server, or JS build tool required. This viewer plots **point centers**, not Gaussian ellipsoids or shaded image formation. For photorealistic 3DGS rendering use the trained backend's viewer.

## Extending without rewriting the project

| Workshop outcome | Plug in / change | Core files to retain |
|---|---|---|
| Vexcel aerial RGB+poses | `adapters/provider.py`, coordinate import, class definitions | `contracts.py`, `geometry.py`, `semantics.py`, `pipeline.py` |
| Matterport indoor panoramas | Panorama adapter, camera poses, movable-fixture ontology | Same core; `projection=equirectangular` supported |
| Existing 3DGS already trained | Import PLY/NPZ point centers, link backend viewer | `gaussians.py`, `semantics.py`, `viewer.py` |
| Want novel 3D semantic model | Replace majority-vote with learned per-Gaussian features | Keep `Scene`, camera geometry, providers and evaluation |
| Choose editable defurnishing | New 3D object grouping/editing & hole completion module | Keep semantic point output, avoid claiming hide==completion |
| Choose 3D change detection | Add timestamped collections/registration & change classifier | `Frame.collection`, `Frame.timestamp` already present |
| Need quality / trustworthy geometry | Add holdout/reprojection and independent ground truth | Add evaluator; preserve manifest/provenance |

## Important invariants

- World coordinates are **right-handed**, locally consistent, and tagged `meters` (externally verified) or `relative` (up-to-scale).
- **Camera c2w transforms OpenCV axes** (x right, y down, z forward) to world. Check transforms and undistortion before projecting labels.
- Occlusion is approximate point-center z-buffering, in scene units; scale and opacity from splats are currently *not* rasterized.
- Per-view labels need same grid and explicit classes; masks cannot silently be warped without corresponding camera update.
- Frame timestamps/collections may be stored, but **temporal change detection is not implemented**.
- No sponsor credentials included. Check event rules and redistribution rights; keep raw sponsor assets out of public git.

## Next workshop decision

See `docs/WORKSHOP_QUESTIONS.md` and `docs/DECISION_LOG.md`. Select a product behavior **only after** confirming what data the sponsor actually grants (raw views vs crops, intrinsics/extrinsics, DSM/DTM, imagery sets, Matterport interiors, model exports).

## Sources

- Nerfstudio custom data: https://docs.nerf.studio/quickstart/custom_dataset.html
- Nerfstudio Splatfacto: https://docs.nerf.studio/nerfology/methods/splat.html
- gsplat: https://docs.gsplat.studio/main/
- Gaussian Grouping (semantic 3D baseline): https://github.com/lkeab/gaussian-grouping

Credit all third-party assets and read each dependency's license before combining with sponsor data.
