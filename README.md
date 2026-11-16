# HackPrinceton

Sunlight analysis, a buyer assistant, and a photo-assisted Gaussian splat walkthrough.

This repository contains application and training code plus the trained HF and VN
models in `models/<sid>/scene.spz`. HF is the completed r5 export at 94,454 steps.
`models/manifest.json` records exact model sizes and SHA-256 hashes.

## Layout

- `demo/`: local demo server, build tools and browser UI.
- `agent/`: buyer assistant and analysis tools.
- `viewer/<sid>/`: the current viewer code for each house.
- `splat/tools/`: splat training, export, geometry and viewer source.
- `experiments/`: sunlight and geometry processing code.
- `semantictwin-starter/`: reconstruction pipeline starter.

## Local setup

Install dependencies from `agent/requirements.txt` and the reconstruction starter
as needed. The training tools target a CUDA/Linux environment; see
`splat/setup_linux.sh` and `splat/run_splat.sh`.

The full demo expects a separately supplied local `data/` tree. Copy viewer code
and the corresponding trained `scene.spz` into `data/splat/web/<sid>/`, alongside
your local viewer configuration, collision mesh, captures and depth maps. Then
run `python3 demo/server.py` and open `http://localhost:8765/`.

Raw datasets, scan/photo/depth captures, listings, recordings, API responses,
third-party Matterport SDK cache, environment files and credentials are excluded.
Configure service credentials only in your local environment. The viewer and demo
are therefore not a self-contained dataset download.

## HF viewer update (W26)

The HF viewer retains full-resolution geometry during motion, preloads along the
3D movement/view direction and uses opaque complementary coverage during capture
changes. Photo coverage is drawn before splats to keep foreground floaters out of
valid photo pixels. Camera eye height stays fixed between resets on stairs.

The local routing/render-state guard, depth/worker guard, module guard and 3,070
HTTP asset checks passed. A post-deployment GPU walkthrough remains unverified
because browser approval review was quota-blocked; no frame-rate guarantee is made.


## HF viewer recovery (W27)

Automatic resolution now recovers at a steady 60 Hz. Photo-worker requests are
isolated across profile resets, including late replies and errors. The local
depth/worker, routing and module guards passed, along with 3,070 HTTP assets and
15 JavaScript modules. Browser smoke tests on both local ports loaded full 256
grids and 2K textures, moved across captures and recovered resolution without
page errors. Photo/depth seams and slow frames during movement remain; this is
not a guarantee of smooth rendering. The trained models are unchanged.
