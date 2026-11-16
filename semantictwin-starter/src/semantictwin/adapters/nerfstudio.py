"""Import known *rectified* Nerfstudio transforms.json into the neutral manifest.

Transforms usually contain OpenGL camera-to-world matrices; flip camera Y and Z.
COLMAP/SfM datasets are often up-to-scale. Import is marked 'relative' by default.
This does NOT run COLMAP or Gaussian Splatting training.
"""
from pathlib import Path
import json
import shutil
import numpy as np
from ..contracts import Camera, Frame, Scene, ContractError


def import_transforms(transforms_json, out_dir, *, metric_scale_verified=False):
    source = Path(transforms_json).resolve()
    root = Path(out_dir).resolve()
    if source.parent == root: raise ValueError('Choose a different destination from source dataset root')
    raw = json.loads(source.read_text())
    root.mkdir(parents=True, exist_ok=True)
    (root/'images').mkdir(exist_ok=True)
    supported = {"OPENCV", "PINHOLE", "SIMPLE_PINHOLE", "PERSPECTIVE"}
    camera_model = raw.get('camera_model', 'OPENCV')
    if camera_model not in supported:
        raise ContractError(f'Camera model {camera_model} not supported by rectified pinhole importer')
    cams, frames = {}, []
    flip = np.diag([1., -1., -1., 1.])
    for i, frame in enumerate(raw['frames']):
        for distortion in ('k1','k2','k3','k4','p1','p2'):
            val = frame.get(distortion, raw.get(distortion, 0))
            if val is not None and abs(float(val)) > 1e-8:
                raise ContractError('Importer requires undistorted/rectified images, found '+distortion)
        cam_id = f'camera_{i:04d}'
        def get(key):
            value = frame.get(key, raw.get(key))
            if value is None: raise ContractError(f'Missing {key} for {cam_id}')
            return value
        c2w_gl = np.asarray(get('transform_matrix'), dtype=float)
        if c2w_gl.shape == (3,4): c2w_gl = np.vstack([c2w_gl, [0,0,0,1]])
        if c2w_gl.shape != (4,4): raise ContractError('Expected 4x4 transform matrix')
        c2w_cv = c2w_gl @ flip
        width, height = int(get('w')), int(get('h'))
        cam = Camera(cam_id, width, height, float(get('fl_x')), float(get('fl_y')),
                     float(get('cx')), float(get('cy')), c2w_cv.tolist())
        cam.validate()
        path = (source.parent / get('file_path')).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        dst = root/'images'/f'view_{i:04d}{path.suffix.lower()}'
        shutil.copy2(path, dst)
        cams[cam_id] = cam
        frames.append(Frame(f'view_{i:04d}', str(dst.relative_to(root)), cam_id))
    scene = Scene(scene_id=source.parent.name, source='nerfstudio_transforms', cameras=cams,
                  frames=frames, classes={}, root=root,
                  world_units='meters' if metric_scale_verified else 'relative',
                  provenance={'source_transforms':str(source), 'synthetic':False,
                              'metric_scale_verified':bool(metric_scale_verified),
                              'camera_axes':'converted_OpenGL_to_OpenCV',
                              'note':'COLMAP/nerfstudio transformed coordinates may have arbitrary scale; do not report physical distances until externally aligned.'})
    scene.save(root/'scene.json')
    return root/'scene.json'
