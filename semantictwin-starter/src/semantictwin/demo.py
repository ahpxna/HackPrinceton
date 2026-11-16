"""Small deterministic SYNTHETIC RGB/multiview scene for CPU smoke tests.

The 3D points and pixel labels here are oracle synthetic, not photogrammetry
or neural segmentation. Do not present these outputs as trained 3DGS.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from .contracts import Scene, Camera, Frame
from .geometry import look_at_opencv, project
from .gaussians import GaussianCloud


def sample_scene():
    rng = np.random.default_rng(23)
    # World: Z up, wall at Y=3, floor Z=0, block/cabinet in foreground.
    # Use regular grids so color/labels are deterministic but views differ.
    def sheet(a, b, c, label, col):
        return a, np.tile(np.asarray(col, np.float32), (len(a), 1)), np.full(len(a), label, dtype=np.int16)
    wall_x, wall_z = np.meshgrid(np.linspace(-2.8, 2.8, 120), np.linspace(0, 2.6, 60))
    wall = np.column_stack([wall_x.ravel(), np.full(wall_x.size, 3), wall_z.ravel()])
    floor_x, floor_y = np.meshgrid(np.linspace(-2.8, 2.8, 112), np.linspace(-1.6, 3, 92))
    floor = np.column_stack([floor_x.ravel(), floor_y.ravel(), np.zeros(floor_x.size)])
    furniture_parts = []
    # Front and two sides of freestanding cabinet.
    xx, zz = np.meshgrid(np.linspace(-0.85, 0.85, 70), np.linspace(0.02, 1.45, 55))
    furniture_parts.append(np.column_stack([xx.ravel(), np.full(xx.size, 1.25), zz.ravel()]))
    yy, zz = np.meshgrid(np.linspace(1.25, 2.05, 35), np.linspace(0.02, 1.45, 55))
    for x in (-0.85, 0.85):
        furniture_parts.append(np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()]))
    fx, fy = np.meshgrid(np.linspace(-0.85, 0.85, 60), np.linspace(1.25, 2.05, 30))
    furniture_parts.append(np.column_stack([fx.ravel(), fy.ravel(), np.full(fx.size, 1.45)]))
    furniture = np.concatenate(furniture_parts)
    chunks = [sheet(wall, None, None, 1, [0.88, 0.85, 0.76]),
              sheet(floor, None, None, 2, [0.6, 0.45, 0.30]),
              sheet(furniture, None, None, 3, [0.23, 0.49, 0.75])]
    xyz = np.concatenate([c[0] for c in chunks]).astype(np.float32)
    rgb = np.concatenate([c[1] for c in chunks]).astype(np.float32)
    label = np.concatenate([c[2] for c in chunks])
    rgb = np.clip(rgb + rng.normal(0, 0.008, rgb.shape), 0, 1).astype(np.float32)
    return xyz, rgb, label


def render_points(xyz, rgb, labels, camera):
    """Fast point raster (nearest point wins at each pixel)."""
    w, h = camera.width, camera.height
    u, v, depth, valid = project(xyz, camera)
    indices = np.flatnonzero(valid)
    # Sort far-to-near so nearer samples overwrite farther samples.
    indices = indices[np.argsort(depth[indices])[::-1]]
    bg = np.array([218, 226, 238], dtype=np.uint8)
    out = np.broadcast_to(bg, (h, w, 3)).copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    dep = np.full((h, w), np.inf, dtype=np.float32)
    # 3x3 approximation of splat footprint; synthetic pinhole render only.
    for i in indices:
        px, py = int(u[i]), int(v[i])
        z = depth[i]
        x0, x1 = max(0, px-1), min(w, px+2)
        y0, y1 = max(0, py-1), min(h, py+2)
        region = dep[y0:y1, x0:x1]
        change = z < region
        region[change] = z
        mask[y0:y1, x0:x1][change] = labels[i]
        out[y0:y1, x0:x1][change] = (255*rgb[i]).astype(np.uint8)
    return out, mask, dep


def create_demo(root):
    root = Path(root).resolve()
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'masks').mkdir(parents=True, exist_ok=True)
    (root / 'depth').mkdir(parents=True, exist_ok=True)
    xyz, rgb, labels = sample_scene()
    GaussianCloud(xyz, rgb, np.full_like(xyz, 0.016), np.ones(len(xyz), np.float32)).save(root/'gaussian_seeds.npz')
    np.save(root/'oracle_point_labels.npy', labels)
    eyes = [(-1.1, -2.2, 1.35), (-0.3, -2.6, 1.6), (0.6, -2.45, 1.45), (1.35, -2.1, 1.65)]
    cameras, frames = {}, []
    for i, eye in enumerate(eyes):
        cid = f'cam_{i:02d}'
        cam = Camera(camera_id=cid, width=320, height=224, fx=270, fy=270,
                     cx=160, cy=112, c2w=look_at_opencv(eye, (0, 1.6, 1.1)).tolist())
        cameras[cid] = cam
        color, mask, depth = render_points(xyz, rgb, labels, cam)
        name = f'view_{i:02d}'
        Image.fromarray(color).save(root/f'images/{name}.png')
        Image.fromarray(mask, mode='L').save(root/f'masks/{name}.png')
        np.save(root/f'depth/{name}.npy', depth)
        frames.append(Frame(name, f'images/{name}.png', cid,
                            f'masks/{name}.png', f'depth/{name}.npy'))
    scene = Scene(scene_id='synthetic_room_demo', source='synthetic_oracle', cameras=cameras,
                  frames=frames, classes={1:'wall',2:'floor',3:'cabinet'}, root=root,
                  provenance={'synthetic':True, 'metric_scale_verified':True,
                              'notice':'Perfect labels and geometry come from a procedural scene. NOT trained 3DGS or AI segmentation.'})
    scene.save(root/'scene.json')
    return root/'scene.json', root/'gaussian_seeds.npz'
