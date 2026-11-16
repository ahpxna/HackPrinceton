"""Shared coordinate math, projection and visibility filtering."""
import numpy as np
from .contracts import Camera


def world_to_camera(points_xyz: np.ndarray, camera: Camera) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    c2w = np.asarray(camera.c2w, dtype=np.float64)
    # OpenCV camera frame c2w, row-vector convention.
    return (pts - c2w[:3, 3]) @ c2w[:3, :3]


def project(points_xyz: np.ndarray, camera: Camera):
    """Return float pixel x/y, positive-range depth, valid-in-frame mask.

    Pinhole depth is camera Z; equirectangular depth is radial range.
    Equirectangular longitude is periodic. Frame v increases downward.
    """
    p = world_to_camera(points_xyz, camera)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    if camera.projection == "pinhole":
        valid = z > 1e-6
        safe_z = np.where(valid, z, 1.0)
        u = camera.fx * x / safe_z + camera.cx
        v = camera.fy * y / safe_z + camera.cy
        depth = z
    else:
        depth = np.linalg.norm(p, axis=1)
        valid = depth > 1e-6
        phi = np.arctan2(x, z)  # [-pi, pi]
        theta = np.arctan2(y, np.hypot(x, z))
        u = ((phi / (2 * np.pi)) + 0.5) * camera.width
        v = ((theta / np.pi) + 0.5) * camera.height
        u = np.mod(u, camera.width)
    valid &= (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return u, v, depth, valid


def zbuffer_visible(u, v, depth, valid, camera: Camera, tolerance: float = 0.05,
                    reference_depth: np.ndarray | None = None):
    """Nearest-surface visibility. Input XYZ are point/gaussian *centers*, not ellipsoid splats.

    Without an independent reference depth, visibility is only determined among supplied
    points. Hidden/absent surfaces cannot be inferred. The tolerance is in manifest scene units (meters or relative).
    """
    xi = np.clip(np.floor(u).astype(int), 0, camera.width - 1)
    yi = np.clip(np.floor(v).astype(int), 0, camera.height - 1)
    flat = yi * camera.width + xi
    buf = np.full(camera.width * camera.height, np.inf, dtype=float)
    np.minimum.at(buf, flat[valid], depth[valid])
    if reference_depth is not None:
        ref = np.asarray(reference_depth)
        if ref.shape != (camera.height, camera.width):
            raise ValueError("Reference depth shape disagrees with camera")
        ref_flat = ref.reshape(-1)
        measured = np.isfinite(ref_flat) & (ref_flat > 0)
        buf[measured] = ref_flat[measured]
    return valid & (depth <= buf[flat] + tolerance)


def look_at_opencv(position, target, up=(0, 0, 1)):
    """Construct c2w using local world up +Z and OpenCV camera axes."""
    eye = np.asarray(position, float)
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray(up, float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    m = np.eye(4)
    m[:3, :3] = np.column_stack([right, down, forward])
    m[:3, 3] = eye
    return m
