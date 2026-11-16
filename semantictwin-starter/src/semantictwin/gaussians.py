"""Point/gaussian center carrier: not a renderer or 3DGS trainer."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class GaussianCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    scale: np.ndarray | None = None
    opacity: np.ndarray | None = None

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=np.float32)
        self.rgb = np.asarray(self.rgb, dtype=np.float32)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or self.rgb.shape != self.xyz.shape:
            raise ValueError("xyz and rgb must both be Nx3")
        if self.scale is not None and np.asarray(self.scale).shape != self.xyz.shape:
            raise ValueError("scale must be Nx3")
        if self.opacity is not None and len(self.opacity) != len(self.xyz):
            raise ValueError("opacity length mismatch")

    @staticmethod
    def load(path: str | Path) -> "GaussianCloud":
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path) as data:
                return GaussianCloud(data["xyz"], data["rgb"],
                                     data["scale"] if "scale" in data else None,
                                     data["opacity"] if "opacity" in data else None)
        if path.suffix.lower() == ".ply":
            try:
                from plyfile import PlyData
            except ImportError as e:
                raise RuntimeError("PLY import needs: pip install -e '.[ply]'") from e
            p = PlyData.read(str(path))["vertex"]
            names = set(p.data.dtype.names)
            xyz = np.column_stack([p[x] for x in ("x", "y", "z")])
            if {"red", "green", "blue"} <= names:
                rgb = np.column_stack([p[x] for x in ("red", "green", "blue")]).astype(float)
                if rgb.max() > 1.0: rgb /= 255.0
            elif {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
                rgb = np.clip(0.5 + 0.28209479177 * np.column_stack([p[f"f_dc_{i}"] for i in range(3)]), 0, 1)
            else:
                rgb = np.full_like(xyz, 0.7)
            scales = np.exp(np.column_stack([p[f"scale_{i}"] for i in range(3)])) if {f"scale_{i}" for i in range(3)} <= names else None
            opacities = 1 / (1 + np.exp(-np.asarray(p["opacity"], float))) if "opacity" in names else None
            return GaussianCloud(xyz, rgb, scales, opacities)
        raise ValueError("Expected .npz or .ply; do not treat mesh files as Gaussians")

    def save(self, path: str | Path):
        kwargs = {"xyz": self.xyz, "rgb": self.rgb}
        if self.scale is not None: kwargs["scale"] = self.scale
        if self.opacity is not None: kwargs["opacity"] = self.opacity
        np.savez_compressed(path, **kwargs)


def write_point_ply(path, xyz, rgb, labels=None, confidence=None):
    """Export a colored point-cloud PLY for inspection. NOT splat SH/covariance format."""
    xyz = np.asarray(xyz)
    colors = np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)
    labs = np.full(len(xyz), -1, dtype=np.int32) if labels is None else np.asarray(labels, np.int32)
    conf = np.zeros(len(xyz)) if confidence is None else np.asarray(confidence)
    with open(path, "w") as fp:
        fp.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fp.write("property int semantic\nproperty float confidence\nend_header\n")
        for p, c, l, cf in zip(xyz, colors, labs, conf):
            fp.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} {int(l)} {float(cf):.5f}\n")
