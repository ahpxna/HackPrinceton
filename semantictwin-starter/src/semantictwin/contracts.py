"""Provider-neutral scene and camera schema.

Coordinate contract: one *local metric* right-handed world frame per scene.
Camera c2w maps OpenCV camera axes (x right, y down, z forward) to world.
No WGS84 latitude/longitude is ever treated as metric XYZ.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import numpy as np


class ContractError(ValueError):
    pass


@dataclass
class Camera:
    camera_id: str
    width: int
    height: int
    fx: float | None
    fy: float | None
    cx: float | None
    cy: float | None
    c2w: list[list[float]]
    projection: str = "pinhole"

    def validate(self) -> None:
        if self.projection not in ("pinhole", "equirectangular"):
            raise ContractError(f"Unsupported projection {self.projection}")
        if self.width <= 0 or self.height <= 0:
            raise ContractError("Image dimensions must be positive")
        if self.projection == "pinhole" and (
            any(v is None or not np.isfinite(v) or v <= 0 for v in (self.fx, self.fy)) or
            any(v is None or not np.isfinite(v) for v in (self.cx, self.cy))
        ):
            raise ContractError("Pinhole cameras need finite fx/fy/cx/cy and positive fx/fy")
        mat = np.asarray(self.c2w, dtype=float)
        if mat.shape != (4, 4) or not np.isfinite(mat).all():
            raise ContractError("c2w must be a finite 4x4 matrix")
        if not np.allclose(mat[3], [0, 0, 0, 1], atol=1e-5):
            raise ContractError("c2w last row must be [0,0,0,1]")
        r = mat[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-3) or np.linalg.det(r) < 0.99:
            raise ContractError("c2w rotation must be right-handed, orthonormal")


@dataclass
class Frame:
    frame_id: str
    image: str
    camera: str
    mask: str | None = None
    depth_npy: str | None = None
    collection: str | None = None
    timestamp: str | None = None


@dataclass
class Scene:
    scene_id: str
    source: str
    cameras: dict[str, Camera]
    frames: list[Frame]
    classes: dict[int, str]
    world_units: str = "meters"
    coordinate_system: str = "local_right_handed"
    provenance: dict[str, Any] = field(default_factory=dict)
    root: Path = Path(".")

    def validate(self, require_files: bool = True) -> None:
        if self.world_units not in ("meters", "relative"):
            raise ContractError("Use local scene units: meters (verified) or relative (scale unknown); convert geographic CRS first")
        if self.coordinate_system != "local_right_handed":
            raise ContractError("Only local_right_handed is supported")
        if not self.frames:
            raise ContractError("Scene needs at least one frame")
        for camera in self.cameras.values():
            camera.validate()
        ids = set()
        for frame in self.frames:
            if frame.frame_id in ids:
                raise ContractError(f"Duplicate frame_id: {frame.frame_id}")
            ids.add(frame.frame_id)
            if frame.camera not in self.cameras:
                raise ContractError(f"Unknown camera id: {frame.camera}")
            for attr in ("image", "mask", "depth_npy"):
                rel = getattr(frame, attr)
                if rel and require_files and not (self.root / rel).is_file():
                    raise ContractError(f"Missing {attr}: {self.root / rel}")

    @staticmethod
    def load(path: str | Path, require_files: bool = True) -> "Scene":
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        cams = {cid: Camera(camera_id=cid, **c) for cid, c in data["cameras"].items()}
        scene = Scene(scene_id=data["scene_id"], source=data.get("source", "local"),
                      cameras=cams, frames=[Frame(**x) for x in data["frames"]],
                      classes={int(k): str(v) for k, v in data.get("classes", {}).items()},
                      world_units=data.get("world_units", "meters"),
                      coordinate_system=data.get("coordinate_system", "local_right_handed"),
                      provenance=data.get("provenance", {}), root=path.parent)
        scene.validate(require_files=require_files)
        return scene

    def save(self, path: str | Path) -> None:
        from dataclasses import asdict
        data = {"schema_version": 1, "scene_id": self.scene_id, "source": self.source,
                "world_units": self.world_units, "coordinate_system": self.coordinate_system,
                "classes": {str(k): v for k, v in self.classes.items()},
                "cameras": {cid: {k: v for k, v in asdict(cam).items() if k != "camera_id"}
                            for cid, cam in self.cameras.items()},
                "frames": [asdict(f) for f in self.frames], "provenance": self.provenance}
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
