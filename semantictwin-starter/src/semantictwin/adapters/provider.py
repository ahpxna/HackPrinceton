"""Replace these interfaces after the CoStar/Vexcel workshop.

No credentials, undocumented endpoints, or claims of accessible sponsor assets.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataRequest:
    scene_id: str
    region: str | None = None
    data_kind: str = 'multi_view_rgb'
    options: dict | None = None


class AssetProvider(ABC):
    @abstractmethod
    def prepare(self, request: DataRequest, output_dir: Path) -> Path:
        """Return a manifest path following semantictwin.contracts.Scene.

        Authenticate and download via official APIs, normalize camera axes,
        verify allowed redistribution, and preserve source provenance here.
        """
        raise NotImplementedError


class VexcelProvider(AssetProvider):
    def prepare(self, request: DataRequest, output_dir: Path) -> Path:
        raise NotImplementedError('Workshop integration pending: use official authorized Vexcel API/MCP docs; do not infer oblique intrinsics/poses from cropped imagery.')


class MatterportProvider(AssetProvider):
    def prepare(self, request: DataRequest, output_dir: Path) -> Path:
        raise NotImplementedError('Workshop integration pending: confirm access to panorama/images, poses and permissible asset use; avoid guessing API queries.')


class LocalManifestProvider(AssetProvider):
    def prepare(self, request: DataRequest, output_dir: Path) -> Path:
        path = Path(request.scene_id).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
