"""Intentionally model-agnostic segmentation entrypoint."""
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image


class SegmentationAdapter(ABC):
    @abstractmethod
    def predict(self, image_path: Path, output_mask: Path, class_map: dict[int,str]) -> Path:
        """Write a single-channel 2D PNG whose pixel values are class IDs.

        0 = unlabeled/background; use scene.json 'classes' for labels.
        Should avoid alpha/RGB palettes and preserve the original image grid.
        """
        raise NotImplementedError


class ExistingMaskAdapter(SegmentationAdapter):
    def __init__(self, mask_path: Path):
        self.mask_path = Path(mask_path)
    def predict(self, image_path: Path, output_mask: Path, class_map: dict[int,str]) -> Path:
        image = Image.open(image_path)
        mask = Image.open(self.mask_path)
        if mask.size != image.size or mask.mode not in ('L','P','I','I;16'):
            raise ValueError('Expected same-resolution single-channel class-index mask')
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        mask.save(output_mask)
        return output_mask

# Future adapters: SAM3, SAM2+open-vocab detector, Mask2Former, GroundingDINO.
# Their model-specific input/output code belongs here, not in fusion.py.
