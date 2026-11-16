"""Model-agnostic segmentation masks and visibility-aware multiview semantic voting."""
from __future__ import annotations
import numpy as np
from PIL import Image
from .contracts import Scene, ContractError
from .gaussians import GaussianCloud
from .geometry import project, zbuffer_visible


def read_label_mask(path, expected_wh):
    """Single-channel integer semantic IDs, NOT RGB-colored palette previews."""
    img = Image.open(path)
    if img.size != expected_wh:
        raise ContractError(f"Mask {path} size {img.size} != {expected_wh}")
    if img.mode not in ("L", "I", "I;16", "P"):
        raise ContractError(f"Mask {path} must have integer label IDs, got {img.mode}")
    return np.asarray(img, dtype=np.int32)


def lift_semantics(scene: Scene, cloud: GaussianCloud, tolerance: float = 0.08,
                   threshold: float = 0.6):
    """Project point/gaussian centers into masks with approximate visibility.

    This is a *baseline*, not differentiable semantic Gaussian training.
    A gaussian center can span multiple pixels and its anisotropic footprint is not
    represented. Conflicting observations are vote-aggregated, not learned.
    """
    n = len(cloud.xyz)
    class_ids = sorted(scene.classes)
    if not class_ids: raise ContractError("Define classes in manifest")
    votes = np.zeros((n, len(class_ids)), dtype=np.float32)
    id_to_col = {cid: k for k, cid in enumerate(class_ids)}
    observations = np.zeros(n, dtype=np.int32)
    observed_frames = 0
    for frame in scene.frames:
        if not frame.mask: continue
        observed_frames += 1
        camera = scene.cameras[frame.camera]
        mask = read_label_mask(scene.root / frame.mask, (camera.width, camera.height))
        u, v, depth, valid = project(cloud.xyz, camera)
        ref = np.load(scene.root / frame.depth_npy) if frame.depth_npy else None
        visible = zbuffer_visible(u, v, depth, valid, camera, tolerance=tolerance, reference_depth=ref)
        indices = np.flatnonzero(visible)
        xs = np.clip(np.floor(u[indices]).astype(int), 0, camera.width - 1)
        ys = np.clip(np.floor(v[indices]).astype(int), 0, camera.height - 1)
        labels = mask[ys, xs]
        columns = np.array([id_to_col.get(int(cid), -1) for cid in labels], dtype=int)
        known = columns >= 0
        np.add.at(votes, (indices[known], columns[known]), 1)
        np.add.at(observations, indices[known], 1)
    if observed_frames == 0: raise ContractError("No labeled masks provided (run a segmentation adapter first)")
    best_cols = np.argmax(votes, axis=1)
    best = np.asarray(class_ids, dtype=np.int32)[best_cols]
    confidences = votes[np.arange(n), best_cols] / np.maximum(observations, 1)
    best[(observations == 0) | (confidences < threshold)] = -1
    return best, confidences, observations, {"frames_with_masks": observed_frames, "point_count": n,
                                          "observed_points": int(np.sum(observations > 0)),
                                          "confident_points": int(np.sum(best >= 0)),
                                          "method": "visibility_aware_center_projection_majority_vote",
                                          "caveat": "No neural semantic training; requires externally produced masks and valid camera poses"}


def label_metrics(predicted, truth, class_ids):
    p, t = np.asarray(predicted), np.asarray(truth)
    if p.shape != t.shape: raise ValueError("Prediction/truth shape mismatch")
    metrics = {}
    for c in class_ids:
        tp = int(np.sum((p == c) & (t == c)))
        fp = int(np.sum((p == c) & (t != c)))
        fn = int(np.sum((p != c) & (t == c)))
        union = tp + fp + fn
        metrics[str(c)] = {"tp":tp,"fp":fp,"fn":fn,
                          "precision": tp/(tp+fp) if tp+fp else None,
                          "recall": tp/(tp+fn) if tp+fn else None,
                          "iou": tp/union if union else None}
    valid = [r["iou"] for r in metrics.values() if r["iou"] is not None]
    return {"per_class": metrics, "mIoU": float(np.mean(valid)) if valid else None}
