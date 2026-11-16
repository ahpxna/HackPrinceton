"""Orchestrate modules without coupling them to data vendors or training frameworks."""
from pathlib import Path
import json
import numpy as np
from .contracts import Scene
from .gaussians import GaussianCloud, write_point_ply
from .semantics import lift_semantics, label_metrics
from .viewer import export_html


def run_fusion(manifest, points, output, *, threshold=0.6, tolerance=0.08,
               oracle_truth=None):
    scene=Scene.load(manifest)
    cloud=GaussianCloud.load(points)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    labels,conf,observations,details=lift_semantics(scene, cloud, tolerance=tolerance, threshold=threshold)
    np.savez_compressed(output/'semantic_points.npz',xyz=cloud.xyz,rgb=cloud.rgb,
                        labels=labels,confidence=conf,observations=observations,
                        **({'scale':cloud.scale} if cloud.scale is not None else {}),
                        **({'opacity':cloud.opacity} if cloud.opacity is not None else {}))
    write_point_ply(output/'semantic_preview.ply',cloud.xyz,cloud.rgb,labels,conf)
    export_html(output/'viewer.html',cloud.xyz,cloud.rgb,labels,conf,scene.classes)
    report={"scene_id":scene.scene_id,"source":scene.source,"world_units":scene.world_units,
            "metric_scale_verified":bool(scene.provenance.get('metric_scale_verified',False)),
            "model_type":"point_or_gaussian_centers_semantic_vote", "trained_3dgs":False,
            "segmentation_backend":"provided_per_view_integer_masks", "fusion":details,
            "threshold":threshold, "visibility_tolerance_in_scene_units":tolerance,
            "classes":scene.classes,"evidence":"Mask projections + supplied point-center geometry; no hidden-surface inference"}
    if oracle_truth is not None:
        truth=np.load(oracle_truth)
        report['oracle_evaluation']=label_metrics(labels,truth,sorted(scene.classes))
        report['oracle_note']='Only valid when oracle labels are independent/reference labels; synthetic demo is self-consistency smoke test'
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report
