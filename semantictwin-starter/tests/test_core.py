import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from semantictwin.geometry import project, look_at_opencv
from semantictwin.contracts import Camera, Scene, ContractError
from semantictwin.semantics import label_metrics
from semantictwin.demo import create_demo
from semantictwin.pipeline import run_fusion

class CoreTests(unittest.TestCase):
    def test_geometry(self):
        cam = Camera('a',100,100,50,50,50,50,np.eye(4).tolist())
        u,v,z,ok=project(np.array([[0,0,2],[1,0,2],[-5,0,-1]]),cam)
        self.assertTrue(ok[0]);self.assertAlmostEqual(u[0],50)
        self.assertAlmostEqual(u[1],75);self.assertFalse(ok[2])
    def test_c2w_orientation(self):
        pose=look_at_opencv((0,-2,1),(0,2,1))
        cam=Camera('a',100,100,50,50,50,50,pose.tolist())
        cam.validate()
        u,v,d,ok=project(np.array([[0,2,1]]),cam)
        self.assertTrue(ok[0]);self.assertAlmostEqual(u[0],50)
    def test_metrics(self):
        r=label_metrics([1,1,2],[1,2,2],[1,2])
        self.assertAlmostEqual(r['per_class']['1']['iou'],.5)
        self.assertAlmostEqual(r['per_class']['2']['iou'],.5)
    def test_manifest_validation(self):
        cam=Camera('a',10,10,5,5,5,5,np.eye(4).tolist())
        cam.c2w[3][3]=0
        with self.assertRaises(ContractError):cam.validate()
    def test_full_demo(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            manifest, pts=create_demo(root/'scene')
            s=Scene.load(manifest)
            self.assertEqual(len(s.frames),4)
            result=run_fusion(manifest,pts,root/'output',oracle_truth=root/'scene/oracle_point_labels.npy')
            self.assertGreater(result['fusion']['observed_points'],1000)
            self.assertTrue((root/'output/viewer.html').stat().st_size>1000)
            self.assertTrue((root/'output/semantic_points.npz').is_file())
            self.assertTrue((root/'output/semantic_preview.ply').is_file())
            self.assertIn('mIoU',result['oracle_evaluation'])
if __name__=='__main__':unittest.main()

class AdapterTests(unittest.TestCase):
    def test_equirectangular_front(self):
        cam=Camera('p', 400, 200, None, None, None, None,np.eye(4).tolist(), 'equirectangular')
        u,v,d,ok=project(np.array([[0.,0.,5.]]),cam)
        self.assertTrue(ok[0]);self.assertAlmostEqual(u[0],200);self.assertAlmostEqual(v[0],100)
    def test_nerfstudio_transform_import(self):
        from semantictwin.adapters.nerfstudio import import_transforms
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);src=r/'input';src.mkdir()
            Image.new('RGB',(12,10)).save(src/'image.png')
            (src/'transforms.json').write_text(json.dumps({'fl_x':8.,'fl_y':8.,'cx':6.,'cy':5.,'w':12,'h':10,
                  'frames':[{'file_path':'image.png','transform_matrix':np.eye(4).tolist()}]}))
            result=import_transforms(src/'transforms.json',r/'imported')
            scene=Scene.load(result)
            self.assertEqual(scene.world_units, 'relative')
            pose=np.asarray(scene.cameras['camera_0000'].c2w)
            self.assertTrue(np.allclose(pose[:3,:3], np.diag([1,-1,-1])))
