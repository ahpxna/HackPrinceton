"""semantictwin: local data first, model adapters later."""
import argparse
import json
from pathlib import Path
from .contracts import Scene
from .demo import create_demo
from .pipeline import run_fusion
from .adapters.nerfstudio import import_transforms


def main(argv=None):
    p=argparse.ArgumentParser(prog='semantictwin',description='Provider-neutral RGB-only semantic 3D starter')
    commands=p.add_subparsers(dest='command', required=True)
    d=commands.add_parser('demo', help='Generate reproducible SYNTHETIC views, masks, points and inspectable output')
    d.add_argument('--out',default='runs/demo')
    v=commands.add_parser('validate',help='Validate scene manifest/camera coordinate contract')
    v.add_argument('--scene',required=True)
    f=commands.add_parser('fuse',help='Lift provided masks onto 3D point/gaussian centers')
    f.add_argument('--scene',required=True)
    f.add_argument('--points',required=True)
    f.add_argument('--out',default='runs/fused')
    f.add_argument('--threshold',type=float,default=0.6)
    f.add_argument('--visibility-tolerance',type=float,default=0.08)
    f.add_argument('--oracle-truth',default=None)
    imp=commands.add_parser('from-nerfstudio',help='Convert rectified Nerfstudio transforms to a scene manifest')
    imp.add_argument('--transforms',required=True)
    imp.add_argument('--out',required=True)
    imp.add_argument('--meters-confirmed',action='store_true')
    args=p.parse_args(argv)
    if args.command=='demo':
        root=Path(args.out).resolve(); manifest,points=create_demo(root/'scene')
        report=run_fusion(manifest,points,root/'result',oracle_truth=root/'scene/oracle_point_labels.npy')
        print(json.dumps({'manifest':str(manifest),'points':str(points), 'viewer':str(root/'result/viewer.html'),
                          'summary':report},indent=2))
    elif args.command=='validate':
        s=Scene.load(args.scene)
        print(json.dumps({'ok':True,'scene':s.scene_id,'views':len(s.frames),'cameras':len(s.cameras),
                          'world_units':s.world_units,'source':s.source},indent=2))
    elif args.command=='fuse':
        report=run_fusion(args.scene,args.points,args.out,threshold=args.threshold,
                          tolerance=args.visibility_tolerance,oracle_truth=args.oracle_truth)
        print(json.dumps(report,indent=2))
    elif args.command=='from-nerfstudio':
        print('Created:',import_transforms(args.transforms,args.out,metric_scale_verified=args.meters_confirmed))

if __name__=='__main__':main()
