"""Build the offline portable derivative without touching the active Mac runner."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
PACK = Path.home() / 'hp_sunpatch_pack'


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def put(name, text):
    target = PACK / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def subset(filename, names, header):
    source = (ROOT / 'experiments' / filename).read_text()
    tree = ast.parse(source)
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    selected = set(names)
    while True:
        needed = {n.id for name in selected for n in ast.walk(definitions[name]) if isinstance(n, ast.Name)}
        expanded = selected | (needed & definitions.keys())
        if expanded == selected:
            break
        selected = expanded
    return header + '\n\n' + '\n\n'.join(ast.get_source_segment(source, n) for n in tree.body
                                                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected) + '\n'


def build():
    PACK.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / 'data/sun_patch/auto/round2_scale/manifest.json').read_text())
    spec = importlib.util.spec_from_file_location('cached_projection', ROOT/'experiments/sun_from_pano.py')
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    from PIL import Image
    import numpy as np
    initial_bases = projection.BASES.copy()
    for index, row in enumerate(manifest['rows']):
        projection.BASES = initial_bases.copy()
        images = [Image.open(ROOT/row['source_folder']/f'face{f}.jpg').convert('RGB') for f in range(6)]
        seams = projection.check_cap_seams(images)
        cache=ROOT/'data/sun_patch/auto/round2_scale/cache'/row['sid']
        if row['sun_sha256'] and row['model_sha256']:
            sun=json.loads((cache/'sun_snapshot.json').read_text())
            room=next((r for r in sun['rooms'] if r['id']==row['pose']['room_id']),None)
            if room:
                model=json.loads((cache/'model_snapshot.json').read_text())['data']['model']
                raw=next(r for r in model['rooms'] if r['id']==room['id'])
                vertices=np.array([[v['position'][k] for k in ('x','y')]
                                   for e in raw['boundary']['edges'] for v in e['vertices']])
                lo,hi=vertices.min(0)-.1,vertices.max(0)+.1
                center=(lo+hi)/2;span=max(hi-lo);lo=center-span/2;hi=center+span/2
                corners=np.array([[x,y,room['floor_z_m']] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
                pos=np.array([row['pose']['position'][k] for k in ('x','y','z')])
                normal=projection.rotation(row['pose']['rotation'])@projection.BASES[0,0]
                assert np.max((corners-pos)@normal)<=0, 'Ceiling may be sampled; cannot omit this face'
        put(str(Path(row['source_folder'])/'cap_seams.json'), json.dumps({
            'cap_seams':seams,'basis':projection.BASES.tolist(),
            'source_sha256':{str(f):sha(ROOT/row['source_folder']/f'face{f}.jpg') for f in range(6)}})+'\n')
        for face in range(1,6):
            rel = Path(row['source_folder']) / f'face{face}.jpg'
            target = PACK / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
        # Ceiling is only read by cap calibration; its exact computed receipt replaces it.
        (PACK/row['source_folder']/'face0.jpg').unlink(missing_ok=True)
        if index%100==0:
            print('Calibration',index,flush=True)
    # Snapshots are copied byte-for-byte: preserve their existing provenance hashes.
    original_manifest = json.loads(json.dumps(manifest))
    for sid in {r['sid'] for r in manifest['rows']}:
        for name in ('sun_snapshot.json', 'model_snapshot.json'):
            rel = Path('data/sun_patch/auto/round2_scale/cache') / sid / name
            if (ROOT / rel).exists():
                target = PACK / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                value = json.loads((ROOT/rel).read_text())
                if name == 'model_snapshot.json':
                    model = value['data']['model']
                    value = {'data': {'model': {k:model[k] for k in ('rooms','geocoordinates')}}}
                else:
                    value = {k:value[k] for k in ('rooms','windows','registration','obstructions')}
                target.write_text(json.dumps(value,separators=(',',':'))+'\n')
                key = 'sun_sha256' if name.startswith('sun') else 'model_sha256'
                for row in manifest['rows']:
                    if row['sid']==sid:
                        row[key]=sha(target)
    for rel in ('data/sun_patch/round2/models/PVTb5_SBU.pth',):
        target = PACK / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    put('data/sun_patch/auto/round2_scale/manifest.json', json.dumps(manifest, indent=2)+'\n')
    scores = json.loads((ROOT / 'data/sun_patch/round2/scores.json').read_text())
    put('data/sun_patch/round2/scores.json', json.dumps({'models': {'silt': {
        'gate_passed': scores['models']['silt']['gate_passed']}}})+'\n')
    put('experiments/sun_from_pano.py', subset('sun_from_pano.py', ['check_cap_seams', 'rotation', 'cube_coordinates'],
        'import math\nimport numpy as np\nfrom PIL import Image\n' +
        '\n'.join(ast.get_source_segment((ROOT/'experiments/sun_from_pano.py').read_text(), n)
          for n in ast.parse((ROOT/'experiments/sun_from_pano.py').read_text()).body
          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='BASES' for t in n.targets))))
    put('experiments/sun_patch_check.py', subset('sun_patch_check.py',
        ['read', 'model', 'frames', 'rectified', 'sample_image', 'window_corners', 'floor_pixel', 'floor_point'],
        "import json, math\nfrom pathlib import Path\nimport numpy as np\nfrom PIL import Image\nimport sun_from_pano as A\nROOT=Path(__file__).resolve().parents[1]\nCACHE=ROOT/'data/sun_patch/auto/round2_scale/cache'"))
    consistency = (ROOT/'experiments/sun_patch_consistency.py').read_text()
    params = next(n for n in ast.parse(consistency).body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id=='PARAMS' for t in n.targets))
    put('experiments/sun_patch_consistency.py', subset('sun_patch_consistency.py',
        ['blocked', 'room_mask', 'fit', 'horizon_proxy', 'band_test'],
        'import math\nimport numpy as np\nimport cv2\nimport sun_patch_check as V\n'+ast.get_source_segment(consistency, params)))
    round2 = (ROOT/'experiments/sun_patch_round2.py').read_text()
    params = next(n for n in ast.parse(round2).body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id=='PARAMS' for t in n.targets))
    helper = subset('sun_patch_round2.py', ['write', 'load_detector'],
        "import json, sys\nfrom pathlib import Path\nimport torch\nimport numpy as np\nimport sun_patch_consistency as S\nV=S.V\nROOT=Path(__file__).resolve().parents[1]\nOUT=ROOT/'data/sun_patch/round2'\nPROMPT='patch of sunlight on the floor'\n"+ast.get_source_segment(round2, params))
    # Keep only SILT; no transformers or unrelated checkpoint dependencies.
    start = helper.index('    from transformers')
    end = helper.index("    if name == 'silt':", start)
    helper = helper[:start]+"    weights = OUT/'models'\n"+helper[end:]
    helper += '\ndef progress(done, next_step):\n    pass\n'
    put('experiments/sun_patch_round2.py', helper)
    vendor = Path('data/sun_patch/round2/vendor/hanyangclarence-SILT-c888520')
    # Only PVT-b5 is instantiated. Remove unused backbone imports, preserving its exact layers.
    network = (ROOT/vendor/'model/network.py').read_text()
    network = '\n'.join(line for line in network.splitlines() if not any(x in line for x in
                        ('from model.backbone.resnext', 'from model.backbone.efficientnet', 'from model.backbone.convnext')))+'\n'
    put(str(vendor/'model/network.py'), network)
    put(str(vendor/'model/backbone/PVT/pvt_v2.py'), (ROOT/vendor/'model/backbone/PVT/pvt_v2.py').read_text())
    for name in ('model/__init__.py', 'model/backbone/__init__.py', 'model/backbone/PVT/__init__.py'):
        put(str(vendor/name), '')
    license_path = ROOT/vendor/'LICENSE'
    if license_path.exists():
        put('SILT_LICENSE', license_path.read_text())
    # The portable runner imports these pruned, cache-only helpers.
    runner = (ROOT/'experiments/sun_patch_round2_scale.py').read_text()
    runner = runner[:runner.index("if __name__=='__main__':")]
    begin = runner.index('spec = importlib.util.spec_from_file_location')
    end = runner.index('V, S = R.V, R.S', begin)
    runner = runner[:begin]+'sys.path.insert(0, str(Path(__file__).resolve().parent))\nimport sun_patch_round2 as R\n'+runner[end:]
    begin = runner.index('def select_device(')
    end = runner.index('\ndef infer(', begin)
    runner = runner[:begin]+'''def select_device(rows):
    device = OPTIONS.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; use --device cpu for a CPU run')
    R.write(OUT/'device.json', {'selected':device, 'torch':torch.__version__,
                             'cuda_runtime':torch.version.cuda})
    return device

'''+runner[end:]
    runner = runner.replace("graph=ROOT/'data/claims/graph/sun_patch_auto_sheet.jpg'", "graph=OUT/'sun_patch_auto_sheet.jpg'")
    runner = runner.replace("    if graph.exists() and not (R.OUT/'checkpoint/sun_patch_auto_sheet.round1.jpg').exists():\n        shutil.copy2(graph,R.OUT/'checkpoint/sun_patch_auto_sheet.round1.jpg')\n", '')
    begin = runner.index('    # Preserve pilot and round-1 rows;')
    end = runner.index('\n\ndef run():', begin)
    runner = runner[:begin]+runner[end:]
    runner = runner.replace("rng.choice(len(faces),100,replace=False)", "rng.choice(len(faces),min(100,len(faces)),replace=False)")
    runner = runner.replace('    data=manifest()', '''    data=json.loads((ROOT/'data/sun_patch/auto/round2_scale/manifest.json').read_text())
    if OPTIONS.limit:
        data['rows']=data['rows'][:OPTIONS.limit]
    R.write(OUT/'manifest.json',data)''')
    runner = runner.replace("'homes':169,'panoramas':1014,'faces':5070,'candidates'", "'homes':len({r['sid'] for r in data['rows']}),'panoramas':len(data['rows']),'faces':5*len(data['rows']),'candidates'")
    runner = runner.replace("digest(OUT/'cache'/sid/name)", "digest(ROOT/'data/sun_patch/auto/round2_scale/cache'/sid/name)")
    runner = runner.replace("V.CACHE = OUT/'cache'", "V.CACHE = ROOT/'data/sun_patch/auto/round2_scale/cache'")
    runner = runner.replace("V.read(OUT/'cache'/r['sid']/'sun_snapshot.json')", "V.read(V.CACHE/r['sid']/'sun_snapshot.json')")
    runner = runner.replace("str(patch.relative_to(ROOT))", "str(patch.relative_to(OUT))")
    runner = runner.replace("Image.open(ROOT/c['mask_path'])", "Image.open(OUT/c['mask_path'])")
    runner = runner.replace("faces = [Image.open(ROOT/row['source_folder']/f'face{f}.jpg').convert('RGB') for f in range(6)]\n    V.A.BASES = INITIAL_BASES.copy()\n    seams = V.A.check_cap_seams(faces)",
        "calibration=V.read(ROOT/row['source_folder']/'cap_seams.json')\n    faces = [Image.new('RGB',(1,1))]+[Image.open(ROOT/row['source_folder']/f'face{f}.jpg').convert('RGB') for f in range(1,6)]\n    V.A.BASES = np.array(calibration['basis'])\n    seams = calibration['cap_seams']")
    runner += '''
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=OUT)
    OPTIONS=parser.parse_args()
    OUT=OPTIONS.output.resolve()
    if OPTIONS.limit is not None and OPTIONS.limit < 1:
        parser.error('--limit must be positive')
    if OPTIONS.run:
        run()
'''
    # Eliminate unused acquisition manifest builder; inputs must already exist.
    begin = runner.index('def manifest():')
    end = runner.index('def select_device(', begin)
    runner = runner[:begin]+runner[end:]
    put('experiments/sun_patch_round2_scale.py', runner)
    originals = ['sun_patch_round2_scale.py','sun_patch_round2.py','sun_patch_consistency.py','sun_patch_check.py','sun_from_pano.py']
    put('PROVENANCE.json', json.dumps({'original_manifest':original_manifest, 'original_sources': {p:sha(ROOT/'experiments'/p) for p in originals},
        'checkpoint_sha256':sha(ROOT/'data/sun_patch/round2/models/PVTb5_SBU.pth'),
        'vendor_revision':'c888520', 'derivative':'Pruned offline helper closure; explicit CUDA; output confined to run folder.'},indent=2)+'\n')
    print('Copied payload bytes:', sum(p.stat().st_size for p in PACK.rglob('*') if p.is_file()))


if __name__ == '__main__':
    build()
