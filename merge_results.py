"""Verify a returned archive and import only whole, previously absent panoramas."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile
import ctypes
import sys

sys.dont_write_bytecode=True


def rename_exclusive(source, destination):
    libc=ctypes.CDLL(None,use_errno=True)
    if sys.platform=='darwin':
        result=libc.renamex_np(os.fsencode(source),os.fsencode(destination),4)
    elif sys.platform.startswith('linux'):
        result=libc.renameat2(-100,os.fsencode(source),-100,os.fsencode(destination),1)
    else:
        raise RuntimeError('Exclusive directory rename requires macOS or Linux')
    if result:
        number=ctypes.get_errno()
        raise OSError(number,os.strerror(number))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def merge(archive_path, root):
    target=root/'data/sun_patch/auto/round2_scale'
    local=json.loads((target/'manifest.json').read_text())
    with zipfile.ZipFile(archive_path) as archive:
        receipt=json.loads(archive.read('RESULT_HASHES.json'))
        original=receipt['provenance']['original_manifest']
        if original != local:
            raise ValueError('Mac manifest differs from the pinned input set')
        config=hashlib.sha256(json.dumps(local['pilot_params'],sort_keys=True).encode()).hexdigest()
        if config!=receipt['config_sha256'] or not receipt['complete']:
            raise ValueError('Configuration mismatch or incomplete archive')
        for name,expected in receipt['provenance']['original_sources'].items():
            if sha(root/'experiments'/name)!=expected:
                raise ValueError('Mac source changed: '+name)
        if sha(root/'data/sun_patch/round2/models/PVTb5_SBU.pth')!=local['checkpoint_sha256']:
            raise ValueError('Mac checkpoint differs')
        allowed={(r['sid'],r['pano_id']):r for r in local['rows']}
        members={info.filename:info for info in archive.infolist()}
        if len(members)!=len(archive.infolist()):
            raise ValueError('Duplicate archive member')
        expected_names={'RESULT_HASHES.json'}|{'round2_scale/'+n for n in receipt['files']}
        if set(members)!=expected_names:
            raise ValueError('Unexpected archive members')
        for name,expected in receipt['files'].items():
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('Unsafe archive member')
            with archive.open('round2_scale/'+name) as source:
                if hashlib.file_digest(source,'sha256').hexdigest()!=expected:
                    raise ValueError('Result hash mismatch: '+name)
        imported=skipped=0
        target.mkdir(parents=True,exist_ok=True)
        for row in receipt['rows']:
            sid,pid=row['sid'],row['pano_id']
            if (sid,pid) not in allowed:
                raise ValueError('Unknown panorama')
            original_row=allowed[sid,pid]
            for key,name in [('sun_sha256','sun_snapshot.json'),('model_sha256','model_snapshot.json')]:
                if original_row[key] and sha(target/'cache'/sid/name)!=original_row[key]:
                    raise ValueError('Mac cached geometry changed')
            for face in range(1,6):
                if sha(root/row['source_folder']/f'face{face}.jpg')!=json.loads(archive.read(
                    f'round2_scale/{sid}/{pid}/face{face}.json'))['source_sha256']:
                    raise ValueError('Mac photo differs')
            prefix=f'{sid}/{pid}/'
            names=[n for n in receipt['files'] if n.startswith(prefix)]
            required=['geometry.json']+[f'face{f}{suffix}' for f in range(1,6)
                       for suffix in ('.json','_prob.png','_mask.png')]
            if any(prefix+n not in receipt['files'] for n in required):
                raise ValueError('Incomplete panorama')
            destination=target/sid/pid
            if destination.exists():
                skipped+=1
                continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=Path(tempfile.mkdtemp(prefix='.sunpatch-merge-',dir=destination.parent))
            try:
                for name in names:
                    path=temporary/name[len(prefix):]
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open('round2_scale/'+name) as source, path.open('xb') as out:
                        shutil.copyfileobj(source,out)
                    if path.suffix=='.json':
                        value=json.loads(path.read_text())
                        def normalize(v):
                            if isinstance(v,dict):
                                for k,x in v.items():
                                    if k=='mask_path' and isinstance(x,str) and x.startswith(prefix):
                                        v[k]='data/sun_patch/auto/round2_scale/'+x
                                    else:
                                        normalize(x)
                            elif isinstance(v,list):
                                for x in v: normalize(x)
                        normalize(value)
                        path.write_text(json.dumps(value,indent=2)+'\n')
                # macOS rename fails for an existing nonempty destination. Never merge files
                # into a live directory: the inference process creates it before any output.
                try:
                    rename_exclusive(temporary,destination)
                    imported+=1
                except OSError:
                    if not destination.exists(): raise
                    skipped+=1
            finally:
                if temporary.exists(): shutil.rmtree(temporary)
        print(json.dumps({'imported_panoramas':imported,'skipped_existing_panoramas':skipped,
                          'global_receipts':'preserved; rerun Mac command after its current process finishes'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    parser.add_argument('--mac-root',type=Path,default=Path.home()/'HackPrinceton')
    args=parser.parse_args()
    merge(args.archive,args.mac_root.resolve())
