"""Losslessly pack only generated home exports; verify bytes before removing originals."""
import gzip
from pathlib import Path
import re
import sys
sys.dontwritebytecode=True
root=Path(__file__).resolve().parent/'site/data'
count=0
for path in root.glob('*.json'):
    if not re.fullmatch(r'[A-Za-z0-9]{11}\.json',path.name):continue
    original=path.read_bytes();packed=path.with_suffix('.json.gz')
    with gzip.open(packed,'wb',compresslevel=5) as stream:stream.write(original)
    with gzip.open(packed,'rb') as stream:assert stream.read()==original
    path.unlink();count+=1
print('Losslessly packed',count,'generated home exports.')
