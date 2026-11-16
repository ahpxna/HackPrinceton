"""Opt-in acquisition. No secrets or download URLs are persisted."""
from pathlib import Path
import base64
import io
import json
import os
import re
import time
import urllib.request
import urllib.error
import urllib.parse
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'demo/cache'
AUTH_FAILURES = 0
ONLINE = False
CALLS = 0

def credential(name):
    if os.environ.get(name): return os.environ[name]
    for line in (ROOT/'.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key,value = line.split('=',1)
            if key.strip()==name: return value.strip().strip('\"\'')
    raise RuntimeError('Required credential unavailable; values suppressed')

def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items() if not any(s in k.lower() for s in ('authorization','api-key','token','secret'))}
    if isinstance(value,list): return [clean(v) for v in value]
    if isinstance(value,str) and re.search(r'https?://',value): return '[URL omitted]'
    return value

def request(path, endpoint=None, params=None, body=None, url=None, service='vexcel', live=False):
    global AUTH_FAILURES, CALLS
    path = CACHE/path
    if path.exists() and not live:
        if path.suffix=='.json':
            result=json.loads(path.read_text())
            if '_error' in result: raise RuntimeError(result['_error'])
            return result
        return path.read_bytes()
    if not ONLINE: raise RuntimeError('Offline cache missing: '+str(path.relative_to(CACHE)))
    if AUTH_FAILURES>=2: raise RuntimeError('Stopped after repeated HTTP 401/403')
    headers={}
    if service=='vexcel':
        url='https://api.vexcelgroup.com/v2/'+endpoint
        headers={'X-API-KEY':credential('VEXCEL_API_KEY')}
    elif service=='matterport':
        url='https://api.matterport.com/api/models/graph'
        token=base64.b64encode((credential('MATTERPORT_API_KEY')+':'+credential('MATTERPORT_API_TOKEN')).encode()).decode()
        headers={'Authorization':'Basic '+token}
    if params: url+='?'+urllib.parse.urlencode(params)
    if body is not None: headers['Content-Type']='application/json'
    req=urllib.request.Request(url,json.dumps(body).encode() if body is not None else None,headers)
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(6):
        try:
            CALLS+=1
            with urllib.request.urlopen(req,timeout=100) as response: raw=response.read()
            if path.suffix=='.json':
                result=json.loads(raw)
                path.write_text(json.dumps(clean(result),separators=(',',':')))
                if result.get('errors'): raise RuntimeError('Remote GraphQL error; details suppressed')
                return result if live else clean(result)
            with Image.open(io.BytesIO(raw)) as im:
                meta=im.info.get('xmp',b'')
                if isinstance(meta,bytes): meta=meta.decode(errors='replace')
                xmp=dict(re.findall(r'<Vexcel:([^>]+)>(.*?)</Vexcel:[^>]+>',meta))
                if xmp: path.with_suffix('.metadata.json').write_text(json.dumps(clean(xmp)))
                im.convert('RGB').save(path,quality=94)
            return path.read_bytes()
        except urllib.error.HTTPError as error:
            if error.code in (401,403): AUTH_FAILURES+=1
            # Cache the outcome of every failed attempt, without response URLs.
            (path.parent/(path.name+'.attempt'+str(attempt)+'.json')).write_text(json.dumps({'_error':'HTTP '+str(error.code)}))
            if error.code==429 and attempt<5:
                time.sleep(min(32,2**(attempt+1)));continue
            raise RuntimeError('Remote HTTP '+str(error.code)+'; details suppressed') from None
        except (urllib.error.URLError,TimeoutError):
            (path.parent/(path.name+'.failure.json')).write_text('{"_error":"Transport error; details suppressed"}')
            raise RuntimeError('Transport error; details suppressed') from None
