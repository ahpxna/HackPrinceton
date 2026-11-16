"""Read-only Model API anchors and measured rigid registration for Showcase."""
import json
import math
import re
import base64
import threading
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
_pose_lock = threading.Lock()
_attempted = set()
_auth_failures = 0


def acquire_poses(sid):
    """Fetch only IDs/positions on demand, using the existing Model API credentials.

    Credential values remain in process memory. No signed URLs are requested.
    A failed model is not retried during this server run; authorization failure
    stops further remote acquisition after two failures.
    """
    global _auth_failures
    path = BASE / 'cache' / sid / 'tour_model_poses.json'
    with _pose_lock:
        if path.is_file() or sid in _attempted or _auth_failures >= 2:
            return path
        _attempted.add(sid)
        if not (BASE / 'site/data' / (sid + '.json.gz')).is_file():
            return path
        try:
            from api_cache import credential
            token = base64.b64encode((credential('MATTERPORT_API_KEY') + ':' + credential('MATTERPORT_API_TOKEN')).encode()).decode()
            body = json.dumps({'query': '{model(id:"' + sid + '"){panoLocations{id position{x y z}}}}'}).encode()
            request = urllib.request.Request('https://api.matterport.com/api/models/graph', data=body,
                headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + token})
            with urllib.request.urlopen(request, timeout=12) as response:
                result = json.load(response)
            poses = result.get('data', {}).get('model', {}).get('panoLocations', [])
            clean = [{'id': p['id'], 'position': {k: p['position'][k] for k in 'xyz'}} for p in poses]
            if clean:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(clean, separators=(',', ':')))
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                _auth_failures += 1
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError):
            pass
    return path


def model_data(sid, acquire=False):
    if not re.fullmatch(r'[A-Za-z0-9]{8,32}', sid):
        raise ValueError('Invalid model identifier')
    paths = [BASE / 'cache' / sid / 'pano_poses.json',
             BASE / 'cache' / sid / 'tour_model_poses.json',
             ROOT / 'data/cache/sun_patch_20261114' / sid / 'panos.json',
             ROOT / 'data/sun_check' / sid / 'pano_locations.json']
    poses, source = [], None
    for path in paths:
        if path.is_file():
            poses = json.loads(path.read_text())
            source = str(path.relative_to(ROOT))
            break
    if not poses and acquire:
        path = acquire_poses(sid)
        if path.is_file():
            poses = json.loads(path.read_text())
            source = str(path.relative_to(ROOT))
    anchors = []
    for pose in poses:
        point = pose.get('position', {})
        if all(isinstance(point.get(k), (float, int)) and math.isfinite(point[k]) for k in 'xyz'):
            anchors.append({'id': pose['id'], 'position': {k: point[k] for k in 'xyz'}})
    path = ROOT / 'data/claims/annotations' / (sid + '.json')
    notes = json.loads(path.read_text()) if path.is_file() else None
    return {'sid': sid, 'source': source, 'model_sweeps': anchors, 'notes': notes}


def fit_sweeps(sid, reported):
    import numpy as np
    source = model_data(sid)
    by_id = {p['id']: p['position'] for p in source['model_sweeps']}
    pairs, seen = [], set()
    for sweep in reported[:10000]:
        if not isinstance(sweep, dict) or not isinstance(sweep.get('position'), dict):
            continue
        ids = [sweep.get(k) for k in ('id', 'uuid', 'sid', 'map_id')] + sweep.get('aliases', [])
        match = next((key for key in ids if isinstance(key, str) and key in by_id), None)
        point = sweep.get('position', {})
        if not match or match in seen or sweep.get('enabled') is False:
            continue
        if not all(isinstance(point.get(k), (float, int)) and math.isfinite(point[k]) for k in 'xyz'):
            continue
        if not isinstance(sweep.get('id'), str):
            continue
        seen.add(match)
        pairs.append((match, sweep['id'], [by_id[match][k] for k in 'xyz'], [point[k] for k in 'xyz']))
    if len(pairs) < 4:
        raise ValueError('Fewer than four sweeps have matching stable Model API and SDK identifiers')
    a = np.array([p[2] for p in pairs]); b = np.array([p[3] for p in pairs])
    ac = a - a.mean(0); bc = b - b.mean(0)
    singular = np.linalg.svd(ac, compute_uv=False)
    if singular[1] < 0.01:
        raise ValueError('Matched sweeps do not span a usable plane')
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.eye(3); correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    shift = b.mean(0) - rotation @ a.mean(0)
    errors = np.linalg.norm(a @ rotation.T + shift - b, axis=1)
    rms, maximum = float(np.sqrt(np.mean(errors ** 2))), float(errors.max())
    return {'method': 'Rigid Kabsch fit; exact stable sweep IDs; metres; proper rotation, no scale',
            'source': source['source'], 'rotation': rotation.tolist(), 'translation': shift.tolist(),
            'count': len(pairs), 'rms_m': rms, 'max_m': maximum,
            'accepted': rms <= 0.05 and maximum <= 0.15,
            'pairs': [{'model_id': p[0], 'sdk_id': p[1], 'error_m': float(e)} for p, e in zip(pairs, errors)]}
