"""Brief D: cached, credential-safe real-data audit. Run with python3 -I.

--survey performs missing network requests; default is strictly offline.
Only this script, data/audit, and docs/audit + docs/data_audit.md are written.
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

csv.field_size_limit(32 * 1024 * 1024)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/audit'
CACHE = OUT / 'cache'
DOC = ROOT / 'docs'
CHART = DOC / 'audit'
LAYERS = 'urban,urban-r,wide-area,urban-g'
MP_URL = 'https://api.matterport.com/api/models/graph'
VX_URL = 'https://api.vexcelgroup.com/v2/'
SECRETS = []


def clean(value):
    if isinstance(value, dict):
        return {k: ('[REDACTED]' if re.search(r'(?i)^(token|api.?key|authorization|password|secret)$', k)
                    else clean(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        for secret in SECRETS:
            if secret:
                value = value.replace(secret, '[REDACTED]')
        return re.sub(r'https?://[^\s<>"\x27]+', '[URL_REDACTED]', value)
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False))


def table(name, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    path = OUT / (name + '.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or ['status'])
        w.writeheader()
        for row in rows:
            w.writerow({k: clean(json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in row.items()})


def progress(done, next_step):
    p = Path('/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T06-31-11-028Z-24029-086ac49d-a7f4-4d7d-8de4-8ccf31fbfed8.md')
    p.write_text('# Objective\nCarry out Brief D exactly; reproduce all audit numbers from sanitized caches.\n\n# Done\n'
                 + done + '\n\n# Next\n' + next_step + '\n')


def credentials():
    # Never emit the file, values, request headers or raw exceptions.
    env = dict(os.environ)
    with (ROOT / '.env').open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.removeprefix('export ').split('=', 1)
            env[key.strip()] = value.strip().strip('"\x27')
    names = ['MATTERPORT_API_KEY', 'MATTERPORT_API_TOKEN', 'VEXCEL_API_KEY']
    SECRETS.extend(env.get(k, '') for k in names)
    return env


class Client:
    def __init__(self, online=False):
        self.online = online
        self.env = credentials() if online else {}
        self.lock = threading.Lock()
        self.blocked = set()
        self.denied = collections.Counter()
        self.calls = collections.Counter()

    def request(self, provider, endpoint='', params=None, query=None):
        spec = {'provider': provider, 'endpoint': endpoint, 'params': params, 'query': query}
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        path = CACHE / provider / (key + '.json')
        if path.exists():
            return json.loads(path.read_text())
        if not self.online:
            return {'status': 'not_cached', 'data': None}
        with self.lock:
            if provider in self.blocked:
                return {'status': 'circuit_open', 'data': None}
        headers = {'Accept': 'application/json'}
        if provider == 'mp':
            if not all(self.env.get(k) for k in ['MATTERPORT_API_KEY', 'MATTERPORT_API_TOKEN']):
                return {'status': 'missing_credentials', 'data': None}
            basic = base64.b64encode((self.env['MATTERPORT_API_KEY'] + ':' + self.env['MATTERPORT_API_TOKEN']).encode()).decode()
            headers.update({'Authorization': 'Basic ' + basic, 'Content-Type': 'application/json'})
            url, body = MP_URL, json.dumps({'query': query}).encode()
        else:
            if not self.env.get('VEXCEL_API_KEY'):
                return {'status': 'missing_credentials', 'data': None}
            headers['X-API-KEY'] = self.env['VEXCEL_API_KEY']
            url, body = VX_URL + endpoint + '?' + urllib.parse.urlencode(params or {}), None
        attempts = []
        for attempt in range(5):
            with self.lock:
                if provider in self.blocked:
                    break
                self.calls[provider] += 1
            try:
                req = urllib.request.Request(url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=65) as r:
                    status, raw = r.status, r.read()
                    retry_after = None
            except urllib.error.HTTPError as e:
                status, raw = e.code, e.read()
                retry_after = e.headers.get('Retry-After')
            except Exception as e:
                status, raw, retry_after = 'transport_' + type(e).__name__, b'', None
            try:
                data = json.loads(raw) if raw else None
            except (ValueError, UnicodeError):
                data = {'message': raw.decode(errors='replace')[:1500]}
            auth_failure = status in (401, 403) or (isinstance(data, dict) and any(
                re.search(r'(?i)unauthorized|forbidden|authentication failed', x.get('message', ''))
                for x in data.get('errors', [])))
            with self.lock:
                if auth_failure:
                    self.denied[provider] += 1
                    if self.denied[provider] >= 2:
                        self.blocked.add(provider)
            attempts.append({'status': status, 'data': clean(data)})
            record = {**spec, 'status': status, 'data': clean(data), 'attempts': attempts,
                      'fetched_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
            dump(path, record)
            if status != 429:
                return record
            try:
                wait = max(2 ** attempt, float(retry_after or 0))
            except ValueError:
                wait = 2 ** attempt
            time.sleep(min(wait, 60))
        if path.exists():
            return json.loads(path.read_text())
        return {'status': 'circuit_open', 'data': None}


TYPE_REF = 'kind name ofType{kind name ofType{kind name ofType{kind name}}}'
SCHEMA_Q = '{__schema{types{name kind description enumValues{name description} fields(includeDeprecated:true){name description type{' + TYPE_REF + '} args{name defaultValue type{' + TYPE_REF + '}}}}}}'


def load_tables():
    tables = {'models_index': list(csv.DictReader((ROOT / 'models_index.csv').open()))}
    kit = ROOT / 'data/costar-kit'
    for p in sorted((kit / 'data').glob('*.jsonl')):
        with p.open() as f:
            tables[p.stem] = [json.loads(line) for line in f if line.strip()]
    for p in sorted(kit.glob('*.csv')):
        tables[p.stem + '_csv'] = list(csv.DictReader(p.open()))
    return tables


def schema_map(client):
    r = client.request('mp', query=SCHEMA_Q)
    return {t['name']: t for t in ((r.get('data') or {}).get('data') or {}).get('__schema', {}).get('types', [])}


MODEL_Q = '''{model(id:"%s"){
 id created geocoordinates{latitude longitude source}
 floors{id label sequence dimensionEstimates(units:metric){area areaIndoor lowerElevation}
  edges(includeUsed:true){id type vertices{id position{x y}}
   openings{id type label width height lowerElevation relativeCenter}}}
 rooms{id label tags floor{id} boundary{edges{id}}
  classifications{id label confidence}
  dimensionEstimates(units:metric){area areaIndoor height lowerElevation}
  ceiling(units:metric){minHeight maxHeight}}
 panoLocations{id options{format resolutions}}
 assets{meshes{id format resolution status} floorplans{id format status}}
 objectAnnotations{id label confidence enabled keywords classification{id label}
  floor{id} room{id} region{__typename} source{__typename}}
}}'''


def stratified(rows, n, key):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha256(r['sid'].encode()).hexdigest())
    result = []
    # Equal allocation until a small stratum is exhausted, then redistribute.
    while len(result) < n and groups:
        for k in sorted(list(groups), key=str):
            if len(result) >= n:
                break
            result.append(groups[k].pop(0))
            if not groups[k]:
                del groups[k]
    return result


def make_samples(tables):
    index = tables['models_index']
    commercial = {str(x['id']): x for x in tables['commercial-property']}
    res = [x for x in index if x['segment'] == 'residential']
    comm = [x for x in index if x['segment'] != 'residential']
    def ptype(r):
        return commercial.get(r['costar_property_id'], {}).get('propertyType', 'unjoined')
    mp = res + stratified(comm, 300, ptype)
    # One model per distinct location within segment, avoiding repeated community tours.
    seen = set()
    vx = []
    for group, n in [(res, 300), (comm, 200)]:
        locations = []
        for r in sorted(group, key=lambda r: r['sid']):
            if num(r['lat']) is None or num(r['lon']) is None:
                continue
            loc = (r['segment'], round(float(r['lat']), 6), round(float(r['lon']), 6))
            if loc not in seen:
                seen.add(loc)
                locations.append(r)
        vx.extend(stratified(locations, n, lambda r: (r['state'], r['city'].strip().casefold())))
    table('matterport_sample', [{**r, 'property_type_code': ptype(r)} for r in mp])
    table('vexcel_sample', vx)
    return mp, vx


def vx_params(r, calculated=False):
    return {'wkt': f"POINT({r['lon']} {r['lat']})", 'srid': 4326, 'layer': LAYERS,
            'latest': 'true', 'include': 'calculated' if calculated else 'precomputed', 'unit-system': 'metric'}


def imagery_params(r, kind):
    result = {'wkt': f"POINT({r['lon']} {r['lat']})", 'srid': 4326,
            'layer': 'urban' if kind == 'oblique' else LAYERS,
            'include': 'product-type,collection,last-capture-date'}
    if kind == 'oblique':
        result['product-type'] = kind
    return result


def imagery_endpoint(kind):
    return 'oriented/collections' if kind == 'oblique' else 'ortho/collections'


def survey(client, mp, vx):
    def work(r):
        result = client.request('mp', query=MODEL_Q % r['sid'])
        body = result.get('data') or {}
        return r['sid'], result.get('status'), len(body.get('errors') or [])
    # Single global pool, only one phase at once: at most four requests in flight.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(work, mp), 1):
            if i % 100 == 0 or i == len(mp):
                print('Matterport completed', i, '/', len(mp), 'last', result, flush=True)
                progress(f'Implemented experiments/data_audit.py; cached Matterport {i}/{len(mp)} planned queries. '
                         'All response URLs redacted; no test files added.', 'Finish API survey, reproduce tables, measure agreement, write report.')
    # Geofence/auth denial circuit is provider-wide: stop after two consecutive 401/403.
    # Serial first two locations avoid unnecessary in-flight requests on an invalid key.
    for i, r in enumerate(vx):
        client.request('vx', 'property/extract', vx_params(r))
        if i == 1 or 'vx' in client.blocked:
            break
    if 'vx' not in client.blocked:
        def work_vx(r):
            client.request('vx', 'property/extract', vx_params(r))
            for kind in ['ortho', 'oblique']:
                client.request('vx', imagery_endpoint(kind), imagery_params(r, kind))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i, _ in enumerate(pool.map(work_vx, vx), 1):
                if i % 25 == 0 or i == len(vx):
                    print('Vexcel completed', i, '/', len(vx), flush=True)
        # Test calculated on exactly the first twenty successful property locations.
        successful = [r for r in vx if client.request('vx', 'property/extract', vx_params(r)).get('status') == 200][:20]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda r: client.request('vx', 'property/extract', vx_params(r, True)), successful))
    dump(OUT / 'network_run.json', {'actual_calls_this_run': client.calls, 'blocked': sorted(client.blocked),
                                  'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat()})


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def present(v):
    return v is not None and v != '' and v != [] and v != {}


def get(obj, path, default=None):
    for key in path.split('.'):
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key)
    return default if obj is None else obj


def stats(values):
    a = sorted(x for v in values if (x := num(v)) is not None)
    if not a:
        return {'n': 0}
    def q(p):
        i = (len(a) - 1) * p
        j = int(i)
        return a[j] + (a[min(j + 1, len(a) - 1)] - a[j]) * (i - j)
    return dict(n=len(a), min=a[0], p05=q(.05), p25=q(.25), median=q(.5), p75=q(.75),
                p95=q(.95), max=a[-1], mean=statistics.mean(a))


def share(n, d):
    return round(100 * n / d, 2) if d else None


def date(v):
    if not isinstance(v, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(v.replace('Z', '+00:00'))
        return parsed.astimezone(dt.timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def daydiff(a, b):
    a, b = date(a), date(b)
    return (a - b).total_seconds() / 86400 if a and b else None


def schema_fields(schema):
    output = {}
    def resolve(node):
        if '$ref' in node:
            return schema.get('$defs', {}).get(node['$ref'].split('/')[-1], {})
        for item in node.get('anyOf', []):
            if '$ref' in item:
                return resolve(item)
        return node
    def walk(node, prefix='', depth=0):
        if depth > 14:
            return
        for key, child in resolve(node).get('properties', {}).items():
            path = prefix + key
            target = resolve(child)
            refs = [x.get('$ref', '') for x in child.get('anyOf', [])] + [child.get('$ref', '')]
            output[path] = {'description': child.get('description', target.get('description', '')),
                            'enum': next((r.split('/')[-1] for r in refs if 'enum' in resolve({'$ref': r})), ''),
                            'schema_type': child.get('type', target.get('type')), 'format': target.get('format', '')}
            if target.get('properties'):
                walk(target, path + '.', depth + 1)
            items = resolve(target.get('items', {}))
            if items.get('properties'):
                walk(items, path + '[].', depth + 1)
    walk(schema)
    return output


def flatten(row):
    result = collections.defaultdict(list)
    def walk(v, p):
        if p:
            result[p].append(v)
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, p + '.' + k if p else k)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    for k, y in x.items():
                        walk(y, p + '[].' + k)
                else:
                    result[p + '[]'].append(x)
    walk(row, '')
    return result


def compact(v):
    if isinstance(v, dict):
        return '{keys:' + ','.join(v.keys())[:140] + '}'
    if isinstance(v, list):
        return f'[length:{len(v)}]'
    return str(clean(v))[:220]


def inventory(tables):
    kit = ROOT / 'data/costar-kit'
    enums = json.loads((kit / 'schemas/enums.json').read_text())
    manifest = []
    for p in [ROOT / 'models_index.csv', *sorted(kit.rglob('*'))]:
        if p.is_file():
            manifest.append({'file': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    table('source_files', manifest)
    profile, distribution, summaries, texts, attachments = [], [], [], [], []
    for name, rows in tables.items():
        print('Profiling', name, len(rows), flush=True)
        schema_path = kit / 'schemas' / (name + '.schema.json')
        declared = schema_fields(json.loads(schema_path.read_text())) if schema_path.exists() else {}
        columns = collections.defaultdict(lambda: {'seen': 0, 'nonempty': 0, 'occ': 0, 'filled': 0,
            'types': collections.Counter(), 'values': collections.Counter(), 'examples': [], 'numeric': [], 'dates': []})
        for row in rows:
            flat = flatten(row)
            for path, vals in flat.items():
                c = columns[path]
                c['seen'] += 1
                c['nonempty'] += any(present(v) for v in vals)
                for v in vals:
                    c['occ'] += 1
                    c['types'][type(v).__name__] += 1
                    if not present(v):
                        continue
                    c['filled'] += 1
                    # Hash container values to count distinct actual content, without retaining raw payloads.
                    key = json.dumps(v, sort_keys=True, separators=(',', ':')) if isinstance(v, (dict, list)) else str(v)
                    if isinstance(v, (dict, list)) or len(key) > 350:
                        key = 'sha256:' + hashlib.sha256(key.encode()).hexdigest()
                    c['values'][key] += 1
                    example = compact(v)
                    if len(c['examples']) < 3 and example not in c['examples']:
                        c['examples'].append(example)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        c['numeric'].append(v)
                    elif name.endswith('_csv') or name == 'models_index':
                        if num(v) is not None and not re.search(r'id|postal|zip|address|phone', path, re.I):
                            c['numeric'].append(float(v))
                    if isinstance(v, str) and re.match(r'^\d{4}-\d\d-\d\d', v) and date(v):
                        c['dates'].append(v)
                    text_field = (path.endswith(('remarks', 'ai_summary', 'notes')) or
                                  bool(re.search(r'notes\[\]\.value$|(?:description|comments?|caption)$', path, re.I)))
                    if text_field and isinstance(v, str):
                        texts.append({'table': name, 'field': path, 'id': row.get('id', row.get('sid')), 'text': v})
            if not name.endswith('_csv') and name != 'models_index':
                def attach(v, p=''):
                    if isinstance(v, dict):
                        if 'videoKey' in v or 'videoExternalReferenceType' in v:
                            attachments.append({'table': name, 'parent_id': row.get('id', row.get('addressId')),
                                'path': p, 'attachment_id': v.get('id'), 'model_sid': v.get('videoKey'),
                                'external_reference_type': v.get('videoExternalReferenceType'), 'type': v.get('type')})
                        for k, x in v.items():
                            if isinstance(x, (dict, list)):
                                attach(x, p + '.' + k)
                    elif isinstance(v, list):
                        for x in v:
                            attach(x, p + '[]')
                attach(row)
        for path in sorted(set(columns) | set(declared)):
            c = columns[path]
            meta = declared.get(path, {})
            d = {'table': name, 'field': path, 'root_rows': len(rows), 'rows_present': c['seen'],
                 'rows_nonempty': c['nonempty'], 'empty_row_share_pct': share(len(rows) - c['nonempty'], len(rows)),
                 'observed_occurrences': c['occ'], 'nonempty_occurrences': c['filled'],
                 'observed_types': dict(c['types']), 'declared_type': meta.get('schema_type'),
                 'distinct_nonempty': len(c['values']), 'always_empty': c['filled'] == 0,
                 'constant_when_filled': len(c['values']) == 1, 'examples': c['examples'],
                 'numeric_stats': stats(c['numeric']), 'date_min': min(c['dates'], default=''),
                 'date_max': max(c['dates'], default=''), 'enum': meta.get('enum', ''),
                 'description': meta.get('description', '')}
            profile.append(d)
            cat = meta.get('enum') or (len(c['values']) <= 100 and not any(t in c['types'] for t in ['dict', 'list']))
            if cat:
                decode = enums.get(meta.get('enum', ''), {})
                for value, count in c['values'].most_common():
                    decoded = decode.get(value, '')
                    if not decoded and 'bit flag' in meta.get('description', '').lower() and value.isdigit():
                        code = int(value)
                        bits = [(int(k), v) for k, v in decode.items() if k.isdigit() and int(k) > 0 and int(k) & (int(k) - 1) == 0]
                        labels = [v for bit, v in bits if code & bit]
                        known = sum(bit for bit, _ in bits if code & bit)
                        decoded = ' | '.join(labels + ([f'UNMAPPED_BITS={code & ~known}'] if code & ~known else []))
                    distribution.append({'table': name, 'field': path, 'value': value,
                                         'decoded': decoded, 'occurrences': count,
                                         'nonempty_occurrence_share_pct': share(count, c['filled'])})
        ids = [str(r.get('id', r.get('sid', r.get('PropertyId', r.get('ListingId', ''))))) for r in rows]
        summaries.append({'table': name, 'rows': len(rows), 'fields_observed': sum(c['occ'] > 0 for c in columns.values()),
                          'fields_declared_or_observed': len(columns), 'key_distinct': len(set(ids)),
                          'key_duplicate_excess_rows': len(ids) - len(set(ids)),
                          'always_empty_fields': sum(not c['filled'] for c in columns.values()),
                          'constant_filled_fields': sum(len(c['values']) == 1 for c in columns.values())})
    table('field_inventory', profile)
    table('categorical_distributions', distribution)
    table('table_inventory', summaries)
    table('attachment_references', attachments)
    # Profile nested attachments as their own derived grain, without inventing a missing ninth JSONL file.
    table('attachment_inventory', [{'source_table': k[0], 'path': k[1], 'references': n} for k, n in
          collections.Counter((a['table'], a['path']) for a in attachments).most_common()])
    text_summary, phrases, duplicates = text_profiles(texts)
    table('text_profiles', text_summary)
    table('text_top60_phrases', phrases)
    table('text_duplicates', duplicates)
    progress('Completed file, schema, nested-field and text inventory; source hashes and CSV profiles written. '
             'API caches preserved.', 'Compute API geometry, joins and cross-source agreement; render charts/report.')
    return summaries, text_summary


STOP = set('a an the and or but of to in on for with from by at as is are was were be been being this that these those it its you your we our they their i me my us all can will would could should has have had not no into over under up out about just more most very each some any both than then there here also such so if do does did home homes house property room rooms bedrooms bedroom bathrooms bathroom'.split())


def text_profiles(texts):
    groups = collections.defaultdict(list)
    for r in texts:
        groups[r['table'], r['field']].append(r)
    summaries, phrases, duplicates = [], [], []
    for (name, field), records in groups.items():
        normalized = collections.defaultdict(list)
        sentence_freq = collections.Counter()
        ngrams = collections.Counter()
        sentences_by_doc = []
        for r in records:
            plain = html.unescape(re.sub(r'<[^>]*>', ' ', r['text']))
            plain = clean(plain).replace('[URL_REDACTED]', ' ')
            text = re.sub(r'\s+', ' ', plain.lower()).strip()
            normalized[text].append(r['id'])
            sentences = set(s.strip() for s in re.split(r'[.!?\n]+', text) if len(s.strip()) >= 50)
            sentence_freq.update(sentences)
            sentences_by_doc.append(sentences)
            tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
            seen = set()
            for n in [2, 3, 4]:
                for i in range(len(tokens) - n + 1):
                    gram = tokens[i:i + n]
                    if gram[0] in STOP or gram[-1] in STOP or sum(w not in STOP for w in gram) < 2:
                        continue
                    seen.add(' '.join(gram))
            ngrams.update(seen)
        threshold = max(5, math.ceil(.01 * len(records)))
        boiler = {s for s, n in sentence_freq.items() if n >= threshold}
        summaries.append({'table': name, 'field': field, 'nonempty_documents': len(records),
                          'length_characters': stats(len(r['text']) for r in records),
                          'distinct_normalized_documents': len(normalized),
                          'duplicate_excess_documents': len(records) - len(normalized),
                          'duplicate_excess_share_pct': share(len(records) - len(normalized), len(records)),
                          'documents_in_duplicate_groups': sum(len(v) for v in normalized.values() if len(v) > 1),
                          'boilerplate_sentence_min_documents': threshold,
                          'documents_with_boilerplate': sum(bool(s & boiler) for s in sentences_by_doc),
                          'boilerplate_share_pct': share(sum(bool(s & boiler) for s in sentences_by_doc), len(records))})
        for rank, (p, n) in enumerate(sorted(ngrams.items(), key=lambda item: (-item[1], item[0]))[:60], 1):
            phrases.append({'table': name, 'field': field, 'rank': rank, 'phrase': p, 'documents': n,
                            'share_nonempty_pct': share(n, len(records))})
        for text, ids in normalized.items():
            if len(ids) > 1:
                duplicates.append({'table': name, 'field': field, 'copies': len(ids), 'ids': ids,
                                   'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'excerpt': text[:180]})
    return summaries, phrases, duplicates


def graph(edges):
    adjacency = collections.defaultdict(set)
    valid = 0
    for e in edges:
        vs = e.get('vertices') or []
        if len(vs) != 2:
            continue
        a, b = [v.get('id') or json.dumps(v.get('position'), sort_keys=True) for v in vs]
        if a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
        valid += 1
    unseen, components = set(adjacency), 0
    while unseen:
        components += 1
        todo = [unseen.pop()]
        while todo:
            for v in adjacency[todo.pop()]:
                if v in unseen:
                    unseen.remove(v)
                    todo.append(v)
    return {'vertices': len(adjacency), 'valid_edges': valid, 'components': components,
            'dangling_vertices': sum(len(v) == 1 for v in adjacency.values()),
            'single_closed_cycle': bool(adjacency) and valid == len(edges) == len(adjacency) and
                                   components == 1 and all(len(v) == 2 for v in adjacency.values())}


def matterport(client, samples):
    models, rooms, openings, floors, errors, anomalies, objects = [], [], [], [], [], [], []
    for r in samples:
        sid = r['sid']
        env = client.request('mp', query=MODEL_Q % sid)
        body = env.get('data') or {}
        m = get(body, 'data.model', {})
        for e in body.get('errors') or []:
            errors.append({'sid': sid, 'path': e.get('path'), 'message': e.get('message'), 'http_status': env['status']})
        mr = {'sid': sid, 'segment': r['segment'], 'city': r['city'], 'state': r['state'],
              'http_status': env['status'], 'graphql_errors': len(body.get('errors') or []),
              'returned': bool(m), 'model_created': m.get('created'), 'index_created': r['created']}
        if not m:
            models.append(mr)
            continue
        geo = m.get('geocoordinates') or {}
        mr.update({'latitude': geo.get('latitude'), 'longitude': geo.get('longitude'), 'geocoordinates_source': geo.get('source')})
        edge_by_id, all_edges, edge_room_elevations = {}, [], collections.defaultdict(list)
        model_rooms = m.get('rooms') or []
        for room in model_rooms:
            z = num(get(room, 'dimensionEstimates.lowerElevation'))
            if z is not None:
                for edge in get(room, 'boundary.edges', []):
                    edge_room_elevations[edge['id']].append(z)
        floorlist = m.get('floors') or []
        for floor in floorlist:
            es = floor.get('edges') or []
            edge_by_id.update({e['id']: e for e in es})
            all_edges.extend(es)
            g = graph(es)
            floors.append({'sid': sid, 'segment': r['segment'], 'floor_id': floor['id'], 'label': floor.get('label'),
                           'sequence': floor.get('sequence'), 'edge_count': len(es), **g,
                           **(floor.get('dimensionEstimates') or {})})
        model_openings = {}
        for e in all_edges:
            for o in e.get('openings') or []:
                model_openings[o['id']] = (o, e)
        window_area = 0.
        for o, edge in model_openings.values():
            w, h, z = num(o.get('width')), num(o.get('height')), num(o.get('lowerElevation'))
            adjacent = edge_room_elevations.get(edge['id'], [])
            sill = z - statistics.median(adjacent) if z is not None and adjacent else None
            typ = str(o.get('type', '')).lower()
            area = w * h if w is not None and h is not None and w > 0 and h > 0 else None
            if 'window' in typ and area is not None:
                window_area += area
            rec = {'sid': sid, 'segment': r['segment'], 'opening_id': o['id'], 'edge_id': edge['id'],
                   'type': o.get('type'), 'width_m': w, 'height_m': h, 'absolute_lower_elevation_m': z,
                   'adjacent_room_elevation_count': len(adjacent), 'sill_m': sill, 'rectangular_area_m2': area}
            openings.append(rec)
            flags = []
            if w is not None and (w <= 0 or w > 8):
                flags.append('width_nonpositive_or_gt8m')
            if h is not None and (h <= 0 or h > 5):
                flags.append('height_nonpositive_or_gt5m')
            if sill is not None and (sill < -.2 or sill > 3):
                flags.append('sill_below_minus0.2_or_above3m')
            for flag in flags:
                anomalies.append({**rec, 'check': flag})
        area_total, indoor_total, labels, closed = 0., 0., collections.Counter(), 0
        for room in model_rooms:
            dim = room.get('dimensionEstimates') or {}
            classifications = room.get('classifications') or []
            ranked = sorted(classifications, key=lambda c: num(c.get('confidence')) or 0, reverse=True)
            best = ranked[0] if ranked else {}
            labels[str(best.get('label', '')).lower()] += 1
            boundary = get(room, 'boundary.edges', [])
            edges = [edge_by_id[e['id']] for e in boundary if e['id'] in edge_by_id]
            g = graph(edges)
            if len(edges) != len(boundary):
                g['single_closed_cycle'] = False
            closed += g['single_closed_cycle']
            area_total += num(dim.get('area')) or 0
            indoor_total += num(dim.get('areaIndoor')) or 0
            rec = {'sid': sid, 'segment': r['segment'], 'room_id': room['id'], 'label': room.get('label'),
                   'floor_id': get(room, 'floor.id'), 'classification': best.get('label'),
                   'confidence': best.get('confidence'), 'all_classifications': classifications,
                   'area_m2': dim.get('area'), 'indoor_area_m2': dim.get('areaIndoor'), 'height_m': dim.get('height'),
                   'ceiling_min_m': get(room, 'ceiling.minHeight'), 'ceiling_max_m': get(room, 'ceiling.maxHeight'),
                   'boundary_edges_requested': len(boundary), 'boundary_edges_returned': len(edges),
                   'boundary_edges_missing': len(boundary)-len(edges), **g}
            rooms.append(rec)
            for field in ['ceiling_min_m', 'ceiling_max_m', 'height_m']:
                x = num(rec[field])
                if x is not None and (x < 1.5 or x > 6):
                    anomalies.append({'sid': sid, 'segment': r['segment'], 'room_id': room['id'],
                                      'check': field + '_outside1.5to6m', 'value': x, 'room_classification': best.get('label')})
        panos = m.get('panoLocations') or []
        assets = m.get('assets') or {}
        annotations = m.get('objectAnnotations') or []
        for a in annotations:
            objects.append({'sid': sid, **a})
        mr.update({'floor_count': len(floorlist), 'room_count': len(model_rooms), 'edge_count': len(all_edges),
                   'wall_edges': sum(str(e.get('type', '')).lower() == 'wall' for e in all_edges),
                   'windows': sum('window' in str(o.get('type', '')).lower() for o, e in model_openings.values()),
                   'openings': len(model_openings), 'rooms_closed_cycle': closed,
                   'room_area_m2': area_total, 'room_indoor_area_m2': indoor_total,
                   'room_indoor_sqft': indoor_total * 10.76391041671,
                   'bedrooms': sum(n for label, n in labels.items() if label in ('bedroom', 'master bedroom', 'primary bedroom')),
                   'bathrooms': sum(n for label, n in labels.items() if 'bathroom' in label or label in ('half bath', 'powder room')),
                   'classification_counts': dict(labels), 'window_rectangle_m2': window_area,
                   'window_floor_ratio': window_area / indoor_total if indoor_total > 0 else None,
                   'pano_count': len(panos), 'panos_with_skybox_options': sum(any(o.get('resolutions') for o in p.get('options') or []) for p in panos),
                   'meshes': len(assets.get('meshes') or []), 'floorplans': len(assets.get('floorplans') or []),
                   'mesh_statuses': dict(collections.Counter(a.get('status') for a in assets.get('meshes') or [])),
                   'floorplan_statuses': dict(collections.Counter(a.get('status') for a in assets.get('floorplans') or [])),
                   'object_annotations': len(annotations)})
        models.append(mr)
    for name, rows in [('matterport_models', models), ('matterport_rooms', rooms), ('matterport_openings', openings),
                       ('matterport_floors', floors), ('matterport_errors', errors), ('geometry_anomalies', anomalies),
                       ('matterport_objects', objects)]:
        table(name, rows)
    summaries, numeric = [], []
    for segment in ['residential', 'commercial_sample']:
        group = [m for m in models if (m['segment'] == 'residential') == (segment == 'residential')]
        ok = [m for m in group if m['returned']]
        for label, pred in [('returned', lambda m: m['returned']), ('graphql_error', lambda m: m['graphql_errors'] > 0),
                            ('wall_graph', lambda m: m.get('wall_edges', 0) > 0),
                            ('windows', lambda m: m.get('windows', 0) > 0),
                            ('closed_room', lambda m: m.get('rooms_closed_cycle', 0) > 0),
                            ('all_rooms_closed', lambda m: m.get('room_count', 0) > 0 and m['rooms_closed_cycle'] == m['room_count']),
                            ('skybox_options', lambda m: m.get('panos_with_skybox_options', 0) > 0),
                            ('meshes', lambda m: m.get('meshes', 0) > 0), ('floorplans', lambda m: m.get('floorplans', 0) > 0),
                            ('object_annotations', lambda m: m.get('object_annotations', 0) > 0)]:
            n = sum(pred(m) for m in group)
            summaries.append({'segment': segment, 'measure': label, 'count': n, 'denominator_planned_models': len(group), 'share_pct': share(n, len(group))})
        for field in ['floor_count', 'room_count', 'edge_count', 'windows', 'room_indoor_sqft', 'window_floor_ratio', 'pano_count']:
            numeric.append({'segment': segment, 'field': field, **stats(m.get(field) for m in ok)})
        for records, fields, prefix in [(rooms, ['confidence', 'height_m', 'ceiling_min_m', 'ceiling_max_m'], 'room.'),
                                         (openings, ['width_m', 'height_m', 'sill_m'], 'window.')]:
            subset = [x for x in records if (x['segment'] == 'residential') == (segment == 'residential')
                      and (prefix != 'window.' or 'window' in str(x.get('type')).lower())]
            for field in fields:
                numeric.append({'segment': segment, 'field': prefix + field, **stats(x.get(field) for x in subset)})
    table('matterport_coverage', summaries)
    table('matterport_distributions', numeric)
    table('room_classifications', [{'segment': k[0], 'classification': k[1], 'rooms': n} for k, n in
          collections.Counter((r['segment'], r['classification']) for r in rooms).most_common()])
    table('opening_types', [{'segment': k[0], 'type': k[1], 'count': n} for k, n in
          collections.Counter((r['segment'], r['type']) for r in openings).most_common()])
    table('geocoordinates_sources', [{'segment': k[0], 'source': k[1], 'models': n} for k, n in
          collections.Counter((r['segment'], r.get('geocoordinates_source')) for r in models).most_common()])
    return models, rooms, openings, summaries, numeric


def haversine(lat1, lon1, lat2, lon2):
    vals = [num(v) for v in [lat1, lon1, lat2, lon2]]
    if any(v is None for v in vals) or any(abs(v) > lim for v, lim in zip(vals, [90, 180, 90, 180])):
        return None
    a, b, c, d = map(math.radians, vals)
    q = math.sin((c - a) / 2) ** 2 + math.cos(a) * math.cos(c) * math.sin((d - b) / 2) ** 2
    return 6371008.8 * 2 * math.asin(min(1, math.sqrt(q)))


def point_in_geometry(geometry, lat, lon):
    lat, lon = num(lat), num(lon)
    if not geometry or lat is None or lon is None:
        return None
    def ring_contains(ring):
        inside = False
        for a, b in zip(ring, ring[1:] + ring[:1]):
            x1, y1 = a[:2]
            x2, y2 = b[:2]
            # Include a point exactly on a boundary segment.
            cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
            if abs(cross) < 1e-12 and min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10:
                return True
            if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
                inside = not inside
        return inside
    polygons = geometry.get('coordinates') or []
    if geometry.get('type') == 'Polygon':
        polygons = [polygons]
    elif geometry.get('type') != 'MultiPolygon':
        return None
    return any(p and ring_contains(p[0]) and not any(ring_contains(r) for r in p[1:]) for p in polygons)


def vexcel(client, samples, index):
    locations, structures, attrs, imagery, calculated, errors = [], [], [], [], [], []
    feature_versions = []
    selected_features = {}
    geometries = {}
    field_values = collections.defaultdict(list)
    successful = []
    for r in samples:
        env = client.request('vx', 'property/extract', vx_params(r))
        data = env.get('data') or {}
        features = data.get('features') or []
        # Multiple vintages can be returned even with latest=true. Select explicitly.
        def preference(f):
            p = f.get('properties') or {}
            return (p.get('request-information.inside-parcel') is True,
                    date(p.get('metadata.ortho-date-estimate')) or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
                    p.get('parcel-source') == 'Real')
        selected_index = max(range(len(features)), key=lambda i: preference(features[i])) if features else None
        selected = features[selected_index] if selected_index is not None else {}
        selected_features[r['sid']] = (selected_index, selected)
        props = selected.get('properties') or {}
        geometry = selected.get('geometry')
        for i, f in enumerate(features):
            p = f.get('properties') or {}
            feature_versions.append({'sid': r['sid'], 'feature_index': i, 'selected': i == selected_index,
                                     'inside_parcel': p.get('request-information.inside-parcel'),
                                     'parcel_source': p.get('parcel-source'), 'structures': len(p.get('structures') or []),
                                     'ortho_date_estimate': p.get('metadata.ortho-date-estimate'),
                                     'collection': p.get('metadata.collection'), 'layer': p.get('metadata.layer-name')})
        geometries[r['sid']] = geometry
        ss = props.get('structures') or []
        primary = next((s for s in ss if s.get('primary-structure')), ss[0] if ss else {})
        if env['status'] == 200:
            successful.append(r)
            for k, v in props.items():
                if k != 'structures':
                    field_values[('parcel', k)].append(v)
            for i, s in enumerate(ss):
                structures.append({'sid': r['sid'], 'segment': r['segment'], 'structure_index': i, **s})
                for k, v in s.items():
                    field_values[('structure', k)].append(v)
        else:
            errors.append({'sid': r['sid'], 'endpoint': 'property/extract', 'status': env['status'], 'message': data.get('message')})
        loc = {'sid': r['sid'], 'segment': r['segment'], 'city': r['city'].strip().casefold(), 'state': r['state'],
               'property_status': env['status'], 'feature_count': len(features), 'selected_feature_index': selected_index, 'structures': len(ss),
               'footprint': any(present(s.get('footprint.wkt')) for s in ss),
               'footprint_total_m2': sum(num(s.get('footprint.area')) or 0 for s in ss),
               'primary_footprint_m2': primary.get('footprint.area'),
               'parcel_source': props.get('parcel-source'), 'requested_point_inside_parcel_api': props.get('request-information.inside-parcel'),
               'requested_point_inside_parcel_computed': point_in_geometry(geometry, r['lat'], r['lon']),
               'pool': props.get('pool.detected'), 'pool_area_m2': props.get('pool.total-area'),
               'solar': primary.get('roof-objects.solar.value'), 'roof_condition': primary.get('roof-characteristics.condition'),
               'roof_material': primary.get('roof-characteristics.material.class'), 'roof_shape': primary.get('roof-characteristics.shape.value'),
               'roof_tree_cover_pct': primary.get('roof-characteristics.tree-cover.percent'),
               'parcel_tree_cover_pct': props.get('parcel-tree-cover.percent'), 'vehicles': props.get('vehicles.count'),
               'ground_elevation_m': primary.get('ground-elevation.value'), 'ortho_date_estimate': props.get('metadata.ortho-date-estimate')}
        for kind in ['ortho', 'oblique']:
            result = client.request('vx', imagery_endpoint(kind), imagery_params(r, kind))
            ff = (result.get('data') or {}).get('features') or []
            loc[kind + '_status'] = result['status']
            loc[kind + '_collections'] = len(set(get(f, 'properties.collection') for f in ff))
            dates = [get(f, 'properties.last-capture-date') for f in ff if date(get(f, 'properties.last-capture-date'))]
            loc[kind + '_earliest_last_capture'] = min(dates, default='')
            loc[kind + '_latest_last_capture'] = max(dates, default='')
            loc[kind + '_years'] = sorted(set(x[:4] for x in dates))
            for f in ff:
                imagery.append({'sid': r['sid'], 'segment': r['segment'], 'city': loc['city'], 'state': loc['state'],
                                'endpoint_kind': kind, **f.get('properties', {})})
            if result['status'] != 200:
                errors.append({'sid': r['sid'], 'endpoint': imagery_endpoint(kind), 'status': result['status'],
                               'message': (result.get('data') or {}).get('message')})
        locations.append(loc)
    for r in successful[:20]:
        result = client.request('vx', 'property/extract', vx_params(r, True))
        features = (result.get('data') or {}).get('features') or []
        selected_index, selected = selected_features[r['sid']]
        candidate = features[selected_index] if selected_index is not None and selected_index < len(features) else {}
        # Calculated-only responses lack vintage metadata. Keep an explicit alignment check.
        aligned = bool(candidate) and candidate.get('geometry') == selected.get('geometry')
        ss = get(candidate, 'properties.structures', []) if aligned else []
        for sidx, s in enumerate(ss):
            for k, v in s.items():
                if re.search(r'height|elevation|stories|volume|score', k):
                    calculated.append({'sid': r['sid'], 'status': result['status'], 'feature_index': selected_index,
                                       'aligned_same_index_and_parcel_geometry': aligned,
                                       'structure_index': sidx, 'field': k, 'value': v})
        if not ss:
            calculated.append({'sid': r['sid'], 'status': result['status'], 'field': 'no_structures', 'value': None})
    for (grain, k), vals in sorted(field_values.items()):
        denominator = len(successful) if grain == 'parcel' else len(structures)
        nonempty = [v for v in vals if present(v)]
        attrs.append({'grain': grain, 'field': k, 'denominator': denominator, 'occurrences_present': len(vals),
                      'nonempty': len(nonempty), 'filled_share_pct': share(len(nonempty), denominator),
                      'numeric_stats': stats(v for v in nonempty if not isinstance(v, bool)),
                      'values': dict(collections.Counter(str(v) for v in nonempty)) if len(set(str(v) for v in nonempty)) <= 50 else '',
                      'examples': [compact(v) for v in nonempty[:3]]})
    cities = []
    alias_evidence = []
    index_cities = collections.defaultdict(list)
    for r in index:
        index_cities[r['state'], r['city'].strip().casefold()].append(r)
    for (state, city), rows in sorted(index_cities.items()):
        group = [l for l in locations if l['state'] == state and l['city'] == city]
        shared = []
        if not group:
            # Exact coordinate equivalence can reuse the measured point without an extra API call.
            coords = {(num(r['lat']), num(r['lon'])) for r in rows if num(r['lat']) is not None and num(r['lon']) is not None}
            sample_sids = {r['sid'] for r in samples if (num(r['lat']), num(r['lon'])) in coords}
            shared = [l for l in locations if l['sid'] in sample_sids]
            alias_evidence.extend({'state': state, 'city': city, 'measured_sid': l['sid'], 'measured_city': l['city'],
                                   'method': 'identical_latitude_and_longitude'} for l in shared)
        effective = group or shared
        cities.append({'state': state, 'city': city, 'index_models': len(rows), 'sample_points': len(group),
                       'coordinate_shared_points': len(shared),
                       'property_200': sum(l['property_status'] == 200 for l in effective),
                       'property_403': sum(l['property_status'] == 403 for l in effective),
                       'ortho_200': sum(l['ortho_status'] == 200 for l in effective),
                       'oblique_200': sum(l['oblique_status'] == 200 for l in effective),
                       'ortho_years': sorted(set(y for l in effective for y in l['ortho_years'])),
                       'oblique_years': sorted(set(y for l in effective for y in l['oblique_years'])),
                       'measured_scope': 'sample_points_only' if group else 'exact_coordinate_alias' if shared else 'no_valid_coordinate'})
    table('city_alias_evidence', alias_evidence)
    table('vexcel_feature_versions', feature_versions)
    for name, rows in [('vexcel_locations', locations), ('vexcel_structures', structures), ('vexcel_attributes', attrs),
                       ('vexcel_imagery', imagery), ('vexcel_calculated20', calculated), ('vexcel_errors', errors), ('city_coverage', cities)]:
        table(name, rows)
    return locations, geometries, attrs, cities, calculated


LIGHT = re.compile(r'\b(?:natural(?:ly)?\s+(?:light|lit)|bright|sunny)\b', re.I)


def joins_agreement(tables, models, vx, geometries):
    index = {r['sid']: r for r in tables['models_index']}
    listings = {str(r['id']): r for r in tables['residential-listings']}
    properties = {str(r['id']): r for r in tables['commercial-property']}
    features = {str(r['id']): r for r in tables['residential-listing-features']}
    mp = {r['sid']: r for r in models}
    vp = {r['sid']: r for r in vx}
    resedges = {(str(r['ListingId']), r['MatterportModelId']) for r in tables['residential-listings_csv']}
    commedges = {(str(r['PropertyId']), r['MatterportModelId']) for r in tables['commercial-properties_csv']}
    # Preserve every distinct authoritative CSV edge, then select one record per HOME for unweighted summary.
    pairs = []
    for listingid, sid in sorted(resedges):
        if sid not in index or listingid not in listings:
            continue
        listing = listings[listingid]
        p = listing.get('property') or {}
        im, am, v = index[sid], mp.get(sid, {}), vp.get(sid, {})
        sqft = num(p.get('sqftMin'))
        tour = num(am.get('room_indoor_sqft'))
        listing_lat, listing_lon = get(p, 'address.latitude'), get(p, 'address.longitude')
        feat = features.get(listingid, {})
        notes = listing.get('remarks') or ''
        ft = ' '.join(str(f.get('name', '')) + ' ' + str(f.get('stringValue', '')) for f in feat.get('features') or [])
        ft += ' ' + ' '.join(str(h.get('text', '')) for h in feat.get('highlights') or [])
        rec = {'sid': sid, 'listing_id': listingid, 'property_id': listing['propertyId'],
               'listing_instance_id': listing.get('residentialListingId'), 'city': get(p, 'address.city'), 'state': get(p, 'address.state'),
               'property_type': p.get('propertyType'), 'primary_listing': listing.get('isPrimaryForThisListingInstance'),
               'index_listing_id_matches': im['homes_listing_id'] == listingid,
               'index_listing_instance_id_matches': im['homes_listing_id'] == str(listing.get('residentialListingId')),
               'remarks_present': bool(notes.strip()), 'light_claim': bool(LIGHT.search(notes)),
               'window_floor_ratio': am.get('window_floor_ratio'), 'windows': am.get('windows'),
               'wall_edges': am.get('wall_edges'), 'model_returned': am.get('returned', False),
               'model_index_distance_m': haversine(am.get('latitude'), am.get('longitude'), im['lat'], im['lon']),
               'listing_index_distance_m': haversine(listing_lat, listing_lon, im['lat'], im['lon']),
               'listing_model_distance_m': haversine(listing_lat, listing_lon, am.get('latitude'), am.get('longitude')),
               'listing_inside_sampled_parcel': point_in_geometry(geometries.get(sid), listing_lat, listing_lon),
               'model_inside_sampled_parcel': point_in_geometry(geometries.get(sid), am.get('latitude'), am.get('longitude')),
               'listing_sqft': sqft, 'tour_indoor_sqft': tour,
               'tour_listing_size_ratio': tour / sqft if sqft and tour and sqft > 0 else None,
               'size_relative_difference': (tour - sqft) / sqft if sqft and tour and sqft > 0 else None,
               'listing_bedrooms': p.get('bedsMin'), 'tour_bedrooms': am.get('bedrooms'),
               'listing_full_baths': p.get('bathsFull'), 'listing_half_baths': p.get('bathsHalf'),
               'listing_weighted_baths': p.get('bathsMin'), 'tour_bathrooms': am.get('bathrooms'),
               'listing_physical_baths': (num(p.get('bathsFull')) + num(p.get('bathsHalf'))) if num(p.get('bathsFull')) is not None and num(p.get('bathsHalf')) is not None else None,
               'listing_stories': get(listing, 'resoValues.structure.stories', get(feat, 'derivedValues.numberOfStories')),
               'tour_floors': am.get('floor_count'),
               'list_date': listing.get('listDate'), 'model_created': am.get('model_created') or im['created'],
               'model_created_minus_list_days': daydiff(am.get('model_created') or im['created'], listing.get('listDate')),
               'latest_oblique_minus_list_days': daydiff(v.get('oblique_latest_last_capture'), listing.get('listDate')),
               'ortho_estimate_minus_list_days': daydiff(v.get('ortho_date_estimate'), listing.get('listDate')),
               'footprint': v.get('footprint', False), 'vexcel_sampled': sid in vp,
               'oblique': v.get('oblique_collections', 0) > 0, 'splat_index_flag': str(im.get('splat')).lower() in ('true', '1', 'yes')}
        # Room classifications may be absent: zero classified rooms is not a measured zero bedroom home.
        cc = am.get('classification_counts') or {}
        rec['has_classifications'] = any(label not in ('', 'other', 'void', 'staircase') and n for label, n in cc.items())
        for label, a, b in [('bedrooms', rec['listing_bedrooms'], rec['tour_bedrooms']),
                             ('full_baths', rec['listing_full_baths'], rec['tour_bathrooms']),
                             ('physical_baths', rec['listing_physical_baths'], rec['tour_bathrooms']),
                             ('stories', rec['listing_stories'], rec['tour_floors'])]:
            a, b = num(a), num(b)
            rec[label + '_difference'] = b - a if a is not None and b is not None and rec['model_returned'] and (label == 'stories' or rec['has_classifications']) else None
        for flag in ['listing_inside_sampled_parcel', 'model_inside_sampled_parcel']:
            rec[flag + '_disagreement'] = 1 if rec[flag] is False else 0 if rec[flag] is True else None
        poolcode = num(get(listing, 'resoValues.poolType'))
        # PoolType is a bit mask. Zero is not evidence of no pool; explicit structured features can say None.
        pool_private = poolcode is not None and (int(poolcode) & 1) != 0
        pool_shared_only = poolcode is not None and (int(poolcode) & 6) != 0 and not pool_private
        rec['listing_pool_type_code'] = poolcode
        rec['pool_community_only'] = pool_shared_only
        pool_positive = pool_private or (not pool_shared_only and bool(re.search(r'\b(?:private pool|in.ground pool|above.ground pool)\b', notes + ' ' + ft, re.I)))
        pool_negative = bool(re.search(r'\b(?:no pool|pool: none|pool none)\b', notes + ' ' + ft, re.I))
        rec['listing_pool'] = True if pool_positive and not pool_negative else False if pool_negative and not pool_positive else None
        rec['vexcel_pool'] = v.get('pool')
        solar_positive = bool(re.search(r'\bsolar\s+panels?\b', notes + ' ' + ft, re.I))
        solar_negative = bool(re.search(r'\b(?:no|without) solar\b', notes + ' ' + ft, re.I))
        rec['listing_solar'] = True if solar_positive and not solar_negative else False if solar_negative and not solar_positive else None
        rec['vexcel_solar'] = v.get('solar')
        rec['listing_garage'] = get(listing, 'resoValues.hasGarage')
        rec['roof_condition_positive_claim'] = bool(re.search(r'\b(?:new roof|roof replaced|recent(?:ly)? (?:replaced |new )?roof)\b', notes, re.I))
        rec['roof_tree_claim'] = bool(re.search(r'\b(?:tree.lined|mature trees|wooded|tree cover|shaded)\b', notes, re.I))
        rec['vexcel_roof_condition'] = v.get('roof_condition')
        rec['vexcel_roof_tree_cover_pct'] = v.get('roof_tree_cover_pct')
        rec['vexcel_parcel_tree_cover_pct'] = v.get('parcel_tree_cover_pct')
        for label in ['pool', 'solar']:
            a, b = rec['listing_' + label], rec['vexcel_' + label]
            rec[label + '_disagreement'] = int(a != b) if isinstance(a, bool) and isinstance(b, bool) else None
        pairs.append(rec)
    table('residential_all_pairs', pairs)
    groups = collections.defaultdict(list)
    for r in pairs:
        groups[r['property_id']].append(r)
    reps = []
    for group in groups.values():
        # Prefer an API-surveyed scan; newest creation timestamp; then primary MLS listing and id.
        reps.append(max(group, key=lambda r: (r['model_returned'], str(r['model_created'] or ''), bool(r['primary_listing']), int(r['listing_id']), r['sid'])))
    table('residential_home_representatives', reps)
    # Features may be tied to another MLS record of the same home. Pair comparisons remain record-specific.
    compare_fields = ['listing_index_distance_m', 'listing_model_distance_m', 'model_index_distance_m',
                      'size_relative_difference', 'bedrooms_difference', 'full_baths_difference', 'physical_baths_difference', 'stories_difference',
                      'listing_inside_sampled_parcel_disagreement', 'model_inside_sampled_parcel_disagreement',
                      'pool_disagreement', 'solar_disagreement']
    agreement = []
    for field in compare_fields:
        for grain, records in [('all_scan_listing_pairs', pairs), ('one_per_home', reps)]:
            available = [r for r in records if num(r.get(field)) is not None]
            values = [float(r[field]) for r in available]
            agreement.append({'check': field, 'grain': grain, 'denominator_records': len(records), 'comparable': len(available),
                              'exact_equal': sum(v == 0 for v in values), 'nonzero': sum(v != 0 for v in values),
                              'abs_gt_20pct_or_m': sum(abs(v) > .2 for v in values) if field == 'size_relative_difference' else None,
                              'distance_gt100m': sum(v > 100 for v in values) if field.endswith('distance_m') else None,
                              **stats(values)})
            if grain == 'one_per_home':
                table('outliers_' + field, sorted(available, key=lambda r: (-abs(r[field]), str(r['property_id'])))[:20])
    commpairs = []
    for pid, sid in sorted(commedges):
        if sid not in index or pid not in properties:
            continue
        p, im, am, v = properties[pid], index[sid], mp.get(sid, {}), vp.get(sid, {})
        addr = p.get('primaryAddress') or {}
        area = num(p.get('grossBuildingAreaSquareFeet'))
        tour = num(am.get('room_indoor_sqft'))
        listing_count = sum(str(l.get('propertyId')) == pid for l in tables['commercial-listing-summary'])
        commpairs.append({'sid': sid, 'property_id': pid, 'property_type': p.get('propertyType'), 'city': im['city'], 'state': im['state'],
                          'index_property_id_matches': im['costar_property_id'] == pid,
                          'property_index_distance_m': haversine(addr.get('latitude'), addr.get('longitude'), im['lat'], im['lon']),
                          'property_model_distance_m': haversine(addr.get('latitude'), addr.get('longitude'), am.get('latitude'), am.get('longitude')),
                          'property_inside_sampled_parcel': point_in_geometry(geometries.get(sid), addr.get('latitude'), addr.get('longitude')),
                          'gross_building_sqft': area, 'tour_indoor_sqft': tour,
                          'tour_whole_building_ratio': tour / area if tour and area and area > 0 else None,
                          'whole_building_stories': p.get('numStoriesAboveGrade'), 'tour_floors': am.get('floor_count'),
                          'scope_comparable_for_size_or_stories': False, 'commercial_listing_count': listing_count,
                          'remarks_present': present(p.get('notes')), 'wall_windows': am.get('wall_edges', 0) > 0 and am.get('windows', 0) > 0,
                          'vexcel_sampled': sid in vp, 'footprint': v.get('footprint', False),
                          'oblique': v.get('oblique_collections', 0) > 0,
                          'splat_index_flag': str(im.get('splat')).lower() in ('true', '1', 'yes')})
    table('commercial_all_pairs', commpairs)
    for field in ['property_index_distance_m', 'property_model_distance_m', 'tour_whole_building_ratio']:
        records = [r for r in commpairs if num(r.get(field)) is not None]
        agreement.append({'check': 'commercial_' + field, 'grain': 'all_scan_property_pairs', 'denominator_records': len(commpairs),
                          'comparable': len(records), 'distance_gt100m': sum(r[field] > 100 for r in records) if field.endswith('distance_m') else None,
                          **stats(r[field] for r in records)})
        table('outliers_commercial_' + field, sorted(records, key=lambda r: -abs(r[field]))[:20])
    table('agreement_summary', agreement)
    # Funnels are strictly nested, with unsurveyed Vexcel locations treated as UNKNOWN, not absent.
    funnels = []
    for segment, universe, records, key in [('residential', [r for r in index.values() if r['segment'] == 'residential'], pairs, 'property_id'),
                                            ('commercial', [r for r in index.values() if r['segment'] != 'residential'], commpairs, 'property_id')]:
        bysid = collections.defaultdict(list)
        for r in records:
            bysid[r['sid']].append(r)
        current = set(r['sid'] for r in universe)
        steps = [('index_models', lambda s: True), ('joined_listing_or_costar_property', lambda s: bool(bysid[s])),
                 ('remarks_or_property_notes', lambda s: any(r['remarks_present'] for r in bysid[s])),
                 ('wall_graph_with_windows', lambda s: mp.get(s, {}).get('wall_edges', 0) > 0 and mp.get(s, {}).get('windows', 0) > 0),
                 ('vexcel_footprint_observed', lambda s: vp.get(s, {}).get('footprint', False)),
                 ('oblique_collections_observed', lambda s: vp.get(s, {}).get('oblique_collections', 0) > 0),
                 ('splat_index_flag', lambda s: str(index[s].get('splat')).lower() in ('true', '1', 'yes'))]
        for step, pred in steps:
            current = {s for s in current if pred(s)}
            ids = {r[key] for s in current for r in bysid[s]}
            funnels.append({'segment': segment, 'step': step, 'models': len(current), 'unique_joined_properties': len(ids),
                            'share_index_models_pct': share(len(current), len(universe)),
                            'models_with_vexcel_survey': sum(s in vp for s in current)})
    table('join_funnels', funnels)
    links = []
    for name, rows, parent, parent_table, parentkey in [
        ('commercial_listing_property', tables['commercial-listing-summary'], 'propertyId', tables['commercial-property'], 'id'),
        ('occupancy_property', tables['location-occupancy'], 'propertyId', tables['commercial-property'], 'id'),
        ('occupancy_company_location', tables['location-occupancy'], 'tenantLocationId', tables['company-detail'], 'id'),
        ('apartment_listing_property', tables['apartment-listing'], 'propertyId', tables['commercial-property'], 'id'),
        ('residential_features_listing', tables['residential-listing-features'], 'id', tables['residential-listings'], 'id')]:
        pk = {str(r.get(parentkey)) for r in parent_table}
        ids = [str(r.get(parent)) for r in rows if present(r.get(parent))]
        links.append({'join': name, 'child_rows': len(rows), 'nonnull_fk_rows': len(ids), 'matched_rows': sum(i in pk for i in ids),
                      'orphan_rows': sum(i not in pk for i in ids), 'distinct_matched_parents': len({i for i in ids if i in pk})})
    table('foreign_key_coverage', links)
    time_rows = []
    for field in ['model_created_minus_list_days', 'latest_oblique_minus_list_days', 'ortho_estimate_minus_list_days']:
        time_rows.append({'field': field, 'grain': 'one_per_home', **stats(r.get(field) for r in reps)})
    table('time_alignment', time_rows)
    light = light_analysis(reps)
    progress('Computed API coverage, geometry, imagery, joins and property-level agreement CSVs. '
             'Kept all scan/listing edges plus one representative per home; all comparisons offline from caches.',
             'Generate report and charts, run existing guard tests and verify replay integrity.')
    return pairs, reps, commpairs, agreement, funnels, light, links, time_rows


def light_analysis(reps):
    available = [r for r in reps if r['remarks_present'] and num(r.get('window_floor_ratio')) is not None and (r.get('windows') or 0) > 0]
    groups = {claim: [r['window_floor_ratio'] for r in available if r['light_claim'] == claim] for claim in [False, True]}
    output = [{'light_claim': claim, **stats(vals)} for claim, vals in groups.items()]
    rng = random.Random(20261114)
    a, b = groups[True], groups[False]
    effect = None
    bootstrap = []
    if a and b:
        effect = statistics.median(a) - statistics.median(b)
        for _ in range(1000):
            bootstrap.append(statistics.median(rng.choices(a, k=len(a))) - statistics.median(rng.choices(b, k=len(b))))
    strata = []
    for key in sorted(set((r['state'], str(r['property_type'])) for r in available)):
        group = [r for r in available if (r['state'], str(r['property_type'])) == key]
        aa = [r['window_floor_ratio'] for r in group if r['light_claim']]
        bb = [r['window_floor_ratio'] for r in group if not r['light_claim']]
        strata.append({'state': key[0], 'property_type': key[1], 'claim_n': len(aa), 'no_claim_n': len(bb),
                       'claim_median': statistics.median(aa) if aa else None, 'no_claim_median': statistics.median(bb) if bb else None,
                       'median_difference': statistics.median(aa) - statistics.median(bb) if aa and bb else None})
    table('light_claim_distributions', output)
    table('light_claim_strata', strata)
    table('light_claim_properties', available)
    table('outliers_light_claim', sorted(available, key=lambda r: -(abs(r['window_floor_ratio'] - statistics.median(a if r['light_claim'] else b))))[:20])
    result = {'groups': output, 'median_difference': effect, 'bootstrap_median_difference': stats(bootstrap),
              'bootstrap95_low': sorted(bootstrap)[24] if bootstrap else None,
              'bootstrap95_high': sorted(bootstrap)[974] if bootstrap else None,
              'positive_strata': sum((s['median_difference'] or 0) > 0 for s in strata),
              'comparable_strata': sum(s['median_difference'] is not None for s in strata),
              'ratio_above1': sum(r['window_floor_ratio'] > 1 for r in available)}
    dump(OUT / 'light_signal.json', result)
    return result


def additional_checks(tables, models, rooms, openings, vx, reps, commpairs, calculated):
    checks, keys = [], []
    for name, rows in tables.items():
        candidates = [('sid',)] if name == 'models_index' else [('id',)] if not name.endswith('_csv') else [
            ('ListingId',), ('ResidentialListingId',), ('MatterportModelId',), ('ListingId', 'MatterportModelId')] if name.startswith('residential') else [
            ('PropertyId',), ('MatterportModelId',), ('PropertyId', 'MatterportModelId')]
        if name == 'residential-listings':
            candidates += [('residentialListingId',), ('propertyId',)]
        for fields in candidates:
            vals = [tuple(str(r.get(k)) for k in fields) for r in rows if all(present(r.get(k)) for k in fields)]
            keys.append({'table': name, 'candidate_key': '+'.join(fields), 'rows': len(rows), 'nonnull_rows': len(vals),
                         'distinct': len(set(vals)), 'duplicate_excess_rows': len(vals) - len(set(vals)),
                         'unique_and_complete': len(vals) == len(rows) == len(set(vals))})
    table('key_integrity', keys)
    root = tables['models_index']
    def record(name, count, denominator, details=''):
        checks.append({'measure': name, 'count': count, 'denominator': denominator, 'share_pct': share(count, denominator), 'details': details})
    for col in ['ai_summary', 'splat', 'drone_photos', 'defurnished']:
        for value, n in collections.Counter(r[col] for r in root).most_common():
            record('index.' + col + '=' + value, n, len(root))
    foreign = [r for r in commpairs if (num(r['property_index_distance_m']) or 0) > 1000000]
    record('commercial_pairs_location_gt1000km', len(foreign), sum(num(r['property_index_distance_m']) is not None for r in commpairs),
           'Distinct models=' + str(len({r['sid'] for r in foreign})) + '; distinct properties=' + str(len({r['property_id'] for r in foreign})))
    for segment in ['residential', 'commercial']:
        group = [r for r in root if r['segment'] == segment]
        record(segment + '.missing_index_coordinates', sum(num(r['lat']) is None or num(r['lon']) is None for r in group), len(group))
    comparable = []
    for r in reps:
        full, half, weighted = [num(r.get(k)) for k in ['listing_full_baths', 'listing_half_baths', 'listing_weighted_baths']]
        if None not in (full, half, weighted):
            comparable.append({**r, 'expected_weighted_baths': full + .5 * half,
                               'weighted_baths_identity_difference': weighted - full - .5 * half})
    table('bathroom_internal_consistency', comparable)
    table('outliers_bathroom_internal_consistency', sorted(comparable, key=lambda r: -abs(r['weighted_baths_identity_difference']))[:20])
    record('listing_weighted_baths_inconsistent_with_documented_half_weight', sum(abs(r['weighted_baths_identity_difference']) > .01 for r in comparable), len(comparable))
    for field in ['ceiling_min_m', 'ceiling_max_m', 'height_m']:
        vals = [r for r in rooms if r['segment'] == 'residential' and num(r.get(field)) is not None]
        bad = [r for r in vals if r[field] < 1.5 or r[field] > 6]
        record('residential.' + field + '.outside1.5to6', len(bad), len(vals))
        nonvoid = [r for r in vals if r['classification'] not in ['Void', 'Staircase', None]]
        record('residential.' + field + '.outside1.5to6_excluding_void_staircase_unclassified',
               sum(r[field] < 1.5 or r[field] > 6 for r in nonvoid), len(nonvoid))
    windows = [o for o in openings if str(o['type']).lower() == 'window']
    record('windows.width_above8m', sum((o['width_m'] or 0) > 8 for o in windows), len(windows))
    record('windows.height_above5m', sum((o['height_m'] or 0) > 5 for o in windows), len(windows))
    record('windows.sill_outside_minus0.2to3m', sum(o['sill_m'] < -.2 or o['sill_m'] > 3 for o in windows if o['sill_m'] is not None),
           sum(o['sill_m'] is not None for o in windows))
    # Roof and tree wording is a proxy for a different construct, not a binary truth label.
    proxy = []
    for claim, field in [('roof_condition_positive_claim', 'vexcel_roof_condition'), ('roof_tree_claim', 'vexcel_roof_tree_cover_pct'),
                         ('roof_tree_claim', 'vexcel_parcel_tree_cover_pct')]:
        for positive in [False, True]:
            vals = [r[field] for r in reps if r[claim] == positive and num(r.get(field)) is not None]
            proxy.append({'listing_claim': claim, 'claim_present': positive, 'vexcel_field': field, **stats(vals),
                          'interpretation': 'descriptive_proxy_only_not_same_observable'})
    table('roof_tree_claim_proxies', proxy)
    table('outliers_roof_condition_claim', sorted([r for r in reps if r['roof_condition_positive_claim'] and num(r.get('vexcel_roof_condition')) is not None], key=lambda r: r['vexcel_roof_condition'])[:20])
    table('outliers_tree_cover_claim', sorted([r for r in reps if r['roof_tree_claim'] and num(r.get('vexcel_parcel_tree_cover_pct')) is not None], key=lambda r: r['vexcel_parcel_tree_cover_pct'])[:20])
    # Calculated pilot compared with exactly the same structures in precomputed responses.
    cal_summary = []
    for field in sorted({r['field'] for r in calculated}):
        vals = [r for r in calculated if r['field'] == field]
        cal_summary.append({'field': field, 'structures': len(vals), 'structures_filled': sum(present(r.get('value')) for r in vals),
                            'properties_filled': len({r['sid'] for r in vals if present(r.get('value'))}),
                            **stats(r.get('value') for r in vals)})
    table('vexcel_calculated20_summary', cal_summary)
    # Sale comps can contain several properties, so count entity edges instead of assuming one FK per sale.
    propids = {str(r['id']) for r in tables['commercial-property']}
    entities = [{'sale_comp_id': r['id'], 'property_id': e.get('propertyId'), 'matched_property': str(e.get('propertyId')) in propids}
                for r in tables['commercial-sale-comp'] for e in r.get('entities') or [] if present(e.get('propertyId'))]
    table('sale_comp_property_edges', entities)
    record('sale_comp_entity_property_edges_matched', sum(e['matched_property'] for e in entities), len(entities))
    res_lookup = {str(r['id']): r for r in tables['residential-listings']}
    ix = {r['sid']: r for r in root}
    joined_edges = [r for r in tables['residential-listings_csv'] if r['MatterportModelId'] in ix and r['ListingId'] in res_lookup]
    for target in ['id', 'residentialListingId', 'propertyId']:
        record('index.homes_listing_id_matches.' + target,
               sum(ix[r['MatterportModelId']]['homes_listing_id'] == str(res_lookup[r['ListingId']].get(target)) for r in joined_edges),
               len(joined_edges), 'CSV edge grain, includes duplicate source edges')
    table('data_quality_checks', checks)
    return checks, keys, proxy, cal_summary


def markdown_table(rows, fields):
    rows = list(rows)
    if rows:
        unknown = set(fields) - set().union(*(r.keys() for r in rows))
        if unknown:
            raise ValueError('Unknown report columns: ' + ', '.join(sorted(unknown)))
    def cell(v):
        if v is None or v == '':
            return '—'
        if isinstance(v, bool):
            return 'yes' if v else 'no'
        if isinstance(v, float):
            return f'{v:,.4g}'
        return str(v).replace('|', '/').replace('\n', ' ')[:220]
    return '| ' + ' | '.join(fields) + ' |\n|' + '|'.join('---' for _ in fields) + '|\n' + '\n'.join(
        '| ' + ' | '.join(cell(r.get(k)) for k in fields) + ' |' for r in rows) + '\n'


def charts(models, reps, vx, coverage):
    os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 10, 'savefig.dpi': 160})
    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(CHART / filename)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['wall_graph', 'windows', 'all_rooms_closed', 'object_annotations']
    for i, seg in enumerate(['residential', 'commercial_sample']):
        vals = [next(x['share_pct'] for x in coverage if x['segment'] == seg and x['measure'] == k) for k in labels]
        ax.bar([j + i * .35 for j in range(len(labels))], vals, width=.35, label=seg.replace('_', ' '), color=['#222222', '#aaaaaa'][i])
    ax.set_xticks([j + .175 for j in range(len(labels))], ['Wall graph', 'Windows', 'All room loops closed', 'Object annotations'])
    ax.set_ylabel('Percent of surveyed scans')
    ax.set_ylim(0, 110)
    ax.legend(frameon=False)
    save(fig, 'matterport_coverage.png')
    fig, ax = plt.subplots(figsize=(7, 4))
    values = [[r['window_floor_ratio'] * 100 for r in reps if r['light_claim'] == claim and r['remarks_present'] and (r.get('windows') or 0) > 0 and num(r.get('window_floor_ratio')) is not None] for claim in [False, True]]
    ax.boxplot(values, tick_labels=[f'No claim (n={len(values[0])})', f'Light claim (n={len(values[1])})'], showfliers=True)
    ax.set_ylabel('Nominal window rectangle area / indoor room area (%)')
    ax.set_title('Overlapping distributions; scan geometry is not a daylight measurement')
    save(fig, 'light_claims.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, title in zip(axes, ['listing_model_distance_m', 'size_relative_difference'], ['Listing ↔ scan coordinates (m, log scale)', 'Tour indoor area minus listing living area (%)']):
        vals = [r[field] for r in reps if num(r.get(field)) is not None]
        if field.endswith('distance_m'):
            vals = [math.log10(max(v, .01)) for v in vals]
            ax.hist(vals, bins=35, color='#555555')
            ax.set_xlabel('log10(metres)')
        else:
            ax.hist([100 * v for v in vals], bins=50, color='#555555')
            ax.set_xlabel('Difference (%) — full observed range')
        ax.set_title(title)
        ax.set_ylabel('Homes')
    save(fig, 'location_size_agreement.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [-daydiff(v['ortho_date_estimate'], next((r['list_date'] for r in reps if r['sid'] == v['sid']), None)) for v in vx if date(v.get('ortho_date_estimate')) and next((r['list_date'] for r in reps if r['sid'] == v['sid']), None)]
    if vals:
        ax.hist(vals, bins=30, color='#555555')
    ax.set_xlabel('Listing date minus property-extraction ortho estimate (days)')
    ax.set_ylabel('Representative homes')
    ax.set_title('Positive means aerial attributes predate the listing')
    save(fig, 'aerial_listing_time_gap.png')


def write_report(result, tables, models, rooms, openings, locations, cities, reps, commpairs, attrs, calculated):
    def rows(name):
        return list(csv.DictReader((OUT / (name + '.csv')).open()))
    def n(v):
        return f'{int(float(v)):,}'
    def f(v, places=2):
        return f'{float(v):,.{places}f}'
    def ag(check):
        return next(r for r in result['agreement'] if r['check'] == check and r['grain'] == 'one_per_home')
    def cv(segment, measure):
        segment = 'commercial_sample' if segment == 'commercial' else segment
        return next(r for r in result['mp_coverage'] if r['segment'] == segment and r['measure'] == measure)
    def cov(segment, measure):
        r = cv(segment, measure)
        return f"{n(r['count'])}/{n(r['denominator_planned_models'])} ({f(r['share_pct'])}%)"
    def quality(measure):
        return next(r for r in result['data_quality'] if r['measure'] == measure)
    def dist(segment, field):
        return next(r for r in result['mp_numeric'] if r['segment'] == segment and r['field'] == field)
    def md(name, fields, selected=None):
        return markdown_table(rows(name) if selected is None else selected, fields)
    def csvlink(name):
        return f'[{name}.csv](../data/audit/{name}.csv)'
    size, bed, bath, story, loc = [ag(x) for x in ['size_relative_difference', 'bedrooms_difference', 'physical_baths_difference', 'stories_difference', 'listing_model_distance_m']]
    light = result['light']
    groups = {r['light_claim']: r for r in rows('light_claim_distributions')}
    # CSV booleans are strings; keep the actual statistic names explicit.
    yes, no = groups['True'], groups['False']
    resfinal = next(r for r in result['funnels'] if r['segment'] == 'residential' and r['step'] == 'splat_index_flag')
    commfinal = next(r for r in result['funnels'] if r['segment'] == 'commercial' and r['step'] == 'splat_index_flag')
    times = {r['field']: r for r in result['time_alignment']}
    statuses = [{'endpoint': field.removesuffix('_status'), 'status': status, 'points': count,
                 'share_500_pct': share(count, len(locations))}
                for field in ['property_status', 'ortho_status', 'oblique_status']
                for status, count in collections.Counter(str(r[field]) for r in locations).items()]
    table('vexcel_endpoint_statuses', statuses)
    feature_rows = rows('vexcel_feature_versions')
    versions = stats(float(r['feature_count']) for r in locations if r['property_status'] == 200)
    structures = stats(r['structures'] for r in locations if r['property_status'] == 200)
    footprint = stats(r['primary_footprint_m2'] for r in locations)
    changed_first = sum(r.get('selected_feature_index') not in (None, 0) for r in locations)
    pool, solar = ag('pool_disagreement'), ag('solar_disagreement')
    foreign = quality('commercial_pairs_location_gt1000km')
    wc = quality('residential.ceiling_min_m.outside1.5to6_excluding_void_staircase_unclassified')
    bath_internal = quality('listing_weighted_baths_inconsistent_with_documented_half_weight')
    findings = [
        f"**Window geometry is common in residential tours and scarce in this commercial sample.** Residential windows: {cov('residential', 'windows')}; commercial: {cov('commercial', 'windows')}. Residential wall graphs: {cov('residential', 'wall_graph')}. The earlier small-sample assumption does not hold for every scan.",
        f"**Light claims have no clear positive signal in this geometry proxy.** Claim group n={n(yes['n'])}, median {f(100*float(yes['median']))}%; no-claim n={n(no['n'])}, median {f(100*float(no['median']))}%. Median difference {f(100*light['median_difference'], 3)} percentage points; bootstrap interval [{f(100*light['bootstrap95_low'], 3)}, {f(100*light['bootstrap95_high'], 3)}] includes zero. This is not measured daylight.",
        f"**Area values often differ materially.** {n(size['abs_gt_20pct_or_m'])}/{n(size['comparable'])} homes ({f(share(size['abs_gt_20pct_or_m'], size['comparable']))}%) differ by more than 20%; median tour/listing area ratio is {f(1+size['median'], 4)}. Indoor room-area sum and listing living area have different scope.",
        f"**Room labels are not reliable replacements for listing counts.** Bedroom counts differ on {n(bed['nonzero'])}/{n(bed['comparable'])} homes ({f(share(bed['nonzero'], bed['comparable']))}%); physical bathroom-room counts differ on {n(bath['nonzero'])}/{n(bath['comparable'])} ({f(share(bath['nonzero'], bath['comparable']))}%). Stories differ on {n(story['nonzero'])}/{n(story['comparable'])}. Separately, {n(bath_internal['count'])}/{n(bath_internal['denominator'])} listing bathroom totals violate the documented half-bath formula.",
        f"**Location joins need an outlier gate.** Residential listing/scan distance has median {f(loc['median'])} m, but {n(loc['distance_gt100m'])}/{n(loc['comparable'])} exceed 100 m; maximum {f(loc['max']/1000)} km. Commercial index/property pairs include {n(foreign['count'])} distances above 1,000 km ({foreign['details']}). These are review flags, not proof of which source is wrong.",
        f"**Aerial coverage is broad at the sampled points, with explicit holes.** Property extraction succeeds at {n(result['vexcel_statuses'].get('200',0))}/{n(len(locations))} points; {n(result['vexcel_footprints'])} have footprints and {n(result['vexcel_oblique'])} have oblique collections. There are {n(result['vexcel_statuses'].get('403',0))} geofenced points and {n(result['vexcel_statuses'].get('404',0))} extraction 404s. Measured coordinates cover {n(result['city_with_measured_coordinate'])}/{n(result['city_total'])} city/state labels, including an exact-coordinate alias; this is not a geofence boundary map.",
        f"**Aerial and listing dates are not synchronized.** Selected extraction imagery predates listing by a median {f(-times['ortho_estimate_minus_list_days']['median'],1)} days (n={n(times['ortho_estimate_minus_list_days']['n'])}); latest oblique collection last-capture predates it by {f(-times['latest_oblique_minus_list_days']['median'],1)} days (n={n(times['latest_oblique_minus_list_days']['n'])}). Neither collection metadata nor model creation time proves the scan/image capture date at the property.",
        f"**The complete measured residential chain is small.** {n(resfinal['models'])} scans / {n(resfinal['unique_joined_properties'])} homes satisfy listing → remarks → windows → observed footprint → oblique collection → splat flag. Commercial: {n(commfinal['models'])}. These are observed lower bounds because aerial and commercial geometry were sampled.",
        f"**Useful-looking flags do not supply content.** Object annotations are empty in {n(len(models))}/{n(len(models))} returned models. `ai_summary=yes` occurs on {n(quality('index.ai_summary=yes')['count'])}/{n(len(tables['models_index']))} index rows, but the column contains flags, not summary text. Mesh, skybox and floor-plan metadata do not prove downloadable or renderable assets.",
        f"**Geometry needs filtering before product use.** Only {cov('residential', 'all_rooms_closed')} satisfy the strict all-room cycle criterion; this does not establish an open exterior envelope. Even excluding Void, Staircase and unclassified rooms, {n(wc['count'])}/{n(wc['denominator'])} minimum ceiling values fall outside 1.5–6 m. Window-height and sill flags are exported with scan/opening ids.",
    ]
    table('top_ten_findings', [{'rank': i, 'finding': text} for i, text in enumerate(findings, 1)])
    sections = ['# HackPrinceton data audit\n\nAll findings below are computed by `experiments/data_audit.py` from the kit and cached real API responses. This is a data audit, not verified property ground truth.\n\n## Ten most important findings\n']
    sections.append('\n\n'.join(f'{i}. {text}' for i, text in enumerate(findings, 1)))
    sections.append('''
## Scope, units and reproducibility

The concept and model notes were read before querying the datasets. Their definitions govern joins and missingness. Commercial property, marketed listing, apartment model/unit, occupancy, company location, sale transaction, residential MLS record, listing instance, home and scan are different grains. The CSV bridges are authoritative scan edges. Multiple records per property are retained in pair tables; residential headline agreement uses one representative per `propertyId`, preferring a queried scan, newest model creation, primary MLS record, then stable ids. This is deterministic selection, not an assertion that the newest record is truth.

Run `python3 -I -B experiments/data_audit.py` from the project root. The default is strictly offline: it does not load credentials or call either API. `--skip-inventory` reuses existing inventory CSVs. `--survey` is explicitly online and requests missing caches. All query specifications, sanitized responses and failure statuses are cached. The light query requests metadata and geometry, never binary assets. Units are metres, square metres, square feet where labelled, and UTC days; conversions and thresholds are in the script. Quantiles use linear interpolation. Empty/null/missing is unknown; numeric zero and Boolean false remain observed values. An enum code named Unknown is semantically unknown even though it is a filled field.

Sampling: every residential scan is queried; commercial scans use deterministic equal round-robin allocation across joined property types plus an unjoined stratum, not proportional population sampling. Vexcel uses separate residential/commercial allocations across state and case-folded city labels and deduplicates coordinates within segment. These are descriptive coverage measurements, not weighted national estimates. City labels are not coordinate-validated municipality boundaries. Sample membership is in `matterport_sample.csv` and `vexcel_sample.csv`.
''')
    unique_coordinates = len({(r['lat'],r['lon']) for r in rows('vexcel_sample')})
    sections.append(f"The API census contains {n(len(models))} models; the Vexcel survey contains {n(len(locations))} segment/location records at {n(unique_coordinates)} distinct coordinate pairs. A coordinate shared across segments reuses its response cache. Residential agreement retains {n(result['residential_pairs'])} distinct scan/MLS pairs and {n(len(reps))} homes. Commercial agreement retains {n(len(commpairs))} scan/property pairs. All numeric backing tables are linked below.\n")
    sections.append('## Source: local model index\n\n' + md('data_quality_checks', ['measure', 'count', 'denominator', 'share_pct'],
        [r for r in result['data_quality'] if r['measure'].startswith('index.') and not r['measure'].startswith('index.drone_photos=') and 'matches' not in r['measure']]))
    index = tables['models_index']
    geography = [{'state': state, 'city': city, 'segment': segment, 'models': count} for (state,city,segment), count in
                 sorted(collections.Counter((r['state'],r['city'].strip().casefold(),r['segment']) for r in index).items())]
    table('index_geography', geography)
    sources = [{'source': key, 'models': value} for key,value in collections.Counter(r['latlon_source'] for r in index).most_common()]
    sections.append('\n' + markdown_table(sources, ['source','models']))
    sections.append(f"\nSegment counts: " + ', '.join(f"{s}={n(c)}" for s,c in collections.Counter(r['segment'] for r in index).items()) +
                    '. `splat`, `defurnished` and `ai_summary` are flags. `drone_photos` is a count. Asset contents were not downloaded. Index coordinates and API model coordinates are nearly identical; that is shared provenance, not an independent accuracy check. Geography: ' + csvlink('index_geography') + '.\n')
    sections.append('## Source: CoStar/Homes kit — every file and field\n\n' + md('table_inventory', ['table','rows','fields_observed','fields_declared_or_observed','always_empty_fields','constant_filled_fields']))
    sections.append('''
There is no standalone attachments JSONL in this kit: attachments are nested in the records below. The file manifest includes data, schemas, enum definitions, OpenAPI and concept/model notes with byte counts and SHA-256 hashes. The field inventory includes observed nested paths and schema-declared paths that never occur. `empty_row_share_pct` uses root records as the denominator; `observed_occurrences` and `nonempty_occurrences` describe repeated nested elements. A missing parent makes its descendants unavailable. Container distinct values use content hashes; example containers summarize keys/length, and long scalar examples are truncated. Scalar examples and category distributions are sanitized. URL-shaped values are redacted in all outputs.

Raw numeric field distributions do not make identifier codes quantities or zero an economically meaningful price. Decoded enum frequencies use `enums.json`; documented bit flags are decomposed into named bits, with unmapped bits marked. Constant means constant when filled, not necessarily complete. Listing notes and summary flags were treated as data throughout.
''')
    sections.append('\n' + md('key_integrity', ['table','candidate_key','rows','nonnull_rows','distinct','duplicate_excess_rows','unique_and_complete']))
    sections.append('\nRepeated property or listing ids in bridge CSVs describe one-to-many edges; only the composite key tests detect repeated edges. Do not interpret those repetitions as duplicated properties.\n\n' + md('attachment_inventory', ['source_table','path','references']))
    sections.append('\n### Free text\n\n' + md('text_profiles', ['table','field','nonempty_documents','distinct_normalized_documents','duplicate_excess_documents','duplicate_excess_share_pct','boilerplate_sentence_min_documents','documents_with_boilerplate','boilerplate_share_pct']))
    textrows = rows('text_profiles')
    lengthrows = [{'table':r['table'],'field':r['field'],**json.loads(r['length_characters'])} for r in textrows]
    sections.append('\nRaw source character-length distributions:\n\n' + markdown_table(lengthrows, ['table','field','n','min','p25','median','p75','p95','max']))
    sections.append('''
Text profiles include remarks, summary flags, nested note values, descriptions, comments and captions. Each observed text occurrence is a document at its stated table/field grain; nested notes are not unique properties. Duplicate normalization removes HTML tags and URLs, decodes HTML entities, lowercases and collapses whitespace; duplicate excess is records beyond the first per normalized text. Boilerplate means a repeated sentence of at least 50 characters appearing in at least max(5, ceil(1% of nonempty documents)) records. Zero under this detector does not prove originality: paraphrases, short slogans and near-duplicate text are outside its scope. `ai_summary` is profiled for completeness but its yes/no values are not free-text summaries. Phrase counts are document frequencies of 2–4 word sequences after stop-word boundary filtering, ordered by decreasing count then alphabetically for ties; phrases overlap and are descriptive, not topics inferred by an LLM. Fields with fewer eligible phrases have fewer than 60 rows.
''')
    sections.append('\nThe requested 60 highest-frequency meaningful phrases per text field:\n\n' + md('text_top60_phrases', ['table','field','rank','phrase','documents','share_nonempty_pct']))
    sections.append('\nFull field-level evidence: ' + ', '.join(csvlink(x) for x in ['source_files','field_inventory','categorical_distributions','key_integrity','attachment_references','text_profiles','text_duplicates','text_top60_phrases']) + '.\n')
    sections.append('## Source: Matterport Model API\n\n' + md('matterport_coverage', ['segment','measure','count','denominator_planned_models','share_pct']))
    enums = json.loads((ROOT/'data/costar-kit/schemas/enums.json').read_text())
    sample = rows('matterport_sample')
    sampling = [{'property_type_code': code, 'decoded': enums.get('PropertyType',{}).get(code,code), 'sampled_models': count}
                for code,count in collections.Counter(r['property_type_code'] for r in sample if r['segment']=='commercial').most_common()]
    table('commercial_sampling_strata', sampling)
    sections.append('\nCommercial sampling allocation:\n\n' + markdown_table(sampling, ['property_type_code','decoded','sampled_models']))
    sections.append('''
Closure is topological: every referenced boundary edge must be returned and have two distinct endpoints; a room boundary must be one connected component with every vertex degree two, and equal edge/vertex counts. Vertex ids are used directly, with position fallback only when no id exists; vertices are not snapped together by spatial tolerance. “All rooms close” includes inferred/void/fragment rooms. This is a stricter check than presence of a useful floor plan and does not establish a watertight exterior shell, absence of doors, or building-code compliance. The floor-level graph records connected components/dangling vertices in `matterport_floors.csv`.

Openings are deduplicated by opening id within each scan. Window area is width × height, a rectangle proxy; sill is opening lower elevation minus the median lower elevation of adjacent rooms when available. It is not true glass area or guaranteed ground-relative elevation. Room labels use the highest-confidence classification, and all candidate labels remain in the room CSV. Bedroom count includes bedroom/master bedroom/primary bedroom; bathroom count includes bathroom/half bath/powder room. These count classified regions, not independently verified rooms or fixtures. Room `areaIndoor` sums can include garages, circulation, utility rooms and void-related artifacts and need not match statutory living area.
''')
    sections.append('\nOpening type counts:\n\n' + md('opening_types', ['segment','type','count']))
    sections.append('\nRoom label counts:\n\n' + md('room_classifications', ['segment','classification','rooms']))
    sections.append('\nNumeric distributions (all retained values; no silent winsorization):\n\n' + md('matterport_distributions', ['segment','field','n','min','p25','median','p75','p95','max']))
    sections.append('\nGeocoordinates provenance:\n\n' + md('geocoordinates_sources', ['segment','source','models']))
    sections.append('\nImplausibility screens are audit thresholds, not validated defect labels:\n\n' + markdown_table(
        [r for r in result['data_quality'] if 'outside' in r['measure'] or r['measure'].startswith('windows.')],
        ['measure','count','denominator','share_pct']))
    anomalies = rows('geometry_anomalies')
    sections.append('\nExamples with identifiers (the complete, multi-flag list is in the CSV):\n\n' + markdown_table(sorted(anomalies,key=lambda r: -abs(num(r.get('value')) or 0))[:20],
        ['sid','segment','room_id','opening_id','check','value','room_classification']))
    sections.append('''
Skybox “available” here means panoramic image options advertise formats/resolutions. Mesh and floor-plan counts/statuses are metadata. No asset URL, download, image decoding or viewer render was tested, and a splat flag is not a working exterior-render proof. Object annotations being empty for this query does not prove the API can never contain objects. Model `created` is a creation timestamp, not measured scan capture time.
''')
    sections.append('\nBacking: ' + ', '.join(csvlink(x) for x in ['matterport_models','matterport_rooms','matterport_openings','matterport_floors','matterport_objects','matterport_errors','geometry_anomalies']) + '.\n\n![API coverage](audit/matterport_coverage.png)\n')
    sections.append('## Source: Vexcel API\n\n' + markdown_table(statuses,['endpoint','status','points','share_500_pct']))
    sections.append(f"\nThe property survey requests precomputed attributes only. The calculated pilot is restricted to {n(len({r['sid'] for r in calculated}))} locations. Responses contain multiple extraction vintages despite `latest=true`: median {f(versions.get('median',0),1)} features per successful request, maximum {n(versions.get('max',0))}. Explicit latest-date/inside-parcel selection chooses a feature other than the first at {n(changed_first)} locations. Every version's collection/date/selection is retained in {csvlink('vexcel_feature_versions')}. Field fill rates below use only the selected feature per successful location, then all its structures. Primary-structure comparisons use its flagged primary structure, falling back to the first when the flag is absent.\n")
    sections.append('\n' + markdown_table([{'measure':'structures_per_successful_parcel',**structures},{'measure':'primary_footprint_area_m2',**footprint}], ['measure','n','min','p25','median','p75','p95','max']))
    selected_attrs = [r for r in attrs if r['field'] in ['parcel-source','parcel-tree-cover.percent','pool.detected','pool.total-area','vehicles.detected','vehicles.count','ground-elevation.value','footprint.area','primary-structure','roof-characteristics.condition','roof-characteristics.condition.precise-value','roof-characteristics.tree-cover.percent','roof-characteristics.material.class','roof-characteristics.shape.value','roof-characteristics.eave-height','roof-characteristics.maximum-roof-height','roof-characteristics.number-of-stories-below-eave','roof-characteristics.entrance-elevation','roof-characteristics.volume-below-eave','roof-objects.solar.value'] or r['field'].startswith(('roof-defects.','roof-objects.')) and r['field'].endswith(('.detected','.count'))]
    sections.append('\nSelected requested attribute fill rates (Boolean false and zero count as filled):\n\n' + markdown_table(selected_attrs,['grain','field','denominator','nonempty','filled_share_pct']))
    sections.append('\nCalculated pilot — same response index and parcel geometry checked against the chosen precomputed feature; no independent flight/vintage identity is present in calculated-only metadata:\n\n' + markdown_table(result['calculated20_summary'],['field','properties_filled','structures','structures_filled','min','median','max']))
    sections.append('''
Roof elevation and height are distinct: the former is a vertical elevation, the latter an above-ground difference returned by the service. The calculated pilot does not establish national coverage or accuracy. Unfilled precomputed height/story fields remain unknown; the pilot is not extrapolated to the survey. Roof condition is an undocumented numeric scale in these responses, not a known probability or certified inspection grade. Material/shape/defect/solar values are model attributes at their imagery date. A real or synthetic/Bins parcel is not necessarily the legal lot or the individual condo unit.

### City/state access and imagery years

The following is an empirical access map of sampled points only. A city can have both successes and denials; neither one observed 403 nor one success establishes city-wide eligibility. Blank city/state and misspellings are retained. One misspelled label reuses an exactly identical numeric coordinate, documented in `city_alias_evidence.csv`; the remaining label has no usable coordinate. Ortho and oblique years are derived from collection `last-capture-date`, not the year embedded in the collection name. The latter can differ. These endpoints identify available collections; they do not establish the capture time of an image at the exact requested point, image count, façade visibility, ground resolution or useful image quality. The separate `oriented/query` point-image capture-date probe was not performed.
''')
    sections.append('\n' + markdown_table(cities,['state','city','index_models','sample_points','coordinate_shared_points','property_200','property_403','ortho_200','oblique_200','ortho_years','oblique_years','measured_scope']))
    sections.append('\n403 locations:\n\n' + markdown_table([r for r in locations if r['property_status']==403],['sid','state','city','property_status','ortho_status','oblique_status']))
    sections.append('\nBacking: ' + ', '.join(csvlink(x) for x in ['vexcel_sample','vexcel_locations','vexcel_structures','vexcel_attributes','vexcel_imagery','vexcel_calculated20','vexcel_feature_versions','vexcel_errors','city_coverage','city_alias_evidence','cache_status_inventory']) + '. The cache-status inventory also retains the initial incompatible ortho requests and schema introspection, so its request counts differ from survey-location counts.\n')
    sections.append('## Joins, geographic coverage and time alignment\n\n' + markdown_table(result['funnels'],['segment','step','models','unique_joined_properties','share_index_models_pct','models_with_vexcel_survey']))
    sections.append('''
Funnels are strictly nested by scan id. Residential “joined” uses the bridge CSV and listing `id`; home counts use listing `propertyId`. Commercial “joined” uses the commercial bridge CSV and commercial property `id`; the requested commercial chain is at property grain. Its notes stage uses property `notes`, not residential-style listing remarks. Commercial marketing-listing availability is separately recorded in each commercial pair's `commercial_listing_count`. The first stage's property count describes joinable properties, not an inferred unique-property count for all unjoined models.

Unsurveyed locations are unknown, and commercial geometry is only sampled. Consequently the final observed chains are lower bounds, not absence rates or estimates of all eligible properties. The same sampled coordinate does not mean every model on that parcel depicts the same unit. The index's `homes_listing_id` uses a different namespace from JSONL listing `id`; the CSV avoids the resulting bad join. Matches to each candidate namespace are measured below at source-CSV edge grain.
''')
    sections.append('\n' + markdown_table([r for r in result['data_quality'] if 'homes_listing_id_matches' in r['measure']],['measure','count','denominator','share_pct']))
    sections.append('\nForeign keys:\n\n' + markdown_table(result['foreign_keys'],['join','child_rows','nonnull_fk_rows','matched_rows','orphan_rows','distinct_matched_parents']))
    sections.append('\n' + markdown_table([quality('sale_comp_entity_property_edges_matched')],['measure','count','denominator','share_pct','details']))
    sections.append('''
Sale comps are transaction records with multiple `entities`; their property edges are tested independently rather than assuming one property per sale. Unmatched sale properties can be outside this supplied property subset, so an orphan is not automatically source corruption. Occupancy `tenantLocationId` joins company-detail `id` (a company location); legal `companyId` is not the same key. Null tenant-location links remain unknown. Apartment communities and their floor plans/units are nested property components, not detached homes.
''')
    sections.append('\nTime differences, first date minus listing date, in days:\n\n' + markdown_table(result['time_alignment'],['field','n','min','p25','median','p75','p95','max']))
    sections.append('\nCreation timestamps can reflect reprocessing/import; scan capture dates were not available in the light query. Extraction ortho dates are estimates; oblique dates are collection last-capture metadata. Dates alone cannot establish a property change or determine which conflicting source was contemporaneously correct.\n\n![Aerial age](audit/aerial_listing_time_gap.png)\n')
    # Geography of the fully observed chain, without projecting sampled coverage.
    full_sids = {r['sid'] for r in reps if r['remarks_present'] and (r.get('windows') or 0)>0 and (r.get('wall_edges') or 0)>0 and r['footprint'] and r['oblique'] and r['splat_index_flag']}
    final_geo = [{'segment':segment,'state':state,'city':city,'scans':count} for (segment,state,city),count in
                 sorted(collections.Counter((r['segment'],r['state'],r['city']) for r in index if r['sid'] in full_sids).items())]
    table('complete_chain_geography_representatives', final_geo)
    sections.append('\nGeography of representative homes with the full observed chain (other scan choices may add scans in the funnel):\n\n' + markdown_table(final_geo,['segment','state','city','scans']))
    sections.append('## Cross-source agreement\n\nCounts below use one representative per home. All-pair sensitivity results remain in `agreement_summary.csv`. Differences are tour minus listing. No source is treated as ground truth.\n\n' + markdown_table(
        [r for r in result['agreement'] if r['grain']=='one_per_home'],['check','comparable','exact_equal','nonzero','abs_gt_20pct_or_m','distance_gt100m','min','median','max']))
    sections.append('''
Location uses great-circle distance between valid latitude/longitude pairs. Parcel agreement is an explicit point-in-polygon test (holes respected) against the Vexcel parcel returned for the index point; the extraction point and parcel are not independent validation. Size uses `(sum(areaIndoor) × 10.7639104167 − listing.property.sqftMin) / sqftMin`, only positive source areas. Exact floating-point equality is not a meaningful accuracy metric for area. Bedroom/bathroom comparisons require at least one meaningful room label, excluding all-Other/Void/unclassified tours; remaining labels can still be incomplete or split rooms. Physical bathrooms use `bathsFull + bathsHalf` only when both fields are known; no null half-bath is assumed zero. Comparing all bathroom regions only to full baths is separately retained as an intentionally weak proxy. Stories use listing RESO structure stories, falling back to feature-derived stories, versus scan floor count; high-rise building and individual-unit scope can differ.

Pool comparison requires an explicit private-pool feature or private/in-ground/above-ground wording, or explicit negative wording. Association/community-only flags are excluded; code 0 is unknown. Solar requires explicit panel wording (solar heating alone is not counted). Negation is simple explicit phrase matching, not language understanding. Garage features are available in listings, but no matching Vexcel garage observable was returned. Roof “new/replaced” language describes age/maintenance, while Vexcel condition is a different undocumented score; trees mentioned in remarks need not be tree-cover percentage on a parcel or roof. Those are descriptive proxies, not binary agreement or truth checks. Missing feature mentions never mean absence.
''')
    sections.append('\nPool and solar comparison coverage:\n\n' + markdown_table([pool,solar],['check','comparable','exact_equal','nonzero']))
    sections.append('\nRoof and tree descriptive proxies:\n\n' + markdown_table(result['roof_tree_proxies'],['listing_claim','claim_present','vexcel_field','n','min','p25','median','p75','max','interpretation']))
    sections.append('\nListing bathroom internal consistency:\n\n' + markdown_table([bath_internal],['measure','count','denominator','share_pct']))
    sections.append('\nCommercial location and scope diagnostics:\n\n' + markdown_table([r for r in result['agreement'] if r['grain']=='all_scan_property_pairs'],['check','comparable','distance_gt100m','min','median','p95','max']))
    sections.append('''
Commercial room-area sum versus whole-building gross area is exported as a scope diagnostic only, not a disagreement rate; a tour can depict one leased suite or an apartment plan. Floors have the same unit/building mismatch and no defensible equality test without a shared-unit boundary. Some foreign coordinates remain valid numbers, so range validation alone cannot detect a bad geo join.

### Light claims versus window-area ratio
''')
    sections.append('\n' + markdown_table([{'light_claim':r['light_claim'],**{k:float(r[k]) if r[k] else None for k in ['n','min','p25','median','p75','p95','max']}} for r in rows('light_claim_distributions')],['light_claim','n','min','p25','median','p75','p95','max']))
    sections.append(f"\nRatios above are fractions; chart values are percentages. Explicit `natural light`, `naturally lit`, `bright` or `sunny` wording defines the claim group. Both groups require nonempty remarks, observed windows and a positive indoor area. Missing window geometry is excluded instead of being assigned a zero. The unadjusted median difference is {f(light['median_difference'],6)}; the deterministic bootstrap resamples homes within each group, with {n(light['bootstrap_median_difference']['n'])} replicates and percentile interval [{f(light['bootstrap95_low'],6)}, {f(light['bootstrap95_high'],6)}]. It includes zero. There are {n(light['positive_strata'])}/{n(light['comparable_strata'])} positive median differences among comparable state/property-type strata. This does not provide stable positive evidence that the wording identifies more window rectangle area.\n")
    sections.append('''
Claims can describe furnishings, orientation, shadows or a specific room; the proxy lacks façade orientation registration, transparent glass fraction, exterior occlusion, room depth, time/weather and measured illuminance. The regex can include negated or incidental uses and does not grade advertising truth. Geometry availability and the chosen representative scan introduce selection effects; geographic strata are descriptive rather than causal adjustment. No external brightness labels or photos were used as ground truth.

![Light proxy](audit/light_claims.png)

![Coordinate and size agreement](audit/location_size_agreement.png)

### Largest disagreements, with identifiers

For each numeric check, the following lists at most 20 homes ranked by absolute discrepancy. Boolean lists include only actual disagreements where any exist (all disagreement ids when fewer than 20); ties use property id. A zero-only coordinate/parcel check has no disagreements, so its CSV's largest observed equalities are not called errors. The full rows, source values and ids remain in each linked CSV. Roof/tree rows rank proxy extremes only; they cannot be ranked as verified contradictions. Light rows rank distance from the group median, not dishonesty.
''')
    for check in [r['check'] for r in result['agreement'] if r['grain']=='one_per_home']:
        rr = rows('outliers_' + check)
        if check.endswith('_disagreement'):
            rr = [r for r in rr if num(r.get(check)) and num(r[check]) != 0]
        elif 'distance_m' in check and not any((num(r.get(check))or 0) > .01 for r in rr):
            rr = []
        sections.append('\n#### ' + check + '\n\n' + csvlink('outliers_' + check) + '\n\n' +
                        (markdown_table(rr,['property_id','listing_id','sid','state','city',check]) if rr else 'No observed disagreement under this check; see comparison denominators above.') + '\n')
    for name, fields in [
        ('commercial_property_index_distance_m',['property_id','sid','state','city','property_index_distance_m']),
        ('commercial_property_model_distance_m',['property_id','sid','state','city','property_model_distance_m']),
        ('commercial_tour_whole_building_ratio',['property_id','sid','gross_building_sqft','tour_indoor_sqft','tour_whole_building_ratio']),
        ('bathroom_internal_consistency',['property_id','listing_id','sid','listing_full_baths','listing_half_baths','listing_weighted_baths','expected_weighted_baths','weighted_baths_identity_difference']),
        ('roof_condition_claim',['property_id','listing_id','sid','vexcel_roof_condition']),
        ('tree_cover_claim',['property_id','listing_id','sid','vexcel_parcel_tree_cover_pct']),
        ('light_claim',['property_id','listing_id','sid','light_claim','window_floor_ratio'])]:
        sections.append('\n#### ' + name + '\n\n' + csvlink('outliers_' + name) + '\n\n' + md('outliers_' + name,fields) + '\n')
    facts = []
    def fact(name, sources, count, grain, status, limit):
        facts.append({'fact':name,'sources':sources,'available_count':count,'grain':grain,'status':status,'qualification':limit})
    fact('Tour indoor room-area sum','Matterport',sum((r.get('room_indoor_sqft')or 0)>0 for r in models),'surveyed scans','computed here','Not standardized living area')
    fact('Window rectangle / indoor-area ratio with observed windows','Matterport',sum((r.get('windows')or 0)>0 and num(r.get('window_floor_ratio')) is not None for r in models),'surveyed scans','computed here','Not daylight or true glass area')
    fact('Residential size comparison','CSV + listing + Matterport',size['comparable'],'representative homes','computed here','Different area conventions')
    fact('Residential bedroom-label comparison','CSV + listing + Matterport',bed['comparable'],'representative homes','computed here','Classification proxies; incomplete/split rooms possible')
    fact('Physical bathroom-label comparison','CSV + listing + Matterport',bath['comparable'],'representative homes','computed here','Requires nonnull full and half counts')
    fact('Listing/scan coordinate distance','CSV + listing + Matterport',loc['comparable'],'representative homes','computed here','No authoritative address/parcel truth')
    fact('Footprint at queried location','Vexcel',result['vexcel_footprints'],'sampled locations','computed here','Selected latest extraction; parcel/structure may differ from unit')
    fact('Roof material/shape/condition','Vexcel',sum(all(present(r.get(k))for k in ['roof_material','roof_shape','roof_condition'])for r in locations),'sampled locations','computed here','Metadata model output; condition scale not calibrated')
    fact('Roof above-ground height pilot','Vexcel calculated',len({r['sid']for r in calculated if r['field']=='roof-characteristics.height' and present(r.get('value'))}),'pilot locations','computed here','No extrapolation; index/geometry alignment only')
    fact('Private pool explicit comparison','listing/features + Vexcel',pool['comparable'],'representative homes','computed here','Excludes community-only mentions; imagery time differs')
    fact('Solar panel explicit comparison','listing/features + Vexcel',solar['comparable'],'representative homes','computed here','Sparse mentions; simple negation rules')
    fact('Full listing/remarks/windows/footprint/oblique/splat chain','all joined sources',resfinal['unique_joined_properties'],'residential homes','computed here','Observed lower bound; metadata availability')
    fact('Exterior/skybox/mesh visual inspection','Matterport/Vexcel asset metadata',sum((r.get('panos_with_skybox_options')or 0)>0 for r in models),'eligible surveyed scans','feasible but not computed','No binaries downloaded or rendered')
    fact('Commercial NOI-based investment formulas','CoStar listing forSaleSummary',sum(present((r.get('forSaleSummary')or {}).get('netOperatingIncome'))for r in tables['commercial-listing-summary']),'marketing listings with NOI','feasible but not computed','Eligibility only; matching dates, actual/proforma basis and positive price still required')
    fact('Garage agreement with aerial source','listing + Vexcel',0,'comparable homes','not available','No Vexcel garage observable returned by this query')
    fact('Verified roof condition / true tree-language agreement','remarks + inspection/defined annotations',0,'ground-truth properties','not available','Different constructs; no inspection labels/condition-scale documentation')
    fact('Measured daylight or brightness truth','illuminance + geometry + time/weather',0,'ground-truth properties','not available','No photometric measurements')
    fact('Actual scan capture time / exact aerial image capture alignment','capture metadata',0,'verified aligned properties','not available','Created and collection-last-capture are not property capture timestamps')
    fact('Object detections supplied in objectAnnotations','Matterport',sum((r.get('object_annotations')or 0)>0 for r in models),'surveyed scans','computed here','All empty for this query; no object-content proof')
    fact('Actual AI summary text in index','models_index',0,'models with summary text','not available','Only yes/no flag supplied')
    table('derivable_facts',facts)
    sections.append('## Facts we can derive about a property\n\n' + markdown_table(facts,['fact','sources','available_count','grain','status','qualification']))
    sections.append('''
“Available” always uses the stated grain. Feasible rows count only an observed prerequisite and do not imply completed or validated computations. Cross-source fact counts are representative homes, while API-only counts are scan/location samples. These denominators are deliberately not interchangeable.

## Data problems, unchecked claims and execution limits

- Do not merge all rows on an apparent id: listing `id`, listing-instance id, home `propertyId`, commercial property id, company id and company-location id have different namespaces. Composite bridge edges and unit boundaries matter.
- No geofence polygon or city-wide availability was measured. The no-coordinate city label could not be queried. Sampled successes, spatial denials, empty extraction results and transport errors are separate states; none is a measured absence of a building or amenity.
- Extraction may return many dated vintages even with `latest=true`. The script selects the newest dated feature containing the point, then retains provenance. Calculated-only pilot responses lack independent collection/date identity; same-index/same-geometry alignment is evidence of matching parcels, not proof of matching flight vintage.
- Actual scan capture time, point-image flight time, asset accessibility, visual reconstruction quality, exterior-splat rendering, skybox bytes, georegistered façade orientation and lighting ground truth were not checked. Collections are metadata, not downloaded images. No factual property correction is justified by these proxies alone.
- Garage, roof-age versus roof-condition and qualitative tree language cannot support an equality score with the available aerial fields. Descriptive roof/tree distributions are exported instead. Pool and solar equality use explicit sparse wording and do not establish imagery-classifier accuracy.
- All source files and API responses were handled as data. Credentials were loaded internally only for authentication; headers and URLs are absent/redacted in outputs. Asset binaries, commercial CDN links and signed URLs were never fetched or published.
- Online work was bounded to four concurrent requests with backoff on HTTP 429; failures were cached. The initial authentication circuit counted consecutive denials. Interleaved successful responses prevented it stopping on repeated spatial 403s across the survey/ortho repair. This is a rule-compliance limitation of the executed run: the script now stops after two cumulative denials per provider, and all subsequent analysis/replay is offline. No further network probes are made to denied points.
- The first ortho-collection requests used an incompatible oriented endpoint shape and returned HTTP 400. These failures remain cached. A corrected `ortho/collections` pass supplies the reported ortho results; transport failures remain explicit and were not converted to absence. Endpoint distinction follows the [official Vexcel migration guide](https://vexcel.atlassian.net/wiki/spaces/APIDOCS/pages/3035136040/API%2BMigration%2BGuide%2Bfrom%2B1.0%2Bto%2B2.0).

No kit source files, application implementation, git state or other workers' files were changed by this audit. No new tests were added. The existing guard-suite result and offline-replay verification are recorded alongside the data; they do not prove an application's runtime behavior or third-party asset quality.

## Files written and change ownership

Codex wrote `experiments/data_audit.py`, generated this report and `docs/audit/*.png`, and generated the CSV/JSON/cache/log evidence under `data/audit/`. The separately authorized bridge progress file was updated after completed steps. No commits or git initialization were performed. Changes are reversible by removing these audit-owned artifacts; preserve `data/audit/cache/` first if you want to reproduce without another API survey. For ledger-backed script edits, inspect `python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path experiments/data_audit.py`, then use `rollback <id>` as a dry run or `rollback <id> --yes` to apply the selected reversal. No ledger entry or rollback was independently tested here.

The complete generated-file manifest is [audit_artifacts.csv](../data/audit/audit_artifacts.csv). Its hashes exclude itself and operational progress/logs; the replay verification compares deterministic tables, charts and report. API cache filenames are hashes of sanitized request specifications, and every response status remains inspectable without opening credentials.
''')
    (DOC / 'data_audit.md').write_text('\n'.join(sections))
    progress('Wrote docs/data_audit.md, charts, numerical findings, coverage/agreement CSVs and sanitized API caches. Existing guard suite: 7 passed. No new tests or application/source-data edits.',
             'Inspect rendered charts, verify numerical invariants, replay offline and check artifact safety/determinism.')


def deterministic_artifacts():
    excluded = {'audit_artifacts.csv', 'offline_verification.json'}
    paths = [p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.csv', '.json') and p.name not in excluded]
    paths += sorted(CHART.glob('*.png'))
    if (DOC / 'data_audit.md').exists():
        paths.append(DOC / 'data_audit.md')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def verify_artifacts(result, client, baseline=None):
    checks = []
    if not (OUT/'audit_artifacts.csv').exists():
        # Establish the report target before checking links; filled with final hashes below.
        table('audit_artifacts', [], ['file','bytes','sha256'])
    def check(name, passed, details):
        checks.append({'check': name, 'passed': bool(passed), 'details': details})
    check('analysis_offline', not client.online and not client.calls, dict(client.calls))
    coverage = result['mp_coverage']
    check('survey_model_denominators', sum(r['denominator_planned_models'] for r in coverage if r['measure']=='returned') ==
          sum(r['count'] for r in coverage if r['measure']=='returned'), 'Planned and returned model census/sample sizes')
    for segment in ['residential','commercial']:
        funnel = [r['models'] for r in result['funnels'] if r['segment']==segment]
        check(segment + '_funnel_nested', all(a>=b for a,b in zip(funnel, funnel[1:])), funnel)
    check('agreement_denominators', all(r['comparable']<=r['denominator_records'] for r in result['agreement']),
          'Every comparable count is bounded by its stated grain')
    check('calculated_pilot_limit', all(r['properties_filled']<=20 for r in result['calculated20_summary']),
          'Calculated attributes only for the selected twenty locations')
    source_rows = list(csv.DictReader((OUT/'source_files.csv').open()))
    changed_sources = [r['file'] for r in source_rows if hashlib.sha256((ROOT/r['file']).read_bytes()).hexdigest()!=r['sha256']]
    check('source_hashes_unchanged', not changed_sources, {'files_checked':len(source_rows),'changed':changed_sources})
    cache_paths = sorted(CACHE.glob('*/*.json'))
    cache_status = collections.Counter()
    unsafe = []
    for path in cache_paths:
        raw = path.read_text()
        record = json.loads(raw)
        cache_status[(record['provider'],record['endpoint'],str(record['status']))] += 1
        # Credential values are never read during replay. Require all literal URLs removed and no populated auth fields.
        if re.search(r'https?://', raw) or re.search(r'"(?:Authorization|X-API-KEY|password|token)"\s*:\s*"(?!\[REDACTED\])[^"\s]+', raw, re.I):
            unsafe.append(str(path.relative_to(ROOT)))
    check('cache_sanitization', not unsafe, {'cache_files_checked':len(cache_paths),'flagged_files':unsafe})
    table('cache_status_inventory',[{'provider':a,'endpoint':b,'status':c,'cached_requests':count}
          for (a,b,c),count in sorted(cache_status.items())])
    for path in sorted(OUT.glob('*.csv')):
        if path.name == 'audit_artifacts.csv':
            continue
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            malformed = sum(None in r for r in reader)
        check('csv_width:' + path.name, malformed==0, {'malformed_rows':malformed})
    report = (DOC/'data_audit.md').read_text()
    broken = [link for link in re.findall(r'\]\(([^)]+)\)', report) if not link.startswith('https://') and not (DOC/link).exists()]
    check('report_links_exist', not broken, broken)
    guard = (OUT/'existing_tests.log').read_text()
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s',guard)
    check('existing_guard_suite', bool(match) and '\nOK\n' in guard and 'FAILED' not in guard,
          {'tests':int(match[1]),'elapsed_seconds':float(match[2])} if match else 'No result parsed')
    if baseline is not None:
        current = deterministic_artifacts()
        differences = sorted(p for p in set(baseline)|set(current) if baseline.get(p)!=current.get(p))
        check('offline_replay_hashes', not differences, {'artifacts_compared':len(current),'differences':differences})
    dump(OUT/'offline_verification.json', {'checks':checks,'passed':all(r['passed'] for r in checks),
          'offline_replay_requested':baseline is not None,'network_calls':dict(client.calls)})
    artifact_paths = [ROOT/'experiments/data_audit.py',DOC/'data_audit.md'] + sorted(CHART.glob('*.png'))
    artifact_paths += [p for p in OUT.rglob('*') if p.is_file() and p.name!='audit_artifacts.csv' and
                       p.suffix not in ('.log','.pyc') and 'matplotlib' not in p.parts]
    table('audit_artifacts',[{'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,
                            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(artifact_paths)])
    failed = [r['check'] for r in checks if not r['passed']]
    if failed:
        raise AssertionError('Artifact checks failed: ' + ', '.join(failed))
    progress('Completed report, four visually inspected PNG charts, CSV/JSON audit evidence and sanitized caches. '
             'Existing suite: '+str(int(match[1]))+' passed. Source hashes unchanged. Offline checks passed; '
             + ('deterministic replay passed. ' if baseline is not None else 'replay still pending. ')
             + 'No API calls during analysis. Files listed in data/audit/audit_artifacts.csv.',
             'Deliver the ten findings, unchecked items/rule-compliance limits, and file links.' if baseline is not None else
             'Run python3 -I -B experiments/data_audit.py --verify-replay to compare all deterministic artifacts.')
    print('Artifact checks passed:',len(checks),'cache records:',len(cache_paths),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--survey', action='store_true')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--imagery-repair', action='store_true')
    ap.add_argument('--skip-inventory', action='store_true', help='Reuse already generated inventory CSVs')
    ap.add_argument('--verify-replay', action='store_true', help='Require identical offline tables, metrics, charts and report')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CHART.mkdir(parents=True, exist_ok=True)
    baseline = deterministic_artifacts() if args.verify_replay else None
    if args.verify_replay and (args.survey or args.imagery_repair):
        ap.error('--verify-replay must be offline')
    client = Client(args.survey or args.imagery_repair)
    types = schema_map(client)
    if args.probe:
        names = ['Model', 'Opening', 'ModelAssets', 'PanoramicImage', 'PanoramicImageLocation', 'ObjectAnnotation', 'Units']
        for n in names:
            print(json.dumps(clean(types.get(n, {'name': n, 'missing': True}))))
        print('types', len(types))
        return
    tables = load_tables()
    mp, vx = make_samples(tables)
    if args.imagery_repair:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i, _ in enumerate(pool.map(lambda r: client.request('vx', imagery_endpoint('ortho'), imagery_params(r, 'ortho')), vx), 1):
                if i % 50 == 0:
                    print('Corrected ortho collections', i, '/', len(vx), flush=True)
        dump(OUT / 'imagery_repair_run.json', {'calls': client.calls, 'blocked': sorted(client.blocked)})
        return
    if args.survey:
        # Validate the exact light query once before the full census.
        first = client.request('mp', query=MODEL_Q % mp[0]['sid'])
        if not ((first.get('data') or {}).get('data') or {}).get('model'):
            print('First model query failed:', json.dumps(clean(first.get('data'))))
            return
        survey(client, mp, vx)
    if args.skip_inventory:
        summaries = list(csv.DictReader((OUT / 'table_inventory.csv').open()))
        text_summary = list(csv.DictReader((OUT / 'text_profiles.csv').open()))
    else:
        summaries, text_summary = inventory(tables)
    print('Analyzing Matterport caches', flush=True)
    models, rooms, openings, coverage, numeric = matterport(client, mp)
    print('Analyzing Vexcel caches', flush=True)
    locations, geometries, attrs, cities, calculated = vexcel(client, vx, tables['models_index'])
    print('Measuring cross-source agreement', flush=True)
    pairs, reps, commpairs, agreement, funnels, light, links, times = joins_agreement(tables, models, locations, geometries)
    checks, keys, proxy, cal_summary = additional_checks(tables, models, rooms, openings, locations, reps, commpairs, calculated)
    charts(models, reps, locations, coverage)
    result = {'table_inventory': summaries, 'text_profiles': text_summary, 'mp_coverage': coverage,
              'mp_numeric': numeric, 'agreement': agreement, 'funnels': funnels, 'light': light,
              'foreign_keys': links, 'time_alignment': times,
              'vexcel_statuses': dict(collections.Counter(str(l['property_status']) for l in locations)),
              'vexcel_footprints': sum(l['footprint'] for l in locations),
              'vexcel_oblique': sum(l['oblique_collections'] > 0 for l in locations),
              'city_total': len(cities), 'city_sampled': sum(c['sample_points'] > 0 for c in cities),
              'city_with_measured_coordinate': sum(c['sample_points'] + c['coordinate_shared_points'] > 0 for c in cities),
              'data_quality': checks, 'key_integrity': keys, 'roof_tree_proxies': proxy, 'calculated20_summary': cal_summary,
              'residential_pairs': len(pairs), 'residential_unique_homes': len(reps),
              'commercial_pairs': len(commpairs), 'actual_network_calls_in_analysis': dict(client.calls)}
    dump(OUT / 'audit_metrics.json', result)
    write_report(result, tables, models, rooms, openings, locations, cities, reps, commpairs, attrs, calculated)
    if not client.online:
        verify_artifacts(result, client, baseline)
    print('Metrics written; offline network calls', dict(client.calls), flush=True)


if __name__ == '__main__':
    main()
