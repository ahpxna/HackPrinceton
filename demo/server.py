"""Serve the demo with a quiet, same-origin bridge to the local agent."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit, parse_qs, unquote

SITE = Path(__file__).resolve().parent / 'site'
sys.path.insert(0, str(SITE.parent))
from tour_data import model_data, fit_sweeps
from flight_data import flight_context
from furniture import upload, lookup, openings, scan
AGENT = 'http://127.0.0.1:8770'
WALK_ROOT = SITE.parent.parent / 'data/splat/web'
SID_PATTERN = re.compile(r'[A-Za-z0-9]{8,32}')


def matterport_dimensions(sid):
    if not SID_PATTERN.fullmatch(sid):
        raise ValueError('Invalid model identifier')
    path = SITE.parent.parent / 'data/cache' / sid / 'matterport.json'
    model = json.loads(path.read_text()).get('data', {}).get('model') or {}
    rooms = []
    for room in model.get('rooms') or []:
        dimensions = room.get('dimensionEstimates') or {}
        units = dimensions.get('units')
        if units not in ('imperial', 'metric'):
            continue
        scale = .3048 if units == 'imperial' else 1
        def metres(value):
            return value * scale if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else None
        values = {key: metres(dimensions.get(key)) for key in ('length', 'width', 'height')}
        ceiling = room.get('ceiling') or {}
        values.update({key: metres(ceiling.get(key)) for key in ('minHeight', 'maxHeight')})
        values = {key: value for key, value in values.items() if value is not None}
        if not values:
            continue
        labels = [c.get('label') for c in room.get('classifications') or [] if c.get('label')]
        rooms.append({'id': room.get('id'), 'label': room.get('label') or ' / '.join(labels) or 'Room (no label)',
                      'floor_id': (room.get('floor') or {}).get('id'), 'dimensions_m': values})
    return {'sid': sid, 'source': "Matterport's cached room dimension estimates", 'rooms': rooms}


def sdk_key():
    value = os.environ.get('MATTERPORT_SDK_KEY', '')
    if value:
        return value
    path = SITE.parent.parent / '.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            name, separator, value = line.strip().removeprefix('export ').partition('=')
            if separator and name.strip() == 'MATTERPORT_SDK_KEY':
                return value.strip().strip('\"\'')
    return ''


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        # Bundle resource URLs can carry applicationKey. Never log request URLs.
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def walk(self, head=False):
        try:
            parts = unquote(urlsplit(self.path).path).split('/')
            if (len(parts) < 4 or not SID_PATTERN.fullmatch(parts[2])
                    or any(p.startswith('.') or '\\' in p or '\x00' in p for p in parts[3:])):
                raise ValueError('Invalid walk path')
            root = WALK_ROOT.resolve()
            folder = root / parts[2]
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError('Missing walk folder')
            target = folder
            for part in parts[3:]:
                target = target / part
                if target.is_symlink():
                    raise ValueError('Walk symlinks are not served')
            target = target.resolve()
            target.relative_to(folder)
            if target.is_dir():
                if (target / 'index.html').is_symlink():
                    raise ValueError('Walk symlinks are not served')
                target = (target / 'index.html').resolve()
                target.relative_to(folder)
            with target.open('rb') as source:
                self.send_response(200)
                content_type = {'.js': 'text/javascript', '.spz': 'application/octet-stream',
                                '.obj': 'text/plain', '.json': 'application/json'}.get(target.suffix.lower(), self.guess_type(str(target)))
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(os.fstat(source.fileno()).st_size))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                if not head:
                    self.copyfile(source, self.wfile)
        except (OSError, ValueError, RuntimeError):
            self.send_error(404, 'Walk asset not found')

    def do_HEAD(self):
        if urlsplit(self.path).path.startswith('/walk/'):
            return self.walk(head=True)
        return super().do_HEAD()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith('/walk/'):
            return self.walk()
        if path.startswith('/measure/data/'):
            try:
                return self.reply(200, matterport_dimensions(path[len('/measure/data/'):]))
            except (OSError, ValueError, KeyError, TypeError):
                return self.reply(200, {'rooms': [], 'available': False})
        if path.startswith('/api/furniture/openings/'):
            try:
                return self.reply(200, openings(path.rsplit('/', 1)[-1]))
            except (OSError, ValueError, KeyError, TypeError):
                return self.reply(200, {'available': False, 'zones': []})
        if path.startswith('/flight/data/'):
            try:
                return self.reply(200, flight_context(path.rsplit('/', 1)[-1]))
            except (OSError, ValueError, KeyError, TypeError):
                return self.reply(400, {'error': 'Flight daylight context unavailable'})
        if path == '/home.html':
            key = sdk_key()
            config = json.dumps({'configured': bool(key), 'key': key}).replace('<', '\\u003c')
            page = (SITE / 'home.html').read_text().replace('</head>',
                '<script type="application/json" id="tour-config">' + config + '</script></head>')
            content = page.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(content)
            return
        if path.startswith('/tour/data/'):
            try:
                return self.reply(200, model_data(path.rsplit('/', 1)[-1], acquire=parse_qs(urlsplit(self.path).query).get('offline') != ['1']))
            except (OSError, ValueError, TypeError, KeyError):
                return self.reply(400, {'error': 'Model anchors unavailable'})
        if path == '/agent/status':
            # A missing/old/unconfigured server is a normal offline state.
            # Never probe /ask: status checks must not invoke a model.
            try:
                with urllib.request.urlopen(AGENT + '/status', timeout=2) as response:
                    value = json.load(response)
                configured = value.get('configured') is True and value.get('real_model_run') is not False and value.get('dry_run') is not True
                return self.reply(200, {'configured': configured})
            except (OSError, ValueError, AttributeError):
                return self.reply(200, {'configured': False})
        if path.startswith('/agent/files/'):
            return self.proxy(path[len('/agent'):])
        return super().do_GET()

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.reply(403, {'error': 'same-origin requests only'})
        path = urlsplit(self.path).path
        if path in ('/api/furniture/lookup', '/api/furniture/scan'):
            try:
                fields = upload(self, 12_000_000 if path.endswith('/lookup') else 100_000_000)
                if path.endswith('/lookup'):
                    return self.reply(200, lookup(fields['file']))
                result, status = scan(fields)
                return self.reply(status, result)
            except (ValueError, OSError, TypeError, KeyError):
                return self.reply(400, {'error': 'Invalid upload; use a photo up to 12 MB or video up to 100 MB'})
            except Exception:
                return self.reply(503, {'error': 'Local photo model unavailable; enter exact size manually'})
        if path.startswith('/tour/fit/'):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2000000:
                    raise ValueError('Invalid sweep payload size')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, list):
                    raise ValueError('Expected SDK sweep list')
                return self.reply(200, fit_sweeps(path.rsplit('/', 1)[-1], body))
            except (OSError, ValueError, TypeError, KeyError):
                return self.reply(400, {'error': 'Sweep fit unavailable: need four non-collinear matching stable IDs'})
        if path not in ('/agent/ask', '/agent/approve', '/agent/check'):
            return self.reply(404, {'error': 'unknown endpoint'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 65536:
                raise ValueError()
        except ValueError:
            return self.reply(400, {'error': 'invalid request size'})
        return self.proxy(path[len('/agent'):], self.rfile.read(size))

    def proxy(self, path, body=None):
        request = urllib.request.Request(AGENT + path, data=body,
                                         headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as error:
            return self.reply(error.code, {'error': 'agent request failed'})
        except OSError:
            return self.reply(502, {'error': 'agent unavailable'})
        with response:
            content_type = response.headers.get('Content-Type', 'application/octet-stream')
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                if 'ndjson' in content_type:
                    for line in response:
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    while chunk := response.read(65536):
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass


if __name__ == '__main__':
    print('Serving demo at http://localhost:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
