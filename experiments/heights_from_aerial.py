"""Brief B. Reproduce with python3 -I experiments/heights_from_aerial.py khbuN7q853q.

Only this script and data/shadow_check are written. Credentials are read at run
time and never included in URLs, logs, cached requests, or output artifacts.
"""
import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.vexcelgroup.com/v2/'
OUT = ROOT / 'data/shadow_check/khbuN7q853q'
os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib-cache')
AUTH_FAILURES = 0
PINNED_IMAGES = {
    'nadir': '2026~us-tx-houston-2026~images~NE_20251219_172901_76_2DD2A436D8065C4_rgb',
    'oblique-north': '2026~us-tx-houston-2026~images~N_20251219_175409_49_2DD2A431351EF7A_rgb',
}


def font(size=20):
    path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def documentation():
    for service in ['oriented', 'property', 'ortho', 'custom-elements']:
        path = OUT / f'schema-{service}.json'
        if not path.exists():
            with urllib.request.urlopen(API + 'api-docs/' + service, timeout=60) as response:
                schema = json.load(response)
            write_json(path.name, schema)


def credentials():
    values = {}
    with (ROOT / '.env').open() as stream:
        for line in stream:
            if '=' in line and not line.lstrip().startswith('#'):
                name, value = line.strip().split('=', 1)
                values[name] = value.strip().strip('\"\'')
    return values


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def request(endpoint, params=None, body=None, cache=None):
    global AUTH_FAILURES
    if cache and (OUT / cache).exists():
        raw = (OUT / cache).read_bytes()
        return json.loads(raw) if cache.endswith('.json') else raw
    if AUTH_FAILURES >= 2:
        raise RuntimeError('Stopped after repeated authorization/geofence failures')
    key = credentials()['VEXCEL_API_KEY']
    url = API + endpoint
    if params:
        url += '?' + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    headers = {'X-API-KEY': key, 'Content-Type': 'application/json'}
    log = {'endpoint': endpoint, 'params': params, 'body': body}
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
                log.update(status=response.status, bytes=len(raw),
                           sha256=hashlib.sha256(raw).hexdigest(),
                           content_type=response.headers.get('Content-Type'))
                # Only these response fields can be persisted; no auth headers.
                log['spatial_headers'] = {k: v for k, v in response.headers.items()
                    if any(s in k.lower() for s in ('image', 'bound', 'extent', 'resolution', 'gsd'))}
            break
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                AUTH_FAILURES += 1
            if error.code == 429 and attempt < 4:
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            log.update(status=error.code)
            # API validation messages are useful, but strip any credential.
            message = error.read().decode(errors='replace').replace(key, '[redacted]')
            log['error'] = message[:3000]
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(log) + '\n')
            raise RuntimeError(f'{endpoint}: HTTP {error.code}: {message[:1000]}') from None
    with (OUT / 'requests.jsonl').open('a') as f:
        f.write(json.dumps(log) + '\n')
    if cache:
        (OUT / cache).write_bytes(raw)
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw


def enu(lon, lat, lon0, lat0):
    lon,lat,lon0,lat0=map(float,(lon,lat,lon0,lat0))
    return np.array([(lon-lon0)*111320*math.cos(math.radians(lat0)),
                     (lat-lat0)*110574])


def ll(east, north, lon0, lat0):
    return [lon0+east/(111320*math.cos(math.radians(lat0))), lat0+north/110574]


def square(lon, lat, radius):
    points = [ll(e, n, lon, lat) for e, n in
              [(-radius,-radius),(radius,-radius),(radius,radius),(-radius,radius),(-radius,-radius)]]
    return 'POLYGON((' + ','.join(f'{x} {y}' for x,y in points) + '))'


def sun(when, lat, lon):
    """Geometric solar position, NOAA/Meeus equations; UTC aware input."""
    jd = when.timestamp()/86400 + 2440587.5
    t = (jd-2451545)/36525
    l = (280.46646+t*(36000.76983+t*0.0003032)) % 360
    m = math.radians(357.52911+t*(35999.05029-0.0001537*t))
    e = 0.016708634-t*(0.000042037+0.0000001267*t)
    c = math.sin(m)*(1.914602-t*(0.004817+0.000014*t)) + math.sin(2*m)*(0.019993-0.000101*t)+math.sin(3*m)*0.000289
    om = math.radians(125.04-1934.136*t)
    lam = math.radians(l+c-0.00569-0.00478*math.sin(om))
    eps = math.radians(23+(26+(21.448-t*(46.815+t*(0.00059-t*0.001813)))/60)/60+0.00256*math.cos(om))
    dec = math.asin(math.sin(eps)*math.sin(lam))
    y = math.tan(eps/2)**2
    lr=math.radians(l)
    eq = 4*math.degrees(y*math.sin(2*lr)-2*e*math.sin(m)+4*e*y*math.sin(m)*math.cos(2*lr)-0.5*y*y*math.sin(4*lr)-1.25*e*e*math.sin(2*m))
    minutes=when.hour*60+when.minute+when.second/60
    ha=math.radians(((minutes+eq+4*lon)%1440)/4-180)
    la=math.radians(lat)
    elevation=math.degrees(math.asin(math.sin(la)*math.sin(dec)+math.cos(la)*math.cos(dec)*math.cos(ha)))
    azimuth=(math.degrees(math.atan2(math.sin(ha),math.cos(ha)*math.sin(la)-math.tan(dec)*math.cos(la)))+180)%360
    return {'azimuth_deg':azimuth,'elevation_deg':elevation,'shadow_azimuth_deg':(azimuth+180)%360}


def utm(lon, lat, zone=15):
    """WGS84 transverse Mercator, Snyder series, metre precision here."""
    a=6378137.; e2=0.0066943799901413165; ep=e2/(1-e2); k=.9996
    p=math.radians(lat); dl=math.radians(lon-(zone*6-183))
    n=a/math.sqrt(1-e2*math.sin(p)**2); tt=math.tan(p)**2; cc=ep*math.cos(p)**2; aa=math.cos(p)*dl
    m=a*((1-e2/4-3*e2**2/64-5*e2**3/256)*p-(3*e2/8+3*e2**2/32+45*e2**3/1024)*math.sin(2*p)+(15*e2**2/256+45*e2**3/1024)*math.sin(4*p)-35*e2**3/3072*math.sin(6*p))
    x=500000+k*n*(aa+(1-tt+cc)*aa**3/6+(5-18*tt+tt**2+72*cc-58*ep)*aa**5/120)
    north=k*(m+n*math.tan(p)*(aa**2/2+(5-tt+9*cc+4*cc**2)*aa**4/24+(61-58*tt+tt**2+600*cc-330*ep)*aa**6/720))
    return np.array([x,north])


def xmp(raw):
    im=Image.open(io.BytesIO(raw))
    return dict(re.findall(r'<Vexcel:([^>]+)>(.*?)</Vexcel:[^>]+>',im.info.get('xmp',b'').decode()))


def project(matrix, points):
    p=np.atleast_2d(points)
    q=np.column_stack([p,np.ones(len(p))])@np.asarray(matrix).T
    return q[:,:2]/q[:,2:3]


def geometry_for(view, props, lon, lat, raw):
    # Fit a local projective camera to explicit-Z API transforms; no DSM is used.
    points=np.array([[e,n,z] for z in [22.5552,42.5552] for n in [-60,0,60] for e in [-60,0,60]])
    world=[[*ll(e,n,lon,lat),z] for e,n,z in points]
    wkt='LINESTRINGZ('+','.join(' '.join(map(str,p)) for p in world)+')'
    response=request('oriented/transform-points',body={'operation':'world-2-pixel','image-name':props['image-name'],'wkt':wkt,'srid':4326},cache=f'transform-{view}.json')
    pixel=np.array([[p['x'],p['y']] for p in response['points']])
    bbox=list(map(float,xmp(raw)['pixel-bbox'].split(',')))
    pixel-=bbox[:2]
    rows=[]; rhs=[]
    for (e,n,z),(u,v) in zip(points,pixel):
        rows += [[e,n,z,1,0,0,0,0,-u*e,-u*n,-u*z], [0,0,0,0,e,n,z,1,-v*e,-v*n,-v*z]]
        rhs += [u,v]
    fit=np.linalg.lstsq(rows,rhs,rcond=None)[0]
    matrix=np.r_[fit,1].reshape(3,4)
    residual=float(np.max(np.linalg.norm(project(matrix,points)-pixel,axis=1)))
    check=np.array([[-31.7,22.4,28.1],[13.2,-29.7,35.2],[41.6,38.8,47.3],[-17.5,-8.4,23.7]])
    check_wkt='LINESTRINGZ('+','.join(' '.join(map(str,[*ll(e,n,lon,lat),z])) for e,n,z in check)+')'
    independent=request('oriented/transform-points',body={'operation':'world-2-pixel','image-name':props['image-name'],'wkt':check_wkt,'srid':4326},cache=f'transform-validation-{view}.json')
    check_pixels=np.array([[p['x'],p['y']] for p in independent['points']])-bbox[:2]
    check_error=float(np.max(np.linalg.norm(project(matrix,check)-check_pixels,axis=1)))
    if check_error>.5:
        raise RuntimeError(f'{view} held-out world/pixel mapping exceeds 0.5px')
    write_json(f'projection-{view}.json',{'matrix':matrix.tolist(),'crop_raw_pixel_bbox':bbox,'max_fit_error_px':residual,
        'held_out_max_error_px':check_error,'held_out_world_points_enu_elevation_m':check.tolist(),
        'reference_lon':lon,'reference_lat':lat,'coordinate_order':'east_m,north_m,elevation_m; output raw crop pixels before EXIF rotation'})
    print(view,'projection residual px',round(residual,4),flush=True)
    return matrix


def rectify(raw,matrix,name,radius=65,res=.1,z=22.5552):
    """North-up rendering on a flat ground plane, preserving elevated parallax."""
    count=round(2*radius/res)
    e,n=np.meshgrid(-radius+(np.arange(count)+.5)*res,radius-(np.arange(count)+.5)*res)
    points=np.column_stack([e.ravel(),n.ravel(),np.full(e.size,z)])
    xy=project(matrix,points)
    im=np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
    x=np.rint(xy[:,0]).astype(int);y=np.rint(xy[:,1]).astype(int)
    valid=(x>=0)&(y>=0)&(x<im.shape[1])&(y<im.shape[0])
    arr=np.zeros((e.size,3),dtype=np.uint8);arr[valid]=im[y[valid],x[valid]]
    out=Image.fromarray(arr.reshape(count,count,3));out.save(OUT/name)
    return out


def acquisition(lon,lat):
    base={'wkt':f'POINT({lon} {lat})','srid':4326,'layer':'urban'}
    fields='image-name,geometry,camera-pos-x,camera-pos-y,camera-pos-z,utm-zone,ground-z,omega,phi,kappa,focal-length,pixel-size,raster-size-width,raster-size-height,pp0-x,pp0-y,capture-date,collection'
    cameras=[]; matrices={}; images={}
    for view,file in [('nadir','query-nadir-full.json'),('oblique-north','query-oblique-full.json')]:
        response=request('oriented/query',{**base,'product-type':view,'collection':'us-tx-houston-2026','include':fields},cache=file)
        props=next(f['properties'] for f in response['features']
                   if f['properties']['image-name']==PINNED_IMAGES[view])
        name='nadir.jpg' if view=='nadir' else 'oblique.jpg'
        raw=request('oriented/extract',body={'wkt':square(lon,lat,65),'srid':4326,'layer':'urban','image-name':props['image-name'],'rotation':'north' if view=='nadir' else 'natural','crop':'bbox','image-format':'jpeg/png'},cache=name)
        meta=xmp(raw)
        if meta['image-name']!=props['image-name']:
            raise RuntimeError('Requested and delivered raw image differ')
        matrix=geometry_for(view,props,lon,lat,raw)
        matrices[view]=matrix
        images[view]=rectify(raw,matrix,f'ground-{view}.jpg')
        when=dt.datetime.fromisoformat(props['capture-date']).replace(tzinfo=dt.timezone.utc)
        hypotheses={label:sun(when+dt.timedelta(hours=offset),lat,lon) for label,offset in [('UTC',0),('America/Chicago_CST',6)]}
        horizontal=float(np.linalg.norm(np.array([props['camera-pos-x'],props['camera-pos-y']])-utm(lon,lat)))
        cameras.append({**props,'capture_time_hypotheses':hypotheses,'xmp':meta,
            'camera_centre':{'easting_m':props['camera-pos-x'],'northing_m':props['camera-pos-y'],'elevation_m':props['camera-pos-z'],'crs':'EPSG:32615; vertical datum unspecified'},
            'horizontal_distance_to_house_m':horizontal,'slant_distance_to_house_ground_m':math.hypot(horizontal,props['camera-pos-z']-22.5552),
            'distance_method':'WGS84 house lon/lat converted to UTM15N, Euclidean camera distance; vertical datum presumed compatible',
            'projection_max_error_px':json.loads((OUT/f'projection-{view}.json').read_text())['max_fit_error_px']})
    write_json('camera.json',{'images':cameras,'time_zone_conclusion':'pending shadow measurement'})
    return cameras,matrices,images


def parse_ring(wkt):
    return np.array([[float(x) for x in p.split()[:2]] for p in re.search(r'\(\(([^)]+)\)',wkt).group(1).split(',')])


def canvas_xy(points):
    p=np.asarray(points)
    return np.column_stack([(p[:,0]+65)/.1,(65-p[:,1])/.1])


def load_buildings(lon,lat):
    response=request('property/extract',{'wkt':f'POINT({lon} {lat})','srid':4326,'layer':'urban','buffer-meter':65,'latest':'true','include':'precomputed,calculated','unit-system':'metric'},cache='properties.json')
    buildings=[]
    for feature in response['features']:
        p=feature['properties']
        if p.get('metadata.collection')!='us-tx-houston-2026':
            continue
        for s in p.get('structures') or []:
            if not s.get('footprint.wkt'):
                continue
            ring=[enu(x,y,lon,lat).tolist() for x,y in parse_ring(s['footprint.wkt'])]
            height=s.get('roof-characteristics.height')
            buildings.append({'id':f'B{len(buildings)+1}','type':'building','outline_enu_m':ring,'outline_lonlat':parse_ring(s['footprint.wkt']).tolist(),
                'height_m':height,'ground_elevation_m':s.get('ground-elevation.value'),
                'roof_elevation_m':s.get('roof-characteristics.elevation'),
                'method':'Vexcel property DSM minus DTM at footprint centroid; constant-height extrusion',
                'height_error_m':None,'height_error_note':'Provider supplies no uncertainty. Centroid height is not eave or ridge; model sensitivity uses +/-2m, not a measured confidence interval.',
                'image_name':p.get('metadata.image-id'),'capture_time':p.get('metadata.ortho-date-estimate'),
                'collection':p['metadata.collection'],'is_own_house':False})
    from matplotlib.path import Path as PolygonPath
    for b in buildings:
        b['is_own_house']=bool(PolygonPath(b['outline_enu_m']).contains_point([0,0]))
    write_json('buildings.json',buildings)
    return buildings


def inspection(images,buildings):
    for view,image in images.items():
        im=image.copy(); d=ImageDraw.Draw(im)
        for value in range(-60,61,10):
            x=(value+65)*10;y=(65-value)*10
            d.line([(x,0),(x,1300)],fill=(150,150,150),width=1)
            d.line([(0,y),(1300,y)],fill=(150,150,150),width=1)
            d.text((x+2,50),f'E{value}',fill='yellow',stroke_width=1,stroke_fill='black')
            d.text((5,y),f'N{value}',fill='yellow',stroke_width=1,stroke_fill='black')
        for b in buildings:
            xy=canvas_xy(b['outline_enu_m']);d.line([tuple(p) for p in xy],fill='cyan',width=2)
            p=xy.mean(0)
            d.text(tuple(p),f"{b['id']} {b['height_m']:.2f}m" if b['height_m'] is not None else b['id'],fill='yellow',stroke_width=2,stroke_fill='black')
        im.save(OUT/f'inspection-{view}.jpg')


def detect_trees(lon,lat,camera):
    oblique='~N_' in camera['image-name']
    body={'wkt':square(lon,lat,65),'images':[{'endpoint':'oriented/extract','request':{'wkt':square(lon,lat,65),'srid':4326,'layer':'urban','image-name':camera['image-name'],'rotation':'natural' if oblique else 'north','crop':'bbox','image-format':'jpeg/png'}}],
        'model-input':{'elements':[{'label':'tree','detection-threshold':.35}],'include':['pixel-mask','pixel-bbox','geometry','confidence']}}
    result=request('elements/detect',body=body,cache='tree-detection-oblique.json' if oblique else 'tree-detection.json')
    meta=result['images'][0]['metadata']
    if meta['image-name']!=camera['image-name']:
        raise RuntimeError('Tree detection used a different raw image')
    if not oblique and (meta['pixel-bbox']!=camera['xmp']['pixel-bbox'] or
                        (meta['width'],meta['height'])!=Image.open(OUT/'nadir.jpg').size):
        raise RuntimeError('Used nadir masks do not match the raw crop pixel grid')
    return result


def model_geometry(sid):
    filename=OUT/'model-geo.json'
    if filename.exists():
        return json.loads(filename.read_text())
    query='''{ model(id:"%s") { floors { edges(includeUsed:true) { type vertices { position { x y } } openings { type } } } geocoordinates { latitude longitude source p0:geoLocationOf(modelLocation:{x:0,y:0,z:0}){lat long} px:geoLocationOf(modelLocation:{x:20,y:0,z:0}){lat long} py:geoLocationOf(modelLocation:{x:0,y:20,z:0}){lat long} } } }'''%sid
    keys=credentials()
    auth=base64.b64encode((keys['MATTERPORT_API_KEY']+':'+keys['MATTERPORT_API_TOKEN']).encode()).decode()
    req=urllib.request.Request('https://api.matterport.com/api/models/graph',json.dumps({'query':query}).encode(),{'Content-Type':'application/json','Authorization':'Basic '+auth})
    with urllib.request.urlopen(req,timeout=90) as response:
        result=json.load(response)
    if result.get('errors'):
        raise RuntimeError('Matterport geometry query failed; response withheld')
    write_json('model-geo.json',result)
    return result


def unproject(matrix,pixels,z):
    p=np.atleast_2d(pixels); P=np.asarray(matrix)
    H=np.column_stack([P[:,0],P[:,1],P[:,2]*z+P[:,3]])
    q=np.column_stack([p,np.ones(len(p))])@np.linalg.inv(H).T
    return q[:,:2]/q[:,2:3]


def convex_hull(points):
    pts=sorted(set(map(tuple,points)))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return np.array(lower[:-1]+upper[:-1])


def sample_image(im,xy):
    x=xy[:,0];y=xy[:,1]
    valid=(x>=0)&(x<im.shape[1]-1)&(y>=0)&(y<im.shape[0]-1)
    x=np.clip(x,0,im.shape[1]-2);y=np.clip(y,0,im.shape[0]-2)
    ix=x.astype(int);iy=y.astype(int);dx=(x-ix)[:,None];dy=(y-iy)[:,None]
    rgb=(im[iy,ix]*(1-dx)*(1-dy)+im[iy,ix+1]*dx*(1-dy)+im[iy+1,ix]*(1-dx)*dy+im[iy+1,ix+1]*dx*dy)
    return rgb,valid


def tree_heights(detection,matrices,cameras,lon,lat):
    P=matrices['nadir'];Q=matrices['oblique-north'];ground=22.5552
    nadir=np.asarray(Image.open(OUT/'nadir.jpg').convert('RGB'),dtype=float)
    oblique=np.asarray(Image.open(OUT/'oblique.jpg').convert('RGB'),dtype=float)
    trees=[];curves=[]
    for index,feature in enumerate(detection['features']):
        det=feature['properties']['detections'][0]
        if det['confidence']<.55:continue
        mask=np.array(Image.open(io.BytesIO(base64.b64decode(det['pixel-mask']))))>0
        yy,xx=np.nonzero(mask)
        if len(xx)<100:continue
        bbox=det['pixel-bbox'];pixels=np.column_stack([xx+bbox[0],yy+bbox[1]])
        # SAM mask coordinates refer to the raw crop, before EXIF orientation.
        apparent=unproject(P,pixels,ground)
        c=apparent.mean(0)
        if np.any(np.abs(c)>63):continue
        take=np.linspace(0,len(pixels)-1,min(5000,len(pixels))).astype(int)
        pixels=pixels[take];col,valid=sample_image(nadir,pixels)
        scores=[];hs=np.arange(0,31,.5)
        for h in hs:
            en=unproject(P,pixels,ground+h)
            target=project(Q,np.column_stack([en,np.full(len(en),ground+h)]))
            rgb,ok=sample_image(oblique,target);ok &=valid
            a=col[ok];b=rgb[ok]
            if len(a)<100:scores.append(-1.);continue
            # Channelwise demean avoids making seasonal brightness a match.
            a=a-a.mean(0);b=b-b.mean(0)
            scores.append(float(np.sum(a*b)/max(1e-9,np.sqrt(np.sum(a*a)*np.sum(b*b)))))
        best=int(np.argmax(scores));height=float(hs[best]);score=scores[best]
        near=hs[np.array(scores)>=score-.08]
        error=max(2.5,float(max(height-near.min(),near.max()-height)))
        accepted=score>=.45 and 2<=height<=27 and error<10
        # Convex crown outline at the fitted top plane, never a geolocated SAM
        # ground polygon: the latter embeds an unspecified elevation model.
        hull=convex_hull(apparent[::max(1,len(apparent)//1200)])
        hullraw=project(P,np.column_stack([hull,np.full(len(hull),ground)]))
        ray_slope=unproject(P,hullraw,ground+1)-hull
        outline=unproject(P,hullraw,ground+height) if accepted else hull
        tid=f'T{len(trees)+1}'
        tree={'id':tid,'type':'tree','detection_index':index,'detection_confidence':det['confidence'],
            'outline_enu_m':outline.tolist(),'outline_lonlat':[ll(e,n,lon,lat) for e,n in outline],
            'apparent_crown_enu_m':hull.tolist(),
            'crown_outline_ray_slope_enu_per_m_height':ray_slope.tolist(),
            'outline_note':'Resolved crowns are projected to the estimated top plane; unresolved outlines are apparent ground-plane rays, not known stem positions.',
            'height_m':height if accepted else None,
            'candidate_height_m':height,'height_error_m':error if accepted else None,
            'error_interpretation':'Conservative matching sensitivity half-width (score within 0.08 of peak), minimum +/-2.5m; not a statistical confidence interval. Crown plane and correspondence can be wrong.',
            'method':'SAM3 crown; two-view calibrated plane-sweep photometric height from nadir/oblique parallax',
            'match_score':score,'status':'provisional_stereo_estimate' if accepted else 'failed_ambiguous_stereo',
            'shadow_height_m':None,'shadow_status':'No unique stem-to-shadow endpoint correspondence measured; visibility and overlap remain unresolved.',
            'ground_elevation_m':ground,'image_name':[c['image-name'] for c in cameras],
            'capture_time':[c['capture-date']+'Z' for c in cameras],
            'crown_note':'Convex hull of visible winter crown; can merge trees or omit bare branches. Ground terrain held flat.'}
        trees.append(tree);curves.append({'id':tid,'detection_index':index,'height_m':hs.tolist(),'correlation':scores})
    write_json('stereo-height-curves.json',curves)
    write_json('tree-estimates.json',trees)
    return trees


def calibration(buildings,camera,image):
    # Manual digitisation in the north-up 0.10m ground raster. These are visible
    # northern shadow-step corners, not inferred tree-covered shadow endpoints.
    observations=[('B18',7,[624.3,538.0],.7),('B17',0,[519.0,538.5],.6),('B14',5,[347.0,541.0],.8)]
    s=camera['capture_time_hypotheses']['UTC'];az=math.radians(s['shadow_azimuth_deg'])
    direction=np.array([math.sin(az),math.cos(az)]);tan=math.tan(math.radians(s['elevation_deg']))
    im=image.copy();draw=ImageDraw.Draw(im);results=[]
    for bid,vertex,tip,error in observations:
        b=next(b for b in buildings if b['id']==bid)
        base=np.array(b['outline_enu_m'][vertex]);end=np.array([tip[0]*.1-65,65-tip[1]*.1])
        delta=end-base;along=float(delta@direction);cross=float(direction[0]*delta[1]-direction[1]*delta[0])
        predicted=b['height_m']/tan;pred=base+direction*predicted
        actual=float(np.linalg.norm(delta));measuredaz=math.degrees(math.atan2(delta[0],delta[1]))%360
        rec={'id':bid,'base_outline_vertex':vertex,'base_enu_m':base.tolist(),'measured_shadow_tip_canvas_px':tip,'measured_shadow_tip_enu_m':end.tolist(),
            'known_centroid_roof_height_m':b['height_m'],'predicted_shadow_length_m':predicted,
            'measured_shadow_length_m':actual,'measured_along_sun_shadow_m':along,'cross_track_error_m':cross,
            'length_error_m':actual-predicted,'endpoint_error_m':float(np.linalg.norm(end-pred)),
            'measured_shadow_azimuth_deg':measuredaz,'shadow_derived_casting_edge_height_m':along*tan,
            'shadow_measurement_error_m':error,'height_error_m':error*tan+.3,
            'result':'Centroid roof height overpredicts the visible eave shadow; roof slopes/silhouette correspondence prevent a calibration of the centroid height.',
            'base_assumption':'Provider footprint vertex approximates vertical projection of this visible eave corner. +/-0.5m registration included in endpoint allowance.'}
        results.append(rec)
        xy=canvas_xy([base,pred,end]);draw.line([tuple(xy[0]),tuple(xy[1])],fill='cyan',width=4);draw.line([tuple(xy[0]),tuple(xy[2])],fill='yellow',width=3)
        for p in xy:draw.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),outline='red',width=2)
        draw.text((xy[0,0]+8,xy[0,1]+15),bid,font=font(22),fill='white',stroke_width=2,stroke_fill='black')
    annotated=Image.new('RGB',(1300,1480),'black');annotated.paste(im,(0,180));legend=ImageDraw.Draw(annotated)
    legend.text((15,10),'Known-height calibration | N up | 0.10 m/px | fixed 2025-12-19 17:29:01 UTC image',font=font(23),fill='white')
    legend.text((15,43),'CYAN: predicted from centroid roof height. YELLOW: manually measured northern eave shadow.',font=font(21),fill='white')
    for j,r in enumerate(results):
        legend.text((15,76+j*27),f"{r['id']}: height {r['known_centroid_roof_height_m']:.2f} m | predicted {r['predicted_shadow_length_m']:.2f} m | measured {r['measured_shadow_length_m']:.2f} m | error {r['length_error_m']:+.2f} m",font=font(20),fill='white')
    legend.text((15,157),'Manual endpoints and footprint/eave correspondence are uncertain; centroid height fails this eave calibration.',font=font(18),fill='white')
    annotated.save(OUT/'calibration.jpg')
    write_json('calibration.json',{'image_name':camera['image-name'],'capture_time':camera['capture-date']+'Z','sun':s,'measurements':results,
        'timezone_conclusion':'UTC supported: measured shadows north-northwest; CST hypothesis has sun below horizon. Detection metadata independently returns the same UTC Unix timestamp.'})
    return results


def registered_origin(sid,buildings,lon,lat,image):
    # Reuse the established wall-fill and FFT raster helpers, without invoking
    # geo_check.main/analyse or writing any of that worker's output paths.
    import importlib.util
    spec=importlib.util.spec_from_file_location('geo_helpers',ROOT/'experiments/geo_check.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    m=model_geometry(sid)['data']['model'];g=m['geocoordinates']
    p0=enu(g['p0']['long'],g['p0']['lat'],lon,lat)
    A=np.column_stack([(enu(g['px']['long'],g['px']['lat'],lon,lat)-p0)/20,
                       (enu(g['py']['long'],g['py']['lat'],lon,lat)-p0)/20])
    segments=[tuple((v['position']['x'],v['position']['y']) for v in e['vertices']) for f in m['floors'] for e in f['edges'] if len(e['vertices'])==2]
    inside=helper.interior_points(segments)
    if inside is None:raise RuntimeError('Cannot register open Matterport wall graph')
    inner=inside@A.T+p0;centre=inner.mean(0)
    own=next(b for b in buildings if b['is_own_house']);ring=np.array(own['outline_enu_m']);origin=ring.mean(0)
    ax=(np.arange(helper.GRID)-helper.GRID//2+.5)*helper.RES
    gx,gy=np.meshgrid(ax+origin[0],ax+origin[1],indexing='ij')
    F=helper.Path(ring).contains_points(np.column_stack([gx.ravel(),gy.ravel()])).reshape(helper.GRID,helper.GRID)
    fF=np.fft.rfft2(F.astype(float));area=F.sum();shifts=np.r_[0:helper.MAX_SHIFT+1,-helper.MAX_SHIFT:0]
    best=(-1,None,None,None);before=None
    for theta in helper.ROT_RANGE:
        t=math.radians(theta);R=np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
        M=helper.raster((inner-centre)@R.T+centre,origin)
        corr=np.fft.irfft2(fF*np.conj(np.fft.rfft2(M.astype(float))),s=F.shape)
        if theta==0:before=float(corr[0,0]/(area+M.sum()-corr[0,0]))
        sub=corr[np.ix_(shifts,shifts)];score=sub/(area+M.sum()-sub)
        k=np.unravel_index(np.argmax(score),score.shape)
        if score[k]>best[0]:best=(float(score[k]),float(theta),float(shifts[k[0]]*helper.RES),float(shifts[k[1]]*helper.RES))
    iou,theta,dx,dy=best;t=math.radians(theta);R=np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
    pano=json.loads((ROOT/f'data/matterport/{sid}/pano_meta.json').read_text())['pano']
    if pano['id']!='td86n1sku5u1rt38abpq787ub':
        raise RuntimeError('Pano origin differs from the assigned Brief B origin')
    pos=pano['position'];original=A@np.array([pos['x'],pos['y']])+p0
    fixed=(original-centre)@R.T+centre+[dx,dy]
    lo,la=ll(*fixed,lon,lat)
    result={'pano_id':pano['id'],'lat':la,'lon':lo,'height_above_ground_m':pos['z'],
        'enu_m':fixed.tolist(),'model_position_m':pos,'model_xy_to_enu_affine':A.tolist(),'model_origin_enu_m':p0.tolist(),
        'footprint_correction':{'rotation_deg':theta,'shift_east_m':dx,'shift_north_m':dy,'rotation_centre_enu_m':centre.tolist(),'iou_before':before,'iou_after':iou,'grid_resolution_m':helper.RES},
        'assumptions':['Model z=0 is local ground/floor datum; pano z used as height above ground (+/-0.4m sensitivity, not surveyed).','Wall-filled interior aligned to aerial roof footprint, which includes eaves; correction is a fitted registration, not independent geodetic validation.','Horizontal origin uncertainty set to +/-1m for sensitivity; not a measured confidence interval.']}
    write_json('origin-registration.json',result)
    im=image.copy();d=ImageDraw.Draw(im)
    d.line([tuple(p) for p in canvas_xy(ring)],fill='cyan',width=3)
    for col,pts in [('red',inner[::30]),('lime',(inner[::30]-centre)@R.T+centre+[dx,dy])]:
        for x,y in canvas_xy(pts):d.ellipse((x-1,y-1,x+1,y+1),fill=col)
    x,y=canvas_xy([fixed])[0];d.ellipse((x-8,y-8,x+8,y+8),fill='yellow');d.text((x+12,y),'Pano origin',fill='white',stroke_fill='black',stroke_width=2)
    d.text((20,20),f'CYAN roof; RED original interior; GREEN fitted interior; IoU {before:.3f} -> {iou:.3f}',fill='white',stroke_fill='black',stroke_width=2)
    im.save(OUT/'origin-registration.jpg')
    return result


def manual_tree_estimates(trees,matrices,cameras,lon,lat):
    P=matrices['nadir'];Q=matrices['oblique-north'];ground=22.5552
    # Approximate whole-crown correspondences digitised from the two displayed
    # ground rasters. These are NOT precise homologous branch/apex matches.
    # Deliberately large errors cover crown-shape change and partial occlusion.
    pairs={'T3':([522,298],[553,260],4.),'T4':([294,850],[326,813],4.),
           'T5':([818,716],[850,677],6.),'T7':([105,660],[148,611],6.),
           'T10':([623,817],[650,781],5.),'T12':([1015,640],[1057,586],5.)}
    matches=[]
    for t in trees:
        if t['id'] not in pairs:continue
        a,b,error=pairs[t['id']]
        def as_raw(M,p):
            en=np.array([[p[0]*.1-65,65-p[1]*.1]])
            return project(M,np.c_[en,[ground]])
        pa=as_raw(P,a);pb=as_raw(Q,b)
        a0=unproject(P,pa,ground)[0];b0=unproject(Q,pb,ground)[0]
        va=unproject(P,pa,ground+1)[0]-a0;vb=unproject(Q,pb,ground+1)[0]-b0
        delta=a0-b0;v=va-vb;h=float(-delta@v/(v@v));res=float(np.linalg.norm(delta+h*v))
        t['height_m']=round(h,2);t['height_error_m']=max(error,res/max(.1,float(np.linalg.norm(v))))
        t['status']='assumed_crown_correspondence_estimate'
        t['method']='Two-view calibrated ray fit to manually matched approximate crown centres; automatic photometric match failed'
        t['manual_crown_match']={'nadir_canvas_px':a,'oblique_canvas_px':b,'ray_horizontal_mismatch_m':res,'height_sensitivity_m_per_m_of_crown_mismatch':1/float(np.linalg.norm(v))}
        t['error_interpretation']='Conservative assumed +/- range, not measured confidence interval. Approximate crown centres are not homologous 3D points; estimate can be biased.'
        hull=np.array(t['apparent_crown_enu_m']);raw=project(P,np.c_[hull,np.full(len(hull),ground)])
        outline=unproject(P,raw,ground+h);t['outline_enu_m']=outline.tolist();t['outline_lonlat']=[ll(e,n,lon,lat) for e,n in outline]
        matches.append({'id':t['id'],**t['manual_crown_match'],'height_m':t['height_m'],'assumed_error_m':t['height_error_m']})
    # Small bare tree in the western neighbour's front lawn: stem foot and
    # isolated ground-shadow tip are visible. SAM omitted its sparse branches.
    foot=np.array([-17.1,12.9]) # canvas (479,521), ground plane
    tip=np.array([-19.1,20.6]) # canvas (459,444), road shadow endpoint
    solar=cameras[0]['capture_time_hypotheses']['UTC'];ang=math.radians(solar['shadow_azimuth_deg']);direction=np.array([math.sin(ang),math.cos(ang)])
    along=float((tip-foot)@direction);shadowh=along*math.tan(math.radians(solar['elevation_deg']))
    topcanvas=[485.,479.];top=project(Q,[[topcanvas[0]*.1-65,65-topcanvas[1]*.1,ground]])
    hs=np.arange(0,15,.01);line=project(Q,np.c_[np.tile(foot,(len(hs),1)),ground+hs]);dist=np.linalg.norm(line-top,axis=1)
    h=float(hs[np.argmin(dist)])
    angles=np.linspace(0,2*math.pi,20,endpoint=False);outline=foot+np.c_[np.cos(angles)*2,np.sin(angles)*2]
    trees.append({'id':'T15','type':'tree','outline_enu_m':outline.tolist(),'outline_lonlat':[ll(e,n,lon,lat) for e,n in outline],
        'height_m':round(shadowh,2),'height_error_m':1.8,'ground_elevation_m':ground,
        'method':'Manual bare-tree stem/shadow digitisation, solar projection; independent single-oblique vertical-displacement check',
        'status':'provisional_manual_shadow_estimate','shadow_height_m':shadowh,'oblique_height_m':h,
        'shadow_status':'Isolated endpoint on road; sparse branches make endpoint uncertain; +/-1.5m shadow allowance plus crown/trunk offset',
        'shadow_measurement':{'stem_foot_canvas_px':[479,521],'shadow_tip_canvas_px':[459,444],'length_m':float(np.linalg.norm(tip-foot)),'along_sun_length_m':along,'azimuth_deg':math.degrees(math.atan2(*(tip-foot)))%360},
        'oblique_measurement':{'assumed_tree_top_canvas_px':topcanvas,'raw_reprojection_residual_px':float(dist.min()),'assumption':'Top branch assumed above stem; horizontal crown offset is not known.'},
        'crown_note':'2m radius manual crown envelope, approximate; bare branches are transmissive, not an opaque disk.',
        'error_interpretation':'Assumed conservative +/- range; neither surveyed height nor statistical confidence interval.',
        'image_name':[c['image-name'] for c in cameras],'capture_time':[c['capture-date']+'Z' for c in cameras]})
    write_json('manual-crown-matches.json',matches)
    return trees


def annotate_trees(trees,images,cameras):
    for view in images:
        im=images[view].copy();d=ImageDraw.Draw(im)
        for t in trees:
            if view=='nadir' and 'apparent_crown_enu_m' in t:
                xy=canvas_xy(t['apparent_crown_enu_m'])
                d.line([tuple(p) for p in xy]+[tuple(xy[0])],fill='yellow',width=2)
                c=xy.mean(0)
            elif view=='oblique-north' and t.get('manual_crown_match'):
                c=np.array(t['manual_crown_match']['oblique_canvas_px']);d.ellipse((c[0]-7,c[1]-7,c[0]+7,c[1]+7),outline='yellow',width=3)
            elif t['id']=='T15':
                c=np.array([479,521]);xy=canvas_xy(t['outline_enu_m']);d.line([tuple(p) for p in xy]+[tuple(xy[0])],fill='cyan',width=2)
                if view=='nadir':d.line([(479,521),(459,444)],fill='cyan',width=3)
                else:d.line([(479,521),(485,479)],fill='cyan',width=3)
            else:continue
            c=np.clip(c,[5,5],[1240,1265])
            d.text(tuple(c),t['id'],font=font(23),fill='white',stroke_fill='black',stroke_width=3)
        annotated=Image.new('RGB',(1770,1390),'black');annotated.paste(im,(0,90));legend=ImageDraw.Draw(annotated)
        legend.text((15,12),'Winter tree crowns | '+view+' | N up | 0.10 m/px | all heights provisional',font=font(25),fill='white')
        legend.text((15,49),'Yellow: SAM crown / manual match. Cyan T15: sparse bare tree and shadow / oblique displacement.',font=font(21),fill='white')
        legend.text((1320,105),'Height +/- assumed error (m)',font=font(23),fill='white')
        for j,t in enumerate(trees):
            h=t['height_m'];err=t['height_error_m']
            label=f"{t['id']}: {h:.1f} +/- {err:.1f} m*" if h is not None else f"{t['id']}: unresolved"
            legend.text((1320,152+j*47),label,font=font(23),fill='cyan' if t['id']=='T15' else 'white')
        for j,line in enumerate(['* Assumed ranges, not confidence', 'intervals or surveyed heights.', '', 'T3/4/5/7/10/12: approximate', 'two-view crown centres.', 'Automatic photometric fits failed.', '', 'T15: shadow 5.66 m;', 'oblique branch check 4.93 m.', '', 'Bare winter branches can be missed.', 'Shadows often overlap roofs/trees.', 'Merged crowns may be multiple trees.']):
            legend.text((1320,900+j*29),line,font=font(21),fill='white')
        annotated.save(OUT/('trees.jpg' if view=='nadir' else 'trees-oblique.jpg'))


def obstruction_scene(buildings,trees,calibrations,cameras,lon,lat):
    scene=[]
    for b in buildings:
        obj=dict(b)
        obj['property_capture_time']=obj.pop('capture_time')
        obj['property_image_name']=obj.pop('image_name')
        obj['image_name']=cameras[0]['image-name'];obj['capture_time']=cameras[0]['capture-date']+'Z'
        obj['image_use']='Pinned nadir used to inspect outline; height comes from property collection, not that raw pixel measurement.'
        obj['sensitivity_error_m']=2.
        obj['include_in_horizon']=not obj['is_own_house']
        scene.append(obj)
    own=next(b for b in buildings if b['is_own_house']);cal=next(c for c in calibrations if c['id']==own['id'])
    eave={**own,'id':'EAVE','type':'own_roof_eave','height_m':cal['shadow_derived_casting_edge_height_m'],
        'height_error_m':cal['height_error_m'],'sensitivity_error_m':cal['height_error_m'],
        'method':'House footprint outline; eave-height proxy from its measured northern eave shadow, applied uniformly to the outline',
        'image_name':cameras[0]['image-name'],'capture_time':cameras[0]['capture-date']+'Z','include_in_horizon':True,
        'assumption':'One observed casting edge represents all eaves; local roof slopes, porch heights and footprint/eave registration remain uncertain.'}
    scene.append(eave)
    statuses={'T3':'Shadow falls on northern neighbour roof: no flat-ground height measured.',
        'T4':'Shadow overlaps neighbour roof/crowns: not measured.',
        'T5':'Merged crowns, concealed stem and overlapping shadows on roofs: not measured.',
        'T7':'Street/roof/crown shadows overlap: no apex-to-shadow correspondence established.',
        'T10':'Bare crown shadow overlaps house/other vegetation: not measured.',
        'T12':'Large shadow visible on road, but hidden stem and overlapping crowns prevent a unique top/endpoint pairing.'}
    for t in trees:
        obj=dict(t);obj['include_in_horizon']=obj['height_m'] is not None
        obj['sensitivity_error_m']=obj['height_error_m']
        if obj['id'] in statuses:obj['shadow_status']=statuses[obj['id']]
        scene.append(obj)
    write_json('heights.json',{'reference':{'lat':lat,'lon':lon,'ground_elevation_m':22.5552,'horizontal_coordinates':'local east/north metres; longitude-latitude outlines also supplied','vertical_datum':'Vexcel elevation datum unspecified; relative heights preferred'},
        'objects':scene,'notes':['Building source height is DSM-DTM at footprint centroid, not eave/ridge. No provider error bars; +/-2m is an assumed sensitivity.','Tree estimates are winter visible-crown proxies, not surveyed maxima. Eight SAM candidates remain unknown; sparse bare trees can be missing altogether.','Eaves are an outline/silhouette proxy, not a ray trace through the indoor ceiling or window frames.']})
    return scene


def horizon_for(scene,origin,shift=(0,0),camera_dz=0,height_sign=0,unknown_height=None):
    from matplotlib.path import Path as PolygonPath
    o=np.array(origin['enu_m'])+shift;zeye=22.5552+origin['height_above_ground_m']+camera_dz
    azimuth=np.arange(0,360,2);angles=np.zeros(180);ids=['none']*180
    unit=np.c_[np.sin(np.radians(azimuth)),np.cos(np.radians(azimuth))]
    for obj in scene:
        if not obj.get('include_in_horizon'):
            if obj['type']=='tree' and obj['height_m'] is None and unknown_height is not None:h=unknown_height
            else:continue
        else:h=obj['height_m']+height_sign*(obj.get('sensitivity_error_m') or 0)
        h=max(0,h);poly=np.array(obj['outline_enu_m']);top=(obj.get('ground_elevation_m') or 22.5552)+h
        if obj['type']=='tree' and 'crown_outline_ray_slope_enu_per_m_height' in obj:
            poly=np.array(obj['apparent_crown_enu_m'])+h*np.array(obj['crown_outline_ray_slope_enu_per_m_height'])
        if not np.array_equal(poly[0],poly[-1]):poly=np.r_[poly,poly[:1]]
        a=poly[:-1]-o;v=np.diff(poly,axis=0)
        den=unit[:,0,None]*v[None,:,1]-unit[:,1,None]*v[None,:,0]
        rnum=a[:,0]*v[:,1]-a[:,1]*v[:,0]
        tnum=a[None,:,0]*unit[:,1,None]-a[None,:,1]*unit[:,0,None]
        with np.errstate(divide='ignore',invalid='ignore'):
            r=rnum[None,:]/den;t=tnum/den
        ok=(np.abs(den)>1e-10)&(r>.01)&(t>=0)&(t<=1)
        distance=np.where(ok,r,np.inf).min(1)
        theta=np.degrees(np.arctan2(max(0,top-zeye),distance))
        if obj['type']=='tree' and PolygonPath(poly).contains_point(o) and top>zeye:
            theta[:]=90. # opaque crown-plane scenario overhead, flagged below
        better=theta>angles
        angles[better]=theta[better]
        for j in np.flatnonzero(better):ids[j]=obj['id']
    return angles,ids


def make_horizon(scene,origin):
    nominal,ids=horizon_for(scene,origin)
    scenarios=[nominal]
    for shift in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
        for sign in [-1,1]:
            scenarios.append(horizon_for(scene,origin,shift,camera_dz=-sign*.4,height_sign=sign)[0])
    low=np.min(scenarios,axis=0);high=np.max(scenarios,axis=0)
    unknown,_=horizon_for(scene,origin,unknown_height=30)
    result={'source':'aerial','origin':{k:origin[k] for k in ['lat','lon','height_above_ground_m']},
        'azimuth_deg':list(range(0,360,2)),'elevation_deg':np.round(nominal,3).tolist(),
        'dominant_object_id':ids,'low_sensitivity_deg':np.round(low,3).tolist(),'high_sensitivity_deg':np.round(high,3).tolist(),
        'unresolved_trees_30m_scenario_deg':np.round(unknown,3).tolist(),
        'unresolved_tree_ids':[s['id'] for s in scene if s['type']=='tree' and s['height_m'] is None],
        'notes':'PROVISIONAL scene-derived silhouette, clockwise true north; not a validated observed horizon. Six crown-centre correspondences are assumed, eight SAM crowns unresolved and omitted from nominal; the isolated bare-tree estimate is provisional. Sensitivity varies stated heights, origin +/-1m and eye height +/-0.4m; it is NOT a confidence interval or guaranteed bound. Unknown-tree 30m scenario is illustrative, not a physical upper bound. Winter bare branches are porous and incompletely detected. Own-house EAVE uses shadow-derived eave proxy; actual opaque ceiling/walls/window frame are not modeled. Flat ground at 22.5552m for trees; no terrain horizon or completeness beyond the 130m image crop. Do not claim agreement with Brief A without resolving these limitations.'}
    write_json('horizon_profile.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4));x=np.arange(0,360,2)
    ax.fill_between(x,low,high,color='.85',label='Assumed sensitivity envelope')
    ax.plot(x,nominal,color='black',label='Provisional nominal scene')
    ax.plot(x,unknown,color='.5',ls=':',label='Unresolved crowns at assumed 30m')
    ax.set(xlim=(0,358),ylim=(0,90),xlabel='Azimuth clockwise from true north (degrees)',ylabel='Obstruction elevation (degrees)',title='Aerial silhouette: provisional heights and incomplete tree coverage')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'horizon_profile.png',dpi=140);plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid', choices=['khbuN7q853q'])
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    row = next(r for r in csv.DictReader((ROOT/'models_index.csv').open()) if r['sid']==args.sid)
    lon, lat = float(row['lon']), float(row['lat'])
    base = {'wkt': f'POINT({lon} {lat})', 'srid':4326, 'layer':'urban'}
    if not args.probe:
        documentation()
        cameras,matrices,images=acquisition(lon,lat)
        buildings=load_buildings(lon,lat)
        inspection(images,buildings)
        calibrations=calibration(buildings,cameras[0],images['nadir'])
        detection=detect_trees(lon,lat,cameras[0])
        # Retain the second SAM experiment as evidence: several oblique masks
        # label shadows rather than crowns, so they are not used for heights.
        oblique_detection=detect_trees(lon,lat,cameras[1])
        trees=tree_heights(detection,matrices,cameras,lon,lat)
        trees=manual_tree_estimates(trees,matrices,cameras,lon,lat)
        write_json('tree-final-estimates.json',trees)
        annotate_trees(trees,images,cameras)
        origin=registered_origin(args.sid,buildings,lon,lat,images['nadir'])
        scene=obstruction_scene(buildings,trees,calibrations,cameras,lon,lat)
        profile=make_horizon(scene,origin)
        for camera,det in zip(cameras,[detection,oblique_detection]):
            camera['capture_time_utc']=camera['capture-date']+'Z'
            camera['sun']=camera['capture_time_hypotheses']['UTC']
            camera['line_of_sight_tilt_at_house_deg']=math.degrees(math.atan2(camera['horizontal_distance_to_house_m'],camera['camera-pos-z']-22.5552))
            camera['api_detection_capture_epoch_ms']=det['images'][0]['metadata']['capture-date']
            dm=det['images'][0]['metadata']
            camera['detection_pixel_grid']={k:dm[k] for k in ['image-name','pixel-bbox','width','height']}
            camera['detection_use']='Nadir masks verified on the same raw crop grid.' if camera is cameras[0] else 'Oblique SAM returned a different crop and resized raster; its masks are not used. Epoch/image identity only; manual parallax uses the calibrated raw JPEG.'
            camera['epoch_time_utc']=dt.datetime.fromtimestamp(camera['api_detection_capture_epoch_ms']/1000,dt.timezone.utc).isoformat()
            camera['raw_crop_sha256']=hashlib.sha256((OUT/('nadir.jpg' if camera is cameras[0] else 'oblique.jpg')).read_bytes()).hexdigest()
            view='nadir' if camera is cameras[0] else 'oblique-north'
            camera['projection_held_out_error_px']=json.loads((OUT/f'projection-{view}.json').read_text())['held_out_max_error_px']
            camera['pixel_to_ground']='Invert homography P[:,(0,1,3)]+P[:,2]*chosen_elevation using projection-*.json. Raw crop pixels before EXIF; add XMP pixel-bbox offset for full raw image. Ground render: east=(x+.5)*.1-65, north=65-(y+.5)*.1 at elevation22.5552m. Elevated objects retain relief displacement.'
        write_json('camera.json',{'images':cameras,'time_zone_conclusion':'UTC verified against north-northwest shadow directions and API detection Unix epoch. America/Chicago local-time hypothesis puts sun below horizon.',
            'observed_shadow_azimuth_deg':[r['measured_shadow_azimuth_deg'] for r in calibrations],
            'geometry_note':'The nadir sensor is not nadir at this house: house is off-axis near image edge. Full calibrated projection used; omega alone is not the local look tilt.',
            'sources':['https://api.vexcelgroup.com/v2/api-docs/oriented','https://api.vexcelgroup.com/v2/api-docs/property','https://api.vexcelgroup.com/v2/api-docs/custom-elements']})
        summary={'command':'python3 -I experiments/heights_from_aerial.py khbuN7q853q','status':'complete_artifacts_with_explicit_measurement_failures',
            'calibration_length_errors_m':{r['id']:r['length_error_m'] for r in calibrations},
            'tree_estimates':{t['id']:{'height_m':t['height_m'],'error_m':t['height_error_m'],'status':t['status']} for t in trees},
            'horizon_samples':len(profile['azimuth_deg']),'horizon_range_deg':[min(profile['elevation_deg']),max(profile['elevation_deg'])],
            'verified':['Exact raw-image names, XMP crop offsets and capture times; API camera centres and explicit-Z world/pixel transforms.','UTC supported by measured shadows and Unix-epoch metadata.','Height field means centroid DSM-DTM (cached property schema); northern eave shadows are shorter than centroid-height predictions.','Footprint registration computed from Matterport geo transform and wall graph.','Required artifacts generated for this real house.'],
            'assumed_or_failed':['Automatic foliage plane-sweep matches failed; six crown-centre correspondences are manual approximations, not exact homologous branches.','Eight SAM crowns have unresolved heights; sparse winter branches and some trees are missed.','Only T15 has an isolated usable ground shadow; its top/stem correspondence remains provisional. Other shadows are on roofs, hidden or overlap.','Roof-centroid height is not a valid eave-height calibration; eave proxy applied to entire own-house outline.','No surveyed vertical datum, tree truth or independent horizon validation; no comparison with Brief A made.','Nominal horizon omits unresolved crowns; sensitivity scenarios are not guaranteed bounds.'],
            'files_written':['experiments/heights_from_aerial.py','data/shadow_check/khbuN7q853q/ (required artifacts and cached evidence)'],
            'authored_by':'Codex (Brief B assigned worker); no delegated workers',
            'reproduction':'The command rebuilds required artifacts from pinned raw-image/API caches, fetching missing inputs at runtime. Manual digitisation coordinates and acceptance rules are explicit in the script. Extra exploratory crops are retained as evidence.',
            'undo':'Inspect snapshots with python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path /Users/phanan/HackPrinceton/experiments/heights_from_aerial.py; use rollback <id> for dry run, then rollback <id> --yes. Generated outputs are confined to data/shadow_check/khbuN7q853q/.'}
        write_json('report.json',summary)
        inventory=[Path(__file__).resolve()]+[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json']
        write_json('artifact-manifest.json',[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inventory])
        print('DONE:',len(buildings),'buildings;',sum(t['height_m'] is not None for t in trees),'provisional tree heights;',len(profile['azimuth_deg']),'horizon bins.',flush=True)
        return
    for view in ['nadir', 'oblique-north']:
        got = request('oriented/query', {**base, 'product-type': view,
            'include':'image-name,capture-date,collection,omega,phi,kappa,focal-length,pixel-size,k1,k2,camera-pos-x,camera-pos-y,camera-z,ground-z,raster-size-width,raster-size-height,pp0-x,pp0-y'},
            cache=f'query-{view}.json')
        print(view, len(got.get('features', [])), flush=True)
        if got.get('features'):
            print(json.dumps(got['features'][0])[:5000], flush=True)
    prop = request('property/extract', {**base,'buffer-meter':65,'latest':'true',
                   'include':'precomputed,calculated','unit-system':'metric'},cache='properties.json')
    print('parcels',len(prop.get('features',[])),flush=True)


if __name__ == '__main__':
    main()
