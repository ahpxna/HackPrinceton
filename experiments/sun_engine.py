"""Brief C geometry experiment. python3 -I experiments/sun_engine.py <sid|--batch>.

Online credentials come only from process environment. Cache responses exclude URLs and secrets.
The output is a conditional geometric estimate, not observed irradiance.
"""
from pathlib import Path
import base64
import ast
import collections
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'data/cache/matplotlib')

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

A=load_module('brief_a',ROOT/'experiments/sun_from_pano.py')

def load_geometry(path):
    tree=ast.parse(path.read_text(),filename=str(path))
    constants={'RES','GRID','MAX_SHIFT','ROT_RANGE','MODEL_Q'}
    functions={'enu','interior_points','raster','parse_ring','dominant_angle','analyse'}
    tree.body=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))
        or isinstance(node,ast.FunctionDef) and node.name in functions
        or isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id in constants for t in node.targets)]
    module=types.ModuleType('geo_geometry_only')
    exec(compile(tree,str(path),'exec'),module.__dict__)
    def cached_only(*args):raise RuntimeError('geometry_requires_injected_cached_inputs')
    module.gql=cached_only;module.vexcel_property=cached_only
    return module

G=load_geometry(ROOT/'experiments/geo_check.py')
from matplotlib.path import Path as PolygonPath

DATES=['2026-12-21','2026-03-20','2026-06-21','2026-11-14']
TARGET='khbuN7q853q'
API_CALLS=collections.Counter()
AUTH_FAILURES=0

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def sanitize(obj):
    if isinstance(obj,dict):
        return {k:sanitize(v) for k,v in obj.items() if not any(s in k.lower() for s in ('token','secret','api-key','authorization'))}
    if isinstance(obj,list): return [sanitize(v) for v in obj]
    if isinstance(obj,str) and ('http://' in obj or 'https://' in obj): return '[URL omitted]'
    return obj

def request_cached(path,service,query):
    global AUTH_FAILURES
    if path.exists():
        result=json.loads(path.read_text())
        if '_error' in result: raise RuntimeError(result['_error'])
        return result
    if os.environ.get('SUN_OFFLINE')=='1': raise RuntimeError('offline_cache_missing')
    if AUTH_FAILURES>=2: raise RuntimeError('authentication_stop')
    required=('MATTERPORT_API_KEY','MATTERPORT_API_TOKEN') if service=='matterport' else ('VEXCEL_API_KEY',)
    if any(not os.environ.get(key) for key in required):
        raise RuntimeError(service+'_process_environment_credentials_missing')
    env={key:os.environ[key] for key in required}
    if service=='matterport':
        auth=base64.b64encode((env['MATTERPORT_API_KEY']+':'+env['MATTERPORT_API_TOKEN']).encode()).decode()
        req=urllib.request.Request('https://api.matterport.com/api/models/graph',json.dumps({'query':query}).encode(),{'Content-Type':'application/json','Authorization':'Basic '+auth})
    else:
        req=urllib.request.Request('https://api.vexcelgroup.com/v2/property/extract?'+urllib.parse.urlencode(query),headers={'X-API-KEY':env['VEXCEL_API_KEY']})
    for attempt in range(6):
        try:
            API_CALLS[service]+=1
            with urllib.request.urlopen(req,timeout=100) as response: result=json.load(response)
            if result.get('errors'): raise RuntimeError(service+'_graphql_error')
            result=sanitize(result);write(path,result);return result
        except urllib.error.HTTPError as error:
            if error.code in (401,403): AUTH_FAILURES+=1
            if error.code==429 and attempt<5:
                time.sleep(min(32,2**(attempt+1)));continue
            reason=service+'_http_'+str(error.code);write(path,{'_error':reason});raise RuntimeError(reason) from None
        except (urllib.error.URLError,TimeoutError):
            reason=service+'_transport_error';write(path,{'_error':reason});raise RuntimeError(reason) from None
        except RuntimeError as error:
            write(path,{'_error':str(error)});raise

MODEL_Q='''{model(id:"%s"){
 floors{id label sequence dimensionEstimates{lowerElevation height units}
 edges(includeUsed:true){id type vertices{position{x y}} openings{type width height lowerElevation relativeCenter}}}
 rooms{id label keywords classifications{label confidence defaultKeywords} floor{id} ceiling{minHeight maxHeight} dimensionEstimates{lowerElevation height units}
 boundary{edges{type vertices{position{x y}}}} holes{edges{vertices{position{x y}}}}}
 geocoordinates{latitude longitude
 p0:geoLocationOf(modelLocation:{x:0,y:0,z:0}){lat long}
 px:geoLocationOf(modelLocation:{x:20,y:0,z:0}){lat long}
 py:geoLocationOf(modelLocation:{x:0,y:20,z:0}){lat long}}}}'''

def inputs(sid):
    cache=ROOT/'data/cache'/sid
    model=request_cached(cache/'matterport.json','matterport',MODEL_Q%sid)['data']['model']
    if not model: raise RuntimeError('matterport_model_missing')
    g=model['geocoordinates']
    prop=request_cached(cache/'vexcel_calculated_65m.json','vexcel',{'wkt':f"POINT({g['longitude']} {g['latitude']})",'srid':4326,'layer':'urban','buffer-meter':65,'latest':'true','include':'precomputed,calculated','unit-system':'metric'})
    return model,prop

def geo_matrix(model):
    g=model['geocoordinates'];lat,lon=float(g['p0']['lat']),float(g['p0']['long'])
    matrix=np.column_stack([G.enu(g[k]['lat'],g[k]['long'],lat,lon)/20 for k in ('px','py')])
    return matrix,lat,lon

def building_rings(prop,lat,lon):
    buildings=[];seen=set()
    for feature in prop.get('features',[]):
        props=feature.get('properties') or {}
        for s in props.get('structures') or []:
            wkt=s.get('footprint.wkt')
            if not wkt or wkt in seen: continue
            seen.add(wkt)
            try: ring=np.array([G.enu(la,lo,lat,lon) for lo,la in G.parse_ring(wkt)])
            except (AttributeError,ValueError): continue
            buildings.append({'id':f'B{len(buildings)}','ring':ring,'height':s.get('roof-characteristics.height'),
                'ground':s.get('ground-elevation.value'),'collection':props.get('metadata.collection'),
                'capture_date':props.get('metadata.ortho-date-estimate') or '', 'structure':s})
    if not buildings: raise RuntimeError('vexcel_no_footprints')
    return buildings

def registration(sid,model,prop):
    cache=ROOT/'data/cache'/sid/'registration.json'
    if cache.exists(): return json.loads(cache.read_text())
    old=ROOT/'data/sun_check'/sid/'heading_fit.json'
    if old.exists():
        fit=json.loads(old.read_text());fit['source']='accepted Brief A footprint fit'
    else:
        matrix,lat,lon=geo_matrix(model);candidates=building_rings(prop,lat,lon)
        seg=[tuple(tuple(v['position'][k] for k in ('x','y')) for v in e['vertices']) for f in model['floors'] for e in f['edges'] if len(e['vertices'])==2]
        inner=G.interior_points(seg)
        if inner is None:raise RuntimeError('registration_walls_do_not_close')
        points=inner[::25]@matrix.T
        scores=[float(PolygonPath(b['ring']).contains_points(points).mean()) for b in candidates]
        viable=[b for b,s in zip(candidates,scores) if s>=.2]
        if not viable:viable=candidates
        newest=max(b['capture_date'] for b in viable)
        viable=[b for b in viable if b['capture_date']==newest]
        selected=max(viable,key=lambda b:float(PolygonPath(b['ring']).contains_points(points).mean())-.001*np.linalg.norm(b['ring'].mean(0)-points.mean(0)))
        selected_prop={'features':[{'properties':{'structures':[selected['structure']]}}]}
        original_gql,original_vexcel=G.gql,G.vexcel_property
        G.gql=lambda query:{'data':{'model':model}}
        G.vexcel_property=lambda lat,lon:selected_prop
        try: fit=G.analyse(sid,{'address':'','city':''})
        finally: G.gql,G.vexcel_property=original_gql,original_vexcel
        if 'skip' in fit: raise RuntimeError('registration_'+fit['skip'].split(':')[0].replace(' ','_'))
        fit['source']='cached Matterport/Vexcel, read-only geo_check.analyse'
        fit['collection']=selected['collection'];fit['capture_date']=selected['capture_date']
    fit['poor']=fit['iou_after']<.55 or fit['contain_after']<.9
    write(cache,fit);return fit

def alignment(fit,matrix):
    th=np.deg2rad(fit['fix_rot_deg']);R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    c=np.array(fit['_plot']['centre']);d=np.array(fit['_plot']['d'])
    return R@matrix,c-c@R.T+d

def own_building(buildings,fit):
    ring=np.array(fit['_plot']['ring'])
    indices=[i for i,b in enumerate(buildings) if not fit.get('collection') or b['collection']==fit['collection']]
    return min(indices,key=lambda i:np.linalg.norm(buildings[i]['ring'].mean(0)-ring.mean(0)))

def model_ring(enu,matrix,shift): return (np.array(enu)-shift)@np.linalg.inv(matrix).T

def ray_directions(points,pos):
    d=np.asarray(points)-pos
    return d/np.linalg.norm(d,axis=-1,keepdims=True)

def nearest_angular(anchors,curve):
    dots=np.clip(anchors@curve.T,-1,1);best=dots.argmax(1)
    return np.rad2deg(np.arccos(dots[np.arange(len(best)),best])),best

def roof_curve(ring,height,pos):
    p=np.concatenate([np.linspace(a,b,max(20,int(np.linalg.norm(b-a)/.015))) for a,b in zip(ring,np.roll(ring,-1,axis=0))])
    return ray_directions(np.column_stack([p,np.full(len(p),height)]),pos),p

def roof_edge_check(model,prop,fit,anchors=None):
    matrix,lat,lon=geo_matrix(model);corrected,shift=alignment(fit,matrix)
    buildings=building_rings(prop,lat,lon);own=buildings[own_building(buildings,fit)]
    raw=model_ring(own['ring'],matrix,np.zeros(2));fixed=model_ring(own['ring'],corrected,shift)
    out=ROOT/'data/sun_check'/TARGET
    if anchors is None:
        return raw,fixed,own
    scans=[]
    for folder,meta_path,prefix in [(out,ROOT/'data/matterport'/TARGET/'pano_meta.json','pano_td86n1_face'),(out/f'second_{A.SECOND_PANO}',out/f'second_{A.SECOND_PANO}'/'pano_meta.json','face')]:
        meta=json.loads(meta_path.read_text());meta=meta.get('pano',meta)
        pos=np.array([meta['position'][k] for k in ('x','y','z')]);q=A.rotation(meta['rotation'])
        source=ROOT/'data/matterport'/TARGET if prefix.startswith('pano') else folder
        faces=[Image.open(source/f'{prefix}{i}.jpg').convert('RGB') for i in range(6)]
        A.check_cap_seams(faces)
        aa=[x for x in anchors if x['pano_id']==meta.get('id',A.PRIMARY_PANO)]
        rays=[]
        for anchor in aa:
            u,v=2*np.array(anchor['pixel_xy'])/(faces[anchor['face']].width-1)-1
            d=A.BASES[anchor['face'],0]+u*A.BASES[anchor['face'],1]+v*A.BASES[anchor['face'],2]
            d=q@d;rays.append(d/np.linalg.norm(d))
        scans.append((meta,faces,pos,q,aa,np.array(rays)))
    heights=np.arange(1.5,5.01,.01);scores=[];raw_scores=[]
    for height in heights:
        residual=[]
        for _,_,pos,_,aa,rays in scans:
            if len(aa): residual.extend(nearest_angular(rays,roof_curve(fixed,height,pos)[0])[0])
        scores.append(float(np.sqrt(np.mean(np.square(residual)))))
        rr=[]
        for _,_,pos,_,aa,rays in scans:
            if len(aa): rr.extend(nearest_angular(rays,roof_curve(raw,height,pos)[0])[0])
        raw_scores.append(float(np.sqrt(np.mean(np.square(rr)))))
    height=float(heights[np.argmin(scores)]);raw_height=float(heights[np.argmin(raw_scores)]);records=[];tiles=[]
    for meta,faces,pos,q,aa,rays in scans:
        record={'pano_id':meta.get('id',A.PRIMARY_PANO),'anchors':aa}
        for name,ring in [('delivered',raw),('corrected',fixed)]:
            if not aa:
                record[name]={'angular_residual_deg':[],'status':'No roof silhouette visible through closed door/blinds; residual unmeasurable'};continue
            curve,points=roof_curve(ring,height,pos);residual,indices=nearest_angular(rays,curve)
            horizontal=np.linalg.norm(points[indices]-pos[:2],axis=1)
            record[name]={'angular_residual_deg':residual.tolist(),'rms_angular_deg':float(np.sqrt(np.mean(residual**2))),
                'horizontal_edge_distance_m':horizontal.tolist(),'horizontal_equivalent_error_m':(np.tan(np.deg2rad(residual))*horizontal).tolist()}
        if aa:
            curve,points=roof_curve(raw,raw_height,pos);residual,indices=nearest_angular(rays,curve)
            horizontal=np.linalg.norm(points[indices]-pos[:2],axis=1)
            record['delivered_independent_height']={'height_m':raw_height,'angular_residual_deg':residual.tolist(),
                'horizontal_edge_distance_m':horizontal.tolist(),'horizontal_equivalent_error_m':(np.tan(np.deg2rad(residual))*horizontal).tolist()}
        for face in sorted(set(x['face'] for x in aa)) or [1]:
            img=faces[face].copy();draw=ImageDraw.Draw(img)
            for ring,color in [(raw,'red'),(fixed,'lime')]:
                curve,_=roof_curve(ring,height,pos);ff,uv=A.cube_coordinates(curve,q)
                for k in range(len(curve)-1):
                    if ff[k]==face and ff[k+1]==face and np.linalg.norm(uv[k+1]-uv[k])<.03:
                        draw.line([tuple(uv[k]*img.width),tuple(uv[k+1]*img.width)],fill=color,width=6)
            for anchor in aa:
                if anchor['face']==face:
                    x,y=anchor['pixel_xy'];draw.ellipse((x-8,y-8,x+8,y+8),fill='cyan')
            draw.rectangle((0,0,img.width,100),fill='black')
            draw.text((15,10),f"{record['pano_id']} face{face}; fitted eave z={height:.2f}m",font=ImageFont.load_default(size=30),fill='white')
            draw.text((15,52),'Red delivered | green corrected | cyan hand-marked roof edge',font=ImageFont.load_default(size=28),fill='white')
            tiles.append(img.resize((1024,1024)))
        records.append(record)
    image=Image.new('RGB',(1024*len(tiles),1024),'white')
    for k,tile in enumerate(tiles):image.paste(tile,(k*1024,0))
    image.save(out/'roof_edge_overlay.jpg',quality=95)
    all_raw=[x for r in records for x in r['delivered']['angular_residual_deg']]
    all_fixed=[x for r in records for x in r['corrected']['angular_residual_deg']]
    floor_z=float(np.median([r['floor_z_m'] for r in room_data(TARGET,model)[0]]))
    fixed_height=[]
    for floor_above_ground in (.1,.3):
        fixed_z=floor_z-floor_above_ground+2.7
        row={'eave_height_above_ground_m':2.7,'floor_above_ground_m':floor_above_ground,
            'reference_floor_model_z_m':floor_z,'eave_model_z_m':fixed_z}
        for name,ring in [('delivered',raw),('corrected',fixed)]:
            angles=[];equiv=[]
            for _,_,pos,_,aa,rays in scans:
                if not aa:continue
                curve,points=roof_curve(ring,fixed_z,pos);residual,indices=nearest_angular(rays,curve)
                angles.extend(residual);equiv.extend(np.tan(np.deg2rad(residual))*np.linalg.norm(points[indices]-pos[:2],axis=1))
            row[name]={'rms_angular_deg':float(np.sqrt(np.mean(np.square(angles)))),
                'rms_horizontal_equivalent_m':float(np.sqrt(np.mean(np.square(equiv)))),'per_anchor_angular_deg':angles}
        fixed_height.append(row)
    report={'eave_height_model_z_m':height,'free_parameters':['one eave height per alignment; horizontal alignment held fixed'],
        'height_datum':{'reference':'Matterport model Z, not camera-relative or Vexcel elevation datum','reference_floor_median_z_m':floor_z,
            'corrected_height_above_median_scan_floor_m':height-floor_z,'delivered_independent_height_above_median_scan_floor_m':raw_height-floor_z,
            'visible_edge':'Hand-marked low porch fascia/beam silhouette, not independently identified as the main aerial eave'},
        'height_search_m':[1.5,5,.01],'height_at_search_boundary':height<1.51 or height>4.99,
        'delivered_rms_deg':float(np.sqrt(np.mean(np.square(all_raw)))),'corrected_rms_deg':float(np.sqrt(np.mean(np.square(all_fixed)))),
        'scan_records':records,'height_scores':{'height_m':heights.tolist(),'rms_deg':scores},
        'independent_height_fit_delivered':{'height_m':raw_height,'rms_deg':float(min(raw_scores))},
        'comparison_basis':'Each alignment gets its own one free eave height. Overlay uses the corrected best height for both; this common-height overlay alone is not independent evidence.',
        'fixed_independent_eave_source':{'eave_above_ground_m':2.7,'source':'Claude reviewer bridge reply d50b701c-da96-499c-aab0-6dc96adc4298,2026-11-14T06:52Z: neighbouring eave shadows3.77–3.90m at sun elevation35.42deg',
            'status':'Independent aerial estimate supplied by reviewer; lengths not remeasured by this worker; transferred to target main eave as a sensitivity assumption'},
        'fixed_independent_eave_checks':fixed_height,
        'comparison':'better' if min(scores)<min(raw_scores)-.1 else ('worse' if min(scores)>min(raw_scores)+.1 else 'same'),
        'limitations':['Eight hand-marked fascia silhouette points on the primary pano; correspondence to aerial footprint may be wrong.','One common height cannot describe sloped roofs or stepped porch eaves.','Second pano closed door and blinds hide the edge, so projection is shown but no residual can be certified.','Horizontal equivalent is tan(angular residual) times horizontal edge distance; height and correspondence errors also contribute.']}
    write(out/'roof_edge_fit.json',report);return report

def segments(region):
    result=[]
    for edge in (region or {}).get('edges') or []:
        if len(edge.get('vertices') or [])==2:
            result.append([[v['position'][k] for k in ('x','y')] for v in edge['vertices']])
    return np.asarray(result,dtype=float).reshape(-1,2,2)

def inside_segments(points,segs):
    points=np.asarray(points);flat=points.reshape(-1,2);inside=np.zeros(len(flat),bool)
    for a,b in segs:
        if abs(b[1]-a[1])<1e-12:continue
        crossing=((a[1]>flat[:,1])!=(b[1]>flat[:,1]))
        crossx=(b[0]-a[0])*(flat[:,1]-a[1])/(b[1]-a[1])+a[0]
        inside^=crossing&(flat[:,0]<crossx)
    return inside.reshape(points.shape[:-1])

def unit_scale(dim):return .3048 if (dim or {}).get('units')=='imperial' else 1.0

def room_data(sid,model):
    extra=ROOT/'data/cache'/sid/'room_labels.json'
    enrich={r['id']:r for r in json.loads(extra.read_text()).get('model',{}).get('rooms',[])} if extra.exists() else {}
    rooms=[];excluded=[]
    for source in model.get('rooms') or []:
        r={**source,**enrich.get(source['id'],{})};classes=r.get('classifications') or []
        labels=[c['label'] for c in classes if c.get('confidence',0)>=.5 and c['label']!='Void']
        if any(c['label']=='Void' for c in classes) or 'nonArea' in (r.get('keywords') or []):
            excluded.append({'id':r['id'],'reason':'Void/nonArea'});continue
        seg=segments(r.get('boundary'))
        if len(seg)<3:
            excluded.append({'id':r['id'],'reason':'missing boundary'});continue
        lo=seg.min((0,1));hi=seg.max((0,1));x,y=np.meshgrid(np.arange(lo[0]+.05,hi[0],.1),np.arange(lo[1]+.05,hi[1],.1))
        grid=np.column_stack([x.ravel(),y.ravel()]);mask=inside_segments(grid,seg)
        holes=[segments(h) for h in r.get('holes') or []]
        for hole in holes:mask&=~inside_segments(grid,hole)
        grid=grid[mask]
        if not len(grid):excluded.append({'id':r['id'],'reason':'empty raster boundary'});continue
        dim=r.get('dimensionEstimates') or {};scale=unit_scale(dim)
        floor_id=(r.get('floor') or {}).get('id')
        if not floor_id:
            excluded.append({'id':r['id'],'reason':'missing floor assignment'});continue
        if dim.get('lowerElevation') is not None:
            z=float(dim['lowerElevation'])*scale;floor_source='Matterport room dimensionEstimates lowerElevation converted from '+dim.get('units','metric')
        else:
            elevations=[float(s['dimensionEstimates']['lowerElevation'])*unit_scale(s['dimensionEstimates']) for s in model['rooms']
                if (s.get('floor') or {}).get('id')==floor_id and (s.get('dimensionEstimates') or {}).get('lowerElevation') is not None]
            z=float(np.median(elevations)) if elevations else 0.
            floor_source='Assumed median of other rooms on same floor' if elevations else 'Assumed model Z=0; no room floor elevations supplied'
        ceil=(r.get('ceiling') or {}).get('minHeight')
        ceiling=z+(float(ceil)*scale if ceil else float(dim.get('height') or (2.5/scale))*scale)
        rooms.append({'id':r['id'],'label':r.get('label') or ' / '.join(labels) or 'Unlabelled room',
            'classifications':classes,'floor_id':floor_id,'floor_z_m':z,'ceiling_z_m':ceiling,
            'floor_height_source':floor_source,
            'ceiling_height_source':'Matterport minHeight above room floor, same unit system as room dimensions' if ceil else ('Matterport dimension height above room floor' if dim.get('height') else 'Assumed2.5m above room floor'),
            'segments':seg,'holes':holes,'grid':grid,'area_m2':round(len(grid)*.01,2)})
    if not rooms:raise RuntimeError('no_usable_room_boundaries')
    for room in rooms:
        median=float(np.median([r['floor_z_m'] for r in rooms if r['floor_id']==room['floor_id']]))
        room['floor_z_minus_same_floor_median_m']=round(room['floor_z_m']-median,4)
        room['floor_elevation_uncertain']=abs(room['floor_z_m']-median)>.3 or room['floor_height_source'].startswith('Assumed')
    return rooms,excluded

def room_contains(room,points):
    mask=inside_segments(points,room['segments'])
    for h in room['holes']:mask&=~inside_segments(points,h)
    return mask

def exterior_openings(model,rooms,matrix):
    windows=[];excluded=[]
    interiors={floor['id']:(G.interior_points(segments(floor)) if len(segments(floor)) else None) for floor in model['floors']}
    for item in A.openings(model):
        if item['type'] not in ('window','doorway'):continue
        if not item.get('width') or not item.get('height'):continue
        tangent=np.array(item['tangent']);n=np.array([-tangent[1],tangent[0]]);center=np.array(item['center'])
        floor_id=model['floors'][item['floor_index']]['id']
        sides=[[r for r in rooms if r['floor_id']==floor_id and room_contains(r,(center+sign*.15*n)[None])[0]] for sign in (1,-1)]
        if bool(sides[0])==bool(sides[1]):
            excluded.append({'id':item['id'],'type':item['type'],'reason':'both sides have rooms (interior)' if sides[0] else 'neither side assigned to room'});continue
        room=sides[0][0] if sides[0] else sides[1][0]
        outward=-n if sides[0] else n
        inner=interiors[floor_id]
        if inner is not None and any(np.linalg.norm(inner-(center+step*outward),axis=1).min()<.15 for step in (.4,.6,.8)):
            excluded.append({'id':item['id'],'type':item['type'],'reason':'opposite side enclosed by wall-graph flood fill (closet/omitted room)'});continue
        en=matrix@outward;az=float(np.rad2deg(np.arctan2(en[0],en[1]))%360)
        item.update(room_id=room['id'],outward=outward,outward_azimuth_deg=round(az,2),
            glazing='Matterport window: assumed fully transparent' if item['type']=='window' else 'Exterior doorway: glazing unknown, conditional full-aperture transmission',
            lower_z_m=float(item['lowerElevation']))
        windows.append(item)
    return windows,excluded

def obstructions(sid,model,prop,fit,rooms,matrix,shift):
    _,lat,lon=geo_matrix(model);buildings=building_rings(prop,lat,lon);oi=own_building(buildings,fit);own=buildings[oi]
    base=min(r['floor_z_m'] for r in rooms);ground_z=base-.45
    roof_fit=ROOT/'data/sun_check'/sid/'roof_edge_fit.json'
    roof=model_ring(own['ring'],matrix,shift)
    eave=max(r['ceiling_z_m'] for r in rooms)+.25;source='highest room ceiling + assumed0.25m'
    roofs=[{'id':'main_roof','ring':roof,'height_model_z_m':eave,'source':source}]
    if sid==TARGET and roof_fit.exists():
        porch_eave=json.loads(roof_fit.read_text())['eave_height_model_z_m']
        back=next(o for o in A.openings(model) if o['id']=='F0E16O1')
        tangent=np.array(back['tangent']);normal=np.array([-tangent[1],tangent[0]])
        center=np.array(back['center'])
        room=next(r for r in rooms if room_contains(r,(center+.15*normal)[None])[0] or room_contains(r,(center-.15*normal)[None])[0])
        if room_contains(room,(center+.15*normal)[None])[0]:normal=-normal
        roofs.append({'id':'low_back_porch','ring':roof,'height_model_z_m':porch_eave,
            'source':'Hand-marked roof-edge fitted height; assumed constant across footprint outside back-door wall plane',
            'outside_wall_center_model_xy_m':center,'outside_wall_normal_model_xy':normal})
    obstacles=[];missing=[]
    for i,b in enumerate(buildings):
        if i==oi:continue
        if b['collection']!=own['collection']:continue
        if np.linalg.norm(b['ring'].mean(0)-own['ring'].mean(0))>75:continue
        if PolygonPath(b['ring']).contains_point(own['ring'].mean(0)) or PolygonPath(own['ring']).contains_point(b['ring'].mean(0)):continue
        height=b['height']
        assumed=height is None or float(height)<=0 or float(height)>80
        if assumed:height=10.;missing.append(b['id'])
        relative_ground=float(b['ground'] or own['ground'] or 0)-float(own['ground'] or b['ground'] or 0)
        bottom=ground_z+relative_ground
        obstacles.append({'id':b['id'],'kind':'neighbour','ring':model_ring(b['ring'],matrix,shift),'bottom':bottom,'top':bottom+float(height),'height_m':float(height),'height_assumed':assumed})
    tree_cache=ROOT/'data/cache'/sid/'trees_from_brief_b.json';tree_source=ROOT/'data/shadow_check'/sid/'heights.json'
    if not tree_cache.exists() and tree_source.exists():write(tree_cache,json.loads(tree_source.read_text()))
    tree_status={'source_present':tree_cache.exists(),'included':[],'excluded':[],'warning':'Unmodelled vegetation may further reduce predicted sun; geometric zero is only conditional on this obstruction model.'}
    if tree_cache.exists():
        tree_data=json.loads(tree_cache.read_text())
        for t in tree_data.get('objects',[]):
            if t.get('type')!='tree':continue
            if t.get('height_m') is None or not t.get('include_in_horizon',False):
                tree_status['excluded'].append({'id':t['id'],'reason':t.get('status','unresolved height')});continue
            ll=t.get('outline_lonlat') or []
            if not ll:tree_status['excluded'].append({'id':t['id'],'reason':'missing georeferenced crown'});continue
            ring=np.array([G.enu(la,lo,lat,lon) for lo,la in ll]);height=float(t['height_m'])
            relative_ground=float(t.get('ground_elevation_m') or own['ground'] or 0)-float(own['ground'] or t.get('ground_elevation_m') or 0)
            bottom=ground_z+relative_ground
            obstacles.append({'id':t['id'],'kind':'tree','ring':model_ring(ring,matrix,shift),'bottom':bottom,'top':bottom+height,'height_m':height,'height_assumed':True})
            tree_status['included'].append({'id':t['id'],'height_m':height,'status':t.get('status')})
    return roofs,obstacles,{'eave_height_model_z_m':eave,'eave_source':source,
        'low_back_porch_eave_model_z_m':roofs[-1]['height_model_z_m'] if len(roofs)>1 else None,
        'vexcel_own_roof_height_m':own['height'],'vexcel_collection':own['collection'],'vexcel_capture_date':own['capture_date'],
        'collection_selection':'One collection matching selected own roof; historical overlapping structures excluded despite API latest=true',
        'vexcel_height_interpretation':'Height above local ground used only for neighbouring solid prisms; own eave fitted/assumed separately',
        'lowest_floor_above_ground_assumed_m':.45,'model_ground_z_m':ground_z,'missing_neighbour_heights_assumed_10m':missing,'trees':tree_status}

def cross2(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def prism_hit(origins,directions,obstacle):
    ring=obstacle['ring'];hit=inside_segments(origins[...,:2],np.array(list(zip(ring,np.roll(ring,-1,axis=0)))))&(origins[...,2]>=obstacle['bottom'])&(origins[...,2]<=obstacle['top'])
    for a,b in zip(ring,np.roll(ring,-1,axis=0)):
        edge=b-a;den=cross2(directions[...,:2],edge);safe=np.abs(den)>1e-10
        den=np.where(safe,den,1);delta=a-origins[...,:2]
        ray_t=cross2(delta,edge)/den;u=cross2(delta,directions[...,:2])/den
        z=origins[...,2]+ray_t*directions[...,2]
        hit|=safe&(ray_t>1e-4)&(u>=0)&(u<=1)&(z>=obstacle['bottom'])&(z<=obstacle['top'])
    return hit

def aperture_verdict(window,directions,roofs,obstacles):
    tangent=np.array(window['tangent']);c=np.array(window['center']);w=window['width'];h=window['height'];z=window['lower_z_m']
    samples=np.array([[*(c+tangent*w*((i+.5)/3-.5)),z+h*(j+.5)/3] for j in range(3) for i in range(3)])
    origins=np.broadcast_to(samples,(len(directions),9,3));d=np.broadcast_to(directions[:,None,:],origins.shape)
    codes=np.zeros((len(directions),9),np.uint8)
    outside=(directions[:,:2]@window['outward'])>1e-7
    codes[~outside]=1;codes[directions[:,2]<=0]=5
    for roof in roofs:
        ray_t=(roof['height_model_z_m']-origins[...,2])/np.maximum(d[...,2],1e-12)
        xy=origins[...,:2]+ray_t[...,None]*d[...,:2]
        own=(ray_t>1e-4)&PolygonPath(roof['ring']).contains_points(xy.reshape(-1,2)).reshape(codes.shape)
        if 'outside_wall_normal_model_xy' in roof:
            own&=((xy-roof['outside_wall_center_model_xy_m'])@roof['outside_wall_normal_model_xy'])>=0
        codes[(codes==0)&own]=2
    for obstacle in obstacles:
        blocked=prism_hit(origins,d,obstacle)
        codes[(codes==0)&blocked]=3 if obstacle['kind']=='neighbour' else 4
    return codes

def floor_patch(window,room,directions,codes):
    points=room['grid'];c=np.array(window['center']);n=window['outward'];tangent=np.array(window['tangent'])
    den=directions[:,:2]@n;safe=np.abs(den)>1e-8
    travel=((c-points)@n)[None,:]/np.where(safe,den,1)[:,None]
    xx=(points-c)@tangent+travel*(directions[:,:2]@tangent)[:,None]
    zz=room['floor_z_m']+travel*directions[:,2,None]
    u=(xx/window['width']+.5)*3;v=(zz-window['lower_z_m'])/window['height']*3
    valid=(travel>0)&safe[:,None]&(u>=0)&(u<3)&(v>=0)&(v<3)&(directions[:,2,None]>0)
    indices=np.clip(v,0,2.999).astype(int)*3+np.clip(u,0,2.999).astype(int)
    return valid&(np.take_along_axis(codes,indices,axis=1)==0)

BLOCK_NAMES=['clear','sun_inside_wall','own_roof','neighbour','tree','below_horizon']

def sample_day(date,lat,lon,matrix):
    rise,set_=A.daylight(date,lat,lon);total=(set_-rise).total_seconds()/60
    offsets=np.arange(0,total,5);times=[rise+dt.timedelta(minutes=float(v)) for v in offsets]
    weights=np.minimum(5,total-offsets);az,el,app=A.solar(times,lat,lon)
    en=np.column_stack([np.sin(np.deg2rad(az))*np.cos(np.deg2rad(app)),np.cos(np.deg2rad(az))*np.cos(np.deg2rad(app))])
    xy=en@np.linalg.inv(matrix).T;directions=np.column_stack([xy,np.sin(np.deg2rad(app))]);directions/=np.linalg.norm(directions,axis=1)[:,None]
    return times,weights,directions,az,app,rise,set_

def compass(az):return ['N','NE','E','SE','S','SW','W','NW'][int((az+22.5)//45)%8]

def draw_floor(sid,rooms,windows,patches,matrix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(2,3,figsize=(18,12),layout='constrained')
    colors={'N':'#1764aa','NE':'#188b93','E':'#338b31','SE':'#9c931d','S':'#d88414','SW':'#c94527','W':'#a52d87','NW':'#5548a3'}
    floors=list(dict.fromkeys(r['floor_id'] for r in rooms));allxy=np.concatenate([r['grid'] for r in rooms]);span=allxy.max(0)-allxy.min(0);offsets={f:np.array([i*(span[0]+4),0]) for i,f in enumerate(floors)}
    for row,date in enumerate(('2026-12-21','2026-06-21')):
        for col,hour in enumerate((9,12,15)):
            ax=axes[row,col]
            for room in rooms:
                off=offsets[room['floor_id']];ax.add_collection(LineCollection(room['segments']+off,colors='#444444',linewidths=.7))
                grid=room['grid'];mask=patches.get((date,hour,room['id']),np.zeros(len(grid),bool))
                if mask.any():ax.scatter(*(grid[mask]+off).T,s=2,color='#f7bd29',marker='s',rasterized=True)
                center=np.mean(grid,0)+off
                label=room['label'].replace(' / ','/').replace('Living Room','Living').replace('Bedroom','Bed')
                ax.text(*center,label,fontsize=5,ha='center',va='center',bbox={'facecolor':'white','alpha':.7,'edgecolor':'none','pad':.3})
            for w in windows:
                room=next(r for r in rooms if r['id']==w['room_id']);off=offsets[room['floor_id']];c=np.array(w['center'])+off;t=np.array(w['tangent'])*w['width']/2
                ax.plot([c[0]-t[0],c[0]+t[0]],[c[1]-t[1],c[1]+t[1]],color=colors[compass(w['outward_azimuth_deg'])],lw=2)
            ax.autoscale();ax.set_aspect('equal');ax.set_title(f'{date} {hour:02d}:00 {A.TZ.key}',fontsize=10);ax.tick_params(labelsize=6)
            north=np.linalg.inv(matrix)@np.array([0.,1.]);north=north/np.linalg.norm(north)*2
            start=allxy.min(0)-np.array([2.,0]);ax.annotate('N',xy=start+north,xytext=start,arrowprops={'arrowstyle':'->'},fontsize=9)
    fig.suptitle(f'{sid}: conditional geometric floor sun (yellow), 0.10m grid\nWindows by outward compass: N blue, E green, S orange, W magenta; separate floors laid side by side',fontsize=12)
    fig.savefig(ROOT/'data/sun'/sid/'floor_sun.jpg',dpi=130);plt.close(fig)

def run_home(sid, skip_floor_images=False):
    started=time.monotonic();before=dict(API_CALLS);model,prop=inputs(sid);fit=registration(sid,model,prop)
    raw_matrix,lat,lon=geo_matrix(model);matrix,shift=alignment(fit,raw_matrix)
    A.TZ=ZoneInfo('America/Los_Angeles' if lon<-114 else ('America/Denver' if lon<-104.5 else 'America/Chicago'))
    rooms,excluded_rooms=room_data(sid,model);windows,excluded_openings=exterior_openings(model,rooms,matrix)
    if not windows:raise RuntimeError('no_exterior_openings_assigned')
    roofs,obstacles,assumptions=obstructions(sid,model,prop,fit,rooms,matrix,shift)
    assumptions.update({'timezone':A.TZ.key,'timezone_source':'Longitude cutoff for this Houston/Los Angeles cohort; Chicago >=-104.5, Denver >=-114, Los Angeles farther west; not a general timezone boundary service',
        'time_step_minutes':5,'sample_time_definition':'Left endpoint of each5min daylight bin; last bin has fractional minutes; first/last lit times are sample times, not exact transition times',
        'aperture_grid':'3x3 equal-area cells; one centre ray determines each entire cell',
        'floor_grid_m':.1,'floor_area_threshold_m2':.1,'window_glazing':'All windows clear, no blinds, frames or transmission loss',
        'exterior_doorways':'All exterior doorways included as conditional full transparent apertures; glazed portions not supplied by graph',
        'opening_z':'Matterport opening lowerElevation used directly as model Z in metres',
        'roof':'Main horizontal opaque plane at highest ceiling+0.25m over full own footprint; target adds lower back-porch plane outside back-door wall. No slopes, dormers, roof thickness or returning facade walls',
        'neighbours':'Opaque ground-to-roof-height prisms; uncertain height semantics and ground datum;65m query buffer',
        'trees_geometry':'Accepted Brief B crown footprints extruded ground-to-top as opaque prisms; no seasonal porosity; unresolved trees omitted',
        'room_floor':'Horizontal floors; boundary even-odd clipping; room-local lit patches unioned on0.1m grid; no furniture/interior occlusion beyond room clip',
        'floor_elevation_outlier_threshold_m':.3,
        'door_glazing_sensitivity':'Main totals assume all exterior doorways fully transparent. windows_only_* totals omit every doorway; neither scenario determines actual closed-door glazing.',
        'weather':'Clear sun assumed; no clouds, diffuse light or irradiance model',
        'orientation_classes':'Morning before11:00, midday11:00–14:00, afternoon at/after14:00; >=5min qualifying floor area',
        'missing_primary_bedroom':'No primary-bedroom role inferred from generic Bedroom labels; index null unless explicitly labelled',
        'blocker_precedence':'own roof, then neighbours, then trees; records first category in that order, not necessarily nearest physical object',
        'zero_interpretation':'0 minutes means geometric model predicts no >=0.1m2 direct floor patch under these assumptions; vegetation omissions can overestimate positive minutes and do not certify observed darkness.'})
    result={'sid':sid,'registration':{k:v for k,v in fit.items() if k!='_plot'},'assumptions':assumptions,'excluded_rooms':excluded_rooms,'excluded_openings':excluded_openings,
        'windows':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in w.items() if k not in ('outward','corners')}|{'dates':{}} for w in windows],
        'rooms':[{k:v for k,v in r.items() if k not in ('segments','holes','grid')}|{'dates':{}} for r in rooms],
        'own_roof_layers':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in roof.items()} for roof in roofs],
        'obstructions':[dict(o,ring_model_xy_m=o['ring'].tolist()) for o in obstacles]}
    for o in result['obstructions']:o.pop('ring')
    for room in result['rooms']:
        room['floor_height_above_model_ground_m']=room['floor_z_m']-assumptions['model_ground_z_m']
    patches={};sample_records={}
    for date in DATES:
        times,weights,directions,az,el,rise,set_=sample_day(date,lat,lon,matrix)
        lit={r['id']:np.zeros((len(times),len(r['grid'])),bool) for r in rooms}
        windows_only={key:np.zeros_like(value) for key,value in lit.items()}
        verdicts={}
        for w,wr in zip(windows,result['windows']):
            room=next(r for r in rooms if r['id']==w['room_id']);codes=aperture_verdict(w,directions,roofs,obstacles);verdicts[w['id']]=codes
            patch=floor_patch(w,room,directions,codes);lit[room['id']]|=patch
            if w['type']=='window':windows_only[room['id']]|=patch
            fractions=np.mean(codes==0,axis=1)
            wr['dates'][date]={'unobstructed_any_aperture_minutes':round(float(weights[fractions>0].sum()),2),
                'unobstructed_full_aperture_minutes':round(float(weights[fractions==1].sum()),2),
                'unobstructed_area_weighted_minutes':round(float(weights@fractions),2),
                'blocker_area_weighted_minutes':{name:round(float(weights@np.mean(codes==k,axis=1)),2) for k,name in enumerate(BLOCK_NAMES) if k},
                'minute_definition':'any=at least1/9 aperture sample clear; full=9/9 clear; area-weighted integrates clear fraction'}
        local=[t.astimezone(A.TZ) for t in times]
        for r,rr in zip(rooms,result['rooms']):
            areas=lit[r['id']].sum(1)*.01;sel=areas>=.1;indices=np.flatnonzero(sel)
            classes={name:round(float(weights[sel&np.array([test(t.hour+t.minute/60) for t in local])].sum()),2) for name,test in [('morning',lambda h:h<11),('midday',lambda h:11<=h<14),('afternoon',lambda h:h>=14)]}
            rr['dates'][date]={'lit_minutes_ge_0_1m2':round(float(weights[sel].sum()),2),'peak_lit_area_m2':round(float(areas.max()),2),
                'windows_only_lit_minutes_ge_0_1m2':round(float(weights[windows_only[r['id']].sum(1)*.01>=.1].sum()),2),
                'windows_only_peak_lit_area_m2':round(float((windows_only[r['id']].sum(1)*.01).max()),2),
                'first_lit_time':local[indices[0]].isoformat() if len(indices) else None,'last_lit_time':local[indices[-1]].isoformat() if len(indices) else None,
                'class':[k for k,v in classes.items() if v>=5] or ['no qualifying geometric sun'],'class_minutes':classes}
        sample_records[date]={'sunrise':rise.astimezone(A.TZ).isoformat(),'sunset':set_.astimezone(A.TZ).isoformat(),
            'local_times':[t.isoformat() for t in local],'weights_minutes':weights.tolist(),'sun_azimuth_deg':az.tolist(),'sun_apparent_elevation_deg':el.tolist(),
            'room_floor_area_m2':{key:(value.sum(1)*.01).tolist() for key,value in lit.items()},
            'room_windows_only_floor_area_m2':{key:(value.sum(1)*.01).tolist() for key,value in windows_only.items()},
            'window_sample_codes':{key:value.tolist() for key,value in verdicts.items()},'code_legend':dict(enumerate(BLOCK_NAMES))}
    for date in ('2026-12-21','2026-06-21'):
        if skip_floor_images:continue
        for hour in (9,12,15):
            instant=dt.datetime.fromisoformat(date).replace(hour=hour,tzinfo=A.TZ);az,_,el=A.solar([instant],lat,lon)
            en=np.array([[np.sin(np.deg2rad(az[0]))*np.cos(np.deg2rad(el[0])),np.cos(np.deg2rad(az[0]))*np.cos(np.deg2rad(el[0]))]])
            d=np.column_stack([en@np.linalg.inv(matrix).T,np.sin(np.deg2rad(el))]);d/=np.linalg.norm(d,axis=1)[:,None]
            for r in rooms:patches[(date,hour,r['id'])]=np.zeros(len(r['grid']),bool)
            for w in windows:
                r=next(r for r in rooms if r['id']==w['room_id']);codes=aperture_verdict(w,d,roofs,obstacles)
                patches[(date,hour,r['id'])]|=floor_patch(w,r,d,codes)[0]
    for rr in result['rooms']:
        dec=rr['dates'][DATES[0]];jun=rr['dates'][DATES[2]]
        rr['sentence']=f"Under the listed clear-aperture geometry assumptions, {rr['label']} has {dec['lit_minutes_ge_0_1m2']:.0f} minutes of qualifying direct floor sun on 21 December and {jun['lit_minutes_ge_0_1m2']:.0f} minutes on 21 June (windows alone: {dec['windows_only_lit_minutes_ge_0_1m2']:.0f} and {jun['windows_only_lit_minutes_ge_0_1m2']:.0f}); omitted vegetation, blinds and uncertain roof heights are not accounted for."
    out=ROOT/'data/sun'/sid;out.mkdir(parents=True,exist_ok=True)
    write(out/'sample_verdicts.json',sample_records)
    result['api_calls_this_run']={k:API_CALLS[k]-before.get(k,0) for k in API_CALLS};result['seconds']=round(time.monotonic()-started,3)
    write(out/'sun.json',result)
    if not skip_floor_images:draw_floor(sid,rooms,windows,patches,matrix)
    result['seconds']=round(time.monotonic()-started,3);write(out/'sun.json',result)
    return result

def batch_selection():
    with (ROOT/'models_index.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    byid={r['sid']:r for r in rows};required=[r['sid'] for r in json.loads((ROOT/'data/geo_check/results.json').read_text())['homes']]
    texas=lambda row:row['state'] in ('TX','Texas') or ', TX,' in row['address']
    comparators=[sid for sid in required if not texas(byid[sid])]
    selected=[]
    for sid in [s for s in required if s not in comparators]+[TARGET]+[r['sid'] for r in rows if r['segment']=='residential' and texas(r)]:
        if sid in {r['sid'] for r in selected}:continue
        row=byid[sid]
        if row['segment']!='residential' or (row['state'] not in ('TX','Texas') and ', TX,' not in row['address']):
            raise RuntimeError('required_home_not_Texas_residential_'+sid)
        selected.append(row)
        if len(selected)==40-len(comparators):break
    if len(selected)!=40-len(comparators):raise RuntimeError('batch_selection_incomplete')
    selected=[dict(r,cohort='Texas38') for r in selected]+[dict(byid[sid],cohort='required_nonTexas_comparator') for sid in comparators]
    if not set(required)<=set(r['sid'] for r in selected):raise RuntimeError('required20_not_covered')
    write(ROOT/'data/sun/batch_selection.json',{'required_20_sids':required,
        'conflict_resolution':'Claude reviewer clarified40 total including20 required:38 Texas+2 California; geography does not matter for this batch',
        'clarification_reference':'Bridge reply d50b701c-da96-499c-aab0-6dc96adc4298,2026-11-14T06:52Z',
        'homes':[{k:r[k] for k in ('sid','address','city','state','segment','cohort')} for r in selected]})
    return selected

def index_row(row,result):
    out={'sid':row['sid'],'address':row['address'],'state':row['state'],'cohort':row.get('cohort',''),'registration_iou':result['registration']['iou_after'],
        'registration_containment':result['registration']['contain_after'],'registration_poor':result['registration']['poor'],
        'windows':sum(w['type']=='window' for w in result['windows']),'conditional_exterior_doorways':sum(w['type']=='doorway' for w in result['windows']),
        'rooms':len(result['rooms']),'seconds':result['seconds'],'failure_reason':''}
    for role in ('living_room','kitchen','primary_bedroom'):
        def matches(room):
            text=(room['label']+' '+ ' '.join(c['label'] for c in room['classifications'])).lower()
            if role=='living_room':return 'living' in text or 'family room' in text
            if role=='kitchen':return 'kitchen' in text
            return ('primary' in text or 'master' in text) and ('bed' in text or 'suite' in text)
        candidates=[r for r in result['rooms'] if matches(r)];room=max(candidates,key=lambda r:r['area_m2']) if candidates else None
        out[role+'_room_id']=room['id'] if room else ''
        out[role+'_selection_note']='largest labelled room among'+str(len(candidates)) if room else 'role not explicitly labelled; null'
        for season,date in [('dec','2026-12-21'),('jun','2026-06-21')]:out[role+'_'+season+'_minutes']=room['dates'][date]['lit_minutes_ge_0_1m2'] if room else ''
        if role=='living_room':
            windows=[w for w in result['windows'] if room and w['room_id']==room['id'] and w['type']=='window']
            main=max(windows,key=lambda w:w['width']*w['height']) if windows else None
            out['main_living_window_id']=main['id'] if main else '';out['main_living_window_orientation']=compass(main['outward_azimuth_deg']) if main else '';out['main_living_window_azimuth_deg']=main['outward_azimuth_deg'] if main else ''
    return out

def run_batch():
    started=time.monotonic();selected=batch_selection();index=[];failures=collections.Counter();runtimes=[]
    for number,row in enumerate(selected,1):
        start=time.monotonic()
        try:
            result=run_home(row['sid']);record=index_row(row,result);runtimes.append(result['seconds'])
            print(json.dumps({'number':number,'sid':row['sid'],'status':'ok','seconds':result['seconds'],'rooms':len(result['rooms']),'windows':len(result['windows'])}),flush=True)
        except Exception as error:
            reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
            if len(reason)>160 or re.search(r'https?://',reason):reason=type(error).__name__
            failures[reason]+=1;record={'sid':row['sid'],'address':row['address'],'state':row['state'],'cohort':row.get('cohort',''),'failure_reason':reason,'seconds':round(time.monotonic()-start,3)}
            write(ROOT/'data/sun'/row['sid']/'failure.json',record)
            print(json.dumps({'number':number,**record}),flush=True)
        index.append(record)
        fields=list(dict.fromkeys(k for item in index for k in item))
        with (ROOT/'data/sun/index.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(index)
        summary={'homes_selected':len(selected),'homes_finished':len(index),'successes':len(runtimes),'failures_by_reason':dict(failures),
            'seconds_per_success_mean':round(float(np.mean(runtimes)),3) if runtimes else None,'seconds_per_success_median':round(float(np.median(runtimes)),3) if runtimes else None,
            'total_seconds':round(time.monotonic()-started,3),'api_calls':dict(API_CALLS),'offline_mode':os.environ.get('SUN_OFFLINE')=='1',
            'cohort_counts':dict(collections.Counter(r.get('cohort') for r in index)),
            'poor_registration_count':sum(bool(r.get('registration_poor')) for r in index),
            'missing_primary_bedroom_role_count':sum(not r.get('primary_bedroom_room_id') for r in index if not r.get('failure_reason')),
            'interpretation':'Conditional clear-glazing geometric estimates, not observed sun or irradiance; consult each sun.json assumptions. Unlabelled roles remain null.'}
        write(ROOT/'data/sun/batch_summary.json',summary)
        if os.environ.get('SUN_OFFLINE')=='1':write(ROOT/'data/sun/offline_batch_summary.json',summary)
    return summary

def pano_agreement():
    A.TZ=ZoneInfo('America/Chicago')
    sid=TARGET;model,prop=inputs(sid);fit=registration(sid,model,prop)
    matrix,lat,lon=geo_matrix(model);matrix,_=alignment(fit,matrix)
    out=ROOT/'data/sun_check'/sid
    meta=json.loads((ROOT/'data/matterport'/sid/'pano_meta.json').read_text());meta=meta.get('pano',meta)
    pos=np.array([meta['position'][k] for k in ('x','y','z')]);q=A.rotation(meta['rotation'])
    faces=[Image.open(ROOT/'data/matterport'/sid/f'pano_td86n1_face{i}.jpg').convert('RGB') for i in range(6)];A.check_cap_seams(faces)
    items=A.openings(model);A.PANO=A.PRIMARY_PANO
    samples=json.loads((ROOT/'data/sun'/sid/'sample_verdicts.json').read_text())
    table=collections.defaultdict(lambda:[0,0.]);records=[];roof_test=collections.Counter();by_window={w['id']:i for i,w in enumerate(items)}
    for date,data in samples.items():
        times,weights,directions,az,el,_,_=sample_day(date,lat,lon,matrix)
        owner=A.owners_for(directions,pos,items,model);rgb,face,uv=A.pano_sample(directions,q,faces)
        labels=A.classify(rgb,face,uv,owner,items)
        for ti,time_value in enumerate(times):
            record={'date':date,'time':time_value.astimezone(A.TZ).isoformat(),'weight_minutes':float(weights[ti]),'sun_azimuth_deg':float(az[ti]),'sun_elevation_deg':float(el[ti])}
            if owner[ti]<0:
                records.append(dict(record,status='solar direction outside the five visible graph apertures'));continue
            wid=items[owner[ti]]['id'];code=int(data['window_sample_codes'][wid][ti][4]);pixel=uv[ti]*2048
            porch=wid=='F0E16O1' and face[ti]==2 and uv[ti,1]<.39-.075*uv[ti,0]-.025
            c=rgb[ti].astype(float);green=c[1]>c[0]*1.07 and c[1]>c[2]*1.04 and c[0]<215;dark=max(c)<105
            foliage=wid=='F0E16O1' and (green or dark)
            category='porch_roof' if porch else ('foliage' if foliage else 'unknown')
            geometry=BLOCK_NAMES[code]
            if geometry in ('sun_inside_wall','below_horizon'):
                assessment='not comparable: sun behind wall/below horizon'
            elif category=='porch_roof':
                assessment='agrees on own roof' if geometry=='own_roof' else ('roof/clear disagreement' if geometry=='clear' else 'both blocked; different reported blocker')
                roof_test[assessment]+=1
            elif category=='foliage':assessment='potential tree agreement; individual crown correspondence unverified' if geometry=='tree' else 'foliage is a separate unvalidated obstruction; no strict agreement score'
            else:assessment='unknown pano evidence; unscorable'
            key=(date,wid,category,geometry);table[key][0]+=1;table[key][1]+=float(weights[ti])
            records.append(dict(record,window_id=wid,face=int(face[ti]),pixel_xy=pixel.tolist(),brief_a_label=int(labels[ti]),pano_category=category,
                geometry_at_opening_center=geometry,clear_aperture_samples=int(np.sum(np.array(data['window_sample_codes'][wid][ti])==0)),assessment=assessment))
    rows=[{'date':key[0],'window_id':key[1],'pano_class':key[2],'engine_verdict':key[3],'samples':value[0],'minutes':round(value[1],2)} for key,value in sorted(table.items())]
    report={'sid':sid,'pano_id':A.PRIMARY_PANO,'table':rows,'roof_evidence_counts':dict(roof_test),'records':records,
        'visible_openings':sorted(A.VISIBLE_OPENINGS),'unseen_window_dates':[{'date':date,'window_id':wid,'status':'no sampled solar direction through this aperture from this pano'} for date in DATES for wid in sorted(A.VISIBLE_OPENINGS) if not any(r['date']==date and r['window_id']==wid for r in rows)],
        'limitations':['Same sample times and solar direction; pano ray starts at camera, engine centre ray starts at opening: parallax can change obstruction.',
            'Porch-roof region and foliage/dark heuristic exactly reuse Brief A face2 threshold and green/dark rules; dark frames/door leaf can be falsely labelled foliage.',
            'Blinded windows and opaque door-panel directions are unknown rather than clear sky. No sky is certified.',
            'Only porch-roof versus own-roof is directly scorable. Foliage and neighbour are not mutually exclusive, and tree identities are unverified.',
            'The3x3 engine verdict shown is its centre ray; clear_aperture_samples preserves partial exposure. Trees are retained as a separate fourth obstruction category.']}
    write(out/'geometry_pano_agreement.json',report)
    with (out/'geometry_pano_agreement.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['date','window_id','pano_class','engine_verdict','samples','minutes']);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'roof_evidence_counts':dict(roof_test),'table_rows':len(rows),'classified_sample_count':sum(r['samples'] for r in rows)}))
    return report

def main():
    sid=sys.argv[1] if len(sys.argv)>1 else TARGET
    if sid=='--batch':print(json.dumps(run_batch()));return
    if sid=='--compare':pano_agreement();return
    if '--roof' in sys.argv:
        model,prop=inputs(sid);fit=registration(sid,model,prop)
        path=ROOT/'data/sun_check'/TARGET/'roof_edge_anchors.json'
        if path.exists():
            r=roof_edge_check(model,prop,fit,json.loads(path.read_text()))
            print(json.dumps({k:r[k] for k in ('eave_height_model_z_m','delivered_rms_deg','corrected_rms_deg')}))
        else:
            raw,fixed,own=roof_edge_check(model,prop,fit)
            print(json.dumps({'raw_model_ring':raw.tolist(),'corrected_model_ring':fixed.tolist(),'vexcel_height':own['height'],'rooms':model['rooms'],'floors':[{k:v for k,v in f.items() if k!='edges'} for f in model['floors']]},indent=2))
    else:
        result=run_home(sid);print(json.dumps({'sid':sid,'rooms':len(result['rooms']),'windows':len(result['windows']),'seconds':result['seconds'],'api_calls':result['api_calls_this_run']}))

if __name__=='__main__': main()
