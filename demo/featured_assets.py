"""Local, graph-projected window crops. Image association is not a survey."""
from pathlib import Path
import ast
import io
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'demo'

def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,separators=(',',':')))

def corners(w):
    c=np.array(w['center']);t=np.array(w['tangent'])*w['width']/2
    return np.array([[*(c-t),w['lower_z_m']],[*(c+t),w['lower_z_m']],
        [*(c+t),w['lower_z_m']+w['height']],[*(c-t),w['lower_z_m']+w['height']]])

def utm(lon,lat,zone):
    # Reuse Brief B's tested, pure conversion without executing its experiment.
    tree=ast.parse((ROOT/'experiments/heights_from_aerial.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='utm']
    scope={'math':math,'np':np};exec(compile(tree,'brief_b_utm','exec'),scope)
    return scope['utm'](lon,lat,zone)

def indoor(sid,data,api,engine,records):
    model=read(ROOT/'data/cache'/sid/'matterport.json')['data']['model']
    query='{model(id:"%s"){panoLocations{id position{x y z} rotation{x y z w} skybox(resolution:"2k"){children}}}}'%sid
    remote=None
    cache=BASE/'cache'/sid
    posepath=cache/'pano_poses.json'
    if posepath.exists():panos=read(posepath)
    else:
        remote=api.request(Path(sid)/'pano_query.json',service='matterport',body={'query':query},live=True)
        panos=[{k:p[k] for k in ['id','position','rotation']} for p in remote['data']['model']['panoLocations']]
        write(posepath,panos)
    all_openings=engine.A.openings(model)
    for w in data['windows']:
        rec=records.setdefault(w['id'],{})
        target=BASE/'site/img'/sid/(w['id']+'-inside.jpg')
        if target.exists() and rec.get('inside'):
            rec['inside_status']='Interior panorama with this opening outlined. Blinds and trim are not modelled.'
            continue
        geom=data['geometry'][w['room_id']];room=next(r for r in data['rooms'] if r['id']==w['room_id'])
        xyz=corners(w);middle=xyz.mean(0);eligible=[]
        for p in panos:
            pos=np.array([p['position'][k] for k in ('x','y','z')])
            inside=engine.inside_segments(pos[None,:2],np.array(geom['segments']))[0]
            if any(engine.inside_segments(pos[None,:2],np.array(h))[0] for h in geom['holes']):inside=False
            if not inside or not room['floor_z_m']+.2<pos[2]<room['ceiling_z_m']+.5:continue
            delta=middle-pos;distance=np.linalg.norm(delta)
            if distance<.3:continue
            clear=engine.A.wall_clear((delta/distance)[None],pos,model,all_openings,np.array([distance]))[0]
            if not clear:continue
            eligible.append((distance,p,pos))
        if not eligible:
            rec['inside_status']='No suitable panorama is available in this room.';continue
        distance,p,pos=min(eligible,key=lambda e:e[0])
        forward=middle-pos;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
        relative=xyz-pos;den=relative@forward
        focal=min(520.,230./max(float(np.max(np.abs(relative@right/den))),float(np.max(np.abs(relative@up/den))),.1))
        size=640;yy,xx=np.mgrid[:size,:size]
        rays=forward+((xx-size/2)/focal)[...,None]*right-((yy-size/2)/focal)[...,None]*up
        face,uv=engine.A.cube_coordinates(rays,engine.A.rotation(p['rotation']))
        rgb=np.zeros((size,size,3),np.uint8)
        for fi in np.unique(face):
            facepath=Path(sid)/'panos'/f"{p['id']}-{fi}.jpg"
            if not (BASE/'cache'/facepath).exists():
                if remote is None:
                    remote=api.request(Path(sid)/'pano_query.json',service='matterport',body={'query':query},live=True)
                live=next(x for x in remote['data']['model']['panoLocations'] if x['id']==p['id'])
                api.request(facepath,url=live['skybox']['children'][int(fi)],service='image')
            source=np.asarray(Image.open(BASE/'cache'/facepath).convert('RGB'));sel=face==fi
            xy=np.clip((uv[sel]*source.shape[0]).astype(int),0,source.shape[0]-1)
            rgb[sel]=source[xy[:,1],xy[:,0]]
        image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
        outline=np.column_stack([size/2+focal*(relative@right)/den,size/2-focal*(relative@up)/den])
        draw.line([tuple(v) for v in np.vstack([outline,outline[:1]])],fill='#16ee99',width=3)
        image.save(target,quality=92)
        rec.update(inside=f'img/{sid}/{target.name}',pano_id=p['id'],pano_distance_m=round(float(distance),2),
            inside_status='Interior panorama with this opening outlined. Blinds and trim are not modelled.',
            indoor_projection={'size_px':size,'focal_px':float(focal),'window_corners_px':outline.tolist(),'selection':'Pano inside assigned room, correct floor, clear centre ray in wall graph.'})
        write(cache/'window_assets.json',records)

def outdoor(sid,data,api,engine,records):
    tr=data['transform'];lat=tr['origin_lat'];lon=tr['origin_lon']
    matrix=np.array(tr['registered_matrix']);shift=np.array(tr['registered_shift'])
    candidates=[]
    for direction in ('north','south','east','west'):
        query_path=Path(sid)/('oblique-'+direction+'.json')
        missing=BASE/'cache'/query_path.parent/(query_path.name+'.attempt0.json')
        if missing.exists() and read(missing).get('_error')=='HTTP 404':continue
        try:
            result=api.request(query_path,'oriented/query',params={
                'wkt':f'POINT({lon} {lat})','srid':4326,'layer':'urban','product-type':'oblique-'+direction,
                'collection':data['assumptions']['vexcel_collection'],
                'include':'image-name,geometry,camera-pos-x,camera-pos-y,camera-pos-z,utm-zone,ground-z,capture-date,collection,product-type',
                'order-by':'image-date-desc,image-center-distance-asc'})
        except RuntimeError as error:
            if str(error).startswith('Remote HTTP 404;'):continue
            raise
        for feature in result.get('features',[]):
            p=feature['properties'];p['direction']=direction;candidates.append(p)
    groups={}
    for w in data['windows']:
        rec=records.setdefault(w['id'],{})
        az=np.deg2rad(w['outward_azimuth_deg']);normal=np.array([np.sin(az),np.cos(az)])
        center=np.array(w['center'])@matrix.T+shift;rank=[]
        for p in candidates:
            zone=int(str(p['utm-zone']).rstrip('NS'))
            origin=utm(lon,lat,zone)
            delta=np.array([p['camera-pos-x'],p['camera-pos-y']])-origin-center
            norm=np.linalg.norm(delta)
            cos=float(delta@normal/norm)
            if cos>.15:rank.append((cos-.00001*norm,p))
        if not rank:rec['outside_status']='No camera-facing oblique candidate in the supplied collection.';continue
        score,p=max(rank,key=lambda x:x[0])
        groups.setdefault(p['image-name'],{'properties':p,'windows':[]})['windows'].append(w)
    prop=read(ROOT/'data/cache'/sid/'vexcel_calculated_65m.json')
    raw,_,_=engine.geo_matrix(read(ROOT/'data/cache'/sid/'matterport.json')['data']['model'])
    fit=read(ROOT/'data/cache'/sid/'registration.json')
    buildings=engine.building_rings(prop,lat,lon);own=buildings[engine.own_building(buildings,fit)]
    ground=float(own['ground'] or 0)
    for number,(name,group) in enumerate(sorted(groups.items())):
        if all(records[w['id']].get('outdoor_projection',{}).get('facade_version')==3 and records[w['id']].get('outside') and (BASE/'site'/records[w['id']]['outside']).exists() for w in group['windows']):continue
        p=group['properties'];tag=p['direction']+'-'+str(number)
        imagepath=Path(sid)/('oblique-'+tag+'.jpg')
        api.request(imagepath,'oriented/extract',body={'wkt':data['aerial']['request_wkt'],'srid':4326,'layer':'urban',
            'image-name':name,'rotation':'natural','crop':'bbox','image-format':'jpeg','downsample':0})
        metadata=read((BASE/'cache'/imagepath).with_suffix('.metadata.json'))
        if metadata.get('image-name')!=name:raise RuntimeError('Oblique image identity mismatch')
        bbox=np.array(list(map(float,metadata['pixel-bbox'].split(','))))
        # Project all openings, plus a contextual rectangle for each selected
        # facade, into the same raw photograph. Resizing is bounded to 3x.
        contexts=[];facades=[];context_widths=[]
        for w in group['windows']:
            tangent=np.array(w['tangent']);normal=np.array([-tangent[1],tangent[0]]);center=np.array(w['center'])
            facade=[o for o in data['windows'] if abs(np.array(o['tangent'])@tangent)>.95 and abs((np.array(o['center'])-center)@normal)<.75]
            xyz=np.concatenate([corners(o) for o in facade]);along=(xyz[:,:2]-center)@tangent
            width=max(18.,float(np.ptp(along))+3);mid=center+tangent*(along.min()+along.max())/2
            left=mid-tangent*width/2;right=mid+tangent*width/2
            bottom=float(xyz[:,2].min())-1.5;top=float(xyz[:,2].max())+2
            contexts.append(np.array([[*left,bottom],[*right,bottom],[*right,top],[*left,top]]));facades.append(facade);context_widths.append(width)
        points=[]
        for xyz in [*(corners(w) for w in data['windows']),*contexts]:
            en=xyz[:,:2]@matrix.T+shift
            points.extend([[lon+e/(111320*np.cos(np.deg2rad(lat))),lat+n/110574,ground+z-data['assumptions']['model_ground_z_m']] for (e,n),z in zip(en,xyz[:,2])])
        wkt='MULTIPOINT Z('+','.join(' '.join(f'{v:.11f}' for v in point) for point in points)+')'
        projection_key=hashlib.sha256((name+wkt).encode()).hexdigest()[:16]
        result=api.request(Path(sid)/('pixels-facade-v3-'+projection_key+'-'+tag+'.json'),'oriented/transform-points',body={'operation':'world-2-pixel','image-name':name,'wkt':wkt,'srid':4326})
        pixels=np.array([[p['x'],p['y']] for p in result['points']])-bbox[:2]
        rawimage=Image.open(BASE/'cache'/imagepath).convert('RGB')
        if rawimage.size!=(int(bbox[2]),int(bbox[3])):raise RuntimeError('Raw oblique crop pixel grid mismatch')
        all_pixels={w['id']:pixels[i*4:i*4+4] for i,w in enumerate(data['windows'])}
        zone=int(str(p['utm-zone']).rstrip('NS'));camera_delta=np.array([p['camera-pos-x'],p['camera-pos-y']])-utm(lon,lat,zone)
        side=['north','northeast','east','southeast','south','southwest','west','northwest'][int((math.degrees(math.atan2(camera_delta[0],camera_delta[1]))%360+22.5)//45)%8]
        for i,w in enumerate(group['windows']):
            uv=all_pixels[w['id']];context=pixels[(len(data['windows'])+i)*4:(len(data['windows'])+i+1)*4]
            lo=np.floor(context.min(0)-8).astype(int);hi=np.ceil(context.max(0)+8).astype(int)
            # Keep roof and ground context when the facade projects to a thin
            # band in a steep aerial view.
            if hi[1]-lo[1]<(hi[0]-lo[0])*.6:
                cy=(lo[1]+hi[1])/2;half=(hi[0]-lo[0])*.3
                lo[1]=math.floor(cy-half);hi[1]=math.ceil(cy+half)
            lo=np.maximum(lo,0);hi=np.minimum(hi,rawimage.size)
            if np.any(hi<=lo):records[w['id']]['outside_status']='Projected aperture falls outside raw crop.';continue
            crop=rawimage.crop((*lo,*hi));scale=min(3.,900/max(crop.size))
            crop=crop.resize(tuple(max(1,round(d*scale)) for d in crop.size),Image.Resampling.LANCZOS)
            draw=ImageDraw.Draw(crop);outline=(uv-lo)*scale
            for opening in facades[i]:
                other=(all_pixels[opening['id']]-lo)*scale
                draw.line([tuple(v) for v in np.vstack([other,other[:1]])],fill='#16ee99',width=1)
            draw.line([tuple(v) for v in np.vstack([outline,outline[:1]])],fill='#16ee99',width=3)
            target=BASE/'site/img'/sid/(w['id']+'-outside.jpg');crop.save(target,quality=94)
            records[w['id']].update(outside=f'img/{sid}/{target.name}',outside_image_name=name,
                outside_status=f'Viewed from the {side}. Thin outlines show openings on this facade; the selected opening is thick. Assumed ground-to-floor height; trees or buildings may hide the opening.',
                outdoor_projection={'facade_version':3,'facade_opening_ids':[o['id'] for o in facades[i]],'context_width_m':context_widths[i],
                    'camera_side':side,'raw_crop_bbox':bbox.tolist(),'window_corners_raw_crop_px':uv.tolist(),'image_crop_lo':lo.tolist(),'enlargement':scale,'ground_elevation_m':ground,
                    'selection':'Best camera-facing oblique by facade angle and distance. Physical opening identity not independently checked.'})
            write(BASE/'cache'/sid/'window_assets.json',records)

def build(sid,data,api,engine):
    manifest=BASE/'cache'/sid/'window_assets.json'
    records=read(manifest) if manifest.exists() else {}
    failures=[]
    for name,function in [('inside',indoor),('outside',outdoor)]:
        try:function(sid,data,api,engine,records)
        except Exception as error:
            failures.append(name+': '+type(error).__name__+'; transport details suppressed')
            if api.AUTH_FAILURES>=2:break
    for w in data['windows']:
        record=records.setdefault(w['id'],{})
        for name in ['inside','outside']:
            if name not in record:record.setdefault(name+'_status','Unavailable; '+('; '.join(failures) or 'no eligible view'))
    write(manifest,records)
    return records
