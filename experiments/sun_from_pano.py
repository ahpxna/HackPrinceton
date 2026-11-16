"""Reproduce Brief A. Run with python3 -I experiments/sun_from_pano.py khbuN7q853q.

Only reads credentials at runtime; never persists credentials, signed URLs or
remote error bodies. All generated data is confined to data/sun_check/.
"""
from pathlib import Path
import base64
import datetime as dt
import json
import math
import os
import sys
import importlib.util
import hashlib
import urllib.request
import urllib.error
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PANO = "td86n1sku5u1rt38abpq787ub"
PRIMARY_PANO = PANO
SECOND_PANO = "7c6sb376gt2qn5y8yembtmidb"
VISIBLE_OPENINGS = {"F0E16O0","F0E22O0","F0E22O1","F0E16O1","F0E22O2"}
# Each row is (centre, image-right, image-down), in the quaternion's local frame.
BASES = np.array([[[0,0,1],[1,0,0],[0,1,0]],
                  [[1,0,0],[0,-1,0],[0,0,-1]],
                  [[0,-1,0],[-1,0,0],[0,0,-1]],
                  [[-1,0,0],[0,1,0],[0,0,-1]],
                  [[0,1,0],[1,0,0],[0,0,-1]],
                  [[0,0,-1],[1,0,0],[0,-1,0]]], dtype=float)


def credentials():
    values = {}
    with (ROOT / ".env").open() as stream:
        for line in stream:
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.strip().split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def gql(query):
    env = credentials()
    auth = base64.b64encode((env["MATTERPORT_API_KEY"] + ":" +
                            env["MATTERPORT_API_TOKEN"]).encode()).decode()
    req = urllib.request.Request("https://api.matterport.com/api/models/graph",
                                 json.dumps({"query": query}).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": "Basic " + auth})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Matterport HTTP {error.code}; response body suppressed") from None
    if result.get("errors"):
        raise RuntimeError("Matterport GraphQL rejected the scoped query; body suppressed")
    return result["data"]


def prepare(sid, out):
    cache = out / "geometry.json"
    if cache.exists():
        return json.loads(cache.read_text())
    query = '''{model(id:"%s"){
      floors{edges(includeUsed:true){type vertices{position{x y}}
        openings{type width height lowerElevation relativeCenter}}}
      geocoordinates{latitude longitude
        p0:geoLocationOf(modelLocation:{x:0,y:0,z:0}){lat long}
        px:geoLocationOf(modelLocation:{x:20,y:0,z:0}){lat long}
        py:geoLocationOf(modelLocation:{x:0,y:20,z:0}){lat long}
        scan:geoLocationOf(modelLocation:{x:-6.844671726226807,y:-3.4381508827209473,z:1.4652230739593506}){lat long}}}}
    ''' % sid
    model = gql(query)["model"]
    cache.write_text(json.dumps(model, indent=2))
    return model


def rotation(q):
    x,y,z,w = [q[k] for k in ("x","y","z","w")]
    x,y,z,w = np.array([x,y,z,w]) / np.linalg.norm([x,y,z,w])
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def openings(model):
    result=[]
    for fi,floor in enumerate(model["floors"]):
        for ei,edge in enumerate(floor["edges"]):
            if len(edge["vertices"]) != 2:
                continue
            a,b = [np.array([v["position"][k] for k in ("x","y")]) for v in edge["vertices"]]
            tangent=(b-a)/np.linalg.norm(b-a)
            for oi,o in enumerate(edge.get("openings") or []):
                center=a+o["relativeCenter"]*(b-a)
                p0=center-tangent*o["width"]/2
                p1=center+tangent*o["width"]/2
                z=o["lowerElevation"]
                item=dict(o, id=f"F{fi}E{ei}O{oi}", center=center.tolist(),
                          tangent=tangent.tolist(), floor_index=fi, edge_index=ei,
                          corners=[[p0[0],p0[1],z],[p1[0],p1[1],z],
                                   [p1[0],p1[1],z+o["height"]],[p0[0],p0[1],z+o["height"]]])
                result.append(item)
    return result


def cube_coordinates(vectors, q):
    local = np.asarray(vectors) @ q
    dots = local @ BASES[:,0].T
    face = dots.argmax(-1)
    den = np.take_along_axis(dots,face[...,None],axis=-1)[...,0]
    uv = np.stack([(local*BASES[face,1]).sum(-1)/den,
                   (local*BASES[face,2]).sum(-1)/den],axis=-1)
    return face, (uv+1)/2


def overlay(items, pos, q, faces, out):
    tiles=[]
    for f,img in enumerate(faces):
        tile=img.copy()
        draw=ImageDraw.Draw(tile)
        draw.text((25,25),f"FACE {f}",fill="red",font=ImageFont.load_default(size=38))
        for w in items:
            color=("lime" if w["type"]=="window" else "orange") if w["id"] in VISIBLE_OPENINGS else "gray"
            corners=np.array(w["corners"])
            allpts=[]
            for a,b in zip(corners,np.roll(corners,-1,axis=0)):
                pts=np.linspace(a,b,100)
                face,uv=cube_coordinates(pts-pos,q)
                good=face==f
                for j in range(len(pts)-1):
                    if good[j] and good[j+1]:
                        draw.line([tuple(uv[j]*img.width),tuple(uv[j+1]*img.width)],fill=color,width=5)
                allpts.extend((uv[good]*img.width).tolist())
            if allpts:
                xy=np.mean(allpts,axis=0)
                draw.text(tuple(xy),w["id"],fill=color,stroke_width=2,stroke_fill="black",font=ImageFont.load_default(size=25))
        tiles.append(tile.resize((768,768)))
    sheet=Image.new("RGB",(2304,1536))
    for f,tile in enumerate(tiles): sheet.paste(tile,((f%3)*768,(f//3)*768))
    draw=ImageDraw.Draw(sheet)
    draw.rectangle((0,1500,2304,1536),fill="white")
    draw.text((12,1506),"Green: visible window geometry | orange: exterior doorway geometry | gray: graph openings behind walls (not visible apertures)",fill="black",font=ImageFont.load_default(size=20))
    sheet.save(out/"window_overlay.jpg",quality=95)


def check_cap_seams(faces):
    """Infer cap rotations/reflections against fixed, geometrically verified sides."""
    a=[np.array(f) for f in faces]
    report={}
    for cap in (0,5):
        scores=[]
        initial=BASES[cap].copy()
        for swap in (False,True):
            for sr in (-1,1):
                for sd in (-1,1):
                    basis=initial.copy()
                    basis[1]=initial[2 if swap else 1]*sr
                    basis[2]=initial[1 if swap else 2]*sd
                    diffs=[]
                    for side in (1,2,3,4):
                        x=np.linspace(-.95,.95,300)
                        d=BASES[side,0]+x[:,None]*BASES[side,1]+(.999 if cap==5 else -.999)*BASES[side,2]
                        def sample(f,b):
                            uv=np.stack([d@b[1]/(d@b[0]),d@b[2]/(d@b[0])],axis=-1)
                            pix=np.clip(((uv+1)/2*(a[f].shape[0]-1)).astype(int),0,a[f].shape[0]-1)
                            return a[f][pix[:,1],pix[:,0]].astype(float)
                        diffs.extend(np.abs(sample(side,BASES[side])-sample(cap,basis)).ravel())
                    scores.append((float(np.mean(diffs)),basis.tolist()))
        scores.sort(key=lambda x:x[0])
        BASES[cap]=np.array(scores[0][1])
        report[str(cap)]={"best_mean_rgb_error":scores[0][0],"runner_up_mean_rgb_error":scores[1][0],"basis":scores[0][1]}
    return report


def heading_fit(sid,out):
    path=out/"heading_fit.json"
    if path.exists(): return json.loads(path.read_text())
    # Import existing routines read-only; do not execute its main or write bytecode.
    sys.dont_write_bytecode=True
    os.environ["MPLCONFIGDIR"]=str(out/"matplotlib_cache")
    spec=importlib.util.spec_from_file_location("geo_readonly",ROOT/"experiments"/"geo_check.py")
    geo=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    result=geo.analyse(sid,{"address":"811 Sara Rose St","city":"Houston"})
    if "skip" in result:
        raise RuntimeError("House-specific footprint fit failed; no guessed correction substituted")
    path.write_text(json.dumps(result,indent=2))
    return result


def solar(times, lat, lon):
    """NOAA/Meeus equations; UTC datetimes, east-positive lon. Degrees."""
    stamp=np.array([t.timestamp() for t in times])
    jd=stamp/86400+2440587.5
    T=(jd-2451545)/36525
    rad=np.deg2rad
    L=(280.46646+T*(36000.76983+.0003032*T))%360
    M=357.52911+T*(35999.05029-.0001537*T)
    ecc=.016708634-T*(.000042037+.0000001267*T)
    C=np.sin(rad(M))*(1.914602-T*(.004817+.000014*T))+np.sin(rad(2*M))*(.019993-.000101*T)+.000289*np.sin(rad(3*M))
    omega=125.04-1934.136*T
    apparent=L+C-.00569-.00478*np.sin(rad(omega))
    eps=23+(26+(21.448-T*(46.815+T*(.00059-T*.001813)))/60)/60+.00256*np.cos(rad(omega))
    dec=np.arcsin(np.sin(rad(eps))*np.sin(rad(apparent)))
    y=np.tan(rad(eps)/2)**2
    eq=4*np.rad2deg(y*np.sin(2*rad(L))-2*ecc*np.sin(rad(M))+4*ecc*y*np.sin(rad(M))*np.cos(2*rad(L))-.5*y*y*np.sin(4*rad(L))-1.25*ecc*ecc*np.sin(2*rad(M)))
    minutes=(stamp%86400)/60
    hour=rad(((minutes+eq+4*lon)%1440)/4-180)
    phi=rad(lat)
    el=np.rad2deg(np.arcsin(np.clip(np.sin(phi)*np.sin(dec)+np.cos(phi)*np.cos(dec)*np.cos(hour),-1,1)))
    az=(np.rad2deg(np.arctan2(np.sin(hour),np.cos(hour)*np.sin(phi)-np.tan(dec)*np.cos(phi)))+180)%360
    tan=np.tan(rad(np.where(abs(el)<.00001,.00001,el)))
    ref=np.where(el>85,0,np.where(el>5,(58.1/tan-.07/tan**3+.000086/tan**5)/3600,
        np.where(el>-.575,(1735-518.2*el+103.4*el**2-12.79*el**3+.711*el**4)/3600,(-20.774/tan)/3600)))
    return az,el,el+ref


def solar_reference(out):
    url="https://data.ntsb.gov/Docket/Document/docBLOB?FileExtension=.PDF&FileName=Weather+Summary-Master.PDF&ID=40458548"
    az,el,app=solar([dt.datetime(2016,6,9,18,9,tzinfo=dt.timezone.utc)],29.634352,-95.284166)
    result={"source":url,"page":2,"published_location":{"lat":29.634352,"lon":-95.284166},
            "time":"2016-06-09T13:09:00-05:00","published_azimuth_deg":158.04,
            "published_apparent_elevation_deg":82.89,"computed_azimuth_deg":float(az[0]),
            "computed_apparent_elevation_deg":float(app[0]),"computed_geometric_elevation_deg":float(el[0])}
    result["passed"]=bool(abs(az[0]-158.04)<.02 and abs(app[0]-82.89)<.02)
    (out/"solar_reference.json").write_text(json.dumps(result,indent=2))
    if not result["passed"]: raise RuntimeError("NOAA Houston published reference mismatch")
    return result


def heading(model,fit):
    g=model["geocoordinates"]
    def en(p):
        return np.array([(float(p["long"])-float(g["p0"]["long"]))*111320*np.cos(np.deg2rad(float(g["p0"]["lat"]))),
                         (float(p["lat"])-float(g["p0"]["lat"]))*110574])/20
    A=np.column_stack([en(g["px"]),en(g["py"])])
    theta=np.deg2rad(fit["fix_rot_deg"])
    R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    # Remove the small scale/shear from local geodesy; directions need a rotation.
    u,s,v=np.linalg.svd(A)
    ortho=u@v
    used=R@ortho
    return used,{"raw_model_xy_to_east_north":A.tolist(),"orthonormal_model_xy_to_east_north":used.tolist(),
        "delivered_model_positive_x_azimuth_deg":float(np.rad2deg(np.arctan2(ortho[0,0],ortho[1,0]))%360),
        "used_model_positive_x_azimuth_deg":float(np.rad2deg(np.arctan2(used[0,0],used[1,0]))%360),
        "used_model_positive_y_azimuth_deg":float(np.rad2deg(np.arctan2(used[0,1],used[1,1]))%360),
        "footprint_rotation_ccw_deg":fit["fix_rot_deg"],"assumed_heading_uncertainty_plus_minus_deg":3.,
        "uncertainty_basis":"Assumed, not a surveyed confidence interval: coarse 0.5 deg footprint search, roof/wall mismatch and dominant-axis discrepancy. No external true-north survey.",
        "footprint_iou_before":fit["iou_before"],"footprint_iou_after":fit["iou_after"]}


def intersect(vectors,pos,items):
    """Ray/rectangle intersections: infinity outside the rectangular opening."""
    d=np.asarray(vectors)
    distances=[]
    for w in items:
        tangent=np.array(w["tangent"])
        normal=np.array([-tangent[1],tangent[0]])
        center=np.array(w["center"])
        den=d[...,:2]@normal
        t=np.divide((center-pos[:2])@normal,den,out=np.full(den.shape,np.inf),where=abs(den)>1e-10)
        with np.errstate(invalid="ignore"):
            xy=pos[:2]+t[...,None]*d[...,:2]
            z=pos[2]+t*d[...,2]
            along=(xy-center)@tangent
        good=(t>0)&(abs(along)<=w["width"]/2)&(z>=w["lowerElevation"])&(z<=w["lowerElevation"]+w["height"])
        distances.append(np.where(good,t,np.inf))
    return np.stack(distances,axis=-1)


def wall_clear(d,pos,model,items,target_distance):
    """All intervening wall segments must be traversed inside an opening."""
    clear=np.ones(d.shape[:-1],bool)
    for fi,floor in enumerate(model["floors"]):
        for ei,edge in enumerate(floor["edges"]):
            if len(edge["vertices"])!=2: continue
            a,b=[np.array([v["position"][k] for k in ("x","y")]) for v in edge["vertices"]]
            length=np.linalg.norm(b-a)
            tangent=(b-a)/length
            n=np.array([-tangent[1],tangent[0]])
            den=d[...,:2]@n
            t=np.divide((a-pos[:2])@n,den,out=np.full(den.shape,np.inf),where=abs(den)>1e-10)
            with np.errstate(invalid="ignore"):
                xy=pos[:2]+t[...,None]*d[...,:2]
                z=pos[2]+t*d[...,2]
                along=(xy-a)@tangent
            hit=(t>1e-4)&(t<target_distance-.03)&(along>=0)&(along<=length)&(z>=0)&(z<=2.6)
            aperture=np.zeros_like(clear)
            for w in items:
                if w["floor_index"]==fi and w["edge_index"]==ei:
                    aperture|=(abs(along-w["relativeCenter"]*length)<=w["width"]/2)&(z>=w["lowerElevation"])&(z<=w["lowerElevation"]+w["height"])
            clear &= ~(hit & ~aperture)
    return clear


def pano_sample(d,q,faces):
    face,uv=cube_coordinates(d,q)
    rgb=np.empty(d.shape,dtype=np.uint8)
    for f,a in enumerate(faces):
        sel=face==f
        pixels=np.clip((uv[sel]*a.width).astype(int),0,a.width-1)
        rgb[sel]=np.asarray(a)[pixels[:,1],pixels[:,0]]
    return rgb,face,uv


def classify(rgb,face,uv,owner,items):
    """Conservative, house-specific visual rules; 0 excluded,1 sky,2 blocked,3 unknown.

    All blinded glazing is unknown. In the open porch portal, texture and foliage
    are blocked; overexposure, low texture and blue patches remain unknown.
    No automatically guessed sky pixels are promoted to confirmed visibility.
    """
    label=np.zeros(owner.shape,np.uint8)
    valid=owner>=0
    label[valid]=3
    # Closed opaque door panels are excluded from possible transmission. Boxes
    # deliberately include a 10px margin around manually inspected glazing.
    glass_boxes=({"F0E22O2":(4,1566,976,1681,1060)} if PANO==PRIMARY_PANO else
                 {"F0E22O2":(4,583,922,628,1020),"F0E16O1":(1,238,945,345,1105)})
    for oid,(f,x0,y0,x1,y1) in glass_boxes.items():
        index=next(i for i,w in enumerate(items) if w["id"]==oid)
        sel=owner==index
        glass=(face==f)&(uv[...,0]*2048>=x0)&(uv[...,0]*2048<=x1)&(uv[...,1]*2048>=y0)&(uv[...,1]*2048<=y1)
        label[sel&~glass]=2
    if PANO!=PRIMARY_PANO:
        return label
    back=next(i for i,w in enumerate(items) if w["id"]=="F0E16O1")
    sel=(owner==back)
    c=rgb.astype(float)
    # Green leaves and dark branches/frames are opaque; blue-painted shed remains unknown.
    green=(c[...,1]>c[...,0]*1.07)&(c[...,1]>c[...,2]*1.04)&(c[...,0]<215)
    dark=c.max(-1)<105
    # Porch roof boundary manually inspected in face2, conservatively inset 0.025.
    px,py=uv[...,0],uv[...,1]
    roof=(face==2)&(py<.39-.075*px-.025)&(PANO==PRIMARY_PANO)
    label[sel&(green|dark|roof)]=2
    # Outside the porch centre face, doorjamb, door leaf or uncertain view stays unknown.
    return label


def owners_for(d,pos,items,model):
    dist=intersect(d,pos,items)
    # Exterior apertures visible from this scan. Other windows lie behind walls.
    for i,w in enumerate(items):
        if w["id"] not in VISIBLE_OPENINGS: dist[...,i]=np.inf
    owner=dist.argmin(-1)
    nearest=dist.min(-1)
    owner[~np.isfinite(nearest)]=-1
    select=owner>=0
    if select.any():
        clear=wall_clear(d[select],pos,model,items,nearest[select])
        current=owner[select]
        current[~clear]=-1
        owner[select]=current
    return owner


def direction(az,el,A):
    az,el=np.deg2rad(az),np.deg2rad(el)
    en=np.stack([np.sin(az)*np.cos(el),np.cos(az)*np.cos(el)],axis=-1)
    return np.concatenate([en@A,np.sin(el)[...,None]],axis=-1)


def make_pano(A,pos,q,faces,items,model,out):
    width,height=2400,1200
    az=(np.arange(width)+.5)*360/width
    el=90-(np.arange(height)+.5)*180/height
    aa,ee=np.meshgrid(az,el)
    d=direction(aa,ee,A)
    rgb,face,uv=pano_sample(d,q,faces)
    owner=owners_for(d,pos,items,model)
    labels=classify(rgb,face,uv,owner,items)
    colors=np.array([[0,0,0],[0,210,255],[230,65,65],[255,195,0]],np.float32)
    blend=rgb.astype(float)
    sel=labels>0
    blend[sel]=.45*blend[sel]+.55*colors[labels[sel]]
    image=Image.fromarray(blend.astype(np.uint8))
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,width,52),fill="white")
    draw.text((12,8),"True-north azimuth 0..360 left to right; elevation +90..-90 top to bottom | cyan sky, red blocked, yellow UNKNOWN, unshaded outside openings",fill="black",font=ImageFont.load_default(size=17))
    image.save(out/"sky_mask.jpg",quality=95)
    Image.fromarray(labels).save(out/"sky_labels.png")
    Image.fromarray(rgb).save(out/"equirectangular.jpg",quality=95)
    profile=[]
    for az0 in range(0,360,2):
        sel=(aa>=az0)&(aa<az0+2)&(labels==1)
        profile.append(float(ee[sel].min()) if sel.any() else None)
    return rgb,labels,owner,profile


TZ=ZoneInfo("America/Chicago")
DATES=("2026-12-21","2026-03-20","2026-06-21","2026-11-14")


def daylight(date,lat,lon):
    start=dt.datetime.fromisoformat(date).replace(tzinfo=TZ).astimezone(dt.timezone.utc)
    times=[start+dt.timedelta(minutes=m) for m in range(1441)]
    az,el,app=solar(times,lat,lon)
    rising=np.flatnonzero((el[:-1]<-.833)&(el[1:]>=-.833))[0]
    falling=np.flatnonzero((el[:-1]>=-.833)&(el[1:]<-.833))[0]
    def solve(index,up):
        lo,hi=times[index],times[index+1]
        for _ in range(24):
            mid=lo+(hi-lo)/2
            above=solar([mid],lat,lon)[1][0]>=-.833
            if above==up: hi=mid
            else: lo=mid
        return lo+(hi-lo)/2
    return solve(rising,True),solve(falling,False)


def evaluate(times,A,pos,q,faces,items,model,lat,lon):
    az,el,app=solar(times,lat,lon)
    d=direction(az,app,A)
    owner=owners_for(d,pos,items,model)
    rgb,face,uv=pano_sample(d,q,faces)
    labels=classify(rgb,face,uv,owner,items)
    labels[app<=0]=2
    return az,app,owner,labels


def timed_interval(sel,times,weights):
    ix=np.flatnonzero(sel)
    groups=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1) if len(ix) else []
    return {"minutes":round(float(weights[sel].sum()),3),
            "first_time":times[ix[0]].astimezone(TZ).isoformat() if len(ix) else None,
            "last_time":(times[ix[-1]]+dt.timedelta(minutes=float(weights[ix[-1]]))).astimezone(TZ).isoformat() if len(ix) else None,
            "intervals":[{"start":times[g[0]].astimezone(TZ).isoformat(),
                          "end":(times[g[-1]]+dt.timedelta(minutes=float(weights[g[-1]]))).astimezone(TZ).isoformat()} for g in groups]}


def exposure(A,pos,q,faces,items,model,out):
    lat=float(model["geocoordinates"]["scan"]["lat"])
    lon=float(model["geocoordinates"]["scan"]["long"])
    dates={}
    tracks=[]
    for date in DATES:
        rise,set_=daylight(date,lat,lon)
        total=(set_-rise).total_seconds()/60
        offsets=np.arange(0,total,5.)
        times=[rise+dt.timedelta(minutes=float(m)) for m in offsets]
        weights=np.minimum(5,total-offsets)
        az,el,owner,label=evaluate(times,A,pos,q,faces,items,model,lat,lon)
        by_window=[]
        for i,w in enumerate(items):
            if w["type"]!="window" and w["id"] not in ("F0E16O1","F0E22O2"): continue
            known=(owner==i)&(label==1)
            unknown=(owner==i)&(label==3)
            confirmed=timed_interval(known,times,weights)
            uncertain=timed_interval(unknown,times,weights)
            by_window.append({"id":w["id"],"type":w["type"],"center_model_xy_m":w["center"],
                "minutes_direct_sun":None if unknown.any() else confirmed["minutes"],
                "confirmed_sky_lower_bound_minutes":confirmed["minutes"],
                "first_time":confirmed["first_time"],"last_time":confirmed["last_time"],
                "unknown_possible_minutes":uncertain["minutes"],
                "upper_bound_minutes":round(confirmed["minutes"]+uncertain["minutes"],3),
                "unknown_first_time":uncertain["first_time"],"unknown_last_time":uncertain["last_time"],
                "confirmed_intervals":confirmed["intervals"],"unknown_intervals":uncertain["intervals"],
                "sampled_geometric_aperture_minutes":timed_interval((owner==i)&(el>0),times,weights)["minutes"],
                "status":"unresolved_view" if unknown.any() else "zero_in_static_geometry_model"})
        rates=np.abs(np.gradient(np.rad2deg(np.unwrap(np.deg2rad(az))),offsets))
        eligible=(owner>=0)&(el>0)
        conversion=3/np.maximum(rates,.001)
        sensitivity=[]
        for angle in (-3,3):
            th=np.deg2rad(angle)
            R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
            _,_,o,l=evaluate(times,R@A,pos,q,faces,items,model,lat,lon)
            sensitivity.append({"extra_rotation_ccw_deg":angle,
                                "confirmed_sky_lower_bound_minutes":timed_interval(l==1,times,weights)["minutes"],
                                "unknown_possible_minutes":timed_interval((o>=0)&(l==3),times,weights)["minutes"]})
        dates[date]={"sunrise":rise.astimezone(TZ).isoformat(),"sunset":set_.astimezone(TZ).isoformat(),
            "daylight_minutes":round(total,3),"sample_count":len(times),"windows_and_exterior_doors":by_window,
            "scan_point_total":{"confirmed_sky_lower_bound_minutes":timed_interval(label==1,times,weights)["minutes"],
                                "unknown_possible_minutes":timed_interval((owner>=0)&(label==3),times,weights)["minutes"],
                                "upper_bound_minutes":timed_interval((owner>=0)&((label==1)|(label==3)),times,weights)["minutes"]},
            "heading_timing_uncertainty_minutes_at_opening_azimuths":{
                "minimum":round(float(conversion[eligible].min()),2) if eligible.any() else None,
                "maximum":round(float(conversion[eligible].max()),2) if eligible.any() else None,
                "method":"3 deg / instantaneous |d(solar azimuth)/dt|; first-order shift of vertical aperture boundaries, not a guarantee for foliage or horizontal edges."},
            "heading_sensitivity":sensitivity}
        tracks.append((date,times,az,el))
    return dates,tracks


def draw_tracks(rgb,tracks,out):
    image=Image.fromarray(rgb)
    draw=ImageDraw.Draw(image)
    width,height=image.size
    colors=["#ff7b00","#e000d0","#00d6ee","#00e060"]
    font=ImageFont.load_default(size=19)
    for k,(date,times,az,el) in enumerate(tracks):
        points=np.column_stack([az/360*width,(90-el)/180*height])
        for a,b in zip(points,points[1:]):
            if abs(a[0]-b[0])<width/2: draw.line([tuple(a),tuple(b)],fill=colors[k],width=4)
        last_hour=None
        for t,p in zip(times,points):
            local=t.astimezone(TZ)
            if local.minute<5 and local.hour!=last_hour:
                last_hour=local.hour
                x,y=p
                draw.ellipse((x-5,y-5,x+5,y+5),fill=colors[k])
                draw.text((x+7,y+6),local.strftime("%H:%M"),font=font,fill=colors[k],stroke_width=2,stroke_fill="black")
        draw.rectangle((10,12+k*29,420,40+k*29),fill="white")
        draw.text((18,15+k*29),date+" (America/Chicago)",fill=colors[k],font=font)
    draw.text((700,15),"Azimuth 0..360 true north; elevation +90..-90 | 5-minute samples",fill="white",stroke_width=2,stroke_fill="black",font=font)
    image.save(out/"sun_paths.jpg",quality=95)


def offset_assessment(A,pos,items,model):
    windows=[w for w in items if w["type"]=="window" and w["id"] in VISIBLE_OPENINGS]
    def distance(w):
        return abs(float((pos[:2]-w["center"])@np.array([-w["tangent"][1],w["tangent"][0]])))
    w=min(windows,key=distance)
    normal=np.array([-w["tangent"][1],w["tangent"][0]])
    normal*=np.sign((pos[:2]-w["center"])@normal)
    moved=pos.copy();moved[:2]+=normal
    ends=np.array(w["center"])+np.array([-1,1])[:,None]*np.array(w["tangent"])*w["width"]/2
    views=[]
    for p in (pos,moved):
        en=(ends-p[:2])@A.T
        views.append({"window_edge_azimuth_deg":(np.rad2deg(np.arctan2(en[:,0],en[:,1]))%360).tolist(),
                      "window_top_centre_elevation_deg":float(np.rad2deg(np.arctan2(w["lowerElevation"]+w["height"]-p[2],np.linalg.norm(np.array(w["center"])-p[:2]))))})
    lat=float(model["geocoordinates"]["scan"]["lat"]);lon=float(model["geocoordinates"]["scan"]["long"])
    counts={}
    index=items.index(w)
    for date in DATES:
        rise,set_=daylight(date,lat,lon)
        total=(set_-rise).total_seconds()/60;offsets=np.arange(0,total,5.)
        times=[rise+dt.timedelta(minutes=float(m)) for m in offsets]
        az,_,el=solar(times,lat,lon);d=direction(az,el,A)
        counts[date]=[timed_interval((owners_for(d,p,items,model)==index)&(el>0),times,np.minimum(5,total-offsets))["minutes"] for p in (pos,moved)]
    return {"nearest_window_id":w["id"],"distance_to_window_centre_m":float(np.linalg.norm(np.array(w["center"])-pos[:2])),
            "plane_distance_m":distance(w),"inward_offset_model_m":[float(normal[0]),float(normal[1]),0.],
            "original_and_1m_inward_views":views,"original_and_1m_inward_geometric_aperture_minutes":counts,
            "limitation":"Geometry-only sensitivity at the same camera height and five-minute grid. No sky mask is reused at a translated viewpoint; exterior parallax and new occlusion are unknown. These are solar rays through graph apertures, not direct-sun minutes."}


def projection_record(items,pos,q,faces,seams,out):
    # Manually read opening frame corners in the original 2048 face4 image.
    observed=({"F0E22O0":(4,[[1177,1166],[1530,1168],[1525,976],[1175,984]],[0,1,2,3]),
               "F0E22O1":(4,[[823,1166],[1093,1169],[1090,995],[822,1001]],[0,1,2,3])} if PANO==PRIMARY_PANO else
              {"F0E16O1":(1,[[363,923],[222,925]],[2,3])})
    checks=[]
    for w in items:
        if w["id"] not in observed: continue
        anchor_face,coords,indices=observed[w["id"]]
        f,uv=cube_coordinates(np.array(w["corners"])[indices]-pos,q)
        predicted=uv*2048
        obs=np.array(coords)
        errors=np.linalg.norm(predicted-obs,axis=-1)
        checks.append({"opening_id":w["id"],"face":anchor_face,"observed_corners_px":obs.tolist(),
                       "projected_corners_px":predicted.tolist(),"rms_corner_error_px":float(np.sqrt(np.mean(errors**2))),
                       "corner_indices":indices,"manual_corner_uncertainty_px":5})
    # Compare the inverse quaternion using the same face and anchors; no fitted yaw.
    inverse=[]
    for w in items:
        if w["id"] in observed:
            anchor_face,coords,indices=observed[w["id"]]
            local=(np.array(w["corners"])[indices]-pos)@q.T
            uv=np.stack([local@BASES[anchor_face,1]/(local@BASES[anchor_face,0]),local@BASES[anchor_face,2]/(local@BASES[anchor_face,0])],axis=-1)
            inverse.append(float(np.sqrt(np.mean(np.sum(((uv+1)*1024-np.array(coords))**2,axis=-1)))))
    report={"mapping":"d_model = R(quaternion_xyzw) @ (face_center + u*image_right + v*image_down), u,v in [-1,1]",
            "face_bases_local":BASES.tolist(),"rotation_local_to_model":q.tolist(),
            "anchor_checks":checks,"inverse_quaternion_rms_px":inverse,"cap_seam_checks":seams,
            "verification":("Visual alignment of near blinded window, back opening and two far windows; quantitative anchors on two far windows." if PANO==PRIMARY_PANO else "Inherited proven cube convention; independently checked cap seams, visible front windows and two top corners of back door. Back door lower edge is occluded by counter; graph window dimensions visibly differ from trim."),
            "orientation_verified":bool(max(c["rms_corner_error_px"] for c in checks)<20 and min(inverse)>100),
            "anchor_limitations":"Near window graph frame dimensions differ from physical glass/frame; do not fit the quaternion to force those dimensions to agree."}
    (out/"projection_validation.json").write_text(json.dumps(report,indent=2))
    if not report["orientation_verified"]: raise RuntimeError("Cube orientation proof failed; stopping before solar exposure")
    return report


def run_scan(sid,out,model,meta,faces,fit):
    pos=np.array([meta["position"][k] for k in ("x","y","z")])
    q=rotation(meta["rotation"])
    items=openings(model)
    seams=check_cap_seams(faces)
    (out/"cube_seams.json").write_text(json.dumps(seams,indent=2))
    overlay(items,pos,q,faces,out)
    projection=projection_record(items,pos,q,faces,seams,out)
    A,head=heading(model,fit)
    reference=solar_reference(out)
    rgb,labels,owner,profile=make_pano(A,pos,q,faces,items,model,out)
    dates,tracks=exposure(A,pos,q,faces,items,model,out)
    draw_tracks(rgb,tracks,out)
    g=model["geocoordinates"]["scan"]
    horizon={"source":"pano","origin":{"lat":float(g["lat"]),"lon":float(g["long"]),"height_above_ground_m":float(pos[2])},
             "azimuth_deg":list(range(0,360,2)),"elevation_deg":profile,
             "notes":"No confidently established sky pixels: null means no verified sky observation, NOT zero-elevation horizon. Window blinds and bright gaps in porch foliage are unknown. Height assumes model floor z=0 is local ground; unverified. Azimuth bins [a,a+2); lowest sky elevation; 0.15 deg image grid. A single lower envelope cannot represent porch overhangs or disconnected gaps."}
    (out/"horizon_profile.json").write_text(json.dumps(horizon,indent=2))
    mask_stats={name:int((labels==i).sum()) for i,name in enumerate(("excluded","sky","not_sky","unknown"))}
    result={"model_id":sid,"pano_id":PANO,"scan_position_model_m":pos.tolist(),"heading":head,
            "projection_verified":projection["orientation_verified"],"solar_reference":reference,
            "mask_pixel_counts":mask_stats,"dates":dates,
            "one_metre_offset_geometry_only":offset_assessment(A,pos,items,model),
            "method":{"timezone":"America/Chicago with date-specific DST","solar_model":"NOAA Meeus with NOAA apparent-elevation correction",
                      "sampling":"Left edge every 5 minutes from -0.833 deg geometric sunrise; last interval clipped at sunset. Whole interval inherits sampled state. Sun centre only; weather ignored.",
                      "mask":"Conservative manual opening assessment plus color/brightness rules. All blind-covered windows unknown; closed-door opaque panels blocked, glass unknown with 10px boundary margin. In primary pano only, green foliage, dark branches and conservatively inset porch roof are not sky. Bright/blue gaps unknown. No confidently verified open-sky patch was found; no sky guessed.",
                      "minutes_direct_sun":"Null if any sampled direction is unknown; otherwise number for this conservative static-view model. Lower bound counts verified sky only, upper bound also unknown. Neither is observed irradiance.",
                      "window_occlusion":"Ray cast through 41 graph wall segments and all openings. Wall height assumed 2.6m. Exterior opening identities visually verified; door panel/glazing ambiguity remains unknown.",
                      "scope":"Sun reaching this scan point at camera height, not whole-room sun hours."},
            "assumed_or_failed":["No confidently verified sky pixels; exact sun durations unresolved where unknown_possible_minutes > 0. All-null horizon is an observation limit.",
                "No independent true-north survey; residual +/-3 deg assumed. Roof footprint match is indirect evidence.",
                "Static blind positions, glazing transmission, vegetation/season, weather and solar-disc edge effects are not measured.",
                "Model floor z=0 is assumed local ground. Graph rectangles and wall height are approximate.",
                "Pano images are not simultaneous: the back door is open in primary and closed in second; neither state predicts later occupancy."],
            "scan_point_offset":{"actual_nearest_visible_window_plane_distance_m":round(min(abs(float((pos[:2]-np.array(w['center']))@np.array([-w['tangent'][1],w['tangent'][0]]))) for w in items if w['id'] in VISIBLE_OPENINGS and w['type']=='window'),3),
                "one_metre_depth_example":{"window_width_m":.944938,"window_height_m":1.398,"camera_height_above_floor_m":float(pos[2]),
                    "horizontal_aperture_deg_at_1m":round(float(np.rad2deg(2*np.arctan(.944938/2))),2),
                    "horizontal_aperture_deg_at_2m":round(float(np.rad2deg(2*np.arctan(.944938/4))),2),
                    "top_elevation_deg_at_1m":round(float(np.rad2deg(np.arctan((2.039-pos[2])/1))),2),
                    "top_elevation_deg_at_2m":round(float(np.rad2deg(np.arctan((2.039-pos[2])/2))),2),
                    "interpretation":"Illustrative nearest kitchen window, centred point: moving from 1m to 2m inward reduces horizontal aperture from 50.6 to 26.6 deg and top elevation from 29.8 to 16.0 deg. This may alter sun access by tens of minutes to hours; primary scan is off-centre, only 0.303m from window plane. Whole-room extrapolation is invalid."}}}
    (out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({"pano_id":PANO,"projection_verified":True,"solar_reference_passed":reference["passed"],"heading":head,
                      "mask_pixel_counts":mask_stats,"scan_point_totals":{date:v["scan_point_total"] for date,v in dates.items()}},indent=2))
    return result


def second_inputs(sid,out):
    path=out/"pano_meta.json"
    meta=json.loads(path.read_text()) if path.exists() else None
    if meta is None or any(not (out/f"face{f}.jpg").exists() for f in range(6)):
        panos=gql('{model(id:"%s"){panoLocations{id position{x y z} rotation{x y z w} skybox(resolution:"2k"){children}}}}'%sid)["model"]["panoLocations"]
        remote=next(p for p in panos if p["id"]==SECOND_PANO)
        meta={k:v for k,v in remote.items() if k!="skybox"}
        for f,url in enumerate(remote["skybox"]["children"]):
            target=out/f"face{f}.jpg"
            if not target.exists():
                try:
                    with urllib.request.urlopen(url,timeout=60) as response: target.write_bytes(response.read())
                except urllib.error.HTTPError as error:
                    raise RuntimeError(f"Second pano HTTP {error.code}; signed URL suppressed") from None
    if "geo_scan" not in meta:
        p=meta["position"]
        query='{model(id:"%s"){geocoordinates{geoLocationOf(modelLocation:{x:%s,y:%s,z:%s}){lat long}}}}'%(sid,p['x'],p['y'],p['z'])
        meta["geo_scan"]=gql(query)["model"]["geocoordinates"]["geoLocationOf"]
    path.write_text(json.dumps(meta,indent=2))
    faces=[Image.open(out/f"face{f}.jpg").convert("RGB") for f in range(6)]
    return meta,faces


def draw_heading_fit(fit,out):
    p=fit["_plot"]
    ring,inner,c=np.array(p["ring"]),np.array(p["inner"]),np.array(p["centre"])
    th=np.deg2rad(p["th"])
    R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    corrected=(inner-c)@R.T+c+np.array(p["d"])
    allpoints=np.concatenate([ring,inner,corrected])
    mid=(allpoints.min(0)+allpoints.max(0))/2
    scale=700/max(np.ptp(allpoints,axis=0))
    def pixels(points): return [(450+(x-mid[0])*scale,470-(y-mid[1])*scale) for x,y in points]
    image=Image.new("RGB",(900,900),"white");draw=ImageDraw.Draw(image)
    draw.polygon(pixels(ring),fill="#dddddd",outline="black",width=3)
    for points,color in [(inner,"#df6f00"),(corrected,"#0071bc")]:
        for x,y in pixels(points): draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
    font=ImageFont.load_default(size=22)
    draw.text((20,20),"ENU footprint fit: north up, east right",fill="black",font=font)
    draw.text((20,50),"Gray Vexcel roof | orange delivered interior | blue corrected",fill="black",font=font)
    draw.text((20,80),f"Rotation {fit['fix_rot_deg']:+.1f} deg CCW; IoU {fit['iou_before']} -> {fit['iou_after']}",fill="black",font=font)
    draw.text((20,845),"Indirect heading evidence; residual +/-3 deg ASSUMED",fill="black",font=font)
    image.save(out/"heading_fit.jpg",quality=95)


def main():
    global PANO
    sys.dont_write_bytecode=True
    sid=sys.argv[1] if len(sys.argv)>1 else "khbuN7q853q"
    if sid!="khbuN7q853q": raise SystemExit("This calibrated experiment supports only khbuN7q853q")
    out=ROOT/"data"/"sun_check"/sid
    out.mkdir(parents=True,exist_ok=True)
    model=prepare(sid,out)
    fit=heading_fit(sid,out)
    draw_heading_fit(fit,out)
    inputs=ROOT/"data"/"matterport"/sid
    meta=json.loads((inputs/"pano_meta.json").read_text())["pano"]
    faces=[Image.open(inputs/f"pano_td86n1_face{f}.jpg").convert("RGB") for f in range(6)]
    PANO=PRIMARY_PANO
    primary=run_scan(sid,out,model,meta,faces,fit)
    second=out/f"second_{SECOND_PANO}"
    second.mkdir(exist_ok=True)
    meta2,faces2=second_inputs(sid,second)
    model2=json.loads(json.dumps(model))
    model2["geocoordinates"]["scan"]=meta2["geo_scan"]
    PANO=SECOND_PANO
    secondary=run_scan(sid,second,model2,meta2,faces2,fit)
    primary["second_pano"]={"pano_id":SECOND_PANO,"result_path":str((second/"result.json").relative_to(ROOT)),
                            "scan_point_totals":{date:v["scan_point_total"] for date,v in secondary["dates"].items()}}
    (out/"result.json").write_text(json.dumps(primary,indent=2,allow_nan=False))
    paths=[Path(__file__),out/"geometry.json",out/"heading_fit.json",inputs/"pano_meta.json",second/"pano_meta.json"]
    paths += [inputs/f"pano_td86n1_face{f}.jpg" for f in range(6)]+[second/f"face{f}.jpg" for f in range(6)]
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/"input_sha256.json").write_text(json.dumps(provenance,indent=2))


if __name__ == "__main__":
    main()
