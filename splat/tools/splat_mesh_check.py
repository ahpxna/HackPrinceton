"""Offline sparse-OBJ depth reprojection, no synthetic colour or GPU required."""
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

spec=importlib.util.spec_from_file_location('pose',Path(__file__).with_name('splat_pose_check.py'))
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
W=256


def read_obj(path):
    vertices=[]; triangles=[]
    for line in path.open():
        words=line.split()
        if not words:
            continue
        if words[0]=='v':
            vertices.append([float(v) for v in words[1:4]])
        elif words[0]=='f':
            ids=[int(v.split('/')[0]) for v in words[1:]]
            ids=[i-1 if i>0 else len(vertices)+i for i in ids]
            triangles.extend([[ids[0],ids[i],ids[i+1]] for i in range(1,len(ids)-1)])
    return np.array(vertices),np.array(triangles)


def depths(vertices,triangles,pano):
    centre=np.array([pano['position'][k] for k in 'xyz'])
    local=(vertices-centre)@P.rotation(pano['rotation'])
    result=[]
    for face in range(6):
        b=P.BASES[face]
        xyz=local@b[[1,2,0]].T
        z=xyz[:,2]
        xy=xyz[:,:2]/np.maximum(z[:,None],.0001)*W/2+(W-1)/2
        depth=np.full((W,W),np.inf)
        for ids in triangles:
            if np.min(z[ids])<.05:
                continue
            uv=xy[ids]
            lo=np.maximum(np.floor(uv.min(0)).astype(int),0)
            hi=np.minimum(np.ceil(uv.max(0)).astype(int),W-1)
            if np.any(hi<lo):
                continue
            a,bp,c=uv
            denom=(bp[1]-c[1])*(a[0]-c[0])+(c[0]-bp[0])*(a[1]-c[1])
            if abs(denom)<1e-9:
                continue
            xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
            wa=((bp[1]-c[1])*(xx-c[0])+(c[0]-bp[0])*(yy-c[1]))/denom
            wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denom
            wc=1-wa-wb
            inv=wa/z[ids[0]]+wb/z[ids[1]]+wc/z[ids[2]]
            candidate=np.where((wa>=0)&(wb>=0)&(wc>=0)&(inv>0),1/np.maximum(inv,1e-12),np.inf)
            region=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            np.minimum(region,candidate,out=region)
        result.append(depth)
    return result


def main(sid):
    home=P.PACK/sid
    panos=json.loads((home/'poses.json').read_text())['panoramas']
    byid={p['id']:p for p in panos}
    pairs=json.loads((home/'proof/pose_metrics.json').read_text())['epipolar_pairs']
    vertices,triangles=read_obj(home/'mesh/mesh_500k.obj')
    reports=[]
    for pair in pairs:
        p,q=[byid[pid] for pid in pair['pair']]
        d1,d2=depths(vertices,triangles,p),depths(vertices,triangles,q)
        images=[P.face_image(home,q['id'],f,W) for f in range(6)]
        canvas=Image.new('RGB',(W*3,40+(W+25)*6),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((8,5),'Capture | real neighbour pixels warped via sparse mesh | absolute RGB difference',fill='black')
        errors=[];visible_count=0;mesh_count=0
        for face in range(6):
            yy,xx=np.mgrid[:W,:W]
            uv=np.stack([(xx+.5)*2/W-1,(yy+.5)*2/W-1],axis=-1)
            ray=P.BASES[face,0]+uv[:,:,0,None]*P.BASES[face,1]+uv[:,:,1,None]*P.BASES[face,2]
            valid=np.isfinite(d1[face])
            point=(ray*np.where(valid,d1[face],0)[:,:,None])@P.rotation(p['rotation']).T
            point+=np.array([p['position'][k]-q['position'][k] for k in 'xyz'])
            local=point@P.rotation(q['rotation'])
            facing=np.argmax(local@P.BASES[:,0].T,axis=-1)
            warped=np.zeros((W,W,3),np.uint8)
            visible=np.zeros((W,W),bool)
            for other in range(6):
                b=P.BASES[other]
                forward=local@b[0]
                mx=((local@b[1])/np.maximum(forward,1e-9)+1)*W/2-.5
                my=((local@b[2])/np.maximum(forward,1e-9)+1)*W/2-.5
                mx,my=mx.astype('float32'),my.astype('float32')
                rgba=cv2.remap(images[other],mx,my,cv2.INTER_LINEAR)
                # Nearest depth sample avoids interpolating through mesh gaps.
                nx,ny=np.clip(np.rint(mx),0,W-1).astype(int),np.clip(np.rint(my),0,W-1).astype(int)
                expected=d2[other][ny,nx]
                keep=valid&(facing==other)&(forward>.05)&np.isfinite(expected)
                difference=np.full_like(forward,np.inf)
                np.subtract(forward,expected,out=difference,where=np.isfinite(expected))
                keep &= abs(difference)<np.maximum(.05,.02*forward)
                warped[keep]=rgba[keep]
                visible|=keep
            current=P.face_image(home,p['id'],face,W)
            diff=abs(current.astype(float)-warped.astype(float))
            errors.extend(np.mean(diff[visible],axis=1).tolist())
            visible_count+=int(visible.sum());mesh_count+=int(valid.sum())
            error_image=np.where(visible[:,:,None],np.minimum(diff*3,255),0).astype('uint8')
            y=40+(W+25)*face
            canvas.paste(Image.fromarray(current),(0,y));canvas.paste(Image.fromarray(warped),(W,y))
            canvas.paste(Image.fromarray(error_image),(2*W,y))
            draw.text((6,y+W+4),f'face {face}: {visible.sum()} mutually visible mesh pixels; black = excluded',fill='black')
        filename='pose_check_mesh_'+p['id']+'.jpg'
        canvas.save(home/'proof'/filename,quality=90)
        reports.append({'pair':pair['pair'],'mesh_pixels':mesh_count,'mutually_visible_pixels':visible_count,
                        'median_RGB_MAE_0_255':float(np.median(errors)) if errors else None,
                        'mean_RGB_MAE_0_255':float(np.mean(errors)) if errors else None,'proof':filename})
        print(json.dumps({'home':sid,**reports[-1]}),flush=True)
    report={'method':'CPU z-buffer of supplied simple_planes OBJ, 256px cube faces; use depth to warp real nearest-neighbour panorama colours, require both mesh depths agree within max(5cm,2%). No exposure correction. This is a mesh/pose consistency diagnostic, not independent mesh texture ground truth. Triangles crossing camera near plane are omitted.',
            'limitation':'Supplied meshes have no UV, material references or vertex colours. A true textured-mesh colour comparison is unavailable; downloaded 50k textures cannot be attached to these meshes.',
            'vertices':len(vertices),'triangles_after_fan_triangulation':len(triangles),'checks':reports}
    (home/'proof/mesh_metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main(sys.argv[1])
