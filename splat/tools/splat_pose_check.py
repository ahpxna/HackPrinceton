"""CPU, image-based convention checks for S0; reads the offline pack only."""
import argparse
import itertools
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PACK = Path(os.environ.get('HP_SPLAT_PACK', str(Path.home() / 'hp_splat_pack')))
BASES = np.array([[[0,0,1],[0,-1,0],[1,0,0]], [[1,0,0],[0,-1,0],[0,0,-1]],
                  [[0,-1,0],[-1,0,0],[0,0,-1]], [[-1,0,0],[0,1,0],[0,0,-1]],
                  [[0,1,0],[1,0,0],[0,0,-1]], [[0,0,-1],[0,-1,0],[-1,0,0]]], float)


def rotation(q):
    x,y,z,w = np.array([q[k] for k in 'xyzw']) / np.linalg.norm(list(q.values()))
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def face_image(home, pid, face, width=1024):
    with Image.open(home / 'faces_2k' / pid / ('face%d.jpg' % face)) as im:
        return np.array(im.convert('RGB').resize((width, width), Image.Resampling.LANCZOS))


def sample(images, face, rays):
    b = BASES[face]
    width = images[face].shape[0]
    u = rays @ b[1] / (rays @ b[0])
    v = rays @ b[2] / (rays @ b[0])
    return cv2.remap(images[face], ((u+1)*width/2-.5).astype('float32'),
                     ((v+1)*width/2-.5).astype('float32'), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def seam_check(home, pano):
    images = [face_image(home, pano['id'], f) for f in range(6)]
    rows, checks = [], []
    for f,g in itertools.combinations(range(6), 2):
        c1,c2 = BASES[f,0],BASES[g,0]
        if abs(c1@c2) > .1:
            continue
        tangent = np.cross(c1,c2)
        along = np.linspace(-.94,.94,768)
        rays = c1+c2+along[:,None]*tangent
        a = sample(images,f,rays[:,None,:])[:,0,:]
        b = sample(images,g,rays[:,None,:])[:,0,:]
        error = np.mean(abs(a.astype(float)-b.astype(float)))
        reversed_error = np.mean(abs(a.astype(float)-b[::-1].astype(float)))
        checks.append({'faces':[f,g], 'mean_absolute_RGB_difference':float(error),
                       'reversed_edge_difference':float(reversed_error)})
        # Show both sides as a continuous geometric strip, 32 pixels on each side.
        left = np.linspace(.94,1,32)
        right = np.linspace(1,.94,32)
        ra = c1+left[None,:,None]*c2+along[:,None,None]*tangent
        rb = right[None,:,None]*c1+c2+along[:,None,None]*tangent
        strip = np.concatenate([sample(images,f,ra), sample(images,g,rb)],axis=1)
        strip = np.transpose(strip,(1,0,2))
        canvas = Image.new('RGB',(800,92),'white')
        canvas.paste(Image.fromarray(strip),(16,25))
        ImageDraw.Draw(canvas).text((16,5),f'faces {f}/{g}, RGB MAE {error:.2f}, reversed {reversed_error:.2f}', fill='black')
        rows.append(canvas)
    proof = Image.new('RGB',(800,45+92*len(rows)),'white')
    ImageDraw.Draw(proof).text((16,12),f'Seam check {pano["id"]}: boundary is at strip centre',fill='black')
    for i,row in enumerate(rows):
        proof.paste(row,(0,45+92*i))
    proof.save(home / 'proof' / ('pose_check_seam_' + pano['id'] + '.jpg'),quality=92)
    return {'panorama':pano['id'],'edges':checks}


def skew(t):
    x,y,z=t
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def pole_calibration(home, panos):
    """Rank all eight image-plane orientations using independent adjacent edges."""
    original = BASES.copy()
    reports = {}
    try:
        for pole in (0, 5):
            scores = []
            for swap in (False, True):
                for a, b in itertools.product((-1, 1), repeat=2):
                    axes = original[pole, [2, 1] if swap else [1, 2]].copy()
                    BASES[pole, 1:] = axes * np.array([a, b])[:, None]
                    errors = []
                    for pano in panos:
                        images = [face_image(home, pano['id'], f, 512) for f in range(6)]
                        for side in (1, 2, 3, 4):
                            c1, c2 = BASES[pole, 0], BASES[side, 0]
                            rays = c1+c2+np.linspace(-.94,.94,384)[:,None]*np.cross(c1,c2)
                            x = sample(images,pole,rays[:,None,:]).astype(float)
                            y = sample(images,side,rays[:,None,:]).astype(float)
                            errors.append(float(np.mean(abs(x-y))))
                    scores.append({'swap_right_down':swap,'right_sign':a,'down_sign':b,
                                   'mean_edge_RGB_MAE':float(np.mean(errors)),
                                   'basis':BASES[pole].tolist()})
            reports[str(pole)] = sorted(scores,key=lambda s:s['mean_edge_RGB_MAE'])
            BASES[:] = original
    finally:
        BASES[:] = original
    return {'method':'All 8 signed/swap orientations per pole; 4 side edges on each of 3 panos; 512px sampling. Lowest score wins.',
            'poles':reports}


def alternatives():
    result = {'corrected':np.eye(3)}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            matrix=np.eye(3)[list(perm)]*np.array(signs)[:,None]
            if np.linalg.det(matrix)>.5 and not np.array_equal(matrix,np.eye(3)):
                result['local_axes_' + ''.join(map(str,perm)) + '_' + ''.join('p' if s>0 else 'm' for s in signs)] = matrix
    return result


def spherical_filter(rows, width):
    """Image-only eight-point RANSAC across all cube faces; no API poses."""
    rays=[[],[]]
    for f,g,x,y,_,_ in rows:
        for index,(face,pixels) in enumerate(((f,x),(g,y))):
            uv=(pixels+.5)*2/width-1
            ray=BASES[face,0]+uv[:,0,None]*BASES[face,1]+uv[:,1,None]*BASES[face,2]
            rays[index].append(ray/np.linalg.norm(ray,axis=1)[:,None])
    if not rows:
        return []
    a,b=[np.concatenate(r) for r in rays]
    if len(a)<16:
        return []
    design=(b[:,:,None]*a[:,None,:]).reshape(-1,9)
    rng=np.random.default_rng(42)
    def fit(indices):
        _,_,v=np.linalg.svd(design[indices],full_matrices=True)
        u,s,v=np.linalg.svd(v[-1].reshape(3,3))
        s[-1]=0
        return (u*s)@v
    def residual(e):
        ea=a@e.T; eb=b@e
        dot=np.sum(b*ea,axis=1)
        denom=(ea*ea).sum(1)+(eb*eb).sum(1)-2*dot**2
        return abs(dot)/np.sqrt(np.maximum(denom,1e-15))
    best=np.zeros(len(a),bool)
    for _ in range(4000):
        e=fit(rng.choice(len(a),8,replace=False))
        keep=residual(e)<.003
        if keep.sum()>best.sum():
            best=keep
    for _ in range(3):
        if best.sum()<16:
            return []
        best=residual(fit(np.flatnonzero(best)))<.003
    filtered=[]; offset=0
    for f,g,x,y,im1,im2 in rows:
        keep=best[offset:offset+len(x)];offset+=len(x)
        if keep.sum():
            filtered.append((f,g,x[keep],y[keep],im1,im2))
    return filtered


def epipolar(home, p, q):
    width = 1024
    sift=cv2.SIFT_create(nfeatures=3500)
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=100))
    correspondences=[]
    face_reports=[]
    features=[]
    for pano in (p,q):
        group=[]
        for face in range(6):
            image=face_image(home,pano['id'],face)
            keypoints,descriptors=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
            group.append((image,keypoints,descriptors))
        features.append(group)
    # Global, mutual ratio matching avoids repeated sky/ceiling patterns matching
    # independently to several faces of the other panorama.
    descriptors=[]; locations=[]
    for group in features:
        descriptors.append(np.concatenate([d for _,_,d in group if d is not None]))
        locations.append([(f,k.pt) for f,(_,keys,d) in enumerate(group) if d is not None for k in keys])
    forward=matcher.knnMatch(descriptors[0],descriptors[1],k=2)
    reverse=matcher.knnMatch(descriptors[1],descriptors[0],k=2)
    backward={m.queryIdx:m.trainIdx for pair in reverse if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance}
    groups={}
    for pair in forward:
        if len(pair)!=2:
            continue
        m,n=pair
        if m.distance>=.7*n.distance or backward.get(m.trainIdx)!=m.queryIdx:
            continue
        f,x=locations[0][m.queryIdx];g,y=locations[1][m.trainIdx]
        groups.setdefault((f,g),[]).append((x,y))
    for (face,other),points in groups.items():
        x,y=np.array(points).transpose(1,0,2)
        correspondences.append((face,other,x,y,features[0][face][0],features[1][other][0]))
    raw_count=sum(len(r[2]) for r in correspondences)
    correspondences=spherical_filter(correspondences,width)
    face_reports=[{'faces':[f,g],'spherical_image_only_inliers':len(x)} for f,g,x,_,_,_ in correspondences]
    r1,r2=rotation(p['rotation']),rotation(q['rotation'])
    t=np.array([q['position'][k]-p['position'][k] for k in 'xyz'])
    k=np.array([[width/2,0,(width-1)/2],[0,width/2,(width-1)/2],[0,0,1.]])
    ki=np.linalg.inv(k)
    scores={}
    matrices={}
    for name,axes in alternatives().items():
        matrices[name]=(r1@axes,r2@axes)
    matrices['inverse_quaternion']=(r1.T,r2.T)
    matrices['ignore_quaternion']=(np.eye(3),np.eye(3))
    matrices['starting_brief']=(r1,r2)
    starting_bases=BASES.copy()
    starting_bases[0]=[[0,0,1],[1,0,0],[0,1,0]]
    starting_bases[5]=[[0,0,-1],[1,0,0],[0,-1,0]]
    for name,(u,v) in matrices.items():
        errors=[]
        bases=starting_bases if name=='starting_brief' else BASES
        for face,other,x,y,_,_ in correspondences:
            # B columns are camera image-right, image-down, optical-forward.
            b=bases[face,[1,2,0]].T
            b2=bases[other,[1,2,0]].T
            f=ki.T@(v@b2).T@skew(t)@(u@b)@ki
            xh=np.c_[x,np.ones(len(x))]; yh=np.c_[y,np.ones(len(y))]
            l2=xh@f.T; l1=yh@f
            error=abs(np.sum(yh*l2,axis=1))/np.sqrt((l2[:,:2]**2).sum(1)+(l1[:,:2]**2).sum(1))
            # Report at native 2048 resolution, not the feature extraction scale.
            errors.extend((error*2).tolist())
        if errors:
            scores[name]={'median_sampson_px_2k':float(np.median(errors)),
                          'p90_sampson_px_2k':float(np.percentile(errors,90)),
                          'matches':len(errors)}
    starting_score=scores.pop('starting_brief',None)
    ranked=sorted(scores,key=lambda name:scores[name]['median_sampson_px_2k'])
    canvas=Image.new('RGB',(1024,60+280*min(3,len(correspondences))),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((10,5),f'{p["id"]} -> {q["id"]}',fill='black')
    if ranked:
        draw.text((10,24),f'2k Sampson median: corrected {scores["corrected"]["median_sampson_px_2k"]:.3f}px; best alternative {ranked[1] if ranked[0]=="corrected" else ranked[0]}',fill='black')
    for i,(face,other,x,y,a,b) in enumerate(sorted(correspondences,key=lambda row:-len(row[2]))[:3]):
        # A horizontal band from each face, preserving 2-D match positions.
        canvas.paste(Image.fromarray(a).resize((256,256)),(0,60+i*280))
        canvas.paste(Image.fromarray(b).resize((256,256)),(512,60+i*280))
        draw.text((270,65+i*280),f'faces {face}/{other}: {len(x)} F-inliers',fill='black')
        for n in np.linspace(0,len(x)-1,min(18,len(x)),dtype=int):
            start=tuple(x[n]/4+np.array([0,60+i*280]))
            end=tuple(y[n]/4+np.array([512,60+i*280]))
            colour=(int(60+n*47%160),int(40+n*79%160),int(40+n*31%160))
            draw.line([start,end],fill=colour,width=1)
    filename='pose_check_pair_'+p['id']+'_'+q['id']+'.jpg'
    canvas.save(home/'proof'/filename,quality=92)
    return {'pair':[p['id'],q['id']], 'baseline_m':float(np.linalg.norm(t)),
            'mutual_ratio_matches_before_ransac':raw_count,
            'faces':face_reports,'scores':scores,'starting_brief_score':starting_score,
            'ranking':ranked,'proof':filename}


def main(sid):
    cv2.setRNGSeed(0)
    cv2.setNumThreads(4)
    home=PACK/sid
    panos=json.loads((home/'poses.json').read_text())['panoramas']
    selected=[panos[i] for i in (0,len(panos)//3,2*len(panos)//3)]
    (home/'proof/pole_calibration.json').write_text(json.dumps(pole_calibration(home,selected),indent=2)+'\n')
    seams=[seam_check(home,p) for p in selected]
    positions=np.array([[p['position'][k] for k in 'xyz'] for p in panos])
    pairs=[]
    rejected=[]
    used=set()
    for p in selected:
        i=panos.index(p)
        distances=np.linalg.norm(positions-positions[i],axis=1)
        j=next(int(j) for j in np.argsort(distances) if distances[j]>.4 and tuple(sorted((i,int(j)))) not in used)
        used.add(tuple(sorted((i,j))))
        result=epipolar(home,p,panos[j])
        if not result['ranking']:
            rejected.append(result)
            # No usable image-only consensus: try the next closest capture,
            # keeping the failed attempt and selection rule in the report.
            j=next(int(j) for j in np.argsort(distances) if distances[j]>.4 and tuple(sorted((i,int(j)))) not in used)
            used.add(tuple(sorted((i,j))))
            result=epipolar(home,p,panos[j])
        anchor_i=i
        for offset in range(1,9):
            if result['ranking'] and result['scores'][result['ranking'][0]]['matches']>=30:
                break
            rejected.append(result)
            i=(anchor_i+offset)%len(panos)
            p=panos[i]
            distances=np.linalg.norm(positions-positions[i],axis=1)
            j=next(int(j) for j in np.argsort(distances) if distances[j]>.4 and tuple(sorted((i,int(j)))) not in used)
            used.add(tuple(sorted((i,j))))
            result=epipolar(home,p,panos[j])
        pairs.append(result)
        print(json.dumps({'home':sid,'pair':pairs[-1]['pair'],'top':pairs[-1]['ranking'][:2],
                          'corrected':pairs[-1]['scores'].get('corrected')}),flush=True)
    confirmed=all(p['ranking'] and p['ranking'][0]=='corrected' and
                  p['scores']['corrected']['matches']>=30 and
                  p['scores']['corrected']['median_sampson_px_2k']<4 and
                  p['scores'][p['ranking'][1]]['median_sampson_px_2k']>3*p['scores']['corrected']['median_sampson_px_2k'] for p in pairs)
    result={'model':sid,'confirmed_epipolar':confirmed,'seams':seams,'epipolar_pairs':pairs,'rejected_pairs':rejected,
            'candidate_face_bases_center_right_down':BASES.tolist(),
            'pair_selection':'Anchors at API indices 0, N//3, 2*N//3; nearest capture >0.4m. If no image-only consensus, retain failed attempt and try next nearest once, then up to 8 next API anchors until >=30 image-only inliers. No selection using pose errors.',
            'method':'SIFT global mutual ratio .70; independent spherical eight-point RANSAC across all faces, angular threshold .003 rad, 4000 deterministic iterations, no API pose used for filtering. Pose Sampson error scaled to native 2048. 24 proper local cube-axis rotations plus inverse/ignored quaternion. Seam checks sample all 12 shared edges, compared to reversed alignment.',
            'training_gate':'requires epipolar confirmation AND visual inspection of seams and mesh projection'}
    (home/'proof/pose_metrics.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('sid');args=parser.parse_args()
    main(args.sid)
