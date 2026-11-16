"""Generate draft notes only where measured evidence has a room/window scan anchor."""
from agent.tools import store


def draft_annotations(sid, document):
    s=store();sun=s.suns.get(sid,{});notes=[]
    for i,claim in enumerate(document['claims']):
        roomids={e.get('room_id') for e in claim.get('evidence',[]) if e.get('room_id')}
        windows=[w for w in sun.get('windows',[]) if w.get('type')=='window' and w.get('room_id') in roomids and len(w.get('center',[]))==2 and w.get('lower_z_m') is not None]
        if not windows:continue
        w=windows[0]
        notes.append(dict(id=f'{sid}:c{i}',x=w['center'][0],y=w['center'][1],z=w['lower_z_m']+w['height']/2,
            room_id=w['room_id'],window_id=w['id'],title=claim['topic'].replace('_',' ').capitalize(),
            text=claim['explanation'],verdict=claim['verdict'],evidence_ids=[e['id'] for e in claim['evidence']],
            anchor_source=f'data/sun/{sid}/sun.json',anchor_kind='Measured window centre in the evidenced room; not a location of the claimed feature.'))
    return dict(sid=sid,coordinate_frame='Matterport scan metres; x/y horizontal, z vertical',published=False,annotations=notes,
                omitted_claims=len(document['claims'])-len(notes),caveats=['Claims without an evidenced room and a recorded window anchor are omitted. Placement caveats remain in each dossier.'])
