"""Buyer wording, with explicit evidence for every rounded or converted number."""
from .tools import DATES, fact


def display(evidence, original, value, unit, operation):
    evidence.append(fact(original['fact']+'_display',value,unit,original['source_file'],
                         derived_from=dict(fact=original['fact'],value=original['value'],operation=operation)))
    return f'{value:g}'


def find(evidence, name):
    return next((e for e in evidence if e['fact']==name and e['value'] is not None),None)


def ratio_words(evidence, scope):
    key='scope' if find(evidence,'scope_window_floor_ratio') else 'home'
    ratio=find(evidence,key+'_window_floor_ratio')
    if not ratio:
        floor=find(evidence,'home_floor_area');windows=find(evidence,'home_window_area')
        if floor and windows:
            area=display(evidence,floor,round(floor['value'],1),'m2','round(value, 1)')
            glass=display(evidence,windows,round(windows['value'],1),'m2','round(value, 1)')
            return f'The scan records {glass} square metres of window rectangles over {area} square metres of indoor floor; missing window measurements prevent a reliable window share or comparison with other homes'
        return 'Window and floor measurements are missing for the named space'
    share=display(evidence,ratio,round(ratio['value']*100,1),'percent','round(value * 100, 1)')
    label=('the '+scope.lower()) if key=='scope' else 'the scanned indoor rooms'
    phrase=f'Window area is about {share}% of floor area in {label}'
    rank=find(evidence,key+'_ratio_cohort_percentile')
    if rank:
        position=display(evidence,rank,round(rank['value'],1),'percentile','round(value, 1)')
        phrase+=f', around percentile {position} among homes with reliable aerial placement'
    return phrase


def sun_words(evidence, scope, date):
    rows=[e for e in evidence if e['fact']=='direct_sun_minutes' and e.get('date')==date]
    if not rows:return None
    best=max(rows,key=lambda e:e['value'])
    duration=(f"{best['value']:g} minutes" if 0<best['value']<60 else display(evidence,best,round(best['value']/60,1),'hours','round(value / 60, 1)')+' hours')
    day=int(date[-2:])
    evidence.append(fact('sun_date_day',day,'day of month',best['source_file'],derived_from=dict(date=date,operation='day of month')))
    month={'03':'March','12':'December','06':'June','09':'September'}[date[5:7]]
    return f'The sunniest measured room in the {scope.lower()} gets about {duration} of direct sun on {day} {month} by geometry'


def explain(claim, tool, evidence, cars=None):
    topic=claim['topic'];verdict=claim['verdict'];scope=claim['scope']
    if topic=='light':
        if tool['registration_poor'] is True:
            return ratio_words(evidence,scope)+'; sun timing could not be computed reliably because the scan could not be placed on the aerial image.'
        if not tool['available'] or tool['registration_poor'] is not False:
            return {'missing_sun_output':'The sun calculation file is missing for this home.',
                    'no_room_in_scope':'The scan has no measured room matching the space named in this light claim.'}.get(tool.get('unavailable_reason'),'The scan’s aerial placement is unknown, so reliable sun timing is unavailable.')
        march=sun_words(evidence,scope,DATES[1])
        words=ratio_words(evidence,scope)
        if march:
            words=march+'; '+words[0].lower()+words[1:]
        if verdict=='contradicted':
            winter=sun_words(evidence,scope,DATES[0])
            if winter:words+='; '+winter[0].lower()+winter[1:]
            words+='; these small windows and lack of direct sun weigh against the description'
        elif verdict=='unclear':words+='; these measurements do not settle how bright the space feels'
        if any(c.startswith('Trees are not') for c in tool['caveats']):words+='; trees are not counted'
        return words+'.'
    if topic=='ceilings':
        heights=[e for e in evidence if e.get('unit')=='m' and isinstance(e['value'],(int,float))]
        if not heights:return 'The scan has no usable ceiling height in a main room.'
        highest=max(heights,key=lambda e:e['value'])
        metres=display(evidence,highest,round(highest['value'],1),'m','round(value, 1)')
        inches=round(highest['value']/0.0254)
        feet=display(evidence,highest,inches//12,'ft','round(value / 0.0254) // 12')
        remaining=display(evidence,highest,inches%12,'in','round(value / 0.0254) % 12')
        stated=find(evidence,'listing_stated_ceiling_height')
        description=f"{stated['value']:g} ft" if stated else next((w for w in ('vaulted','cathedral','soaring','high','tall') if w in claim['text'].lower()),'a ceiling spanning multiple storeys')
        return f'Highest measured ceiling in a main room is about {metres} m ({feet} ft {remaining} in); the listing says {description}.'
    if topic=='open_plan':
        shared=find(evidence,'shared_kitchen_living_classifications')
        if shared and shared['value']:return 'The scan labels the kitchen and living or dining area as the same space.'
        widths=[e for e in evidence if e['fact'] in ('kitchen_connection_width','kitchen_doorway_width') and e['value'] is not None]
        if verdict=='supported':
            divider=next((e for e in evidence if e.get('type')=='divider'),None)
            if divider:
                if divider['value'] is not None:
                    width=display(evidence,divider,round(divider['value'],2),'m','round(value, 2)')
                    return f'The scan records an open divider about {width} m wide between the kitchen and a living or dining space.'
                return 'The scan records an open divider between the kitchen and a living or dining space.'
        if widths:
            widest=max(widths,key=lambda e:e['value'])
            width=display(evidence,widest,round(widest['value'],2),'m','round(value, 2)')
            if verdict=='contradicted':return f'Every recorded kitchen connection is a doorway, and the widest is {width} m across; the kitchen walls are fully recorded.'
            if verdict=='supported':return f'The kitchen connects to a living or dining space through an opening about {width} m wide.'
            return f'The widest recorded kitchen-to-living connection is {width} m; missing wall or opening details prevent a firm open-plan check.'
        return 'Kitchen walls and openings are not recorded well enough to check whether the living spaces are joined.'
    if topic=='garage':
        area=tool['area_m2']
        if area is None:return 'The scan contains no measured room labelled Garage.'
        source=evidence[0]['source_file']
        total=fact('total_garage_area',area,'m2',source,derived_from=[dict(fact=e['fact'],value=e['value']) for e in evidence])
        evidence.append(total)
        area_text=display(evidence,total,round(area,1),'m2','round(value, 1)')
        evidence.append(fact('listing_garage_cars',cars,'cars','data/costar-kit/data/residential-listings.jsonl' if claim['source']=='remarks' else 'data/costar-kit/data/residential-listing-features.jsonl',quote=claim['text']))
        tail={'supported':'enough floor area for that approximate capacity, subject to shape and parking arrangement',
              'contradicted':'too little floor area for that approximate capacity',
              'unclear':'shape and parking arrangement could decide whether the cars fit',
              'not_checkable':'this parking capacity is beyond the area checks available here'}[verdict]
        return f'The garage measures about {area_text} square metres and the listing advertises space for {cars} cars; {tail}.'
    if topic in ('pool','solar'):
        name='a pool' if topic=='pool' else 'solar panels'
        if verdict=='not_checkable':return f'No usable aerial check of {name} is available on a confirmed matching parcel.'
        words=f'The aerial check found {name} on this parcel' if verdict=='supported' else f'The aerial check did not find {name}; that does not establish absence'
        age=find(evidence,'image_age_at_listing_days')
        if age:
            days=display(evidence,age,abs(age['value']),'days','abs(value)')
            words+=f"; the image was taken {days} days {'before' if age['value']>=0 else 'after'} the listing"
        return words+'.'
    raise ValueError('No explanation template for '+topic)
