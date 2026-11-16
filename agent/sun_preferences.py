"""Buyer habits and auditable room-linked window exposure fit, never indoor heat."""
import csv
import gzip
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agent.tools import ROOT, store
from agent.buyer_tools import PROFILE_LOCK, profile_path, profile_read, matches, validate_needs

BOOL_HABITS = ('early_riser', 'works_from_home', 'young_kids_nap',
               'low_cooling_bills', 'houseplants', 'avoid_hot_bedrooms')
QUESTIONS = {
    'early_riser': 'Are you an early riser who wants morning kitchen sun?',
    'works_from_home': 'Do you work from home, and which labelled room or room ID is your office?',
    'young_kids_nap': 'Do young children nap at home? Which room and hours?',
    'low_cooling_bills': 'Is avoiding summer solar exposure for lower cooling demand a priority?',
    'houseplants': 'Do you keep houseplants, and in which room?',
}


def validate_habits(habits):
    if not isinstance(habits, dict) or set(habits)-set(BOOL_HABITS)-{'work_room', 'work_room_ids'}:
        raise ValueError('Unknown habit')
    for key in BOOL_HABITS:
        if key in habits and not isinstance(habits[key], bool):
            raise ValueError('Habit flags must be explicit booleans')
    if 'work_room' in habits and (not isinstance(habits['work_room'], str) or not 0 < len(habits['work_room'].strip()) <= 100):
        raise ValueError('Name the work room')
    if 'work_room_ids' in habits:
        if not isinstance(habits['work_room_ids'], dict):
            raise ValueError('Expected home to room ID mapping')
        for sid, rid in habits['work_room_ids'].items():
            if rid not in {r['id'] for r in store().suns.get(sid, {}).get('rooms', [])}:
                raise ValueError('Unknown work room ID')
    return dict(habits)


def read_sun_habits(profile_id='default'):
    """Read approved habits and ask unanswered habit questions; missing never means false."""
    habits = profile_read(profile_id).get('habits', {})
    questions = [v for k, v in QUESTIONS.items() if k not in habits]
    if habits.get('works_from_home') and not habits.get('work_room') and not habits.get('work_room_ids'):
        questions.insert(0, QUESTIONS['works_from_home'])
    return dict(habits=habits, questions=questions, memory_path=f'/memories/{profile_id}.json')


def commit_sun_habits(habits, profile_id='default'):
    """Merge explicit habits only after framework human approval; preserve needs and feedback."""
    habits = validate_habits(habits)
    with PROFILE_LOCK:
        previous = profile_read(profile_id)
        previous['habits'] = dict(previous.get('habits', {}), **habits)
        path = profile_path(profile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(previous, indent=2))
        tmp.replace(path)
    return dict(status='saved_approved_habits', **read_sun_habits(profile_id))


def rules_for(habits):
    rules = []
    def add(name, room, start, end, season, weight):
        rules.append(dict(rule=name, room=room, local_hours=[start, end], season=season, weight=weight))
    if habits.get('early_riser'):
        add('wanted morning kitchen sun', 'kitchen', 6, 10, 'all', 1)
    if habits.get('works_from_home'):
        add('avoided office direct-sun glare risk', 'office', 9, 17, 'all', -1)
    if habits.get('young_kids_nap'):
        add('avoided bedroom sun during default nap band; confirm room and hours', 'bedroom', 12, 15, 'all', -1)
    if habits.get('avoid_hot_bedrooms') or habits.get('low_cooling_bills'):
        add('avoided summer bedroom noon/afternoon exposure', 'bedroom', 12, 18, 'summer', -1)
    if habits.get('low_cooling_bills'):
        add('avoided summer daytime solar exposure; cooling-demand proxy only', 'all', 10, 18, 'summer', -1)
    if habits.get('houseplants'):
        add('wanted winter midday living-room sun; confirm plant placement', 'living', 10, 14, 'winter', 1)
    return rules


def read_f4():
    """Optional F4 output contract; absent values remain explicitly unavailable."""
    path = ROOT/'data/facts/daylight/room_sun_f4.json'
    if not path.exists(): return {}, None
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or not isinstance(data.get('rooms'), list):
        raise ValueError('Unsupported F4 room sun schema')
    return {(r['sid'], r['room_id']): r for r in data['rooms']}, str(path.relative_to(ROOT))


def rank_sun_fit(needs: dict, profile_id: str = 'default') -> dict:
    """Rank reliable hourly-data homes by approved habits; report missing criteria and evidence."""
    needs = validate_needs(needs)
    memory = read_sun_habits(profile_id)
    f4, f4_path = read_f4()
    if f4:
        return rank_f4_fit(needs, memory, f4, f4_path)
    habits = memory['habits']; rules = rules_for(habits)
    directory = ROOT/'data/facts/daylight'
    hourly_path = directory/'hourly_featured_featured.json'
    if not hourly_path.exists(): hourly_path = directory/'hourly_featured.json'
    if not hourly_path.exists():
        return dict(status='unavailable', ranked_homes=[], reason='No hourly facts', **memory)
    records = json.loads(hourly_path.read_text())
    windows = {(r['sid'], r['window_id']): r for r in csv.DictReader((directory/'windows.csv').open())}
    by_home = {}
    for row in records:
        if row['scenario'] == 'geometry': by_home.setdefault(row['sid'], []).append(row)
    ranked = []; skipped = []
    f4, f4_path = read_f4()
    for sid, rows in sorted(by_home.items()):
        if not matches(sid, needs): continue
        if not store().placement_reliable(sid):
            skipped.append(dict(sid=sid, reason=store().placement_caveat(sid))); continue
        sun = store().suns[sid]
        tz = ZoneInfo(sun['assumptions']['timezone'])
        rooms = {r['id']: r for r in sun['rooms']}
        exposures = {}
        for row in rows:
            window = windows.get((sid, row['window_id']))
            if not window or window['room_id'] not in rooms: continue
            rid = window['room_id']; area = float(window['glazed_area_m2'])
            if f4.get((sid, rid), {}).get('status') == 'not_ranked_no_sun': continue
            for i, hour in enumerate(row['hours_utc_ending']):
                # UTC hour-ending bins: midpoint is h-0.5, including the prior UTC date for h=0.
                local = (datetime.fromisoformat(row['date']).replace(tzinfo=timezone.utc)+timedelta(hours=hour-.5)).astimezone(tz)
                key = (rid, row['date'], local.hour)
                value = exposures.setdefault(key, [0., 0.])
                fraction = row['aperture_clear_fraction'][i] if row['incidence_cosine'][i] > 0 else 0
                value[0] += area*fraction; value[1] += area
        reasons = []; unresolved = []; contributions = []
        office = None
        explicit = habits.get('work_room_ids', {}).get(sid)
        named = [r for r in rooms.values() if r['label'].casefold() == habits.get('work_room', '').casefold()]
        if habits.get('works_from_home'):
            if explicit: office = rooms[explicit]
            elif len(named) == 1: office = named[0]
            else:
                candidates = []
                for room in rooms.values():
                    if 'bedroom' not in room['label'].casefold(): continue
                    if any(word in room['label'].casefold() for word in ('primary', 'master')): continue
                    dates = sorted({d for rid, d, h in exposures if rid == room['id']})
                    samples = [exposures.get((room['id'], d, h)) for d in dates for h in range(9, 17)]
                    if dates and samples and all(v is not None and v[1] > 0 for v in samples):
                        candidates.append((sum(v[0]/v[1] for v in samples)/len(samples), room['id']))
                if candidates: office = rooms[min(candidates)[1]]
        suggested = bool(office and not explicit and len(named) != 1)
        for rule in rules:
            if rule['room'] == 'office':
                selected = [office] if office else []
                if len(selected) != 1:
                    unresolved.append('No complete office candidate; confirm which bedroom to use as an office.'); continue
            else:
                selected = [r for r in rooms.values() if rule['room'] == 'all' or rule['room'] in r['label'].casefold()]
                if rule['room'] == 'bedroom' and office:
                    selected = [r for r in selected if r['id'] != office['id']]
            room_values = []
            for room in selected:
                dates = sorted({d for rid, d, h in exposures if rid == room['id'] and (rule['season']=='all' or d[5:] == ('06-21' if rule['season']=='summer' else '12-21'))})
                for date in dates:
                    samples = [exposures.get((room['id'], date, hour)) for hour in range(*rule['local_hours'])]
                    if any(v is None or v[1] <= 0 for v in samples): continue
                    minutes = sum(v[0]/v[1]*60 for v in samples)
                    fraction = minutes/(60*(rule['local_hours'][1]-rule['local_hours'][0]))
                    room_values.append(fraction)
                    reasons.append(dict(room_id=room['id'], date=date, **rule,
                                        window_exposure_equivalent_minutes=round(minutes, 2), exposure_fraction=round(fraction, 6)))
                    reasons[-1]['room'] = room['label']
            if room_values: contributions.append(rule['weight']*sum(room_values)/len(room_values))
            else: unresolved.append('No complete hourly window coverage for '+rule['rule'])
        if rules and not contributions:
            skipped.append(dict(sid=sid, reason='not ranked: no sun coverage for active rules')); continue
        score = round(sum(contributions)/len(contributions), 6) if contributions else None
        evidence = [str(hourly_path.relative_to(ROOT)), 'data/facts/daylight/windows.csv', f'data/sun/{sid}/sun.json']
        f4_rooms = []
        for rid in rooms:
            if (sid, rid) in f4:
                item = f4[(sid, rid)]
                f4_rooms.append(dict(item, heat=item.get('heat') if item.get('heat') is not None else 'not computed yet',
                                     hot_sun_hours_jja=item.get('hot_sun_hours_jja') if item.get('hot_sun_hours_jja') is not None else 'not computed yet'))
        if f4_rooms: evidence.append(f4_path)
        ranked.append(dict(sid=sid, fit_score=score, scored_rules=len(contributions), requested_rules=len(rules),
                           reasons=reasons, unresolved=unresolved, evidence_files=evidence, timezone=str(tz),
                           office_room=dict(room_id=office['id'], label=office['label'], suggested=suggested) if office else None,
                           room_assignment_note='Second is not a source room label. Office suggestion minimizes mean work-hour window exposure; confirm this proposed room use.' if suggested else None,
                           f4_rooms=f4_rooms,
                           heat={r['room_id']: r['heat'] for r in f4_rooms} if f4_rooms else 'not computed yet',
                           hot_sun_hours={r['room_id']: r['hot_sun_hours_jja'] for r in f4_rooms} if f4_rooms else 'not computed yet'))
    ranked.sort(key=lambda r: (-r['scored_rules'], -(r['fit_score'] if r['fit_score'] is not None else -2), r['sid']))
    return dict(status='provisional_ranking' if ranked else 'unavailable', ranked_homes=ranked, skipped=skipped,
                habits=habits, questions=memory['questions'], rules=rules,
                method='Equal-weight preference heuristic: signed mean area-weighted clear-window aperture fraction during local clock bands; higher fits better. Missing rules are unscored. Compare coverage before score.',
                limitations=['Window exposure proxy, not measured room-floor sun, heat, indoor temperature or actual glare.',
                             'Geometry-only scenario consistently used; DSM and weather are not included in this score.',
                             'Four representative dates only; no yearly or full-season coverage. No inferred second-bedroom mapping.',
                             'Plant placement and nap room/hours need confirmation; default bands are adjustable heuristics.'],
                heat='see per-room F4 fields' if any(r['f4_rooms'] for r in ranked) else 'not computed yet',
                hot_sun_hours='see per-room F4 fields' if any(r['f4_rooms'] for r in ranked) else 'not computed yet')


def rank_f4_fit(needs, memory, facts, evidence_path):
    """F4 cohort ranking with explicit dimensionless preference heuristics."""
    habits = memory['habits']; rules = rules_for(habits)
    by_home = {}
    for (sid, _), room in facts.items():
        by_home.setdefault(sid, []).append(room)
    ranked = []; skipped = []
    for sid, all_rooms in sorted(by_home.items()):
        if not matches(sid, needs): continue
        if not store().placement_reliable(sid):
            skipped.append(dict(sid=sid, reason=store().placement_caveat(sid))); continue
        rooms = [r for r in all_rooms if r['status'] == 'ok']
        cache = ROOT/'data/cache/daylight_f4/computed'/f'{sid}.json.gz'
        if not cache.exists(): cache = ROOT/'data/facts/daylight/f4_computed'/f'{sid}.json.gz'
        if not rooms or not cache.exists():
            skipped.append(dict(sid=sid, reason='No ranked F4 sun rooms or hourly evidence')); continue
        with gzip.open(cache, 'rt') as stream: computed = json.load(stream)
        clocks = {r['room_id']: r['hourly_minutes'] for r in computed['room_clocks']}
        def minutes(room, start, end, months):
            hourly = clocks.get(room['room_id'])
            if hourly is None: return None
            return sum(sum(hourly[m-1][start:end]) for m in months)/len(months)
        explicit = habits.get('work_room_ids', {}).get(sid)
        named = [r for r in rooms if r['label'].casefold() == habits.get('work_room', '').casefold()]
        office = next((r for r in rooms if r['room_id'] == explicit), None)
        if not office and len(named) == 1: office = named[0]
        suggested = False
        if habits.get('works_from_home') and not office and not explicit:
            candidates = [r for r in rooms if 'bedroom' in r['label'].casefold()
                          and not any(w in r['label'].casefold() for w in ('primary', 'master'))
                          and r['room_id'] in clocks]
            if candidates:
                office = min(candidates, key=lambda r: (minutes(r, 9, 17, range(1, 13)), r['room_id']))
                suggested = True
        reasons = []; contributions = []; unresolved = []
        for rule in rules:
            if rule['room'] == 'office': selected = [office] if office else []
            else:
                selected = [r for r in rooms if rule['room'] == 'all' or rule['room'] in r['label'].casefold()]
                if rule['room'] == 'bedroom' and office:
                    selected = [r for r in selected if r['room_id'] != office['room_id']]
            values = []
            for room in selected:
                months = [6, 7, 8] if rule['season'] == 'summer' else [12, 1, 2] if rule['season'] == 'winter' else list(range(1, 13))
                exposure = minutes(room, *rule['local_hours'], months)
                if exposure is None: continue
                flags = {f['id']: f for f in room['flags']}
                flag_id = {'office':'office_workhours_glare', 'kitchen':'kitchen_morning_sun',
                           'living':'living_winter_midday_sun'}.get(rule['room'])
                components = dict(local_band_sun_fraction=exposure/(60*(rule['local_hours'][1]-rule['local_hours'][0])))
                if flag_id in flags:
                    components = dict(f4_flag_fraction=flags[flag_id]['value']/(60*(rule['local_hours'][1]-rule['local_hours'][0])))
                if rule['season'] == 'summer':
                    heat = room.get('heat'); hot = room.get('hot_sun_hours_jja')
                    if heat is None or hot is None:
                        unresolved.append(f"{room['room_id']}: heat or hot-sun hours not computed yet")
                        continue
                    # Bounded monotone scaling, not a calibrated comfort or utility model.
                    components['peak_heat_clear_scaled'] = heat['peak_w_clear']/(heat['peak_w_clear']+1500)
                    components['hot_sun_hours_fraction'] = hot/24
                    if 'bedroom_summer_afternoon_sun' in flags:
                        components['bedroom_afternoon_flag_fraction'] = flags['bedroom_summer_afternoon_sun']['value']/240
                value = sum(components.values())/len(components); values.append(value)
                reasons.append(dict(rule, room=room['label'], room_id=room['room_id'],
                                    date='representative months '+','.join(map(str, months)),
                                    window_exposure_equivalent_minutes=round(exposure, 2),
                                    score_components=components, heat=room.get('heat', 'not computed yet'),
                                    hot_sun_hours_jja=room.get('hot_sun_hours_jja', 'not computed yet'),
                                    flags=room['flags']))
            if values: contributions.append(rule['weight']*sum(values)/len(values))
            else: unresolved.append('Unscored: '+rule['rule'])
        if not contributions:
            skipped.append(dict(sid=sid, reason='No covered active preference rules')); continue
        ranked.append(dict(sid=sid, basis='F4', provisional=False, fit_score=round(sum(contributions)/len(contributions), 6),
                           scored_rules=len(contributions), requested_rules=len(rules), reasons=reasons, unresolved=unresolved,
                           evidence_files=[evidence_path, str(cache.relative_to(ROOT)), f'data/sun/{sid}/sun.json'],
                           timezone=computed['timezone'],
                           office_room=dict(room_id=office['room_id'], label=office['label'], suggested=suggested) if office else None,
                           room_assignment_note='Second is not a source room label. Proposed office minimizes F4 work-hour direct sun; confirm this room use.' if suggested else None,
                           heat={r['room_id']:r.get('heat') for r in rooms},
                           hot_sun_hours={r['room_id']:r.get('hot_sun_hours_jja') for r in rooms}))
    ranked.sort(key=lambda r: (-r['scored_rules'], -r['fit_score'], r['sid']))
    return dict(status='f4_ranking' if ranked else 'unavailable', ranked_homes=ranked[:10],
                ranked_home_count=len(ranked), cohort_home_count=len(by_home), returned_limit=10,
                skipped=skipped[:5], skipped_home_count=len(skipped), skipped_details_limit=5,
                habits=habits, questions=memory['questions'], rules=rules,
                heat='computed F4 heat at window glass', hot_sun_hours='computed F4 JJA climatological hot-sun hours',
                method='F4: compare rule coverage, then equal-weight signed means of room components. Sun and flags divided by band duration; hot-sun hours divided by 24; clear-glass peak W divided by (peak W + 1500 W). Summer rules combine all available components equally. Higher is better. These are explicit preference heuristics, not calibrated comfort scores.',
                limitations=['F4 is modelled sun and solar heat at window glass; never indoor temperature or actual glare.',
                             'Twelve representative days and historical weather, not a forecast. No HVAC, insulation or blinds model.',
                             'No-sun rooms and unreliable placements withheld; missing criteria remain unscored.',
                             'Office use, plant placement and nap bands need buyer confirmation. Ranking returns the top ten.'])
