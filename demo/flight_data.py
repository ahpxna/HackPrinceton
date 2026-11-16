"""Read optional daylight horizons; never acquire or mutate another session's data."""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOMES = {'khbuN7q853q', 'VN4b1MGnnCR'}


def flight_context(sid):
    if sid not in HOMES:
        raise ValueError('Flight not enabled for this home')
    table = ROOT / 'data/facts/daylight/window_months.csv'
    daily = []
    if table.is_file():
        with table.open() as stream:
            for row in csv.DictReader(stream):
                value = row.get('clear_sky_dsm_horizon_minutes')
                if row['sid'] == sid and value:
                    daily.append({'window_id': row['window_id'], 'date': row['date'],
                                  'minutes': float(value)})
    path = ROOT / 'data/cache/daylight_dsm' / (sid + '_horizons.npz')
    horizons = {}
    if path.is_file():
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                values = arrays[key]
                if values.shape == (360,) and np.isfinite(values).all() and ((values >= 0) & (values <= 90)).all():
                    horizons[key] = values.astype(float).tolist()
    return {'daily': daily, 'horizons': horizons,
            'source': 'data/facts/daylight/window_months.csv',
            'status': 'available' if horizons else 'unavailable'}
