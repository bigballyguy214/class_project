import json
from db import ROOT, connect, execute, initialize


def seed():
    conn = connect()
    try:
        initialize(conn)
        rows = json.loads((ROOT / 'data' / 'venues.json').read_text())
        columns = ['osm_id', 'name', 'amenity', 'lat', 'lon', 'addr', 'postcode',
                   'opening_hours', 'outdoor_seating', 'website']
        with conn:
            for row in rows:
                execute(conn, 'INSERT INTO venues (' + ','.join(columns) + ') VALUES ('
                        + ','.join(['%s'] * len(columns)) + ') ON CONFLICT (osm_id) DO NOTHING',
                        [row[k] for k in columns])
        print(f'Seed complete: {len(rows)} source records processed. Existing IDs preserved.')
    finally:
        conn.close()


if __name__ == '__main__':
    seed()
