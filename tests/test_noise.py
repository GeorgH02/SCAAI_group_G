import os
import sqlite3
import struct
import tempfile
import unittest
import uuid
from pathlib import Path

from ingest import noise

ROOT = Path(__file__).resolve().parent.parent


def gpkg_blob(wkb, srs_id=31287, envelope=0, empty=False):
    flags = (envelope << 1) | 1 | (0b10000 if empty else 0)
    return b'GP' + bytes([0, flags]) + struct.pack('<i', srs_id) + b'\x00' * noise.ENVELOPE_BYTES[envelope] + wkb


WKB_POINT = struct.pack('<BIdd', 1, 1, 1.0, 2.0)


def make_gpkg(path, rows, geom_column='geom'):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE gpkg_geometry_columns (table_name text, column_name text)')
    db.execute("INSERT INTO gpkg_geometry_columns VALUES ('road_lnight', ?)", (geom_column,))
    db.execute(f'CREATE TABLE road_lnight (fid integer primary key, db_band text, {geom_column} blob)')
    db.executemany('INSERT INTO road_lnight (db_band, ' + geom_column + ') VALUES (?, ?)', rows)
    db.commit()
    db.close()


class NoiseHelperTests(unittest.TestCase):
    def test_geometry_header_with_and_without_envelope(self):
        self.assertEqual(noise.split_gpkg_geometry(gpkg_blob(WKB_POINT)), (31287, WKB_POINT))
        self.assertEqual(noise.split_gpkg_geometry(gpkg_blob(WKB_POINT, envelope=1)), (31287, WKB_POINT))
        self.assertEqual(noise.split_gpkg_geometry(gpkg_blob(WKB_POINT, envelope=4)), (31287, WKB_POINT))
        self.assertIsNone(noise.split_gpkg_geometry(gpkg_blob(b'', empty=True)))
        with self.assertRaises(ValueError):
            noise.split_gpkg_geometry(b'not a geometry')

    def test_band_floor_orders_worst_band(self):
        self.assertEqual(noise.band_floor('55-59'), 55)
        self.assertEqual(noise.band_floor('>= 70 dB'), 70)
        self.assertGreater(noise.band_floor('60-64'), noise.band_floor('55-59'))
        with self.assertRaises(ValueError):
            noise.band_floor('unknown')

    def test_reads_layer_and_skips_empty_or_unlabelled_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'noise.gpkg'
            make_gpkg(path, [('55-59', gpkg_blob(WKB_POINT)), ('60-64', gpkg_blob(b'', empty=True)),
                             (None, gpkg_blob(WKB_POINT))])
            zones = list(noise.gpkg_zones(path, 'road_lnight', 'db_band'))
            self.assertEqual(zones, [('55-59', 31287, WKB_POINT)])
            with self.assertRaises(ValueError):
                list(noise.gpkg_zones(path, 'missing_layer', 'db_band'))

    def test_arguments_require_existing_file_and_known_indicator(self):
        with self.assertRaises(SystemExit):
            noise.parse_args(['--gpkg', 'missing.gpkg', '--indicator', 'road', '--layer', 'x', '--band-column', 'b'])
        with self.assertRaises(SystemExit):
            noise.parse_args(['--gpkg', __file__, '--indicator', 'air', '--layer', 'x', '--band-column', 'b'])


def polygon_blob(x0, y0, x1, y1, srs_id):
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    wkb = struct.pack('<BIII', 1, 3, 1, 5) + b''.join(struct.pack('<dd', *point) for point in ring)
    return gpkg_blob(wkb, srs_id)


@unittest.skipUnless(os.getenv('VSA_TEST_DATABASE_URL'), 'Requires an isolated vsa_test_* PostGIS database')
class NoiseIntegrationTests(unittest.TestCase):
    """Real PostGIS run on synthetic zones in EPSG:31287 (Austria Lambert); rolled back at the end.

    Each listing sits 5 km from the others, so only its own zones can be within 150 m.
    """
    SCHEMA = 'vienna_stay_assistant'

    def test_worst_band_within_150_m(self):
        import psycopg
        with psycopg.connect(os.environ['VSA_TEST_DATABASE_URL']) as conn, tempfile.TemporaryDirectory() as folder:
            self.assertTrue(conn.execute('SELECT current_database()').fetchone()[0].startswith('vsa_test_'),
                            'Refusing a non-test database')
            conn.execute((ROOT / 'docker/postgres/init/20-schema.sql').read_text(encoding='utf-8'))
            x0, y0 = conn.execute('SELECT ST_X(g), ST_Y(g) FROM (SELECT ST_Transform('
                                  'ST_SetSRID(ST_MakePoint(16.3, 48.2), 4326), 31287) AS g) t').fetchone()
            centres = [(x0, y0 + 5000 * i) for i in range(7)]

            load_id = uuid.uuid4()
            conn.execute(f"INSERT INTO {self.SCHEMA}.ingestion_runs VALUES (%s, now(), '2026-09-17', '{{}}', '{{}}')", (load_id,))
            conn.execute(f"""INSERT INTO {self.SCHEMA}.neighbourhoods (neighbourhood, district_number, geometry, snapshot_date, load_id)
                VALUES ('Neubau', 7, ST_Multi(ST_MakeEnvelope(15, 47, 17, 49, 4326)), '2026-09-17', %s) ON CONFLICT DO NOTHING""", (load_id,))
            conn.execute(f"""INSERT INTO {self.SCHEMA}.hosts (host_id, source_listing_id, last_scraped, snapshot_date, load_id)
                VALUES (999000000, 999000000, '2026-09-17', '2026-09-17', %s)""", (load_id,))
            ids = []
            for i, (x, y) in enumerate(centres):
                ids.append(999000001 + i)
                conn.execute(f"""INSERT INTO {self.SCHEMA}.listings (listing_id, host_id, neighbourhood, latitude, longitude,
                    last_scraped, snapshot_date, load_id)
                    SELECT %s, 999000000, 'Neubau', ST_Y(g), ST_X(g), '2026-09-17', '2026-09-17', %s
                    FROM (SELECT ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), 31287), 4326) AS g) t""", (ids[-1], load_id, x, y))

            def box(i, east_from, east_to):  # tall strip east_from..east_to metres east of listing i
                x, y = centres[i]
                return x + east_from, y - 500, x + east_to, y + 500

            road = [  # (listing index, band, strip)
                (0, '55-59', box(0, -50, 50)),    # inside
                (1, '60-64', box(1, 100, 400)),   # edge 100 m away: inside the buffer
                (2, '65-69', box(2, 300, 600)),   # edge 300 m away: outside the buffer
                (3, '50-54', box(3, -50, 50)),    # three zones: highest band must win
                (3, '60-64', box(3, -50, 50)),
                (3, '75-79', box(3, 120, 300)),
                (4, '95-99', box(4, -50, 50)),    # numeric, not alphabetical: 100 > 95
                (4, '100-104', box(4, -50, 50)),
            ]
            rail = [(5, '60-64', box(5, -50, 50))]
            expected_road = {0: '55-59', 1: '60-64', 2: None, 3: '75-79', 4: '100-104', 5: None, 6: None}
            expected_rail = {i: None for i in range(7)} | {5: '60-64'}

            def write_gpkg(name, zones):
                path = Path(folder) / name
                make_gpkg(path, [(band, polygon_blob(*strip, 31287)) for _, band, strip in zones])
                return path

            def bands(column):
                rows = dict(conn.execute(f'SELECT listing_id, {column} FROM {self.SCHEMA}.listings WHERE listing_id = ANY(%s)', (ids,)))
                return {i: rows[ids[i]] for i in range(7)}

            road_file, rail_file = write_gpkg('road.gpkg', road), write_gpkg('rail.gpkg', rail)
            for _ in range(2):  # second pass proves a rerun gives the same result
                self.assertEqual(noise.load_zones(conn, road_file, 'road_lnight', 'db_band', 'road'), len(road))
                noise.assign_bands(conn, 'road')
                self.assertEqual(bands('lnight_road_band'), expected_road)
            self.assertEqual(noise.load_zones(conn, rail_file, 'road_lnight', 'db_band', 'rail'), len(rail))
            noise.assign_bands(conn, 'rail')
            self.assertEqual(bands('lnight_rail_band'), expected_rail)
            self.assertEqual(bands('lnight_road_band'), expected_road)  # rail load left road untouched
            self.assertEqual(conn.execute(f"SELECT count(*) FROM {self.SCHEMA}.noise_zones WHERE indicator = 'road'").fetchone()[0], len(road))
            conn.rollback()


if __name__ == '__main__':
    unittest.main()
