"""Load Lnight noise zones (road or rail) from a GeoPackage and store the worst band within 150 m per listing.

Airbnb offsets listing coordinates by up to ~150 m, hence the buffer. Rerunnable per indicator.
A --replace load in load_sql.py clears the bands, so run this again afterwards.
"""

import argparse
import logging
import os
import re
import sqlite3
import struct
import sys
from pathlib import Path

SCHEMA = 'vienna_stay_assistant'
BUFFER_M = 150
INDICATORS = ('road', 'rail')
ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

log = logging.getLogger('ingest.noise')


def split_gpkg_geometry(blob):
    """Return (srs_id, WKB bytes) from a GeoPackage geometry blob, or None for empty geometry."""
    if blob is None or blob[:2] != b'GP':
        raise ValueError('Not a GeoPackage geometry blob')
    flags = blob[3]
    if flags & 0b10000:  # empty geometry flag
        return None
    envelope = (flags >> 1) & 0b111
    if envelope not in ENVELOPE_BYTES:
        raise ValueError('Unsupported GeoPackage envelope type')
    srs_id = struct.unpack('<i' if flags & 1 else '>i', blob[4:8])[0]
    return srs_id, bytes(blob[8 + ENVELOPE_BYTES[envelope]:])


def band_floor(label):
    """Lower bound in dB of a band label such as '55-59', '55 - 60' or '>= 70'."""
    match = re.search(r'\d+', label or '')
    if not match:
        raise ValueError('Band label has no number')
    return int(match.group())


def gpkg_zones(path, layer, band_column):
    """Yield (band label, srs_id, WKB) for every non-empty feature of the layer."""
    db = sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True)
    try:
        geom = db.execute('SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?', (layer,)).fetchone()
        if not geom:
            raise ValueError('Layer not found in GeoPackage')
        # Identifiers cannot be bound; they are quoted.
        query = 'SELECT "{}", "{}" FROM "{}"'.format(
            band_column.replace('"', '""'), geom[0].replace('"', '""'), layer.replace('"', '""'))
        for band, blob in db.execute(query):
            split = split_gpkg_geometry(blob)
            if split and band is not None:
                yield (str(band), *split)
    finally:
        db.close()


def count(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def load_zones(conn, path, layer, band_column, indicator):
    before = count(conn, f'SELECT count(*) FROM {SCHEMA}.noise_zones WHERE indicator = %s', (indicator,))
    log.info('noise_zones[%s] rows before: %d', indicator, before)
    conn.execute(f'DELETE FROM {SCHEMA}.noise_zones WHERE indicator = %s', (indicator,))
    statement = f'''INSERT INTO {SCHEMA}.noise_zones (source, indicator, db_band, geometry)
        VALUES (%s, %s, %s, ST_Multi(ST_CollectionExtract(ST_MakeValid(
            ST_Transform(ST_SetSRID(ST_GeomFromWKB(%s), %s), 4326)), 3)))'''
    loaded = 0
    with conn.cursor() as cur:
        for band, srs_id, wkb in gpkg_zones(path, layer, band_column):
            band_floor(band)  # fail early on unparseable labels
            cur.execute(statement, (Path(path).name, indicator, band, wkb, srs_id))
            loaded += 1
            if loaded % 5000 == 0:
                log.info('noise_zones[%s] %d rows inserted', indicator, loaded)
    if not loaded:
        raise ValueError('GeoPackage layer contained no usable zones')
    after = count(conn, f'SELECT count(*) FROM {SCHEMA}.noise_zones WHERE indicator = %s', (indicator,))
    log.info('noise_zones[%s] rows after: %d', indicator, after)
    return after


def assign_bands(conn, indicator):
    if indicator not in INDICATORS:
        raise ValueError('Unknown indicator')
    column = f'lnight_{indicator}_band'
    before = count(conn, f'SELECT count(*) FROM {SCHEMA}.listings WHERE {column} IS NOT NULL')
    log.info('listings.%s non-null before: %d', column, before)
    conn.execute(f'UPDATE {SCHEMA}.listings SET {column} = NULL WHERE {column} IS NOT NULL')
    # The && test (degree margin) uses the GiST index; ST_DWithin on geography then measures in metres.
    conn.execute(f'''
        UPDATE {SCHEMA}.listings l SET {column} = worst.db_band
        FROM (SELECT DISTINCT ON (l2.listing_id) l2.listing_id, z.db_band
              FROM {SCHEMA}.listings l2
              JOIN {SCHEMA}.noise_zones z
                ON z.indicator = %s
               AND z.geometry && ST_Expand(l2.location::geometry, 0.004)
               AND ST_DWithin(z.geometry::geography, l2.location, %s)
              ORDER BY l2.listing_id, (substring(z.db_band FROM '\\d+'))::integer DESC) worst
        WHERE l.listing_id = worst.listing_id''', (indicator, BUFFER_M))
    after = count(conn, f'SELECT count(*) FROM {SCHEMA}.listings WHERE {column} IS NOT NULL')
    total = count(conn, f'SELECT count(*) FROM {SCHEMA}.listings')
    log.info('listings.%s non-null after: %d of %d listings', column, after, total)
    return after, total


def connect(psycopg):
    dsn = os.getenv('DATABASE_URL', '')
    defaults = {} if dsn else {key: os.getenv(env, default) for key, env, default in (
        ('host', 'PGHOST', '127.0.0.1'), ('port', 'PGPORT', '5433'), ('dbname', 'PGDATABASE', 'vienna_stay'),
        ('user', 'PGUSER', 'postgres'), ('password', 'PGPASSWORD', 'postgres'))}
    return psycopg.connect(dsn, **defaults, connect_timeout=10, autocommit=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpkg', type=Path, required=True, help='Path to the noise GeoPackage')
    parser.add_argument('--indicator', choices=INDICATORS, required=True)
    parser.add_argument('--layer', required=True, help='Feature table inside the GeoPackage')
    parser.add_argument('--band-column', required=True, help='Column holding the dB band label, e.g. "55-59"')
    args = parser.parse_args(argv)
    if not args.gpkg.is_file():
        parser.error('GeoPackage not found')
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import psycopg
    with connect(psycopg) as conn:
        with conn.transaction():
            load_zones(conn, args.gpkg, args.layer, args.band_column, args.indicator)
            assign_bands(conn, args.indicator)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Noise load failed ({type(error).__name__}): {error}. No changes were committed.', file=sys.stderr)
        sys.exit(1)
