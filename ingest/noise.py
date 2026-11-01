"""Load Lnight noise zones (road or rail) and store the worst band within 150 m per listing.

Zones come from a GeoPackage (--gpkg) or from the Austrian 2022 strategic noise map OGC API
(--ogc-collection, Umweltbundesamt via gis.lfrz.gv.at, CC BY 4.0), limited to the Vienna area.
API downloads are cached in derived/noise/ so a rerun does not download again (--refresh does).
Airbnb offsets listing coordinates by up to ~150 m, hence the buffer. Rerunnable per indicator.
A --replace load in load_sql.py clears the bands, so run this again afterwards.
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import struct
import sys
import time
from pathlib import Path

SCHEMA = 'vienna_stay_assistant'
BUFFER_M = 150
INDICATORS = ('road', 'rail')
ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
OGC_API = 'https://gis.lfrz.gv.at/api/geodata/i000804/ogc/features/v1/collections'
CACHE = Path(__file__).resolve().parent.parent / 'derived' / 'noise'
# Vienna's districts, padded by about 1 km; used when the neighbourhoods table is empty.
VIENNA_BBOX = (16.17, 48.11, 16.59, 48.33)

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


def normalize_band(label):
    """Band label in the '55-59' / '>=70' form the queries expect.

    The 2022 map writes 'Lnight5559' and 'LnightGreaterThan70'; reading the first number of those would
    give 5559 dB. Labels already in the short form are returned unchanged.
    """
    text = str(label).strip()
    if match := re.fullmatch(r'(?:Lnight|Lden)?(\d{2})(\d{2})', text, re.IGNORECASE):
        return f'{match.group(1)}-{match.group(2)}'
    if match := re.fullmatch(r'(?:Lnight|Lden)?GreaterThan(?:OrEqual(?:To)?)?(\d+)', text, re.IGNORECASE):
        return f'>={match.group(1)}'
    if re.search(r'LessThan', text, re.IGNORECASE) or not re.search(r'\d', text):
        raise ValueError(f'Unsupported noise band label: {text!r}')
    return text


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


def get_json(client, url, params=None, attempts=5):
    """GET JSON with retries and growing pauses; the public API occasionally times out."""
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == attempts:
                raise
            log.warning('request failed (attempt %d of %d), retrying', attempt, attempts)
            time.sleep(2 * attempt)


def download_collection(collection, bbox, path, page_size=100, client=None):
    """Page through an OGC API collection inside bbox and write one GeoJSON feature per line to path.

    The file is written under a temporary name and renamed only when the download is complete.
    """
    import httpx
    own_client = client is None
    client = client or httpx.Client(timeout=120, follow_redirects=True)
    url = f'{OGC_API}/i000804:{collection}/items'
    params = {'f': 'application/geo+json', 'limit': page_size, 'bbox': ','.join(f'{v:.5f}' for v in bbox)}
    partial = path.with_suffix('.partial')
    path.parent.mkdir(parents=True, exist_ok=True)
    seen, pages = set(), 0
    try:
        with partial.open('w', encoding='utf-8') as out:
            while url:
                page = get_json(client, url, params)
                pages += 1
                for feature in page.get('features', []):
                    if feature.get('id') in seen:
                        continue
                    seen.add(feature.get('id'))
                    out.write(json.dumps(feature, separators=(',', ':')) + '\n')
                if pages % 50 == 0:
                    log.info('%s: %d features of %s downloaded', collection, len(seen), page.get('numberMatched'))
                url = next((link['href'] for link in page.get('links', []) if link.get('rel') == 'next'), None)
                params = None  # the next link carries all parameters
    finally:
        if own_client:
            client.close()
    partial.replace(path)
    log.info('%s: %d features downloaded in %d pages', collection, len(seen), pages)
    return len(seen)


def geojson_zones(path):
    """Yield (band, geometry, srs_id, format) from a downloaded GeoJSON-lines file (CRS84 = EPSG:4326)."""
    with Path(path).open(encoding='utf-8') as lines:
        for line in lines:
            feature = json.loads(line)
            band, geometry = (feature.get('properties') or {}).get('category'), feature.get('geometry')
            if band is not None and geometry:
                yield str(band), json.dumps(geometry), 4326, 'geojson'


def vienna_bbox(conn):
    """Bounding box of the loaded districts padded by about 1 km, or a fixed Vienna box."""
    row = conn.execute(f'SELECT ST_XMin(e), ST_YMin(e), ST_XMax(e), ST_YMax(e) FROM '
                       f'(SELECT ST_Extent(geometry) AS e FROM {SCHEMA}.neighbourhoods) t').fetchone()
    if row[0] is None:
        return VIENNA_BBOX
    return row[0] - 0.015, row[1] - 0.01, row[2] + 0.015, row[3] + 0.01


GEOMETRY_SQL = {
    'wkb': 'ST_GeomFromWKB(%s)',
    'geojson': 'ST_GeomFromGeoJSON(%s)',
}


def load_zones(conn, path, layer, band_column, indicator):
    """Load zones from a GeoPackage layer."""
    zones = ((band, wkb, srs_id, 'wkb') for band, srs_id, wkb in gpkg_zones(path, layer, band_column))
    return load_zone_rows(conn, zones, Path(path).name, indicator)


def load_zone_rows(conn, zones, source, indicator):
    """Replace the indicator's zones with (band, geometry, srs_id, format) rows; labels are normalised."""
    before = count(conn, f'SELECT count(*) FROM {SCHEMA}.noise_zones WHERE indicator = %s', (indicator,))
    log.info('noise_zones[%s] rows before: %d', indicator, before)
    conn.execute(f'DELETE FROM {SCHEMA}.noise_zones WHERE indicator = %s', (indicator,))
    statements = {fmt: f'''INSERT INTO {SCHEMA}.noise_zones (source, indicator, db_band, geometry)
        VALUES (%s, %s, %s, ST_Multi(ST_CollectionExtract(ST_MakeValid(
            ST_Transform(ST_SetSRID({expression}, %s), 4326)), 3)))''' for fmt, expression in GEOMETRY_SQL.items()}
    loaded = 0
    with conn.cursor() as cur:
        for band, geometry, srs_id, fmt in zones:
            band = normalize_band(band)
            band_floor(band)  # fail early on unparseable labels
            cur.execute(statements[fmt], (source, indicator, band, geometry, srs_id))
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--gpkg', type=Path, help='Path to a noise GeoPackage')
    source.add_argument('--ogc-collection', help='Collection of the 2022 noise map API, '
                                                  'e.g. laerm_2022_strasse_lnight or laerm_2022_schiene_lnight')
    parser.add_argument('--indicator', choices=INDICATORS, required=True)
    parser.add_argument('--layer', help='Feature table inside the GeoPackage')
    parser.add_argument('--band-column', help='GeoPackage column holding the dB band label, e.g. "55-59"')
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
                        help='Download area in degrees (default: the loaded Vienna districts plus about 1 km)')
    parser.add_argument('--refresh', action='store_true', help='Download again even if a cached file exists')
    args = parser.parse_args(argv)
    if args.gpkg is not None:
        if not args.gpkg.is_file():
            parser.error('GeoPackage not found')
        if not args.layer or not args.band_column:
            parser.error('--gpkg needs --layer and --band-column')
    elif not re.fullmatch(r'[a-z0-9_]+', args.ogc_collection):
        parser.error('Collection names contain only lowercase letters, digits and underscores')
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import psycopg
    with connect(psycopg) as conn:
        if args.ogc_collection:
            # Download before opening the transaction, so a slow download does not hold it open.
            path = CACHE / f'{args.ogc_collection}.geojsonl'
            if args.refresh or not path.is_file():
                download_collection(args.ogc_collection, args.bbox or vienna_bbox(conn), path)
            else:
                log.info('using cached download %s (pass --refresh to download again)', path)
        with conn.transaction():
            if args.ogc_collection:
                load_zone_rows(conn, geojson_zones(path), args.ogc_collection, args.indicator)
            else:
                load_zones(conn, args.gpkg, args.layer, args.band_column, args.indicator)
            assign_bands(conn, args.indicator)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Noise load failed ({type(error).__name__}): {error}. No changes were committed.', file=sys.stderr)
        sys.exit(1)
