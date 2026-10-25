"""Load the fixed Vienna snapshot into PostgreSQL/PostGIS atomically.

Existing data is protected unless --replace is explicit. No embedding model is used.
"""

import argparse
import json
import logging
import os
import re
import sys
import uuid
from collections import Counter
from datetime import date
from decimal import Decimal
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from data_setup.load_vectors import ROOT, SNAPSHOT_DATE, Masker, clean_text, documents, file_hash, rows, source_id

SCHEMA = 'vienna_stay_assistant'

FILES = ('listings.csv', 'reviews.csv', 'calendar.csv', 'neighbourhoods.csv', 'neighbourhoods.geojson')
TABLES = ('neighbourhoods', 'hosts', 'listings', 'reviews', 'calendar')


class DependencyError(RuntimeError):
    """Actionable startup error safe to display without raw library context."""


class ExistingDataError(ValueError):
    """Existing data requires an explicit replacement request."""


def postgres_driver():
    try:
        return import_module('psycopg')
    except ImportError:
        raise DependencyError(
            'PostgreSQL driver unavailable: psycopg needs its binary component or libpq. '
            f'Install dependencies in the interpreter used by the notebook: '
            f'"{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}". '
            'The requirement psycopg[binary] includes the binary component.'
        ) from None


DISTRICTS = dict(zip((
    'Innere Stadt', 'Leopoldstadt', 'Landstraße', 'Wieden', 'Margareten',
    'Mariahilf', 'Neubau', 'Josefstadt', 'Alsergrund', 'Favoriten', 'Simmering',
    'Meidling', 'Hietzing', 'Penzing', 'Rudolfsheim-Fünfhaus', 'Ottakring',
    'Hernals', 'Währing', 'Döbling', 'Brigittenau', 'Floridsdorf', 'Donaustadt', 'Liesing',
), range(1, 24)))
# Explicit repairs observed in all three local district sources. Never mutate raw files.
DISTRICT_REPAIRS = {'Landstra\xa7e': 'Landstraße', 'Rudolfsheim-F\x9fnfhaus': 'Rudolfsheim-Fünfhaus',
                    'W\x8ahring': 'Währing', 'D\x9abling': 'Döbling'}
SCORES = ('rating', 'accuracy', 'cleanliness', 'checkin', 'communication', 'location', 'value')


def district(value):
    value = DISTRICT_REPAIRS.get(value, value)
    if value not in DISTRICTS:
        raise ValueError('Unknown district name')
    return value


def number(value, integer=False, price=False):
    value = (value or '').strip()
    if not value or value == 'N/A':
        return None
    if price:
        if not re.fullmatch(r'[$€£]?\s*(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{1,2})?', value):
            raise ValueError('Invalid price formatting')
        value = re.sub(r'[$€£,\s]', '', value)
    result = Decimal(value)
    if not result.is_finite() or result < 0 or (integer and result != result.to_integral_value()):
        raise ValueError('Invalid numeric field')
    return int(result) if integer else result


def boolean(value):
    if not value:
        return None
    if value not in ('t', 'f'):
        raise ValueError('Invalid boolean field')
    return value == 't'


def optional_date(value):
    return date.fromisoformat(value) if value else None


def masked(value, names, masker):
    text = clean_text(value)
    return masker.mask(text, names) if text else None


def listing_sources(data_dir, limit, stats):
    selected, all_ids = {}, set()
    required = ('id', 'host_id', 'host_name', 'description', 'neighbourhood_cleansed',
                'last_scraped', 'latitude', 'longitude', 'price', 'amenities', 'room_type', 'accommodates')
    for row in rows(data_dir / 'listings.csv', required):
        lid = source_id(row['id'])
        if lid in all_ids:
            raise ValueError('Duplicate listing_id')
        all_ids.add(lid)
        stats['source_listings'] += 1
        if limit is None or len(selected) < limit:
            source_id(row['host_id'])
            date.fromisoformat(row['last_scraped'])
            district(row['neighbourhood_cleansed'])
            selected[lid] = row
    if not selected:
        raise ValueError('No listings to load')
    return selected, all_ids


def neighbourhood_records(data_dir, stats):
    features = json.loads((data_dir / 'neighbourhoods.geojson').read_text(encoding='utf-8-sig'))
    if features.get('crs', {}).get('properties', {}).get('name') not in (
        None, 'urn:ogc:def:crs:OGC:1.3:CRS84', 'urn:ogc:def:crs:EPSG::4326', 'EPSG:4326'
    ):
        raise ValueError('District geometry must use WGS84')
    geometries = {}
    for feature in features['features']:
        name = district(feature['properties']['neighbourhood'])
        if name in geometries or feature['geometry']['type'] not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Duplicate or invalid district geometry')
        geometries[name] = json.dumps(feature['geometry'])
    seen = set()
    for row in rows(data_dir / 'neighbourhoods.csv', ('neighbourhood',)):
        name = district(row['neighbourhood'])
        if name in seen:
            raise ValueError('Duplicate district')
        seen.add(name)
        stats['neighbourhoods_loaded'] += 1
        yield name, DISTRICTS[name], geometries[name]
    if seen != set(DISTRICTS) or set(geometries) != seen:
        raise ValueError('Expected exactly 23 matching district boundaries')


def host_sources(listings, stats):
    # Prefer most recent scrape, then lowest listing ID. Count differing host data.
    hosts = {}
    fields = ('host_name', 'host_since', 'host_about', 'host_is_superhost',
              'host_listings_count', 'host_response_rate')
    conflicts = set()
    for lid, row in sorted(listings.items(), key=lambda item: (-date.fromisoformat(item[1]['last_scraped']).toordinal(), item[0])):
        hid = source_id(row['host_id'])
        if hid in hosts:
            if any(row.get(f, '') != hosts[hid][1].get(f, '') for f in fields):
                conflicts.add(hid)
        else:
            hosts[hid] = lid, row
    stats['hosts_with_conflicting_fields'] = len(conflicts)
    return hosts


def host_records(hosts, masker):
    for hid, (lid, row) in hosts.items():
        rate = row.get('host_response_rate', '')
        yield (hid, '<PERSON_0>' if row['host_name'] else None,
               optional_date(row.get('host_since')), masked(row.get('host_about'), [row['host_name']], masker),
               boolean(row.get('host_is_superhost')), number(row.get('host_listings_count'), integer=True),
               number(rate.removesuffix('%')), lid, date.fromisoformat(row['last_scraped']))


def listing_records(listings, masker, stats):
    for lid, row in listings.items():
        names = [row['host_name']]
        price = number(row['price'], price=True)
        stats['listings_missing_price'] += price is None
        amenities = json.loads(row['amenities']) if row['amenities'] else None
        if amenities is not None and (not isinstance(amenities, list) or any(not isinstance(a, str) for a in amenities)):
            raise ValueError('Amenities must be an array of strings')
        # Amenities are free text too; protect unusual custom entries.
        amenities = [masked(a, names, masker) or '' for a in amenities] if amenities is not None else None
        lat = Decimal(row['latitude']) if row['latitude'] else None
        lon = Decimal(row['longitude']) if row['longitude'] else None
        if any(v is not None and not v.is_finite() for v in (lat, lon)):
            raise ValueError('Invalid coordinates')
        yield (lid, source_id(row['host_id']), district(row['neighbourhood_cleansed']),
               *(masked(row.get(f), names, masker) for f in ('name', 'description', 'neighborhood_overview')),
               masked(row.get('property_type'), names, masker), row['room_type'] or None,
               number(row['accommodates'], integer=True),
               *(number(row.get(f)) for f in ('bedrooms', 'beds', 'bathrooms')),
               price, amenities,
               *(number(row.get(f), integer=True) for f in ('minimum_nights', 'maximum_nights')),
               *(number(row.get('review_scores_' + f)) for f in SCORES),
               lat, lon, date.fromisoformat(row['last_scraped']))


def review_records(args, listings, masker, language, stats):
    # Reuse vector source selection: identical duplicate collapse, exclude ALL
    # conflicting IDs, inclusive dates, cancellation filtering, known-name masking.
    source_stats = Counter()
    candidates = 0
    for text, names, metadata in documents(args.data_dir, args.reviews_since, source_stats):
        if metadata['source_type'] != 'review':
            continue
        if metadata['listing_id'] not in listings:
            stats['reviews_outside_selected_listings'] += 1
            continue
        if args.limit is not None and candidates >= args.limit:
            continue
        candidates += 1
        text = masker.mask(text, names)
        if len(text) < args.min_review_chars:
            stats['short_reviews'] += 1
            continue
        yield metadata['review_id'], metadata['listing_id'], date.fromisoformat(metadata['review_date']), text, language(text)
    stats.update({'review_selection_' + key: value for key, value in source_stats.items()})


def is_active(row):
    # Project definition of an active listing: at least one review in the last 12 months.
    return (row.get('number_of_reviews_ltm') or '0') != '0'


def calendar_records(data_dir, listings, all_ids, stats):
    for row in rows(data_dir / 'calendar.csv', ('listing_id', 'date', 'available', 'minimum_nights', 'maximum_nights')):
        stats['calendar_rows_read'] += 1
        lid = source_id(row['listing_id'])
        if lid not in all_ids:
            stats['orphan_calendar_rows_excluded'] += 1
            continue
        if lid not in listings:
            stats['calendar_outside_selected_listings'] += 1
            continue
        if not is_active(listings[lid]):
            stats['calendar_inactive_listing_rows_excluded'] += 1
            continue
        yield (lid, date.fromisoformat(row['date']), boolean(row['available']),
               number(row['minimum_nights'], integer=True), number(row['maximum_nights'], integer=True))


def copy_records(conn, table, columns, records, load_id, stats):
    from psycopg import sql
    columns = columns.split() + ['snapshot_date', 'load_id']
    statement = sql.SQL('COPY {} ({}) FROM STDIN').format(
        sql.Identifier(SCHEMA, table), sql.SQL(', ').join(map(sql.Identifier, columns)))
    with conn.cursor().copy(statement) as copy:
        for record in records:
            copy.write_row((*record, date.fromisoformat(SNAPSHOT_DATE), load_id))
            stats[table + '_loaded'] += 1
            if stats[table + '_loaded'] % 10000 == 0:
                print(f'{table}: {stats[table + "_loaded"]} rows prepared', flush=True)


def validate(conn):
    counts = {table: conn.execute(f'SELECT count(*) FROM {SCHEMA}.{table}').fetchone()[0] for table in TABLES}
    counts['missing_prices'] = conn.execute('SELECT count(*) FROM vienna_stay_assistant.listings WHERE price IS NULL').fetchone()[0]
    for table in ('reviews', 'calendar'):
        low, high = conn.execute(f'SELECT min(date), max(date) FROM {SCHEMA}.{table}').fetchone()
        counts[table + '_date_range'] = [str(low) if low else None, str(high) if high else None]
    coverage = conn.execute('''
        WITH coverage AS (
            SELECT l.listing_id, count(c.date) AS nights, min(c.date) AS first, max(c.date) AS last
            FROM vienna_stay_assistant.listings l LEFT JOIN vienna_stay_assistant.calendar c USING (listing_id) GROUP BY l.listing_id
        ) SELECT count(*) FILTER (WHERE nights = 0),
                 count(*) FILTER (WHERE nights > 0 AND nights <> last - first + 1),
                 min(nights), max(nights)
          FROM coverage
    ''').fetchone()
    counts['calendar_coverage'] = dict(zip(('listings_without_calendar', 'listings_with_internal_gaps',
                                          'minimum_nights_present', 'maximum_nights_present'), coverage))
    counts['key_and_foreign_key_constraints'] = 'enforced; duplicate calendar keys abort the transaction'
    return counts


def ingest(conn, args, masker, language):
    from psycopg.types.json import Jsonb
    stats = Counter()
    hashes = {name: file_hash(args.data_dir / name) for name in FILES}
    listings, all_ids = listing_sources(args.data_dir, args.limit, stats)
    hosts = host_sources(listings, stats)
    load_id = uuid.uuid4()
    config = {'pipeline_version': 2, 'schema': SCHEMA, 'snapshot_date': SNAPSHOT_DATE, 'source_sha256': hashes,
              'schema_sha256': file_hash(ROOT / 'docker/postgres/init/20-schema.sql'),
              'loader_sha256': file_hash(Path(__file__)),
              'reviews_since': args.reviews_since.isoformat(), 'min_review_chars': args.min_review_chars,
              'limit': args.limit, 'scope': 'sample' if args.limit else 'full',
              'dependencies': {n: version(n) for n in ('psycopg', 'presidio-analyzer', 'spacy',
                                                     'en-core-web-lg', 'de-core-news-lg', 'langdetect')}}
    with conn.transaction():
        conn.execute('SET LOCAL search_path TO vienna_stay_assistant, public')
        conn.execute("SELECT pg_advisory_xact_lock(729104617)")
        conn.execute((ROOT / 'docker/postgres/init/20-schema.sql').read_text(encoding='utf-8'))
        occupied = any(conn.execute(f'SELECT EXISTS (SELECT 1 FROM {SCHEMA}.{t})').fetchone()[0] for t in TABLES)
        if occupied and not args.replace:
            raise ExistingDataError(
                'Database already contains accommodation data. To replace it, set '
                'SQL_REPLACE = True in run_pipeline.ipynb and rerun the SQL cell, '
                'or pass --replace on the command line. To keep it, skip the SQL load.'
            )
        if occupied:
            # Transactional DELETE avoids TRUNCATE's visibility behavior and never cascades.
            for table in reversed(TABLES):
                conn.execute(f'DELETE FROM {SCHEMA}.{table}')
        conn.execute('INSERT INTO vienna_stay_assistant.ingestion_runs (load_id, snapshot_date, config, validation) VALUES (%s,%s,%s,%s)',
                     (load_id, SNAPSHOT_DATE, Jsonb(config), Jsonb({})))
        for name, num, geometry in neighbourhood_records(args.data_dir, stats):
            conn.execute('''INSERT INTO vienna_stay_assistant.neighbourhoods (neighbourhood, district_number, geometry, snapshot_date, load_id)
                VALUES (%s,%s,ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s),4326)),%s,%s)''',
                         (name, num, geometry, SNAPSHOT_DATE, load_id))
        copy_records(conn, 'hosts', 'host_id host_name host_since host_about host_is_superhost host_listings_count '
                     'host_response_rate source_listing_id last_scraped', host_records(hosts, masker), load_id, stats)
        columns = ('listing_id host_id neighbourhood name description neighborhood_overview property_type room_type '
                   'accommodates bedrooms beds bathrooms price amenities minimum_nights maximum_nights ' +
                   ' '.join('review_scores_' + f for f in SCORES) + ' latitude longitude last_scraped')
        copy_records(conn, 'listings', columns, listing_records(listings, masker, stats), load_id, stats)
        copy_records(conn, 'reviews', 'review_id listing_id date comments language',
                     review_records(args, listings, masker, language, stats), load_id, stats)
        copy_records(conn, 'calendar', 'listing_id date available minimum_nights maximum_nights',
                     calendar_records(args.data_dir, listings, all_ids, stats), load_id, stats)
        validation = validate(conn)
        if any(validation[t] != stats[t + '_loaded'] for t in TABLES):
            raise ValueError('Loaded counts do not match database counts')
        if hashes != {name: file_hash(args.data_dir / name) for name in FILES}:
            raise ValueError('Source files changed during ingestion')
        report = {'load_id': str(load_id), 'status': 'complete', 'config': config,
                  'stats': dict(stats), 'validation': validation}
        conn.execute('UPDATE vienna_stay_assistant.ingestion_runs SET completed_at=clock_timestamp(), validation=%s WHERE load_id=%s',
                     (Jsonb(report), load_id))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--reviews-since', type=date.fromisoformat, default=date(2024, 9, 17))
    parser.add_argument('--min-review-chars', type=int, default=30)
    parser.add_argument('--limit', type=int, help='Sample: first N listings and at most N candidate reviews; all their calendar rows')
    parser.add_argument('--replace', action='store_true', help='Atomically replace existing accommodation tables; keeps run history')
    args = parser.parse_args(argv)
    if args.min_review_chars < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Minimum length and limit must be positive')
    if args.reviews_since > date.fromisoformat(SNAPSHOT_DATE):
        parser.error('Review window starts after the snapshot')
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.disable(logging.CRITICAL)  # Third-party errors must not expose raw text.
    psycopg = postgres_driver()
    from langdetect import DetectorFactory, detect, LangDetectException
    DetectorFactory.seed = 0

    def language(text):
        try:
            return detect(text)
        except LangDetectException:
            return 'und'

    # Standard PG* environment variables are supported when DATABASE_URL is absent.
    dsn = os.getenv('DATABASE_URL', '')
    defaults = {} if dsn else {key: os.getenv(env, default) for key, env, default in (
        ('host', 'PGHOST', '127.0.0.1'), ('port', 'PGPORT', '5433'), ('dbname', 'PGDATABASE', 'vienna_stay'),
        ('user', 'PGUSER', 'postgres'), ('password', 'PGPASSWORD', 'postgres'))}
    # autocommit keeps connectivity checks outside the atomic ingestion transaction.
    with psycopg.connect(dsn, **defaults, connect_timeout=10, autocommit=True) as conn:
        print('Connected. Loading German/English privacy models (no embedding model)...', flush=True)
        report = ingest(conn, args, Masker(), language)
    output = ROOT / 'derived' / 'sql'
    path = output / (report['load_id'] + '.json')
    try:
        output.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix('.tmp')
        staging.write_text(json.dumps(report, indent=2), encoding='utf-8')
        staging.replace(path)
    except OSError:
        print('SQL committed. File report could not be written; recover validation from ingestion_runs.', file=sys.stderr)
        return
    print(f'Complete ({report["config"]["scope"]}). Validation report: {path}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except DependencyError as error:
        print(f'SQL load could not start. {error} No database changes were made.', file=sys.stderr)
        sys.exit(1)
    except ExistingDataError as error:
        print(f'SQL load stopped. {error} Existing data is preserved.', file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        # Do not print database errors: COPY context can include text and credentials.
        print(f'SQL load failed ({type(error).__name__}). No ingestion transaction was committed. '
              'Check database connectivity, dependencies/privacy models, CSV values, and whether '
              'existing SQL data requires --replace. Existing data is preserved.', file=sys.stderr)
        sys.exit(1)
