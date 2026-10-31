"""Read-only, evidence-bearing tools over the ingested snapshot.

Pass an idle psycopg connection (preferably using a dedicated SELECT-only role).
Results are JSON-compatible; monetary values are decimal strings, never floats.
No generator, raw CSV access, embedding model, or automatic connection is involved.
"""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from uuid import UUID


SCHEMA = 'vienna_stay_assistant'
PRICE_NOTICE = 'Prices are snapshot base prices, not date-specific quotes; no currency conversion is performed.'
NOISE_NOTICE = ('Night noise is the worst Lnight road/rail band within 150 m of the listing; the filter compares the '
                'start of the band (e.g. 50-54 starts at 50). Listings with no band (not mapped or noise data not loaded) are excluded.')
LISTING_COLUMNS = '''l.listing_id, l.host_id, l.neighbourhood, l.name,
    l.room_type, l.accommodates, l.bedrooms, l.price, l.amenities,
    h.host_is_superhost, l.last_scraped, l.snapshot_date,
    l.load_id, h.load_id AS host_load_id'''
PROVENANCE_SQL = '''SELECT DISTINCT r.load_id, r.snapshot_date,
    r.config->>'scope' AS scope, r.config->>'limit' AS sample_limit
    FROM vienna_stay_assistant.ingestion_runs r
    JOIN (
      SELECT load_id FROM vienna_stay_assistant.listings
      UNION SELECT load_id FROM vienna_stay_assistant.hosts
      UNION SELECT load_id FROM vienna_stay_assistant.reviews
      UNION SELECT load_id FROM vienna_stay_assistant.calendar
      UNION SELECT load_id FROM vienna_stay_assistant.neighbourhoods
    ) active USING (load_id) ORDER BY r.load_id'''


class QueryError(RuntimeError):
    """Safe database error; backend exception text is never exposed."""


def _json(value):
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID)):
        return str(value)
    return value


def _integer(value, name, low=1, high=9223372036854775807):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer between {low} and {high}')
    return value


def _text(value, name, maximum=200):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or '\x00' in value:
        raise ValueError(f'{name} must be nonempty text of at most {maximum} characters')
    return value.strip()


def _day(value, name):
    if type(value) is date:
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f'{name} must be an ISO date (YYYY-MM-DD)')


def _money(value, name):
    try:
        if isinstance(value, bool):
            raise ValueError()
        parsed = Decimal(str(value))
        if not parsed.is_finite() or parsed < 0 or parsed > Decimal('999999999999.99'):
            raise ValueError()
        return parsed
    except (InvalidOperation, ValueError):
        raise ValueError(f'{name} must be a finite nonnegative price') from None


def _ids(values):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 100:
        raise ValueError('listing_ids must contain between 1 and 100 IDs')
    return sorted({_integer(v, 'listing_id') for v in values})


class _Session:
    def __init__(self, connection):
        self.connection = connection
        self.evidence = []

    def fetch(self, sql, parameters=()):
        with self.connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            names = [c.name for c in cursor.description]
            rows = [dict(row) if isinstance(row, dict) else dict(zip(names, row)) for row in cursor.fetchall()]
        self.evidence.append({'sql': sql, 'parameters': parameters})
        return rows

    def page(self, sql, parameters, limit, offset):
        total = self.fetch('SELECT count(*) AS total FROM (' + sql + ') matched', parameters)[0]['total']
        rows = self.fetch(sql + ' LIMIT %s OFFSET %s', (*parameters, limit, offset))
        return {'rows': rows, 'total_matches': total, 'limit': limit, 'offset': offset,
                'has_more': offset + len(rows) < total}


def _run(connection, tool, action, notices=()):
    # SET TRANSACTION must precede every read: reject caller-owned transactions
    # instead of weakening their isolation or committing somebody else's work.
    if connection.info.transaction_status != 0:
        raise ValueError('Tools require an idle connection; finish the current transaction first')
    try:
        with connection.transaction():
            connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
            connection.execute("SET LOCAL statement_timeout = '5s'")
            session = _Session(connection)
            provenance = session.fetch(PROVENANCE_SQL)
            result = action(session)
            warnings = list(notices)
            if not provenance:
                warnings.append('No ingested source rows are present.')
            if any(p['scope'] != 'full' for p in provenance):
                warnings.append('Data includes a sample or unknown scope; results do not represent the full snapshot.')
            return _json({'tool': tool, **result, 'provenance': provenance,
                          'evidence': session.evidence, 'notices': warnings})
    except Exception:
        raise QueryError('Query failed; check the database, schema, permissions, and query timeout.') from None


def search_listings(connection, *, district=None, room_type=None, accommodates=None,
                    min_price=None, max_price=None, amenities=None, superhost=None,
                    near_landmark=None, radius_m=500, max_night_noise_db=None,
                    bedrooms=None, listing_ids=None, limit=20, offset=0):
    """Filter listings; capacity is a minimum, price bounds are inclusive.

    Amenities must all match stored array values exactly. Missing prices are
    excluded only when a price bound is supplied. near_landmark (a curated landmark
    name, case-insensitive) keeps listings within radius_m metres and sorts nearest
    first. max_night_noise_db keeps listings whose worst road/rail Lnight band starts
    below that value; listings without any band are excluded and counted. Otherwise
    results sort by listing ID.
    """
    _integer(limit, 'limit', high=100)
    _integer(offset, 'offset', low=0, high=10000)
    landmark = _text(near_landmark, 'near_landmark') if near_landmark is not None else None
    radius = _integer(radius_m, 'radius_m', high=10000)
    noise_limit = _integer(max_night_noise_db, 'max_night_noise_db', low=30, high=120) \
        if max_night_noise_db is not None else None
    clauses, params = [], []
    if landmark is not None:  # the JOIN placeholder comes first in the SQL text
        params.append(landmark)
        clauses.append('ST_DWithin(l.location, g.location, %s)')
        params.append(radius)
    if listing_ids is not None:
        clauses.append('l.listing_id = ANY(%s::bigint[])')
        params.append(_ids(listing_ids))
    for value, name, expression in ((district, 'district', 'l.neighbourhood = %s'),
                                     (room_type, 'room_type', 'l.room_type = %s')):
        if value is not None:
            clauses.append(expression)
            params.append(_text(value, name))
    if accommodates is not None:
        clauses.append('l.accommodates >= %s')
        params.append(_integer(accommodates, 'accommodates', high=1000))
    if bedrooms is not None:
        clauses.append('l.bedrooms = %s')
        params.append(_integer(bedrooms, 'bedrooms', low=0, high=1000))
    if superhost is not None:
        if type(superhost) is not bool:
            raise ValueError('superhost must be a boolean')
        clauses.append('h.host_is_superhost = %s')
        params.append(superhost)
    if amenities is not None:
        if not isinstance(amenities, (list, tuple)) or len(amenities) > 50:
            raise ValueError('amenities must be a list of at most 50 values')
        clauses.append('l.amenities @> %s::text[]')
        params.append([_text(v, 'amenity') for v in amenities])
    base = (' FROM vienna_stay_assistant.listings l JOIN vienna_stay_assistant.hosts h USING (host_id)'
            + (' JOIN vienna_stay_assistant.landmarks g ON lower(g.name) = lower(%s)' if landmark is not None else '')
            + ' WHERE ' + (' AND '.join(clauses) if clauses else 'TRUE'))
    prices, price_params = [], []
    for value, name, operator in ((min_price, 'min_price', '>='), (max_price, 'max_price', '<=')):
        if value is not None:
            prices.append(f'l.price {operator} %s')
            price_params.append(_money(value, name))
    if min_price is not None and max_price is not None and price_params[0] > price_params[1]:
        raise ValueError('min_price must not exceed max_price')
    if noise_limit is not None:  # GREATEST ignores NULLs, so it is NULL only when both bands are missing
        prices.append("GREATEST((substring(l.lnight_road_band FROM '[0-9]+'))::integer, "
                      "(substring(l.lnight_rail_band FROM '[0-9]+'))::integer) < %s")
        price_params.append(noise_limit)
    columns = LISTING_COLUMNS + ', l.lnight_road_band, l.lnight_rail_band'
    order = ' ORDER BY l.listing_id'
    if landmark is not None:
        columns += ', round(ST_Distance(l.location, g.location)::numeric) AS distance_m'
        order = ' ORDER BY distance_m, l.listing_id'

    def action(s):
        extra = {}
        if landmark is not None:
            found = s.fetch('SELECT name, latitude, longitude FROM vienna_stay_assistant.landmarks '
                            'WHERE lower(name) = lower(%s)', (landmark,))
            if not found:
                names = [r['name'] for r in s.fetch('SELECT name FROM vienna_stay_assistant.landmarks ORDER BY name')]
                return {'landmark_found': False, 'known_landmarks': names, 'rows': [], 'total_matches': 0,
                        'limit': limit, 'offset': offset, 'has_more': False,
                        'missing_price_exclusions': 0, 'missing_noise_exclusions': 0}
            extra = {'landmark_found': True, 'landmark': found[0], 'radius_m': radius}
        missing_price = s.fetch('SELECT count(*) AS count' + base + ' AND l.price IS NULL', params)[0]['count'] \
            if (min_price is not None or max_price is not None) else 0
        missing_noise = s.fetch('SELECT count(*) AS count' + base + ' AND l.lnight_road_band IS NULL '
                                'AND l.lnight_rail_band IS NULL', params)[0]['count'] if noise_limit is not None else 0
        sql = 'SELECT ' + columns + base + ((' AND ' + ' AND '.join(prices)) if prices else '')
        return {**s.page(sql + order, (*params, *price_params), limit, offset), **extra,
                'missing_price_exclusions': missing_price, 'missing_noise_exclusions': missing_noise}
    notices = [PRICE_NOTICE]
    if noise_limit is not None:
        notices.append(NOISE_NOTICE)
    return _run(connection, 'search_listings', action, notices)


def get_host_listings(connection, host_id, *, limit=20, offset=0):
    """Resolve a host's listings. Unknown hosts return host_found=False."""
    _integer(host_id, 'host_id')
    _integer(limit, 'limit', high=100)
    _integer(offset, 'offset', low=0, high=10000)

    def action(s):
        hosts = s.fetch('''SELECT host_id, host_is_superhost, snapshot_date, load_id
            FROM vienna_stay_assistant.hosts WHERE host_id = %s''', (host_id,))
        sql = 'SELECT ' + LISTING_COLUMNS + ''' FROM vienna_stay_assistant.listings l
            JOIN vienna_stay_assistant.hosts h USING (host_id)
            WHERE l.host_id = %s ORDER BY l.listing_id'''
        return {'host_found': bool(hosts), 'host': hosts[0] if hosts else None,
                **s.page(sql, (host_id,), limit, offset)}
    return _run(connection, 'get_host_listings', action, [PRICE_NOTICE])


def compare_host_prices(connection, district):
    """Listing-weighted averages, including a separate unknown-status group.

    Every group includes all member IDs and the exact priced contributing IDs.
    These arrays are deliberately complete, not paginated or truncated.
    """
    district = _text(district, 'district')
    sql = '''SELECT h.host_is_superhost, count(*) AS listing_count,
        count(l.price) AS priced_listing_count,
        count(*) FILTER (WHERE l.price IS NULL) AS missing_price_count,
        avg(l.price) AS average_base_price,
        array_agg(l.listing_id ORDER BY l.listing_id) AS listing_ids,
        array_agg(DISTINCT l.host_id ORDER BY l.host_id) AS host_ids,
        coalesce(array_agg(l.listing_id ORDER BY l.listing_id)
          FILTER (WHERE l.price IS NOT NULL), ARRAY[]::bigint[]) AS contributing_listing_ids,
        array_agg(DISTINCT l.load_id) AS listing_load_ids,
        array_agg(DISTINCT h.load_id) AS host_load_ids
        FROM vienna_stay_assistant.listings l JOIN vienna_stay_assistant.hosts h USING (host_id)
        WHERE l.neighbourhood = %s GROUP BY h.host_is_superhost
        ORDER BY h.host_is_superhost DESC NULLS LAST'''
    return _run(connection, 'compare_host_prices',
                lambda s: {'district': district, 'groups': s.fetch(sql, (district,))}, [PRICE_NOTICE])


def check_availability(connection, listing_ids, check_in, check_out):
    """Conservative per-night rules; missing rows/bounds mean unknown.

    Occupied nights are [check_in, check_out). Total stay length must meet
    minimum AND maximum on every occupied night. Definite failures outweigh
    unknowns; snapshot availability does not establish current bookability.
    """
    ids = _ids(listing_ids)
    start, end = _day(check_in, 'check_in'), _day(check_out, 'check_out')
    nights = (end - start).days
    _integer(nights, 'stay length in nights', high=366)
    sql = '''SELECT requested.listing_id, l.listing_id IS NOT NULL AS listing_found,
        l.snapshot_date, l.load_id AS listing_load_id, night::date AS date,
        c.date IS NOT NULL AS calendar_present, c.available, c.minimum_nights,
        c.maximum_nights, c.snapshot_date AS calendar_snapshot_date, c.load_id AS calendar_load_id
        FROM unnest(%s::bigint[]) AS requested(listing_id)
        LEFT JOIN vienna_stay_assistant.listings l USING (listing_id)
        CROSS JOIN generate_series(%s::timestamp, (%s::date - 1)::timestamp, interval '1 day') night
        LEFT JOIN vienna_stay_assistant.calendar c
          ON c.listing_id = requested.listing_id AND c.date = night::date
        ORDER BY requested.listing_id, night'''

    def action(s):
        groups = {lid: [] for lid in ids}
        for row in s.fetch(sql, (ids, start, end)):
            groups[row['listing_id']].append(row)
        results = []
        for lid, rows in groups.items():
            failures, unknown = [], []
            for row in rows:
                if not row['calendar_present']:
                    unknown.append({'date': row['date'], 'reason': 'missing_calendar'})
                    continue
                if row['available'] is False:
                    failures.append({'date': row['date'], 'reason': 'unavailable'})
                for field, violates in (('minimum_nights', lambda v: nights < v),
                                        ('maximum_nights', lambda v: nights > v)):
                    if row[field] is None:
                        unknown.append({'date': row['date'], 'reason': 'missing_' + field})
                    elif violates(row[field]):
                        failures.append({'date': row['date'], 'reason': field})
            status = ('not_found' if not rows[0]['listing_found'] else
                      'unavailable' if failures else 'unknown' if unknown else 'available')
            results.append({'listing_id': lid, 'status': status, 'failures': failures,
                            'unknowns': unknown, 'calendar_evidence': rows})
        return {'check_in': start, 'check_out': end, 'nights': nights,
                'policy': 'per_night_total_stay_length_v1', 'results': results}
    return _run(connection, 'check_availability', action,
                ['Snapshot availability only; checkout is excluded. Missing calendar rows or stay bounds prevent confirmation.'])


def search_reviews(connection, query, *, listing_ids=None, language=None,
                   since='2024-09-17', until='2026-09-17', limit=20, offset=0):
    """Literal keyword AND search over masked SQL reviews, not semantic/BM25.

    PostgreSQL's simple tokenizer supports both German and English without
    translating or stemming. Scores are ranking signals, not confidence values.
    Callers should mask user input before passing tool results to any model.
    """
    query = _text(query, 'query', maximum=500)
    _integer(limit, 'limit', high=100)
    _integer(offset, 'offset', low=0, high=10000)
    start, end = _day(since, 'since'), _day(until, 'until')
    if start > end or end > date(2026, 9, 17):
        raise ValueError('Review dates must be ordered and end no later than the snapshot')
    clauses, params = ['r.date BETWEEN %s AND %s'], [query, start, end]
    if listing_ids is not None:
        clauses.append('r.listing_id = ANY(%s::bigint[])')
        params.append(_ids(listing_ids))
    if language is not None:
        clauses.append('r.language = %s')
        params.append(_text(language, 'language', maximum=10))
    sql = '''SELECT r.review_id, r.listing_id, r.date, r.comments, r.language,
        r.source_file, r.snapshot_date, r.load_id,
        ts_rank_cd(to_tsvector('simple', r.comments), q.terms) AS score
        FROM vienna_stay_assistant.reviews r
        CROSS JOIN plainto_tsquery('simple', %s) AS q(terms)
        WHERE to_tsvector('simple', r.comments) @@ q.terms AND ''' + ' AND '.join(clauses)
    return _run(connection, 'search_reviews',
                lambda s: {'retrieval_method': 'postgres_simple_keyword',
                           **s.page(sql + ' ORDER BY score DESC, r.review_id', params, limit, offset)},
                ['Keyword search only; no translation, semantic retrieval, BM25, or reranking. Stored reviews are masked, but masking does not guarantee zero PII leakage.'])
