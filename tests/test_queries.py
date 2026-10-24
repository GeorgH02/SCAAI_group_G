import json
import os
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID

from langchain_agent import queries


class QueryValidationTests(unittest.TestCase):
    def setUp(self):
        self.connection = Mock()

    def test_invalid_arguments_never_reach_database(self):
        calls = [
            lambda: queries.search_listings(self.connection, limit=True),
            lambda: queries.search_listings(self.connection, offset=-1),
            lambda: queries.search_listings(self.connection, max_price='NaN'),
            lambda: queries.search_listings(self.connection, min_price=200, max_price=100),
            lambda: queries.search_listings(self.connection, amenities='wifi'),
            lambda: queries.search_listings(self.connection, superhost='true'),
            lambda: queries.search_listings(self.connection, accommodates=0),
            lambda: queries.search_listings(self.connection, listing_ids=[]),
            lambda: queries.get_host_listings(self.connection, True),
            lambda: queries.compare_host_prices(self.connection, ''),
            lambda: queries.check_availability(self.connection, [], '2026-10-01', '2026-10-02'),
            lambda: queries.check_availability(self.connection, [1], '2026-10-02', '2026-10-01'),
            lambda: queries.check_availability(self.connection, [1], '2026-01-01', '2028-01-01'),
            lambda: queries.check_availability(self.connection, [1], 'bad', '2026-10-01'),
            lambda: queries.search_reviews(self.connection, 'quiet', listing_ids=[]),
            lambda: queries.search_reviews(self.connection, 'quiet', until='2026-09-18'),
            lambda: queries.search_reviews(self.connection, 'quiet', since='2026-09-17', until='2025-01-01'),
            lambda: queries.search_reviews(self.connection, '  '),
        ]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()
        self.connection.execute.assert_not_called()
        self.connection.transaction.assert_not_called()

    def test_caller_transaction_is_not_modified(self):
        self.connection.info.transaction_status = 2
        with self.assertRaises(ValueError):
            queries.compare_host_prices(self.connection, 'Neubau')
        self.connection.execute.assert_not_called()

    def test_backend_errors_are_safe(self):
        self.connection.info.transaction_status = 0
        self.connection.execute.side_effect = RuntimeError('private text and credentials')
        with self.assertRaises(queries.QueryError) as caught:
            queries.compare_host_prices(self.connection, 'Neubau')
        self.assertNotIn('private', str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_json_preserves_money_and_keys(self):
        value = queries._json({'price': Decimal('100.10'), 'date': date(2026, 10, 1),
                               'load_id': UUID(int=1)})
        self.assertEqual(value['price'], '100.10')
        self.assertEqual(value['date'], '2026-10-01')
        json.dumps(value)

    def test_availability_distinguishes_unknown_failure_and_not_found(self):
        def row(lid, **overrides):
            return dict(listing_id=lid, listing_found=True, date=date(2026, 10, 1),
                        calendar_present=True, available=True, minimum_nights=1,
                        maximum_nights=30, **overrides)
        rows = [row(1), {**row(2), 'minimum_nights': None},
                {**row(3), 'available': False, 'maximum_nights': None},
                {**row(4), 'listing_found': False, 'calendar_present': False}]
        def run(connection, tool, action, notices):
            session = Mock()
            session.fetch.return_value = rows
            return action(session)
        with patch('langchain_agent.queries._run', side_effect=run):
            result = queries.check_availability(self.connection, [1, 2, 3, 4], '2026-10-01', '2026-10-02')
        self.assertEqual([r['status'] for r in result['results']],
                         ['available', 'unknown', 'unavailable', 'not_found'])


@unittest.skipUnless(os.getenv('VSA_QUERY_TEST_DATABASE_URL'),
                     'Requires a separate empty vsa_test_* PostGIS database')
class QueryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg
        cls.connection = psycopg.connect(os.environ['VSA_QUERY_TEST_DATABASE_URL'], autocommit=True)
        cls.addClassCleanup(cls.connection.close)
        database = cls.connection.execute('SELECT current_database()').fetchone()[0]
        if not database.startswith('vsa_test_'):
            raise RuntimeError('Refusing a non-test database')
        if cls.connection.execute("SELECT to_regnamespace('vienna_stay_assistant')").fetchone()[0] is not None:
            raise RuntimeError('Query tests require a separate empty application schema')
        schema = Path(__file__).resolve().parents[1] / 'docker/postgres/init/20-schema.sql'
        cls.connection.execute(schema.read_text(encoding='utf-8'))
        cls.addClassCleanup(cls.connection.execute, 'DROP SCHEMA vienna_stay_assistant CASCADE')
        cls.connection.execute('''INSERT INTO vienna_stay_assistant.ingestion_runs
            (load_id, snapshot_date, config, validation)
            VALUES ('00000000-0000-0000-0000-000000000001', '2026-09-17',
            '{"scope":"sample","limit":5}', '{}')''')
        cls.connection.execute('''INSERT INTO vienna_stay_assistant.neighbourhoods
            (neighbourhood, district_number, geometry, snapshot_date, load_id)
            VALUES ('Neubau', 7, ST_Multi(ST_GeomFromText(
            'POLYGON((16 48,17 48,17 49,16 49,16 48))',4326)), '2026-09-17',
            '00000000-0000-0000-0000-000000000001')''')
        for hid, superhost in [(10, True), (11, True), (20, False), (30, None)]:
            cls.connection.execute('''INSERT INTO vienna_stay_assistant.hosts
                (host_id, host_is_superhost, source_listing_id, last_scraped, snapshot_date, load_id)
                VALUES (%s,%s,1,'2026-09-19','2026-09-17','00000000-0000-0000-0000-000000000001')''',
                (hid, superhost))
        for lid, hid, price in [(1, 10, 100), (2, 10, 200), (3, 11, 600), (4, 20, None), (5, 30, 50)]:
            cls.connection.execute('''INSERT INTO vienna_stay_assistant.listings
                (listing_id, host_id, neighbourhood, room_type, accommodates, price, amenities,
                 last_scraped, snapshot_date, load_id)
                VALUES (%s,%s,'Neubau','Entire home/apt',4,%s,ARRAY['Wifi','Kitchen'],
                 '2026-09-19','2026-09-17','00000000-0000-0000-0000-000000000001')''', (lid, hid, price))
        for lid, day, available, minimum, maximum in [
            (1, '2026-10-01', True, 2, 30), (1, '2026-10-02', True, 1, 30),
            (1, '2026-10-03', False, 1, 30), (2, '2026-10-01', True, 1, 30),
            (3, '2026-10-01', True, 1, 30), (3, '2026-10-02', True, 3, 30),
            (4, '2026-10-01', True, None, 30), (5, '2026-10-01', True, 1, 1),
            (5, '2026-10-02', True, 1, 1)]:
            cls.connection.execute('''INSERT INTO vienna_stay_assistant.calendar
                (listing_id,date,available,minimum_nights,maximum_nights,snapshot_date,load_id)
                VALUES (%s,%s,%s,%s,%s,'2026-09-17','00000000-0000-0000-0000-000000000001')''',
                (lid, day, available, minimum, maximum))
        for rid, lid, day, language, text in [
            (100, 1, '2026-01-01', 'en', 'Quiet comfortable apartment with a useful kitchen.'),
            (101, 2, '2026-02-01', 'en', 'Quiet comfortable apartment with a kitchen.'),
            (102, 1, '2025-01-01', 'de', 'Sehr ruhige Wohnung mit einer praktischen Küche.'),
            (103, 3, '2026-01-01', 'en', 'Quiet apartment close to transport.'),
            (104, 1, '2026-09-18', 'en', 'Quiet apartment after the snapshot.')]:
            cls.connection.execute('''INSERT INTO vienna_stay_assistant.reviews
                (review_id,listing_id,date,comments,language,snapshot_date,load_id)
                VALUES (%s,%s,%s,%s,%s,'2026-09-17','00000000-0000-0000-0000-000000000001')''',
                (rid, lid, day, text, language))

    def test_listing_filters_pagination_and_missing_prices(self):
        result = queries.search_listings(self.connection, district='Neubau', accommodates=4,
            room_type='Entire home/apt', min_price=100, max_price=200,
            amenities=['Wifi'], superhost=True, limit=1)
        self.assertEqual([r['listing_id'] for r in result['rows']], [1])
        self.assertEqual(result['total_matches'], 2)
        self.assertTrue(result['has_more'])
        self.assertEqual(result['rows'][0]['price'], '100.00')
        second = queries.search_listings(self.connection, max_price=200, offset=1, limit=1)
        self.assertEqual(second['rows'][0]['listing_id'], 2)
        self.assertEqual(second['missing_price_exclusions'], 1)
        empty = queries.search_listings(self.connection, max_price=200, offset=99)
        self.assertEqual(empty['rows'], [])
        self.assertEqual(empty['total_matches'], 3)
        self.assertEqual(queries.search_listings(self.connection)['total_matches'], 5)
        self.assertEqual(queries.search_listings(self.connection, amenities=['Pool'])['rows'], [])
        targeted = queries.search_listings(self.connection, listing_ids=[3, 1, 3])
        self.assertEqual([r['listing_id'] for r in targeted['rows']], [1, 3])
        json.dumps(result)

    def test_aggregation_is_listing_weighted_and_fully_citable(self):
        result = queries.compare_host_prices(self.connection, 'Neubau')
        yes, no, unknown = result['groups']
        self.assertEqual(Decimal(yes['average_base_price']), Decimal('300'))
        self.assertEqual(yes['contributing_listing_ids'], [1, 2, 3])
        self.assertEqual(yes['host_ids'], [10, 11])
        self.assertEqual(no['listing_count'], 1)
        self.assertEqual(no['missing_price_count'], 1)
        self.assertEqual(no['priced_listing_count'], 0)
        self.assertEqual(no['contributing_listing_ids'], [])
        self.assertIsNone(no['average_base_price'])
        self.assertIsNone(unknown['host_is_superhost'])
        self.assertEqual(result['provenance'][0]['scope'], 'sample')
        self.assertTrue(any('sample' in n for n in result['notices']))
        self.assertEqual(queries.compare_host_prices(self.connection, 'No district')['groups'], [])

    def test_parameter_injection_does_not_change_query(self):
        attack = "Neubau' OR TRUE; DROP TABLE listings; --"
        for result in (queries.compare_host_prices(self.connection, attack),
                       queries.search_listings(self.connection, district=attack)):
            self.assertFalse(result.get('groups', result.get('rows')))
            self.assertNotIn(attack, result['evidence'][1]['sql'])
        self.assertEqual(queries.search_listings(self.connection)['total_matches'], 5)

    def test_host_lookup(self):
        result = queries.get_host_listings(self.connection, 10)
        self.assertTrue(result['host_found'])
        self.assertEqual([r['listing_id'] for r in result['rows']], [1, 2])
        self.assertFalse(queries.get_host_listings(self.connection, 999)['host_found'])

    def test_availability_coverage_checkout_and_per_night_limits(self):
        result = queries.check_availability(self.connection, [1, 2, 3, 5, 999], '2026-10-01', '2026-10-03')
        self.assertEqual([r['status'] for r in result['results']],
                         ['available', 'unknown', 'unavailable', 'unavailable', 'not_found'])
        self.assertEqual(len(result['results'][0]['calendar_evidence']), 2)
        self.assertEqual(result['results'][2]['failures'][0]['reason'], 'minimum_nights')
        self.assertEqual(result['results'][3]['failures'][0]['reason'], 'maximum_nights')
        short = queries.check_availability(self.connection, [1], '2026-10-01', '2026-10-02')
        self.assertEqual(short['results'][0]['status'], 'unavailable')
        nulls = queries.check_availability(self.connection, [4], '2026-10-01', '2026-10-02')
        self.assertEqual(nulls['results'][0]['status'], 'unknown')
        blocked = queries.check_availability(self.connection, [1], '2026-10-01', '2026-10-04')
        self.assertIn('unavailable', [f['reason'] for f in blocked['results'][0]['failures']])
        outside = queries.check_availability(self.connection, [1], '2028-01-01', '2028-01-02')
        self.assertEqual(outside['results'][0]['status'], 'unknown')

    def test_review_search_filters_languages_and_snapshot(self):
        result = queries.search_reviews(self.connection, 'quiet comfortable', listing_ids=[1], language='en')
        self.assertEqual([r['review_id'] for r in result['rows']], [100])
        german = queries.search_reviews(self.connection, 'ruhige Küche', listing_ids=[1], language='de')
        self.assertEqual([r['review_id'] for r in german['rows']], [102])
        recent = queries.search_reviews(self.connection, 'quiet', since='2026-02-01')
        self.assertEqual([r['review_id'] for r in recent['rows']], [101])
        self.assertEqual(queries.search_reviews(self.connection, 'missingword')['rows'], [])
        page = queries.search_reviews(self.connection, 'quiet', limit=1, offset=1)
        self.assertEqual(page['total_matches'], 3)
        self.assertTrue(page['has_more'])
        literal = queries.search_reviews(self.connection, "quiet'); DELETE FROM reviews; --")
        self.assertEqual(literal['rows'], [])

    def test_transaction_enforces_read_only_and_restores_connection(self):
        with self.assertRaises(queries.QueryError):
            queries._run(self.connection, 'test', lambda s: s.fetch(
                'DELETE FROM vienna_stay_assistant.listings RETURNING listing_id'))
        self.assertEqual(self.connection.info.transaction_status, 0)
        self.assertEqual(queries.search_listings(self.connection)['total_matches'], 5)
        with self.connection.transaction():
            with self.assertRaises(ValueError):
                queries.search_listings(self.connection)


if __name__ == '__main__':
    unittest.main()
