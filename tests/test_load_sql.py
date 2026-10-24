import csv
import json
import os
import tempfile
import unittest
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from data_setup import load_sql as loader


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def fixture(root):
    listing = dict(id='1', host_id='10', host_name='Secret', description='<p>Quiet flat with Secret.</p>',
                   name='Flat of Secret', neighborhood_overview='', neighbourhood_cleansed='Neubau',
                   last_scraped='2026-09-19', latitude='48.2', longitude='16.3', price='$1,234.50',
                   amenities='["Wi-Fi"]', room_type='Entire home/apt', accommodates='4', host_about='Secret lives here',
                   host_since='2015-01-01', host_response_rate='99%', host_is_superhost='t', host_listings_count='2')
    write_csv(root / 'listings.csv', [listing, {**listing, 'id': '2', 'price': '', 'description': ''}])
    review = dict(id='100', listing_id='1', date='2026-01-01', reviewer_name='Secret',
                  comments='Secret says this is a very quiet and comfortable place to stay.')
    write_csv(root / 'reviews.csv', [review, review, {**review, 'id': '101'},
              {**review, 'id': '101', 'listing_id': '2'}, {**review, 'id': '102', 'date': '2026-09-18'},
              {**review, 'id': '103', 'listing_id': '999'},
              {**review, 'id': '104', 'comments': 'Too short'},
              {**review, 'id': '105', 'comments': 'The host canceled this reservation 2 days before arrival.'}])
    write_csv(root / 'calendar.csv', [dict(listing_id='1', date='2026-09-19', available='t', minimum_nights='2', maximum_nights='30'),
              dict(listing_id='1', date='2026-09-21', available='f', minimum_nights='', maximum_nights='30'),
              dict(listing_id='999', date='2026-09-19', available='t', minimum_nights='1', maximum_nights='30')])
    write_csv(root / 'neighbourhoods.csv', [dict(neighbourhood=n) for n in loader.DISTRICTS])
    geometry = {'type': 'Polygon', 'coordinates': [[[16, 48], [17, 48], [17, 49], [16, 49], [16, 48]]]}
    (root / 'neighbourhoods.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'neighbourhood': n}, 'geometry': geometry} for n in loader.DISTRICTS
    ]}), encoding='utf-8')


class SqlFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.args = loader.parse_args(['--data-dir', str(self.root)])
        self.masker = Mock()
        self.masker.mask.side_effect = lambda text, names: text.replace('Secret', '<PERSON_0>')


class SqlSourceTests(SqlFixture):
    def test_driver_failure_has_safe_install_instructions(self):
        with patch('data_setup.load_sql.import_module', side_effect=ImportError('private library context')):
            with self.assertRaises(loader.DependencyError) as caught:
                loader.postgres_driver()
        self.assertIn('psycopg[binary]', str(caught.exception))
        self.assertIn('requirements.txt', str(caught.exception))
        self.assertNotIn('private library context', str(caught.exception))

    def test_prices_and_strict_types(self):
        self.assertEqual(loader.number('$1,234.50', price=True), Decimal('1234.50'))
        self.assertIsNone(loader.number('', price=True))
        for value in ('NaN', '-1', '1.2'):
            with self.assertRaises(ValueError):
                loader.number(value, integer=True)
        for value in ('12,34', 'EUR 12', '1,2,3'):
            with self.assertRaises(ValueError):
                loader.number(value, price=True)
        with self.assertRaises(ValueError):
            loader.boolean('yes')

    def test_district_repairs_and_matching_boundaries(self):
        self.assertEqual(loader.district('D\x9abling'), 'Döbling')
        self.assertEqual(loader.district('Landstra\xa7e'), 'Landstraße')
        self.assertEqual(len(list(loader.neighbourhood_records(self.root, Counter()))), 23)
        with self.assertRaises(ValueError):
            loader.district('Unknown district')

    def test_mapping_masks_text_and_preserves_nulls(self):
        stats = Counter()
        listings, _ = loader.listing_sources(self.root, None, stats)
        records = list(loader.listing_records(listings, self.masker, stats))
        self.assertNotIn('Secret', repr(records))
        self.assertEqual(records[0][12], Decimal('1234.50'))
        self.assertIsNone(records[1][12])
        self.assertIsNone(records[1][4])
        self.assertEqual(stats['listings_missing_price'], 1)
        hosts = loader.host_sources(listings, stats)
        self.assertEqual(list(loader.host_records(hosts, self.masker))[0][1], '<PERSON_0>')

    def test_review_contract_and_language_after_masking(self):
        detector = Mock(return_value='en')
        stats = Counter()
        records = list(loader.review_records(self.args, {1: {}, 2: {}}, self.masker, detector, stats))
        self.assertEqual([r[0] for r in records], [100])
        self.assertNotIn('Secret', detector.call_args.args[0])
        self.assertEqual(stats['review_selection_conflicting_review_rows_excluded'], 2)
        self.assertEqual(stats['review_selection_identical_review_duplicates_skipped'], 1)
        self.assertEqual(stats['short_reviews'], 1)

    def test_calendar_parent_filter_and_missing_constraints(self):
        stats = Counter()
        records = list(loader.calendar_records(self.root, {1: {}}, {1, 2}, stats))
        self.assertEqual(records[0], (1, date(2026, 9, 19), True, 2, 30))
        self.assertIsNone(records[1][3])
        self.assertEqual(stats['orphan_calendar_rows_excluded'], 1)

    def test_duplicate_listing_outside_sample_still_fails(self):
        path = self.root / 'listings.csv'
        records = list(loader.rows(path, ['id']))
        write_csv(path, records + records[:1])
        with self.assertRaises(ValueError):
            loader.listing_sources(self.root, 1, Counter())

    def test_host_conflict_resolution_is_deterministic(self):
        listings, _ = loader.listing_sources(self.root, None, Counter())
        listings[2]['host_is_superhost'] = 'f'
        stats = Counter()
        self.assertEqual(loader.host_sources(listings, stats)[10][0], 1)
        self.assertEqual(stats['hosts_with_conflicting_fields'], 1)
        listings[2]['last_scraped'] = '2026-09-20'
        self.assertEqual(loader.host_sources(listings, stats)[10][0], 2)

    def test_masking_failure_aborts(self):
        self.masker.mask.side_effect = RuntimeError('unavailable')
        listings, _ = loader.listing_sources(self.root, None, Counter())
        with self.assertRaises(RuntimeError):
            list(loader.listing_records(listings, self.masker, Counter()))


@unittest.skipUnless(os.getenv('VSA_TEST_DATABASE_URL'), 'Requires an isolated vsa_test_* PostGIS database')
class SqlIntegrationTests(SqlFixture):
    # Real SQL transactions with a fake masker.
    # This validates integration, never statistical privacy recall.
    def test_atomic_import_replacement_and_rollback(self):
        import psycopg
        with psycopg.connect(os.environ['VSA_TEST_DATABASE_URL'], autocommit=True) as conn:
            database = conn.execute('SELECT current_database()').fetchone()[0]
            self.assertTrue(database.startswith('vsa_test_'), 'Refusing a non-test database')
            conn.execute('CREATE TABLE public.listings (sentinel text)')
            conn.execute("INSERT INTO public.listings VALUES ('unrelated')")
            with patch('data_setup.load_sql.version', return_value='test'):
                report = loader.ingest(conn, self.args, self.masker, lambda text: 'en')
                self.assertEqual(report['validation']['listings'], 2)
                self.assertEqual(report['validation']['reviews'], 1)
                self.assertEqual(report['validation']['calendar'], 2)
                self.assertEqual(report['validation']['calendar_coverage']['listings_with_internal_gaps'], 1)
                self.assertEqual(report['validation']['calendar_coverage']['listings_without_calendar'], 1)
                self.assertEqual(conn.execute('SELECT ST_SRID(geometry) FROM vienna_stay_assistant.neighbourhoods LIMIT 1').fetchone()[0], 4326)
                self.assertEqual(conn.execute('SELECT ST_X(location::geometry) FROM vienna_stay_assistant.listings LIMIT 1').fetchone()[0], 16.3)
                with self.assertRaises(ValueError):
                    loader.ingest(conn, self.args, self.masker, lambda text: 'en')
                self.args.replace = True
                report2 = loader.ingest(conn, self.args, self.masker, lambda text: 'en')
                self.assertNotEqual(report['load_id'], report2['load_id'])
                path = self.root / 'calendar.csv'
                records = list(loader.rows(path, ['listing_id']))
                write_csv(path, records + records[:1])
                with self.assertRaises(psycopg.errors.UniqueViolation):
                    loader.ingest(conn, self.args, self.masker, lambda text: 'en')
                self.assertEqual(conn.execute('SELECT count(*) FROM vienna_stay_assistant.calendar').fetchone()[0], 2)
                self.assertEqual(conn.execute('SELECT count(*) FROM vienna_stay_assistant.ingestion_runs').fetchone()[0], 2)
                self.assertEqual(str(conn.execute('SELECT DISTINCT load_id FROM vienna_stay_assistant.listings').fetchone()[0]), report2['load_id'])
                self.assertEqual(conn.execute('SELECT sentinel FROM public.listings').fetchone()[0], 'unrelated')


if __name__ == '__main__':
    unittest.main()
