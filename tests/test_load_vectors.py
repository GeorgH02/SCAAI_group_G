import csv
import tempfile
import unittest
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data_setup.load_vectors import chunks, clean_text, documents, is_cancellation, parse_args, prepare, Masker, load, select_device


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class LoaderTests(unittest.TestCase):
    def test_device_defaults_to_auto(self):
        self.assertEqual(parse_args([]).device, 'auto')

    def test_auto_device_uses_cuda_when_available(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {'torch': torch}):
            self.assertEqual(select_device('auto'), 'cuda')

    def test_auto_device_falls_back_to_cpu(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {'torch': torch}):
            self.assertEqual(select_device('auto'), 'cpu')

    def test_explicit_device_is_preserved(self):
        torch = Mock()
        with patch.dict(sys.modules, {'torch': torch}):
            for device in ('cpu', 'cuda', 'cuda:1', 'mps'):
                self.assertEqual(select_device(device), device)
            torch.cuda.is_available.assert_not_called()

    def test_html_and_unicode(self):
        self.assertEqual(clean_text('<p>Quiet&nbsp;flat</p><br>near\n the station'),
                         'Quiet flat near the station')

    def test_chunk_coverage_and_overlap(self):
        text = 'abcdefghijklmnop'
        result = list(chunks(text, CharacterTokenizer(), 6, 2))
        self.assertEqual(result, ['abcdef', 'efghij', 'ijklmn', 'mnop'])
        self.assertEqual(result[0] + ''.join(c[2:] for c in result[1:]), text)
        self.assertEqual(list(chunks('', CharacterTokenizer(), 6, 2)), [])

    def test_masking_combines_languages_and_overlaps(self):
        masker = Masker.__new__(Masker)
        masker.analyzer = Mock()
        masker.analyzer.analyze.side_effect = [
            [SimpleNamespace(start=0, end=4, entity_type='PERSON')],
            [SimpleNamespace(start=0, end=8, entity_type='PERSON')],
        ]
        self.assertEqual(masker.mask('Jane Doe says hello'), '<PERSON_1> says hello')
        self.assertEqual(masker.analyzer.analyze.call_count, 2)

    def test_known_names_are_removed_before_analysis(self):
        masker = Masker.__new__(Masker)
        masker.analyzer = Mock()
        masker.analyzer.analyze.return_value = []
        result = masker.mask('Thanks Anna, from ANNABEL.', ['Anna'])
        self.assertEqual(result, 'Thanks <PERSON_0>, from ANNABEL.')
        for call in masker.analyzer.analyze.call_args_list:
            self.assertNotIn('Thanks Anna', call.kwargs['text'])

    def test_masks_before_language_detection_and_embedding_payload(self):
        masker = Mock()
        masker.mask.return_value = 'A quiet <PERSON_1> apartment'
        detector = Mock(return_value='en')
        args = SimpleNamespace(min_review_chars=5, chunk_tokens=100, overlap_tokens=5)
        result = list(prepare(iter([('raw private name', ['name'], {
            'source_type': 'review', 'source_id': 7, 'review_id': 7, 'listing_id': 2,
        })]), masker, CharacterTokenizer(), detector, args, Counter()))
        detector.assert_called_once_with(masker.mask.return_value)
        self.assertEqual(result[0]['text'], masker.mask.return_value)
        self.assertEqual(result[0]['review_id'], 7)
        self.assertEqual(result[0]['snapshot_date'], '2026-09-17')

    def test_missing_masker_does_not_pass_raw_text(self):
        masker = Mock()
        masker.mask.side_effect = RuntimeError('masking unavailable')
        with self.assertRaises(RuntimeError):
            list(prepare(iter([('raw', [], {})]), masker, CharacterTokenizer(), Mock(),
                         parse_args([]), Counter()))

    def test_source_selection_dates_and_orphans(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (root / 'listings.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'description', 'host_name', 'last_scraped'])
                writer.writerow([2, '<p>Quiet apartment</p>', 'Host', '2026-09-19'])
            with (root / 'reviews.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'listing_id', 'date', 'reviewer_name', 'comments'])
                for rid, lid, day, text in [
                    (1, 2, '2024-09-17', 'Nice stay'),
                    (2, 2, '2026-09-17', 'Nice stay'),
                    (3, 2, '2024-09-16', 'Old stay'),
                    (4, 2, '2026-09-18', 'Future stay'),
                    (5, 999, '2026-01-01', 'Orphan'),
                    (6, 2, '2026-01-01', 'The host canceled this reservation 2 days before arrival.'),
                    (7, 2, '2026-01-01', 'Ambiguous listing link'),
                    (7, 999, '2026-01-01', 'Ambiguous listing link'),
                    (1, 2, '2024-09-17', 'Nice stay'),
                ]:
                    writer.writerow([rid, lid, day, 'Reviewer', text])
            stats = Counter()
            output = list(documents(root, date(2024, 9, 17), stats))
            self.assertEqual([p[2].get('review_id') for p in output], [None, 1, 2])
            self.assertEqual(stats['reviews_outside_window'], 2)
            self.assertEqual(stats['orphan_reviews'], 1)
            self.assertEqual(stats['cancellation_reviews'], 1)
            self.assertEqual(stats['conflicting_review_ids'], 1)
            self.assertEqual(stats['conflicting_review_rows_excluded'], 2)
            self.assertEqual(stats['identical_review_duplicates_skipped'], 1)

    def test_cancellation_languages(self):
        self.assertTrue(is_cancellation('The host cancelled this reservation 5 days before arrival.'))
        self.assertTrue(is_cancellation('Der Gastgeber stornierte diese Buchung.'))
        self.assertFalse(is_cancellation('The apartment was quiet and comfortable.'))

    def test_invalid_chunk_options(self):
        with self.assertRaises(SystemExit):
            parse_args(['--chunk-tokens', '5', '--overlap-tokens', '5'])

    def test_upload_and_resume_preserve_ids_and_masked_text(self):
        # Exercise orchestration without downloading large language models.
        fake_models = SimpleNamespace(PointStruct=lambda **kwargs: SimpleNamespace(**kwargs))
        client = Mock()
        client.retrieve.return_value = []
        vector = Mock()
        vector.tolist.return_value = [0.1, 0.2]
        model = Mock()
        model.encode.return_value = [vector]
        payload = {'chunk_id': 'review:7:0', 'text': 'Hello <PERSON_1>', 'review_id': 7, 'listing_id': 2}
        with patch.dict(sys.modules, {'qdrant_client': SimpleNamespace(models=fake_models)}), patch('builtins.print'):
            stats = Counter()
            load(client, model, iter([payload]), 'test_collection', 'fingerprint', 16, stats)
            point = client.upsert.call_args.kwargs['points'][0]
            self.assertEqual(point.payload, payload)
            self.assertTrue(client.upsert.call_args.kwargs['wait'])
            self.assertEqual(stats['chunks_uploaded'], 1)
            model.encode.assert_called_once_with([payload['text']], batch_size=16,
                                                  normalize_embeddings=True, show_progress_bar=False)
            client.retrieve.return_value = [SimpleNamespace(id=point.id)]
            client.upsert.reset_mock()
            model.encode.reset_mock()
            load(client, model, iter([payload]), 'test_collection', 'fingerprint', 16, stats)
            self.assertEqual(client.retrieve.call_args.kwargs['ids'], [point.id])
            model.encode.assert_not_called()
            client.upsert.assert_not_called()
            self.assertEqual(stats['chunks_already_present'], 1)


if __name__ == '__main__':
    unittest.main()
