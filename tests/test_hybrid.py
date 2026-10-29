import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval import hybrid
from retrieval.hybrid import HybridRetriever, rrf, terms

LANGCHAIN = importlib.util.find_spec('langchain_core') is not None
QDRANT = importlib.util.find_spec('qdrant_client') is not None


def review(review_id, listing_id=1, day=date(2026, 1, 1), text='text'):
    return {'review_id': review_id, 'listing_id': listing_id, 'date': day, 'comments': text,
            'language': 'en', 'source_file': 'reviews.csv', 'snapshot_date': date(2026, 9, 17), 'load_id': 'L1'}


class FakeSession:
    def __init__(self, lexical, stored):
        self.lexical, self.stored, self.evidence, self.calls = lexical, stored, [], []

    def fetch(self, sql, parameters=()):
        self.calls.append((sql, parameters))
        if 'lexical_score' in sql:
            return [{'review_id': rid, 'lexical_score': score} for rid, score in self.lexical]
        wanted = parameters[0]
        return [self.stored[rid] for rid in wanted if rid in self.stored]


def fake_qdrant(hits):
    client = Mock()
    client.query_points.return_value = SimpleNamespace(points=[
        SimpleNamespace(score=score, payload={'review_id': rid, 'chunk_id': chunk}) for rid, chunk, score in hits])
    return client


@unittest.skipUnless(QDRANT, 'Requires qdrant-client')
class HybridSearchTests(unittest.TestCase):
    def run_search(self, *, dense, lexical, stored, rerank=None, **kwargs):
        session = FakeSession(lexical, stored)

        def run(connection, tool, action, notices):
            result = action(session)
            return {'tool': tool, **result, 'evidence': session.evidence, 'notices': list(notices)}

        retriever = HybridRetriever(Mock(), fake_qdrant(dense), 'vienna_test', lambda texts: [[0.1, 0.2]], rerank)
        with patch('langchain_agent.queries._run', side_effect=run):
            return retriever.search('quiet at night', **kwargs), session, retriever

    def test_both_signals_are_fused_and_keep_citation_keys(self):
        stored = {1: review(1), 2: review(2), 3: review(3)}
        result, _, _ = self.run_search(dense=[(1, 'review:1:0', 0.9), (2, 'review:2:1', 0.8)],
                                       lexical=[(2, 0.5), (3, 0.4)], stored=stored)
        ids = [row['review_id'] for row in result['rows']]
        self.assertEqual(ids[0], 2)
        self.assertEqual(set(ids), {1, 2, 3})
        top = result['rows'][0]
        self.assertEqual(top['chunk_id'], 'review:2:1')
        self.assertEqual((top['retrieval']['dense_rank'], top['retrieval']['lexical_rank']), (2, 1))
        lexical_only = next(r for r in result['rows'] if r['review_id'] == 3)
        self.assertEqual(lexical_only['chunk_id'], 'review:3:0')
        self.assertIsNone(lexical_only['retrieval']['dense_rank'])
        self.assertEqual(result['retrieval_method'], 'hybrid_rrf')
        self.assertIn('qdrant', result['evidence'][0])

    def test_chunks_of_one_review_collapse_to_the_best_chunk(self):
        result, _, _ = self.run_search(dense=[(1, 'review:1:2', 0.9), (1, 'review:1:0', 0.7)],
                                       lexical=[], stored={1: review(1)})
        self.assertEqual(len(result['rows']), 1)
        self.assertEqual(result['rows'][0]['chunk_id'], 'review:1:2')

    def test_reranker_decides_the_final_order(self):
        stored = {1: review(1, text='loud street'), 2: review(2, text='very quiet nights')}
        result, _, _ = self.run_search(dense=[(1, 'review:1:0', 0.9), (2, 'review:2:0', 0.8)], lexical=[],
                                       stored=stored, rerank=lambda q, texts: [0.9 if 'quiet' in t else 0.1 for t in texts])
        self.assertEqual([r['review_id'] for r in result['rows']], [2, 1])
        self.assertEqual(result['retrieval_method'], 'hybrid_rrf_rerank')

    def test_reranker_failure_falls_back_to_fused_order(self):
        def broken(query, texts):
            raise RuntimeError('model missing')
        result, _, _ = self.run_search(dense=[(1, 'review:1:0', 0.9)], lexical=[], stored={1: review(1)}, rerank=broken)
        self.assertEqual(result['retrieval_method'], 'hybrid_rrf')
        self.assertTrue(any('Reranker' in n for n in result['notices']))

    def test_qdrant_failure_falls_back_to_keyword_search(self):
        session = FakeSession([(5, 0.3)], {5: review(5)})

        def run(connection, tool, action, notices):
            return {**action(session), 'notices': list(notices)}
        client = Mock()
        client.query_points.side_effect = RuntimeError('connection refused')
        with patch('langchain_agent.queries._run', side_effect=run):
            result = HybridRetriever(Mock(), client, 'c', lambda t: [[0.1]]).search('quiet')
        self.assertEqual([r['review_id'] for r in result['rows']], [5])
        self.assertTrue(any('Semantic' in n for n in result['notices']))

    def test_hits_outside_the_filters_are_dropped_at_hydration(self):
        result, session, _ = self.run_search(dense=[(1, 'review:1:0', 0.9), (2, 'review:2:0', 0.8)], lexical=[],
                                             stored={1: review(1)}, listing_ids=[1], since='2025-01-01')
        self.assertEqual([r['review_id'] for r in result['rows']], [1])
        self.assertEqual(result['filtered_out'], 1)
        hydrate_sql, params = session.calls[-1]
        self.assertIn('r.date BETWEEN', hydrate_sql)
        self.assertIn([1], params)

    def test_listing_filter_reaches_qdrant(self):
        _, _, retriever = self.run_search(dense=[], lexical=[], stored={}, listing_ids=[7, 3])
        query_filter = retriever.qdrant.query_points.call_args.kwargs['query_filter']
        listing = next(c for c in query_filter.must if c.key == 'listing_id')
        self.assertEqual(listing.match.any, [3, 7])

    def test_limit_is_applied_after_fusion(self):
        stored = {i: review(i) for i in range(1, 11)}
        result, _, _ = self.run_search(dense=[(i, f'review:{i}:0', 1 - i / 100) for i in range(1, 11)],
                                       lexical=[], stored=stored, limit=3)
        self.assertEqual(len(result['rows']), 3)
        self.assertEqual(result['total_matches'], 10)

    def test_invalid_arguments_never_reach_the_stores(self):
        retriever = HybridRetriever(Mock(), Mock(), 'c', Mock())
        for kwargs in ({'query': '  '}, {'query': 'x', 'limit': 31}, {'query': 'x', 'listing_ids': []},
                       {'query': 'x', 'until': '2026-09-18'}, {'query': 'x', 'since': '2026-09-01', 'until': '2026-01-01'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                retriever.search(**kwargs)
        retriever.qdrant.query_points.assert_not_called()
        retriever.connection.transaction.assert_not_called()


class HelperTests(unittest.TestCase):
    def test_rrf_rewards_agreement(self):
        scores = rrf([['a', 'b'], ['b', 'c']], k=60)
        self.assertAlmostEqual(scores['b'], 1 / 62 + 1 / 61)
        self.assertGreater(scores['b'], scores['a'])
        self.assertGreater(scores['a'], scores['c'])

    def test_terms_drop_stopwords_and_syntax(self):
        self.assertEqual(terms('Quiet at night & clean!'), ['quiet', 'night', 'clean'])
        self.assertEqual(terms('Sehr ruhig und sauber'), ['ruhig', 'sauber'])
        self.assertEqual(terms('the and'), [])

    def test_latest_collection_ignores_incomplete_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'a.json').write_text(json.dumps({'collection': 'done', 'status': 'complete'}))
            Path(folder, 'b.json').write_text(json.dumps({'collection': 'partial', 'status': 'loading'}))
            self.assertEqual(hybrid.latest_collection(folder), 'done')
            Path(folder, 'a.json').unlink()
            with self.assertRaises(FileNotFoundError):
                hybrid.latest_collection(folder)


@unittest.skipUnless(LANGCHAIN, 'Requires langchain-core')
class HybridToolTests(unittest.TestCase):
    def test_tool_is_added_only_with_a_retriever(self):
        from langchain_agent.tools import build_tools
        self.assertNotIn('hybrid_search_reviews', [t.name for t in build_tools(Mock())])
        retriever = Mock()
        retriever.search.return_value = {'tool': 'hybrid_search_reviews', 'rows': []}
        tool = {t.name: t for t in build_tools(Mock(), retriever)}['hybrid_search_reviews']
        tool.invoke({'query': 'quiet', 'listing_ids': [1, 2], 'since': None})
        retriever.search.assert_called_once_with(query='quiet', listing_ids=[1, 2], limit=8)

    def test_tool_schema_rejects_bad_input(self):
        from langchain_agent.tools import build_tools
        tool = {t.name: t for t in build_tools(Mock(), Mock())}['hybrid_search_reviews']
        for args in ({'query': ''}, {'query': 'x', 'limit': 99}, {'query': 'x', 'listing_ids': ['1']}):
            with self.subTest(args=args), self.assertRaises(Exception):
                tool.invoke(args)


if __name__ == '__main__':
    unittest.main()
