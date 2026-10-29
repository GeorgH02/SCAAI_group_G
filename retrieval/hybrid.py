import json
import re
from datetime import date
from pathlib import Path

from langchain_agent import queries

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_END = date(2026, 9, 17)
DEFAULT_SINCE = '2024-09-17'
RRF_K = 60

STOPWORDS = set('''
a an and are as at be but by for from has have i in is it its of on or so that the this to was were with
we you our my me very really just also all any there their they them he she his her not no
der die das den dem des ein eine einen einem einer und oder ist sind war waren mit von zu im in am an auf
für fur bei aus es wir ich sie er uns unser unsere sehr auch nicht kein keine noch nur so wie als
'''.split())
MAX_TERMS = 12


def terms(query):
    words = []
    for word in re.findall(r'\w+', query.lower()):
        if len(word) > 1 and word not in STOPWORDS and word not in words:
            words.append(word)
    return words[:MAX_TERMS]


def rrf(rankings, k=RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def latest_collection(directory=ROOT / 'derived' / 'vectors'):
    manifests = []
    for path in Path(directory).glob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if data.get('status') == 'complete' and data.get('collection'):
            manifests.append((path.stat().st_mtime, data['collection']))
    if not manifests:
        raise FileNotFoundError('No complete vector manifest found; run data_setup/load_vectors.py first')
    return max(manifests)[1]


def bge_reranker(model='BAAI/bge-reranker-v2-m3', device=None, max_chars=2000):
    from sentence_transformers import CrossEncoder
    encoder = CrossEncoder(model, device=device, max_length=512)
    return lambda query, texts: [float(s) for s in encoder.predict([(query, t[:max_chars]) for t in texts])]


class HybridRetriever:
    def __init__(self, connection, qdrant, collection, embed, rerank=None, *,
                 dense_k=50, lexical_k=50, rerank_k=30, rrf_k=RRF_K):
        self.connection, self.qdrant, self.collection = connection, qdrant, collection
        self.embed, self.rerank = embed, rerank
        self.dense_k, self.lexical_k, self.rerank_k, self.rrf_k = dense_k, lexical_k, rerank_k, rrf_k

    def _dense(self, query, listing_ids, language):
        from qdrant_client import models
        must = [models.FieldCondition(key='source_type', match=models.MatchValue(value='review'))]
        if listing_ids:
            must.append(models.FieldCondition(key='listing_id', match=models.MatchAny(any=listing_ids)))
        if language:
            must.append(models.FieldCondition(key='language', match=models.MatchValue(value=language)))
        vector = [float(x) for x in self.embed([query])[0]]
        points = self.qdrant.query_points(
            collection_name=self.collection, query=vector, query_filter=models.Filter(must=must),
            limit=self.dense_k, with_payload=['review_id', 'chunk_id'],
        ).points
        ranked, best = [], {}
        for point in points:
            review_id = point.payload['review_id']
            if review_id not in best:
                best[review_id] = {'chunk_id': point.payload['chunk_id'], 'dense_score': float(point.score)}
                ranked.append(review_id)
        return ranked, best

    def search(self, query, *, listing_ids=None, language=None, since=DEFAULT_SINCE, until=None, limit=8):
        query = queries._text(query, 'query', maximum=500)
        queries._integer(limit, 'limit', high=30)
        ids = queries._ids(listing_ids) if listing_ids is not None else None
        if language is not None:
            language = queries._text(language, 'language', maximum=10)
        start, end = queries._day(since, 'since'), queries._day(until or SNAPSHOT_END.isoformat(), 'until')
        if start > end or end > SNAPSHOT_END:
            raise ValueError('Review dates must be ordered and end no later than the snapshot')

        notices = []
        try:
            dense_ranked, dense_info = self._dense(query, ids, language)
        except Exception:
            dense_ranked, dense_info = [], {}
            notices.append('Semantic (vector) search was unavailable; results use keyword search only.')

        words = terms(query)
        filters, filter_params = ['r.date BETWEEN %s AND %s'], [start, end]
        if ids is not None:
            filters.append('r.listing_id = ANY(%s::bigint[])')
            filter_params.append(ids)
        if language is not None:
            filters.append('r.language = %s')
            filter_params.append(language)
        where = ' AND '.join(filters)

        def action(s):
            lexical_ranked, lexical_scores = [], {}
            if words:
                sql = ('''SELECT r.review_id, ts_rank(to_tsvector('simple', r.comments), q.terms) AS lexical_score
                    FROM vienna_stay_assistant.reviews r CROSS JOIN to_tsquery('simple', %s) AS q(terms)
                    WHERE to_tsvector('simple', r.comments) @@ q.terms AND ''' + where
                       + ' ORDER BY lexical_score DESC, r.review_id LIMIT %s')
                for row in s.fetch(sql, (' | '.join(words), *filter_params, self.lexical_k)):
                    lexical_ranked.append(row['review_id'])
                    lexical_scores[row['review_id']] = float(row['lexical_score'])
            s.evidence.append({'qdrant': {'collection': self.collection, 'source_type': 'review',
                                          'listing_ids': ids, 'language': language, 'limit': self.dense_k}})
            fused = rrf([dense_ranked, lexical_ranked], self.rrf_k)
            order = sorted(fused, key=lambda key: (-fused[key], key))
            rows = {}
            if order:
                sql = ('''SELECT r.review_id, r.listing_id, r.date, r.comments, r.language,
                    r.source_file, r.snapshot_date, r.load_id
                    FROM vienna_stay_assistant.reviews r WHERE r.review_id = ANY(%s::bigint[]) AND ''' + where)
                rows = {row['review_id']: row for row in s.fetch(sql, (order, *filter_params))}
            dropped = len(order) - len(rows)
            candidates = []
            for review_id in order:
                row = rows.get(review_id)
                if row is None:
                    continue
                info = dense_info.get(review_id, {})
                candidates.append({
                    'review_id': review_id, 'listing_id': row['listing_id'], 'date': row['date'],
                    'chunk_id': info.get('chunk_id', f'review:{review_id}:0'),
                    'text': row['comments'], 'language': row['language'],
                    'source_file': row['source_file'], 'snapshot_date': row['snapshot_date'], 'load_id': row['load_id'],
                    'retrieval': {
                        'dense_rank': dense_ranked.index(review_id) + 1 if review_id in dense_info else None,
                        'dense_score': info.get('dense_score'),
                        'lexical_rank': lexical_ranked.index(review_id) + 1 if review_id in lexical_scores else None,
                        'lexical_score': lexical_scores.get(review_id),
                        'rrf_score': round(fused[review_id], 6),
                        'rerank_score': None,
                    },
                })
            method = 'hybrid_rrf'
            if self.rerank and candidates:
                top = candidates[:self.rerank_k]
                try:
                    scores = self.rerank(query, [c['text'] for c in top])
                    for candidate, score in zip(top, scores):
                        candidate['retrieval']['rerank_score'] = round(float(score), 6)
                    candidates = sorted(top, key=lambda c: -c['retrieval']['rerank_score']) + candidates[self.rerank_k:]
                    method = 'hybrid_rrf_rerank'
                except Exception:
                    notices.append('Reranker was unavailable; results are in fused (RRF) order.')
            return {'retrieval_method': method, 'query_terms': words,
                    'dense_candidates': len(dense_ranked), 'lexical_candidates': len(lexical_ranked),
                    'fused_candidates': len(order), 'filtered_out': dropped,
                    'rows': candidates[:limit], 'total_matches': len(candidates), 'limit': limit}

        notices += ['Reviews are masked before storage; masking does not guarantee zero PII leakage.',
                    'Lexical signal is PostgreSQL full-text ranking (ts_rank), not BM25.']
        return queries._run(self.connection, 'hybrid_search_reviews', action, notices)
