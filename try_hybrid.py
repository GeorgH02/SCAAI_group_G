"""Try the hybrid review search directly (no LLM): dense + keyword, RRF fusion, optional reranker.

Usage (Postgres and Qdrant running, vectors and SQL loaded):
    .venv/bin/python try_hybrid.py "quiet at night"
    .venv/bin/python try_hybrid.py "sehr sauber" --listing-ids 123 456 --no-rerank
    .venv/bin/python try_hybrid.py                 # built-in example queries

Set QDRANT_COLLECTION to pick a collection; otherwise the newest complete manifest in derived/vectors is used.
"""

import argparse
import os
import sys
import time

from qdrant_client import QdrantClient

from guardrails.input_guard import bge_m3_embedder
from retrieval.hybrid import HybridRetriever, bge_reranker, latest_collection
from try_agent_db import connect

QUERIES = ['quiet at night', 'ruhig in der Nacht', 'host gave great restaurant tips', 'very clean and central']


def show(query, result, seconds):
    print(f'\n=== {query!r}  ({seconds:.2f} s, {result["retrieval_method"]})')
    print(f'terms {result["query_terms"]} | dense {result["dense_candidates"]} | keyword {result["lexical_candidates"]}'
          f' | fused {result["fused_candidates"]} | filtered out {result["filtered_out"]}')
    for row in result['rows']:
        r = row['retrieval']
        text = ' '.join(row['text'].split())[:110]
        print(f'  {row["chunk_id"]:<22} listing {row["listing_id"]:<20} dense#{r["dense_rank"]} kw#{r["lexical_rank"]}'
              f' rrf {r["rrf_score"]:.4f} rerank {r["rerank_score"]}\n      {text}')
    for notice in result['notices']:
        print(f'  notice: {notice}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('queries', nargs='*', default=QUERIES)
    parser.add_argument('--listing-ids', type=int, nargs='+')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--no-rerank', action='store_true')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    collection = os.getenv('QDRANT_COLLECTION') or latest_collection()
    print(f'Collection: {collection}\nLoading bge-m3{"" if args.no_rerank else " and bge-reranker-v2-m3"}...', flush=True)
    embed = bge_m3_embedder()
    rerank = None if args.no_rerank else bge_reranker()
    qdrant = QdrantClient(url=os.getenv('QDRANT_URL', 'http://127.0.0.1:6353'))
    with connect() as connection:
        retriever = HybridRetriever(connection, qdrant, collection, embed, rerank)
        for query in args.queries:
            start = time.perf_counter()
            result = retriever.search(query, listing_ids=args.listing_ids, limit=args.limit)
            show(query, result, time.perf_counter() - start)


if __name__ == '__main__':
    main()
