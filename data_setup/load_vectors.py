"""Load masked reviews and listing descriptions into a versioned Qdrant collection.

Python 3.11+. See readme.md for dependencies and operational choices.
"""

import argparse
import csv
import hashlib
import html
import json
import logging
import os
import re
import sys
import unicodedata
import uuid
from collections import Counter
from datetime import date
from html.parser import HTMLParser
from importlib.metadata import version
from itertools import islice
from pathlib import Path

SNAPSHOT_DATE = "2026-09-17"
MODEL = "BAAI/bge-m3"
ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = 1


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        self.parts.append(" ")

    def handle_endtag(self, tag):
        self.parts.append(" ")


def clean_text(text):
    parser = TextExtractor()
    parser.feed(text or "")
    parser.close()
    return " ".join(unicodedata.normalize("NFKC", html.unescape("".join(parser.parts))).split())


def rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        yield from reader


def source_id(value):
    result = int(value)
    if result <= 0:
        raise ValueError("Source IDs must be positive integers")
    return result


def is_cancellation(text):
    return bool(re.search(
        r"^(?:the host cancel(?:ed|led) this reservation|"
        r"(?:der|die) gastgeber(?:in)? (?:hat .*storniert|stornierte))", text, re.I
    ))


class Masker:
    """Run both language recognizers before embedding; never save PII mappings."""

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_lg"},
                {"lang_code": "de", "model_name": "de_core_news_lg"},
            ],
        }).create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["en", "de"])

    def mask(self, text, names=()):
        # Explicit source names supplement statistical detection. Longest first.
        for name in sorted({clean_text(n) for n in names if n}, key=len, reverse=True):
            if name:
                text = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", "<PERSON_0>", text, flags=re.I)
        spans = []
        for language in ("en", "de"):
            spans.extend((r.start, r.end, r.entity_type) for r in self.analyzer.analyze(
                text=text, language=language,
                entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"], score_threshold=0.35,
            ))
        # Merge overlaps to avoid exposing part of a recognized entity.
        merged = []
        for start, end, kind in sorted(spans):
            if merged and start < merged[-1][1]:
                old_start, old_end, old_kind = merged[-1]
                merged[-1] = (old_start, max(end, old_end), old_kind)
            else:
                merged.append((start, end, kind))
        for index, (start, end, kind) in reversed(list(enumerate(merged, 1))):
            text = text[:start] + f"<{kind}_{index}>" + text[end:]
        return text


def chunks(text, tokenizer, size, overlap):
    # Token offsets preserve original masked wording and prevent model truncation.
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    for start in range(0, len(offsets), size - overlap):
        window = offsets[start:start + size]
        yield text[window[0][0]:window[-1][1]]
        if start + size >= len(offsets):
            break


def documents(data_dir, since, stats, limit=None):
    # A review key must resolve to one source row. Exclude every conflicting copy,
    # rather than arbitrarily attaching its evidence to the first listing seen.
    signatures = {}
    conflicts = set()
    review_columns = ["id", "listing_id", "date", "reviewer_name", "comments"]
    for row in rows(data_dir / "reviews.csv", review_columns):
        review_id = source_id(row["id"])
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).digest()
        if review_id in signatures and signatures[review_id] != digest:
            conflicts.add(review_id)
        signatures[review_id] = digest
    stats["conflicting_review_ids"] = len(conflicts)
    del signatures
    listings = {}
    yielded = 0
    for row in rows(data_dir / "listings.csv", ["id", "description", "host_name", "last_scraped"]):
        listing_id = source_id(row["id"])
        if listing_id in listings:
            raise ValueError("Duplicate listing_id")
        listings[listing_id] = row.get("host_name", "")
        stats["listings_read"] += 1
        text = clean_text(row["description"])
        if not text:
            stats["empty_descriptions"] += 1
            continue
        if limit is None or yielded < limit:
            yielded += 1
            yield text, [row.get("host_name", "")], {
                "source_type": "listing_description", "listing_id": listing_id,
                "source_id": listing_id, "source_file": "listings.csv",
                "source_field": "description", "last_scraped": row["last_scraped"],
            }
    seen = set()
    yielded = 0
    for row in rows(data_dir / "reviews.csv", review_columns):
        stats["reviews_read"] += 1
        review_id, listing_id = source_id(row["id"]), source_id(row["listing_id"])
        if review_id in conflicts:
            stats["conflicting_review_rows_excluded"] += 1
            continue
        if review_id in seen:
            stats["identical_review_duplicates_skipped"] += 1
            continue
        seen.add(review_id)
        review_date = date.fromisoformat(row["date"])
        if not since <= review_date <= date.fromisoformat(SNAPSHOT_DATE):
            stats["reviews_outside_window"] += 1
            continue
        if listing_id not in listings:
            stats["orphan_reviews"] += 1
            continue
        text = clean_text(row["comments"])
        if is_cancellation(text):
            stats["cancellation_reviews"] += 1
            continue
        if limit is None or yielded < limit:
            yielded += 1
            yield text, [row["reviewer_name"], listings[listing_id]], {
                "source_type": "review", "listing_id": listing_id,
                "review_id": review_id, "source_id": review_id,
                "source_file": "reviews.csv", "source_field": "comments",
                "review_date": review_date.isoformat(),
            }


def prepare(documents_iter, masker, tokenizer, detect_language, args, stats):
    for text, names, metadata in documents_iter:
        masked = masker.mask(text, names)
        if metadata["source_type"] == "review" and len(masked) < args.min_review_chars:
            stats["short_reviews"] += 1
            continue
        if not masked.strip():
            continue
        language = detect_language(masked)
        stats["documents_accepted"] += 1
        for index, chunk in enumerate(chunks(masked, tokenizer, args.chunk_tokens, args.overlap_tokens)):
            chunk_id = f"{metadata['source_type']}:{metadata['source_id']}:{index}"
            yield {
                **metadata, "text": chunk, "language": language,
                "chunk_id": chunk_id, "chunk_index": index,
                "snapshot_date": SNAPSHOT_DATE,
            }


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load(client, model, payloads, collection, fingerprint, batch_size, stats):
    from qdrant_client import models

    while batch := list(islice(payloads, batch_size)):
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint + p["chunk_id"])) for p in batch]
        existing = {str(p.id) for p in client.retrieve(
            collection_name=collection, ids=ids, with_payload=False, with_vectors=False
        )}
        pending = [(point_id, payload) for point_id, payload in zip(ids, batch) if point_id not in existing]
        stats["chunks_already_present"] += len(batch) - len(pending)
        if pending:
            vectors = model.encode([p["text"] for _, p in pending], batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
            client.upsert(collection_name=collection, wait=True, points=[
                models.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
                for (point_id, payload), vector in zip(pending, vectors)
            ])
            stats["chunks_uploaded"] += len(pending)
        print(json.dumps({"collection": collection, **stats}), flush=True)


def select_device(requested):
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--device", default="auto",
                        help="auto (CUDA when available, otherwise CPU), cpu, cuda, or another PyTorch device")
    parser.add_argument("--model-revision", default="main", help="Use a commit hash for pinned model weights")
    parser.add_argument("--reviews-since", type=date.fromisoformat, default=date(2024, 9, 17))
    parser.add_argument("--min-review-chars", type=int, default=30)
    parser.add_argument("--chunk-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Smoke test: at most this many candidate documents per source")
    args = parser.parse_args(argv)
    if not 0 <= args.overlap_tokens < args.chunk_tokens <= 8000:
        parser.error("Require 0 <= overlap-tokens < chunk-tokens <= 8000")
    if args.batch_size <= 0 or args.min_review_chars < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size, minimum length, and limit must be positive")
    if args.reviews_since > date.fromisoformat(SNAPSHOT_DATE):
        parser.error("Review window starts after the snapshot")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Avoid third-party logs containing document text. Our progress is counts only.
    logging.disable(logging.CRITICAL)
    from langdetect import DetectorFactory, detect, LangDetectException
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    DetectorFactory.seed = 0

    def language(text):
        try:
            return detect(text)
        except LangDetectException:
            return "und"

    print("Checking source files and Qdrant connection...", flush=True)
    hashes = {name: file_hash(args.data_dir / name) for name in ("listings.csv", "reviews.csv")}
    client = QdrantClient(url=args.qdrant_url, api_key=os.getenv("QDRANT_API_KEY"), timeout=120)
    client.get_collections()  # Fail before loading large models when Qdrant is unavailable.
    print("Loading Presidio and German/English spaCy models...", flush=True)
    masker = Masker()  # Fail closed if either language model is missing.
    device = select_device(args.device)
    print(f"Loading multilingual embedding model on {device}...", flush=True)
    model = SentenceTransformer(MODEL, revision=args.model_revision, device=device,
                                trust_remote_code=False)
    if not model.tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    if args.chunk_tokens + model.tokenizer.num_special_tokens_to_add() > model.max_seq_length:
        raise ValueError("Chunk size exceeds the embedding model input length")
    revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("Unable to resolve embedding model commit")
    config = {
        "pipeline_version": PIPELINE_VERSION, "snapshot_date": SNAPSHOT_DATE,
        "source_sha256": hashes, "model": MODEL, "model_revision": revision,
        "reviews_since": args.reviews_since.isoformat(), "min_review_chars": args.min_review_chars,
        "chunk_tokens": args.chunk_tokens, "overlap_tokens": args.overlap_tokens, "limit": args.limit,
        "dependencies": {n: version(n) for n in (
            "en-core-web-lg", "de-core-news-lg", "spacy", "presidio-analyzer",
            "sentence-transformers", "transformers", "tokenizers", "langdetect",
        )},
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    collection = f"vienna_{SNAPSHOT_DATE.replace('-', '_')}_{fingerprint[:16]}"
    size = model.get_sentence_embedding_dimension()
    if not client.collection_exists(collection):
        client.create_collection(collection_name=collection,
                                 vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE))
    else:
        vectors = client.get_collection(collection).config.params.vectors
        if not isinstance(vectors, models.VectorParams) or vectors.size != size or vectors.distance != models.Distance.COSINE:
            raise ValueError("Existing collection has incompatible vectors")
    for field, schema in [("listing_id", models.PayloadSchemaType.INTEGER),
                          ("review_id", models.PayloadSchemaType.INTEGER),
                          ("source_type", models.PayloadSchemaType.KEYWORD),
                          ("language", models.PayloadSchemaType.KEYWORD),
                          ("review_date", models.PayloadSchemaType.KEYWORD)]:
        client.create_payload_index(collection_name=collection, field_name=field, field_schema=schema, wait=True)
    stats = Counter()
    payloads = prepare(documents(args.data_dir, args.reviews_since, stats, args.limit),
                       masker, model.tokenizer, language, args, stats)
    print(json.dumps({"collection": collection, "status": "loading"}), flush=True)
    load(client, model, payloads, collection, fingerprint, args.batch_size, stats)
    # Detect edits during ingestion before declaring the collection ready.
    if hashes != {name: file_hash(args.data_dir / name) for name in hashes}:
        raise ValueError("Source files changed during ingestion; collection is incomplete")
    output = ROOT / "derived" / "vectors"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"{collection}.json"
    staging = manifest.with_suffix(".tmp")
    staging.write_text(json.dumps({"collection": collection, "status": "complete", "config": config,
                                  "stats": dict(stats)}, indent=2), encoding="utf-8")
    staging.replace(manifest)
    print(f"Complete. Manifest: {manifest}", flush=True)
    client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Exceptions from model/database libraries can contain input text or credentials.
        print(f"Load failed ({type(error).__name__}). Check dependencies, models, CSV schema and Qdrant. "
              "The collection may be partial; rerun the same command to resume.", file=sys.stderr)
        sys.exit(1)
