# Query tools

`queries.py` exposes five Python functions for a future tool-calling assistant.
They execute fixed, parameterized SQL against `vienna_stay_assistant`; they do
not accept generated SQL or call a language model. No extra dependency beyond
the existing psycopg requirement is needed.

## Calling the tools

Use an idle psycopg connection, preferably with a dedicated SELECT-only login.
Each call owns a repeatable-read, read-only transaction with a five-second
statement timeout. It rejects an existing transaction instead of committing or
altering the caller's work. Both default and autocommit connections are supported.
Connection creation and credentials belong to the caller; tools do not log
queries, database errors, credentials, or text. Database failures raise a generic
`QueryError`; argument errors raise `ValueError` before querying.

```python
import os
import psycopg
from queries import (
    search_listings, check_availability, compare_host_prices,
    get_host_listings, search_reviews,
)

with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
    listings = search_listings(
        connection, district="Neubau", room_type="Entire home/apt",
        accommodates=4, max_price="150.00", amenities=["Wifi"],
    )
    ids = [row["listing_id"] for row in listings["rows"]]
    if ids:
        stay = check_availability(connection, ids, "2026-12-31", "2027-01-02")
        eligible = [r["listing_id"] for r in stay["results"] if r["status"] == "available"]
        if eligible:
            reviews = search_reviews(connection, "quiet", listing_ids=eligible, language="en")

    comparison = compare_host_prices(connection, "Leopoldstadt")
    target = search_listings(connection, listing_ids=[48604277])
    if target["rows"]:
        related = get_host_listings(connection, target["rows"][0]["host_id"])
```

The example searches only the first page. Exhaust pagination when all eligible
listings are needed. Empty listing-ID filters are rejected: they must never
silently broaden a review search to all listings.

| Function | Parameters and behavior |
| --- | --- |
| `search_listings` | Optional `district`, `room_type`, minimum `accommodates`, inclusive `min_price`/`max_price`, exact array `amenities` (all required), boolean `superhost`, and `listing_ids`. Stable listing-ID order. |
| `check_availability` | Required `listing_ids`, `check_in`, `check_out`. Returns one status per distinct ID plus every requested night's calendar evidence. |
| `compare_host_prices` | Required `district`. Groups by true/false/unknown superhost status and averages listing prices, not host averages. No pagination of contributing IDs. |
| `get_host_listings` | Required `host_id`. Returns host existence/provenance and that host's listings, sorted by listing ID. |
| `search_reviews` | Required `query`; optional `listing_ids`, `language`, inclusive `since`/`until`. Default window is 2024-09-17 through 2026-09-17; dates after the snapshot are rejected. |

Listing, host-listing, and review searches default to `limit=20`, `offset=0`;
limits are 1–100 and offsets 0–10,000. Responses include `total_matches` and
`has_more`, including when an offset yields an empty page. ID lists accept
1–100 positive bigint values. Stays accept 1–366 nights. All inputs have bounds;
the timeout also bounds each SQL statement. Counts and active-load provenance
can scan source tables; these initial tools are not a full-scale latency benchmark.

## Evidence and numeric semantics

Every result includes `tool`, `provenance`, executed `evidence` SQL/parameters,
and `notices`. Source results carry primary keys, snapshot dates, and load IDs;
calendar results use `(listing_id, date)` when `calendar_present` is true.
Generated missing-night rows are diagnostic expectations, not source citations.
Active run provenance comes from the source tables and `ingestion_runs`, so
empty searches still disclose sample/full scope. Historical, inactive runs are
excluded. No active data produces an explicit notice. Unknown scope is never
assumed to mean full ingestion.

Results are JSON-compatible: prices/averages are decimal strings, dates are ISO
strings, and UUIDs are strings. Prices remain snapshot base prices without
currency conversion. Price bounds exclude NULL prices and report the exclusion
count among listings meeting the other filters, before pagination. Unfiltered
searches retain NULL prices.

Comparison groups include total and priced listing counts, missing-price counts,
all member listing IDs, distinct host IDs, exact priced contributing listing IDs,
and source load IDs. All-missing groups have a NULL average and no contributors.
No matching district returns no groups, not invented zero averages. These
results are factual evidence, not validated natural-language answers.

## Availability policy v1

This is an explicit implementation choice, not a claim about Airbnb booking rules:

- Occupied nights are `[check_in, check_out)`; checkout needs no calendar row.
- Every occupied night must have `available=true`.
- Total stay length must satisfy each occupied night's minimum and maximum.
- NULL bounds are unknown; listing-level bounds are not substituted.
- A definite blocked night or violated bound returns `unavailable`, even if other
  evidence is incomplete. Otherwise missing rows/bounds return `unknown`.
- An absent listing returns `not_found`. Only complete passing evidence returns
  `available`. Dates beyond source coverage therefore cannot be confirmed.

Each result includes failures, unknowns, the policy version, and per-night
evidence. Snapshot availability is never a guarantee of current bookability.

## Review search limitations and privacy

The implemented method is `postgres_simple_keyword`: PostgreSQL simple-tokenizer
full-text search with AND terms and `ts_rank_cd`, then review-ID tie breaking.
Terms are literal input to `plainto_tsquery`, not executable SQL or tsquery syntax.
The simple tokenizer can match German and English tokens but provides no
translation, stemming, semantic matching, BM25, fusion, or reranking. Only stored
masked comments are searched/returned; no raw CSV or reviewer identity is read.
There is currently no full-text index, so scans are bounded by the statement
timeout. Dense Qdrant integration remains a later extension.

The future application must mask user input before model inference, treat all
review text as untrusted data, and validate generated output. Search does not
establish privacy recall, sentence grounding, or theme support. Returned scores
are ranking signals, not confidence or NLI scores. The query and its SQL evidence
may contain user-entered text; do not log or send them to models before masking.

## Database role and tests

For deployment, have a database administrator create a dedicated login and grant
only `CONNECT` on `vienna_stay`, `USAGE` on `vienna_stay_assistant`, and `SELECT`
on its six tables. Do not give the assistant the importer/superuser credentials.
Role creation/password provisioning is deliberately not performed by tool calls.

Run the default suite with `py -3.11 -m unittest discover -s tests -v`.
For real query tests, create a **separate empty disposable** database named
`vsa_test_*`, set `VSA_QUERY_TEST_DATABASE_URL` to its connection string, then run
`py -3.11 -m unittest discover -s tests -p test_queries.py -v` in an environment
with psycopg installed. Do not point it at the loader test database:
`VSA_TEST_DATABASE_URL` is a different test fixture. Query tests refuse non-test
database names or an existing application schema; they create synthetic fixtures
and remove only their application schema afterward. Remove the disposable
database after testing. No models or full-source ingestion are needed.
