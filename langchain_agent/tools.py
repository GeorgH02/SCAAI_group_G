"""Typed LangChain adapters for the existing fixed, read-only SQL queries."""

from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_agent import queries

ID = Annotated[int, Field(strict=True, gt=0, le=9223372036854775807)]
IDs = Annotated[list[ID], Field(min_length=1, max_length=100)]
Text = Annotated[str, Field(min_length=1, max_length=200)]


class Arguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Page(Arguments):
    limit: Annotated[int, Field(ge=1, le=100)] = 20
    offset: Annotated[int, Field(ge=0, le=10000)] = 0


class ListingSearch(Page):
    district: Text | None = None
    room_type: Text | None = None
    accommodates: Annotated[int, Field(ge=1, le=1000)] | None = None
    min_price: Annotated[str, Field(max_length=30)] | None = None
    max_price: Annotated[str, Field(max_length=30)] | None = None
    amenities: Annotated[list[Text], Field(max_length=50)] | None = None
    superhost: bool | None = None
    near_landmark: Text | None = None
    radius_m: Annotated[int, Field(ge=1, le=10000)] = 500
    max_night_noise_db: Annotated[int, Field(ge=30, le=120)] | None = None
    listing_ids: IDs | None = None


class HostListings(Page):
    host_id: ID


class PriceComparison(Arguments):
    district: Text


class Availability(Arguments):
    listing_ids: IDs
    check_in: Annotated[str, Field(pattern=r'^\d{4}-\d{2}-\d{2}$')]
    check_out: Annotated[str, Field(pattern=r'^\d{4}-\d{2}-\d{2}$')]


class ReviewSearch(Page):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    listing_ids: IDs | None = None
    language: Annotated[str, Field(min_length=1, max_length=10)] | None = None
    since: str | None = None  # None means the default window
    until: str | None = None


class HybridSearch(Arguments):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    listing_ids: IDs | None = None
    language: Annotated[str, Field(min_length=1, max_length=10)] | None = None
    since: str | None = None
    until: str | None = None
    limit: Annotated[int, Field(ge=1, le=30)] = 8


HYBRID_DESCRIPTION = (
    'Semantic + keyword search of masked guest reviews (multilingual, German and English), fused and reranked. '
    'Use for themes or impressions such as "quiet at night", "clean", "helpful host"; pass listing_ids from '
    'search_listings to search only those listings. Returns review_id, listing_id and chunk_id per row for citations. '
    'Snapshot ends 2026-09-17.')

DEFINITIONS = (
    ('search_listings', ListingSearch,
     'Filter Vienna snapshot listings by exact district/room/amenities, minimum capacity, '
     'room_type must be exactly one of: Entire home/apt, Private room, Shared room, Hotel room; omit it unless asked. '
     'inclusive base prices (decimal strings), superhost status or IDs. near_landmark (a curated station or '
     'sight such as "Wien Westbahnhof"; unknown names return the known list) with radius_m keeps listings within that '
     'distance, nearest first. max_night_noise_db keeps listings whose worst Lnight band starts below that dB. '
     'Returns paginated evidence.'),
    ('check_availability', Availability,
     'Check every occupied night in [check_in, check_out), applying total stay length to '
     'each night. Missing coverage is unknown. Snapshot conditions are not live availability.'),
    ('compare_host_prices', PriceComparison,
     'Compare listing-weighted snapshot base prices by superhost status in an exact district. '
     'Preserves missing prices, unknown status, SQL and all contributing IDs.'),
    ('get_host_listings', HostListings,
     'Get paginated listings belonging to a known host_id, including source evidence.'),
    ('search_reviews', ReviewSearch,
     'AND keyword search of masked reviews, optionally constrained to eligible listing_ids, '
     'language and inclusive dates. No translation, semantic search or BM25. Snapshot ends 2026-09-17.'),
)


def build_tools(connection, retriever=None):
    """Bind tools to one caller-owned idle connection. Use sequential execution.

    Results retain the complete queries.py evidence contract. The caller owns
    masking/guards before model use; these adapters perform no model inference.
    With a retrieval.hybrid.HybridRetriever, a hybrid_search_reviews tool is added.
    """
    def bind(name):
        def execute(**arguments):
            # a null from the model means "not given"; the query then uses its own default
            given = {k: v for k, v in arguments.items() if v is not None or k not in ('since', 'until')}
            return getattr(queries, name)(connection, **given)
        return execute

    tools = [StructuredTool.from_function(
        func=bind(name), name=name, description=description, args_schema=schema,
    ) for name, schema, description in DEFINITIONS]
    if retriever is not None:
        def hybrid(**arguments):
            return retriever.search(**{k: v for k, v in arguments.items() if v is not None})
        tools.append(StructuredTool.from_function(
            func=hybrid, name='hybrid_search_reviews', description=HYBRID_DESCRIPTION, args_schema=HybridSearch))
    return tools
