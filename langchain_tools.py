"""Typed LangChain adapters for the existing fixed, read-only SQL queries."""

from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

import queries

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
    since: str = '2024-09-17'
    until: str = '2026-09-17'


DEFINITIONS = (
    ('search_listings', ListingSearch,
     'Filter Vienna snapshot listings by exact district/room/amenities, minimum capacity, '
     'inclusive base prices (decimal strings), superhost status or IDs. Returns paginated evidence.'),
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


def build_tools(connection):
    """Bind tools to one caller-owned idle connection. Use sequential execution.

    Results retain the complete queries.py evidence contract. The caller owns
    masking/guards before model use; these adapters perform no model inference.
    """
    def bind(name):
        def execute(**arguments):
            return getattr(queries, name)(connection, **arguments)
        return execute

    return [StructuredTool.from_function(
        func=bind(name), name=name, description=description, args_schema=schema,
    ) for name, schema, description in DEFINITIONS]
