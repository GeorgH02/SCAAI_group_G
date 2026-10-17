-- Applied on first Docker initialization and by load_sql.py for existing volumes.
-- CREATE IF NOT EXISTS is bootstrap, not a migration mechanism.
CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;
CREATE SCHEMA IF NOT EXISTS vienna_stay_assistant;

CREATE TABLE IF NOT EXISTS vienna_stay_assistant.ingestion_runs (
    load_id uuid PRIMARY KEY,
    completed_at timestamptz NOT NULL DEFAULT now(),
    snapshot_date date NOT NULL,
    config jsonb NOT NULL,
    validation jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS vienna_stay_assistant.neighbourhoods (
    neighbourhood text PRIMARY KEY,
    district_number smallint NOT NULL UNIQUE CHECK (district_number BETWEEN 1 AND 23),
    geometry geometry(MultiPolygon, 4326) NOT NULL CHECK (ST_IsValid(geometry)),
    source_file text NOT NULL DEFAULT 'neighbourhoods.csv + neighbourhoods.geojson',
    snapshot_date date NOT NULL,
    load_id uuid NOT NULL REFERENCES vienna_stay_assistant.ingestion_runs
);

CREATE TABLE IF NOT EXISTS vienna_stay_assistant.hosts (
    host_id bigint PRIMARY KEY CHECK (host_id > 0),
    host_name text, -- fixed placeholder; no source name or reversible mapping
    host_since date,
    host_about text,
    host_is_superhost boolean,
    host_listings_count integer CHECK (host_listings_count >= 0),
    host_response_rate numeric CHECK (host_response_rate BETWEEN 0 AND 100),
    source_listing_id bigint NOT NULL, -- selected source row; provenance, not another FK
    last_scraped date NOT NULL,
    source_file text NOT NULL DEFAULT 'listings.csv',
    snapshot_date date NOT NULL,
    load_id uuid NOT NULL REFERENCES vienna_stay_assistant.ingestion_runs
);

CREATE TABLE IF NOT EXISTS vienna_stay_assistant.listings (
    listing_id bigint PRIMARY KEY CHECK (listing_id > 0),
    host_id bigint NOT NULL REFERENCES vienna_stay_assistant.hosts,
    neighbourhood text NOT NULL REFERENCES vienna_stay_assistant.neighbourhoods,
    name text,
    description text,
    neighborhood_overview text,
    property_type text,
    room_type text,
    accommodates integer CHECK (accommodates >= 0),
    bedrooms numeric CHECK (bedrooms >= 0),
    beds numeric CHECK (beds >= 0),
    bathrooms numeric CHECK (bathrooms >= 0),
    price numeric(14,2) CHECK (price >= 0), -- snapshot base price; no currency conversion
    amenities text[],
    minimum_nights integer CHECK (minimum_nights >= 0),
    maximum_nights integer CHECK (maximum_nights >= 0),
    review_scores_rating numeric,
    review_scores_accuracy numeric,
    review_scores_cleanliness numeric,
    review_scores_checkin numeric,
    review_scores_communication numeric,
    review_scores_location numeric,
    review_scores_value numeric,
    latitude double precision CHECK (latitude BETWEEN -90 AND 90),
    longitude double precision CHECK (longitude BETWEEN -180 AND 180),
    location geography(Point, 4326) GENERATED ALWAYS AS (
        CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL
        THEN ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography END
    ) STORED,
    last_scraped date NOT NULL,
    source_file text NOT NULL DEFAULT 'listings.csv',
    snapshot_date date NOT NULL,
    load_id uuid NOT NULL REFERENCES vienna_stay_assistant.ingestion_runs
);

CREATE TABLE IF NOT EXISTS vienna_stay_assistant.reviews (
    review_id bigint PRIMARY KEY CHECK (review_id > 0),
    listing_id bigint NOT NULL REFERENCES vienna_stay_assistant.listings,
    date date NOT NULL,
    comments text NOT NULL,
    language text NOT NULL,
    source_file text NOT NULL DEFAULT 'reviews.csv',
    snapshot_date date NOT NULL,
    load_id uuid NOT NULL REFERENCES vienna_stay_assistant.ingestion_runs
    -- Reviewer identity is unnecessary for evidence citations and is not stored.
);

CREATE TABLE IF NOT EXISTS vienna_stay_assistant.calendar (
    listing_id bigint NOT NULL REFERENCES vienna_stay_assistant.listings,
    date date NOT NULL,
    available boolean NOT NULL,
    minimum_nights integer CHECK (minimum_nights >= 0),
    maximum_nights integer CHECK (maximum_nights >= 0),
    source_file text NOT NULL DEFAULT 'calendar.csv',
    snapshot_date date NOT NULL,
    load_id uuid NOT NULL REFERENCES vienna_stay_assistant.ingestion_runs,
    PRIMARY KEY (listing_id, date)
);

CREATE INDEX IF NOT EXISTS listings_host_idx ON vienna_stay_assistant.listings (host_id);
CREATE INDEX IF NOT EXISTS listings_filter_idx ON vienna_stay_assistant.listings (neighbourhood, room_type, accommodates, price);
CREATE INDEX IF NOT EXISTS listings_location_idx ON vienna_stay_assistant.listings USING gist (location);
CREATE INDEX IF NOT EXISTS neighbourhoods_geometry_idx ON vienna_stay_assistant.neighbourhoods USING gist (geometry);
CREATE INDEX IF NOT EXISTS reviews_listing_date_idx ON vienna_stay_assistant.reviews (listing_id, date);
CREATE INDEX IF NOT EXISTS calendar_available_date_idx ON vienna_stay_assistant.calendar (date, listing_id) WHERE available;
