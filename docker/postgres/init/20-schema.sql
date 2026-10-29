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
CREATE INDEX IF NOT EXISTS reviews_comments_fts_idx ON vienna_stay_assistant.reviews
    USING gin (to_tsvector('simple', comments));
CREATE INDEX IF NOT EXISTS calendar_available_date_idx ON vienna_stay_assistant.calendar (date, listing_id) WHERE available;

-- Strategic noise map (Lnight), loaded by ingest/noise.py. Linked to listings by spatial join, not a foreign key.
CREATE TABLE IF NOT EXISTS vienna_stay_assistant.noise_zones (
    zone_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source text NOT NULL,
    indicator text NOT NULL CHECK (indicator IN ('road', 'rail')),
    db_band text NOT NULL,
    geometry geometry(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS noise_zones_geometry_idx ON vienna_stay_assistant.noise_zones USING gist (geometry);
CREATE INDEX IF NOT EXISTS noise_zones_indicator_idx ON vienna_stay_assistant.noise_zones (indicator);

-- Worst noise band within 150 m of the listing. Added with ALTER so existing databases pick it up.
ALTER TABLE vienna_stay_assistant.listings ADD COLUMN IF NOT EXISTS lnight_road_band text;
ALTER TABLE vienna_stay_assistant.listings ADD COLUMN IF NOT EXISTS lnight_rail_band text;

-- Curated landmarks for queries like "near Westbahnhof"; coordinates from OpenStreetMap.
CREATE TABLE IF NOT EXISTS vienna_stay_assistant.landmarks (
    landmark_id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE,
    kind text NOT NULL CHECK (kind IN ('station', 'sight')),
    latitude double precision NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude double precision NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    location geography(Point, 4326) GENERATED ALWAYS AS (
        ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography) STORED
);
CREATE INDEX IF NOT EXISTS landmarks_location_idx ON vienna_stay_assistant.landmarks USING gist (location);

INSERT INTO vienna_stay_assistant.landmarks (name, kind, latitude, longitude) VALUES
    ('Wien Hauptbahnhof', 'station', 48.1850, 16.3779),
    ('Wien Westbahnhof', 'station', 48.1965, 16.3363),
    ('Wien Mitte', 'station', 48.2056, 16.3842),
    ('Praterstern', 'station', 48.2198, 16.3931),
    ('Franz-Josefs-Bahnhof', 'station', 48.2255, 16.3611),
    ('Stephansdom', 'sight', 48.2085, 16.3731),
    ('Schloss Schönbrunn', 'sight', 48.1850, 16.3116),
    ('Hofburg', 'sight', 48.2077, 16.3662),
    ('Schloss Belvedere', 'sight', 48.1915, 16.3809),
    ('Wiener Staatsoper', 'sight', 48.2033, 16.3692),
    ('Rathaus', 'sight', 48.2108, 16.3566),
    ('Parlament', 'sight', 48.2081, 16.3585),
    ('Riesenrad', 'sight', 48.2167, 16.3959),
    ('Naschmarkt', 'sight', 48.1982, 16.3626),
    ('MuseumsQuartier', 'sight', 48.2032, 16.3583),
    ('Kunsthistorisches Museum', 'sight', 48.2037, 16.3618),
    ('Votivkirche', 'sight', 48.2154, 16.3591),
    ('Hundertwasserhaus', 'sight', 48.2074, 16.3939),
    ('Karlskirche', 'sight', 48.1983, 16.3719),
    ('Albertina', 'sight', 48.2046, 16.3683)
ON CONFLICT (name) DO NOTHING;
