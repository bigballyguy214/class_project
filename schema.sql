-- Nightlife Autopilot | Group 7 | PostgreSQL schema
-- Contains no connection strings or credentials. Apply before running seed.py.
BEGIN;
CREATE TABLE IF NOT EXISTS venues (
    osm_id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    amenity TEXT NOT NULL,
    lat DOUBLE PRECISION CHECK (lat BETWEEN -90 AND 90),
    lon DOUBLE PRECISION CHECK (lon BETWEEN -180 AND 180),
    addr TEXT,
    postcode TEXT,
    opening_hours TEXT,
    outdoor_seating TEXT,
    website TEXT
);
CREATE INDEX IF NOT EXISTS venues_postcode_idx ON venues(postcode);
CREATE INDEX IF NOT EXISTS venues_amenity_idx ON venues(amenity);
-- Future Scout feature: empty placeholders, not invented observations.
CREATE TABLE IF NOT EXISTS app_users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    age INTEGER CHECK (age >= 18)
);
CREATE TABLE IF NOT EXISTS scout_reports (
    report_id TEXT PRIMARY KEY,
    venue_id TEXT NOT NULL REFERENCES venues(osm_id),
    user_id TEXT NOT NULL REFERENCES app_users(user_id),
    crowd_level INTEGER NOT NULL CHECK (crowd_level BETWEEN 1 AND 5),
    wait_minutes INTEGER CHECK (wait_minutes >= 0),
    reported_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMIT;
