-- The recommender's own tables, in its own database: nothing here belongs to TypeType-Server.

CREATE EXTENSION IF NOT EXISTS vector;

-- Title vectors, by model build and cleaned title. They hold public video titles only, never
-- who watched them. The column takes any dimension: the model decides it.
CREATE TABLE IF NOT EXISTS title_vectors (
    model text NOT NULL,
    text_hash bytea NOT NULL,
    text text NOT NULL,
    vector vector NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, text_hash)
);

-- What videos say about themselves (tags, category), kept when Server sends them.
CREATE TABLE IF NOT EXISTS catalogue_videos (
    video_id text PRIMARY KEY,
    title text NOT NULL DEFAULT '',
    channel_url text NOT NULL DEFAULT '',
    category text NOT NULL DEFAULT '',
    tags jsonb NOT NULL DEFAULT '[]',
    description text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now()
);
