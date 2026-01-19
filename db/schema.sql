
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS emails (
  id           BIGSERIAL PRIMARY KEY,
  user_id      TEXT NOT NULL,
  message_id   TEXT NOT NULL,

  sender       TEXT,
  recipients   TEXT,
  subject      TEXT,
  sent_at      TIMESTAMPTZ,

  body         TEXT,
  url          TEXT,

  embedding    VECTOR(384) NOT NULL,
  metadata     JSONB DEFAULT '{}'::jsonb
);

ALTER TABLE emails
  ADD CONSTRAINT emails_user_message_unique UNIQUE (user_id, message_id);

CREATE INDEX IF NOT EXISTS idx_emails_user_id ON emails(user_id);
CREATE INDEX IF NOT EXISTS idx_emails_sent_at ON emails(sent_at);
CREATE INDEX IF NOT EXISTS idx_emails_metadata_gin ON emails USING GIN (metadata);

ANALYZE emails;

