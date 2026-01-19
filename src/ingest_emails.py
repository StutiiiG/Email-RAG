import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384

# You can change these paths to wherever your sample email JSON lives.
DEFAULT_EMAIL_FILES = [
    "data/user1_emails.json",
    "data/user2_emails.json",
]

def must_getenv(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"{name} not set. Create a .env in repo root or export it.")
    return val

def connect_db():
    # Loads .env from current working directory (repo root if you run from there)
    load_dotenv()
    db_url = must_getenv("DATABASE_URL")
    return psycopg2.connect(db_url)

def normalize_recipients(x: Any) -> List[str]:
    """
    DB column is text[] so we must send a Python list.
    Accepts:
      - string: "a@b.com"
      - list: ["a@b.com", "c@d.com"]
      - None: []
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [str(x)]

def embed_text(embedder: SentenceTransformer, text: str) -> Tuple[str, float]:
    t0 = time.time()
    vec = embedder.encode(text, normalize_embeddings=True)
    ms = (time.time() - t0) * 1000
    emb_str = "[" + ",".join(f"{v:.6f}" for v in vec.tolist()) + "]"
    return emb_str, ms

def load_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing sample emails file: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "emails" in data:
        data = data["emails"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of email objects (or {{'emails': [...]}}).")
    return data

def infer_user_id_from_filename(filename: str) -> str:
    # If your JSON already includes user_id per email, we use that.
    # Otherwise map based on filename.
    lower = filename.lower()
    if "user1" in lower or "user_1" in lower:
        return "user_1"
    if "user2" in lower or "user_2" in lower:
        return "user_2"
    return "user_1"

def main():
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")
    print(f"[ingest_emails] DATABASE_URL = {db_url!r}")

    embedder = SentenceTransformer(EMBED_MODEL)

    conn = connect_db()
    cur = conn.cursor()

    total_inserted = 0
    total_embed_ms = 0.0
    total_insert_ms = 0.0
    t_all = time.time()

    for file_path in DEFAULT_EMAIL_FILES:
        user_default = infer_user_id_from_filename(file_path)
        emails = load_json(file_path)

        for e in emails:
            user_id = str(e.get("user_id") or user_default)
            message_id = str(e.get("message_id") or e.get("id") or e.get("msg_id") or "")
            if not message_id.strip():
                raise ValueError(f"Email missing message_id in {file_path}: {e}")

            thread_id = e.get("thread_id")
            sender = e.get("sender")
            recipients = normalize_recipients(e.get("recipients"))
            subject = e.get("subject")
            body = e.get("body") or ""
            sent_at = e.get("sent_at")  # can be ISO string; Postgres can parse
            source = e.get("source") or "synthetic"
            url = e.get("url")
            metadata = e.get("metadata") or {}

            # Embed subject+body for better retrieval
            text_for_embedding = f"Subject: {subject or ''}\n\n{body}".strip()
            emb_str, emb_ms = embed_text(embedder, text_for_embedding)
            total_embed_ms += emb_ms

            t1 = time.time()
            cur.execute(
                """
                INSERT INTO emails
                    (user_id, thread_id, message_id, sender, recipients, subject, body, sent_at, source, url, metadata, embedding)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT ON CONSTRAINT emails_user_message_unique
                DO UPDATE SET
                    thread_id   = EXCLUDED.thread_id,
                    sender      = EXCLUDED.sender,
                    recipients  = EXCLUDED.recipients,
                    subject     = EXCLUDED.subject,
                    body        = EXCLUDED.body,
                    sent_at     = EXCLUDED.sent_at,
                    source      = EXCLUDED.source,
                    url         = EXCLUDED.url,
                    metadata    = EXCLUDED.metadata,
                    embedding   = EXCLUDED.embedding;
                """,
                (
                    user_id,
                    thread_id,
                    message_id,
                    sender,
                    recipients,          # <-- list -> text[]
                    subject,
                    body,
                    sent_at,
                    source,
                    url,
                    Json(metadata),
                    emb_str,
                ),
            )
            total_insert_ms += (time.time() - t1) * 1000
            total_inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    print("Ingestion complete")
    print(f"Rows processed (insert/upsert attempts): {total_inserted}")
    print(f"Total time: {time.time() - t_all:.3f}s")
    print(f"Embedding time (sum): {total_embed_ms/1000:.3f}s")
    print(f"Insert time (sum): {total_insert_ms/1000:.3f}s")

if __name__ == "__main__":
    main()
