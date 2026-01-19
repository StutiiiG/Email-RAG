import os
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Optional parsers (only used if the file type needs them)
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def connect_db():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Create a .env with DATABASE_URL=... in repo root.")
    return psycopg2.connect(db_url)


def stable_message_id(user_id: str, file_path: Path) -> str:
    """
    Create a deterministic message_id per user + file.
    This must be stable across runs so upsert works.
    """
    rel = str(file_path).replace("\\", "/")
    raw = f"drive::{user_id}::{rel}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"drive_{h}"


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. Add pypdf to requirements.txt and pip install -r requirements.txt")
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def read_docx(path: Path) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed. Add python-docx to requirements.txt and pip install -r requirements.txt")
    d = docx.Document(str(path))
    return "\n".join([p.text for p in d.paragraphs if p.text])


def load_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".md"]:
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    # fallback: try as text
    return read_txt(path)


def ingest_one(conn, embedder: SentenceTransformer, user_id: str, path: Path) -> Tuple[bool, str, float, int]:
    text = load_file_text(path).strip()
    if len(text) < 40:
        return False, "skipped(short)", 0.0, len(text)

    t0 = time.time()
    vec = embedder.encode(text).tolist()
    embed_ms = (time.time() - t0) * 1000.0

    msg_id = stable_message_id(user_id, path)
    subject = f"[Drive] {path.name}"
    url = f"file://{path.resolve()}"
    metadata = {"filename": path.name, "path": str(path), "source": "drive"}

    with conn.cursor() as cur:
        # IMPORTANT:
        # Use ON CONFLICT ON CONSTRAINT so Postgres matches the exact UNIQUE(user_id,message_id)
        cur.execute(
            """
            INSERT INTO emails (
              user_id, message_id, thread_id, sender, recipients, subject, body, sent_at, source, url, metadata, embedding
            )
            VALUES (
              %(user_id)s, %(message_id)s, NULL, NULL, NULL, %(subject)s, %(body)s, NULL, 'drive', %(url)s, %(metadata)s::jsonb, %(embedding)s
            )
            ON CONFLICT ON CONSTRAINT emails_user_message_unique
            DO UPDATE SET
              subject   = EXCLUDED.subject,
              body      = EXCLUDED.body,
              url       = EXCLUDED.url,
              metadata  = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding,
              source    = 'drive'
            ;
            """,
            {
                "user_id": user_id,
                "message_id": msg_id,
                "subject": subject,
                "body": text,
                "url": url,
                "metadata": str(metadata).replace("'", '"'),
                "embedding": vec,
            },
        )

    conn.commit()
    return True, "inserted/upserted", embed_ms, len(text)


def main():
    conn = connect_db()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    base = Path("data/drive")
    user_dirs = [("user_1", base / "user_1"), ("user_2", base / "user_2")]

    total_files = 0
    inserted = 0
    skipped = 0

    for user_id, udir in user_dirs:
        if not udir.exists():
            continue

        for path in sorted(udir.rglob("*")):
            if path.is_dir():
                continue
            if path.name.startswith("."):
                continue

            total_files += 1
            ok, status, embed_ms, chars = ingest_one(conn, embedder, user_id, path)

            if ok:
                inserted += 1
                print(f"[ingest_drive_files] user={user_id} file={path.name} chars={chars} embed_ms={embed_ms:.1f}")
            else:
                skipped += 1
                print(f"[ingest_drive_files] user={user_id} file={path.name} chars={chars} -> {status}")

    print("\n[ingest_drive_files] DONE")
    print(f"  total_files    = {total_files}")
    print(f"  inserted       = {inserted}")
    print(f"  skipped(short) = {skipped}")

    conn.close()


if __name__ == "__main__":
    main()
