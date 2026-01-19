# import os
# import time
# from typing import List, Dict, Any

# import psycopg2
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer


# def connect_db():
#     load_dotenv()
#     db_url = os.environ.get("DATABASE_URL")
#     if not db_url:
#         raise RuntimeError("DATABASE_URL not set. Put it in .env")
#     return psycopg2.connect(db_url)


# def retrieve_top_k(
#     user_id: str,
#     query: str,
#     k: int = 5,
# ) -> List[Dict[str, Any]]:
#     """
#     Retrieve top-k emails for a given user_id using cosine distance in pgvector.
#     IMPORTANT: user_id filter enforces multi-user isolation.
#     """
#     embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#     t0 = time.time()
#     q_emb = embedder.encode(query, normalize_embeddings=True)
#     embed_ms = (time.time() - t0) * 1000

#     # pgvector input format: '[0.1,0.2,...]'
#     emb_str = "[" + ",".join(f"{x:.6f}" for x in q_emb.tolist()) + "]"

#     conn = connect_db()
#     cur = conn.cursor()

#     t1 = time.time()
#     cur.execute(
#         """
#         SELECT
#           id,
#           sender,
#           subject,
#           sent_at,
#           body,
#           url,
#           (embedding <=> %s::vector) AS cosine_distance
#         FROM emails
#         WHERE user_id = %s
#         ORDER BY embedding <=> %s::vector
#         LIMIT %s;
#         """,
#         (emb_str, user_id, emb_str, k),
#     )
#     rows = cur.fetchall()
#     retrieval_ms = (time.time() - t1) * 1000

#     cur.close()
#     conn.close()

#     results = []
#     for r in rows:
#         results.append(
#             {
#                 "id": r[0],
#                 "sender": r[1],
#                 "subject": r[2],
#                 "sent_at": str(r[3]),
#                 "body_preview": (r[4] or "")[:220].replace("\n", " "),
#                 "url": r[5],
#                 "cosine_distance": float(r[6]),
#             }
#         )

#     return results, embed_ms, retrieval_ms


# def pretty_print(user_id: str, query: str, k: int = 5):
#     results, embed_ms, retrieval_ms = retrieve_top_k(user_id=user_id, query=query, k=k)

#     print("\n" + "=" * 80)
#     print(f"USER: {user_id}")
#     print(f"QUERY: {query}")
#     print(f"Embedding latency: {embed_ms:.1f} ms | Retrieval latency: {retrieval_ms:.1f} ms")
#     print("-" * 80)

#     if not results:
#         print("No results.")
#         return

#     for i, r in enumerate(results, 1):
#         print(f"{i}. [id={r['id']}] dist={r['cosine_distance']:.4f}")
#         print(f"   From: {r['sender']}")
#         print(f"   Subject: {r['subject']}")
#         print(f"   Sent: {r['sent_at']}")
#         print(f"   Preview: {r['body_preview']}")
#         print(f"   URL: {r['url']}")
#         print("-" * 80)


# if __name__ == "__main__":
#     # Test queries
#     q1 = "What did Sarah say about the Q4 budget?"
#     pretty_print("user_1", q1, k=3)
#     pretty_print("user_2", q1, k=3)

#     q2 = "When is the proposed product launch date?"
#     pretty_print("user_1", q2, k=3)
#     pretty_print("user_2", q2, k=3)


import os
import time
from typing import List, Dict, Any, Tuple

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env once at import time
load_dotenv()

# (3) Confidence thresholds (same idea as qa.py)
DIST_CUTOFF = float(os.environ.get("DIST_CUTOFF", "0.75"))
MIN_BODY_CHARS = int(os.environ.get("MIN_BODY_CHARS", "50"))

# (2) Efficiency: reuse embedder instead of re-initializing each query
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def connect_db():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set. Put it in .env")
    return psycopg2.connect(db_url)


def retrieve_top_k(
    user_id: str,
    query: str,
    k: int = 5,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Retrieve top-k emails for a given user_id using cosine distance in pgvector.
    IMPORTANT: user_id filter enforces multi-user isolation.
    """
    t0 = time.time()
    q_emb = EMBEDDER.encode(query, normalize_embeddings=True)
    embed_ms = (time.time() - t0) * 1000

    # pgvector input format: '[0.1,0.2,...]'
    emb_str = "[" + ",".join(f"{x:.6f}" for x in q_emb.tolist()) + "]"

    conn = connect_db()
    cur = conn.cursor()

    t1 = time.time()
    cur.execute(
        """
        SELECT
          id,
          sender,
          subject,
          sent_at,
          body,
          url,
          (embedding <=> %s::vector) AS cosine_distance
        FROM emails
        WHERE user_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (emb_str, user_id, emb_str, k),
    )
    rows = cur.fetchall()
    retrieval_ms = (time.time() - t1) * 1000

    cur.close()
    conn.close()

    results = []
    for r in rows:
        body = (r[4] or "")
        results.append(
            {
                "id": r[0],
                "sender": r[1],
                "subject": r[2],
                "sent_at": str(r[3]),
                "body": body,
                "body_preview": body[:220].replace("\n", " "),
                "url": r[5],
                "cosine_distance": float(r[6]),
            }
        )

    return results, embed_ms, retrieval_ms


def pretty_print(user_id: str, query: str, k: int = 5):
    results, embed_ms, retrieval_ms = retrieve_top_k(user_id=user_id, query=query, k=k)

    print("\n" + "=" * 80)
    print(f"USER: {user_id}")
    print(f"QUERY: {query}")
    print(f"Embedding latency: {embed_ms:.1f} ms | Retrieval latency: {retrieval_ms:.1f} ms")
    print("-" * 80)

    if not results:
        print("No results.")
        return

    # (3) Confidence gate messaging (retrieval-only)
    best = results[0]
    best_dist = best["cosine_distance"]
    best_body_len = len(best["body"].strip())

    if best_dist > DIST_CUTOFF or best_body_len < MIN_BODY_CHARS:
        print(
            f"Low-confidence retrieval: best_dist={best_dist:.4f}, "
            f"best_body_chars={best_body_len}. (Thresholds: dist<={DIST_CUTOFF}, body>={MIN_BODY_CHARS})"
        )
        print("-" * 80)

    for i, r in enumerate(results, 1):
        print(f"{i}. [id={r['id']}] dist={r['cosine_distance']:.4f}")
        print(f"   From: {r['sender']}")
        print(f"   Subject: {r['subject']}")
        print(f"   Sent: {r['sent_at']}")
        print(f"   Preview: {r['body_preview']}")
        print(f"   URL: {r['url']}")
        print("-" * 80)


if __name__ == "__main__":
    # Test queries
    q1 = "What did Sarah say about the Q4 budget?"
    pretty_print("user_1", q1, k=3)
    pretty_print("user_2", q1, k=3)

    q2 = "When is the proposed product launch date?"
    pretty_print("user_1", q2, k=3)
    pretty_print("user_2", q2, k=3)

