import os
import time
from typing import List, Dict, Any, Tuple

import psycopg2
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NO_ANSWER = "I don't have enough information in your emails to answer that."


def connect_db():
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set. Put it in .env")
    return psycopg2.connect(db_url)


def embed_query(query: str) -> Tuple[str, float]:
    embedder = SentenceTransformer(EMBED_MODEL)
    t0 = time.time()
    vec = embedder.encode(query, normalize_embeddings=True)
    ms = (time.time() - t0) * 1000
    emb_str = "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"
    return emb_str, ms


def retrieve_context(user_id: str, query: str, k: int = 5) -> Tuple[List[Dict[str, Any]], float, float]:
    q_emb_str, embed_ms = embed_query(query)

    conn = connect_db()
    cur = conn.cursor()

    t1 = time.time()
    cur.execute(
        """
        SELECT
          id,
          user_id,
          sender,
          subject,
          sent_at,
          body,
          url,
          metadata,
          (embedding <=> %s::vector) AS cosine_distance
        FROM emails
        WHERE user_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (q_emb_str, user_id, q_emb_str, k),
    )
    rows = cur.fetchall()
    retrieval_ms = (time.time() - t1) * 1000

    cur.close()
    conn.close()

    results: List[Dict[str, Any]] = []
    for r in rows:
        metadata = r[7] or {}
        source = (metadata.get("source") or "email").lower()

        results.append(
            {
                "id": r[0],
                "user_id": r[1],
                "sender": r[2],
                "subject": r[3],
                "sent_at": str(r[4]) if r[4] else None,
                "body": r[5] or "",
                "url": r[6],
                "source": source,
                "metadata": metadata,
                "cosine_distance": float(r[8]),
            }
        )

    return results, embed_ms, retrieval_ms


def build_prompt(user_id: str, query: str, docs: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for d in docs:
        tag = "EMAIL" if d["source"] == "email" else "DRIVE"
        sent = d["sent_at"] or "N/A"
        frm = d["sender"] or "N/A"

        context_blocks.append(
            f"[{tag} id={d['id']} user={d['user_id']} sent_at={sent} from={frm} subject={d['subject']} url={d['url']}]\n"
            f"{d['body']}\n"
        )

    context = "\n---\n".join(context_blocks) if context_blocks else "NO_DOCS_FOUND"

    prompt = f"""You are a careful assistant answering questions about a user's private data.

User: {user_id}

Hard rules:
- Use ONLY the information contained in the provided documents below.
- If the documents do not contain the answer, reply exactly with:
{NO_ANSWER}
- After your answer, output EXACTLY one blank line, then EXACTLY one line in this format:
Sources: [EMAIL id=...], [DRIVE id=...]
Do NOT add user=, do NOT add newlines in Sources, do NOT add any extra text.


Question:
{query}

Documents:
{context}

Answer:"""
    return prompt


def call_ollama(prompt: str) -> Tuple[str, float]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 160, "top_p": 0.9, "temperature": 0.2},
    }

    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()
    ms = (time.time() - t0) * 1000
    return out.get("response", "").strip(), ms


def answer_question(user_id: str, query: str, k: int = 5):
    docs, embed_ms, retrieval_ms = retrieve_context(user_id, query, k=k)
    prompt = build_prompt(user_id, query, docs)

    answer, llm_ms = call_ollama(prompt)

    print("\n" + "=" * 80)
    print(f"USER: {user_id}")
    print(f"QUERY: {query}")
    print(f"Embed: {embed_ms:.1f} ms | Retrieve: {retrieval_ms:.1f} ms | LLM: {llm_ms:.1f} ms")
    print("-" * 80)

    if answer.strip() == NO_ANSWER:
        print(NO_ANSWER)
        print("\n(Top retrieved candidates)")
        for d in docs:
            print(f"- id={d['id']} dist={d['cosine_distance']:.4f} subject={d['subject']} url={d['url']}")
        return

    print(answer)
    print("\n(Top retrieved candidates)")
    for d in docs:
        print(f"- id={d['id']} dist={d['cosine_distance']:.4f} subject={d['subject']} url={d['url']}")


if __name__ == "__main__":
    # Email-only queries
    answer_question("user_1", "What did Sarah say about the Q4 budget?", k=3)
    answer_question("user_2", "What did Sarah say about the Q4 budget?", k=3)

    # Drive + Email fusion
    answer_question("user_2", "What are the reliability targets for P1 incidents and MTTR?", k=3)

    # Isolation test (should fail safely)
    answer_question("user_1", "What are the reliability targets for P1 incidents and MTTR?", k=2)

    # Cross-source product question
    answer_question("user_1", "When is the proposed product launch date?", k=3)
