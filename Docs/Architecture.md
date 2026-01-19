# Architecture: Personal Email RAG System (Local-Only)

This document explains the architecture and model/database choices for the **Personal Email RAG** take-home. The system runs fully locally, ingests emails, stores embeddings in Postgres + pgvector, and answers questions using a local LLM via Ollama with strict grounding and per-user isolation.

---

## 1) System Design 

### High-level pipeline
**Ingest → Embed → Store → Retrieve → Generate**

1. **Email ingestion**
   - Input sources: 
     - Gmail export / integration or
     - JSON email datasets (as used in this repo: `data/user1_emails.json`, `data/user2_emails.json`)
     - Exported Google Drive files stored locally under:( `data/drive/user_1` and `data/drive/user_2`)

   - Supported Drive formats: `.txt`, `.md`, `.pdf`, `.docx`

   - Extracted fields (minimum): `user_id`, `sender`, `recipients`, `subject`, `body`, `sent_at`, `url`, optional `metadata/tags`, `source` field (`email` or `drive`)
   - Output: normalized email and drive doc rows ready for embedding + persistence

2. **Embedding**
   - Convert each document’s searchable content into a dense vector using a local embedding model. ( For emails: `subject + "\n\n" + body` and for Drive File: `filename + "\n\n" + extracted_text`)
   - Store vectors in Postgres `vector` column.

3. **Storage**
   - Postgres stores:
     - document metadata (sender, subject, timestamps, url, tags)
     - raw text content
     - embeddings (pgvector)

4. **Retrieval**
   - At query time:
     - embed the user’s question
     - retrieve top-k most similar emails using cosine distance in pgvector
     - **filter by `user_id`** to enforce multi-user isolation
  - Emails and Drive files share the same table schema and are differentiated via the `source` field.
  - A unique constraint on `(user_id, message_id)` ensures idempotent ingestion.

5. **Answer generation**
   - Build a strict prompt containing:
     - the user question
     - the retrieved documents (email and/or Drive Files)
     - rules to prevent hallucination and enforce citations
   - Use Ollama to generate an answer locally and print:
     - embedding time
     - retrieval latency
     - LLM generation time
     - sources used

### Code-to-component mapping 
- **Email Ingestion:** `src/ingest_emails.py`
- **Drive Ingestion:** `src/ingest_drive_files.py`
- **Retrieval-only sanity checks:** `src/retrieve.py`
- **RAG Q&A + metrics:** `src/qa.py`
- **DB schema:** `db/schema.sql` 

---

## 2) Embedding Model Selection Rationale

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`

### Why this model
- **Runs locally on CPU** comfortably (fast embeddings, low memory)
- **Strong baseline semantic retrieval** for short-to-medium text (emails are typically short/medium)
- **Widely used + reliable** for RAG prototypes and take-homes
- **Normalized embeddings** (`normalize_embeddings=True`) work well with cosine similarity and make similarity scores stable

### What text is embedded
- Emails: `subject + "\n\n" + body` , Drive documents: `filename + "\n\n" + extracted_text` 

Including the subject/filename improves retrieval for intent-based queries such as “Q4 budget”, “API integration”, or “product launch”.

### Tradeoffs / future upgrades
- For much larger corpora or higher precision needs, upgrades could include:
  - bigger embedding models (higher quality, slower)
  - hybrid retrieval (BM25 + vector)
  - reranking (cross-encoder)
  - chunking long documents 

---

## 3) LLM Selection and Quantization Strategy

**LLM runtime:** Ollama  
**Default LLM model:** `mistral:7b-instruct` (configurable via environment variables)

### Why Mistral 7B Instruct via Ollama
- **Fully local inference**
- Strong instruction-following for “answer only from context” prompts
- Simple local HTTP API makes the system easy to demo and reproduce

### Quantization strategy 
Ollama distributions typically run **GGUF quantized weights** under the hood (exact quantization depends on the model tag/version installed). Quantization reduces memory and improves speed.

Recommended strategy:
- If your laptop has limited RAM/VRAM: use **4-bit** quantization variants (commonly `Q4_*`).
- If you want better quality and can afford more memory: use **5-bit or 8-bit** variants (`Q5_*`, `Q8_0`).

**How this maps to the project:**
- The system is designed so the LLM is swappable using environment variables:
  - `OLLAMA_MODEL=mistral:7b-instruct`
- You can demonstrate quality/perf tradeoffs by switching to a smaller/faster model or a higher-quality quantization variant (if installed in Ollama).

### Guardrails to prevent hallucination
The QA prompt is strict:
- Use **ONLY** the retrieved documents. 
- If answer not present: return  
  `"I don't have enough information in your emails to answer that."`
- Always print a **Sources** section with email ids used.

This design is deliberate for correctness and evaluability.

---  

## 4) Vector DB Choice and Configuration

**Vector DB:** Postgres + pgvector

### Why pgvector
- Keeps metadata + vectors in the **same system** (simple and robust)
- SQL filtering makes **multi-user isolation** straightforward and auditable
- Easy local setup with Docker 
- Good performance for small–medium datasets 

### Configuration details
- Stored embedding column type: `vector(<dim>)`
  - For `all-MiniLM-L6-v2`, dimension is **384**
- Similarity metric: 
  - cosine distance via pgvector operator `<=>`
- Retrieval query structure (key security property shown here conceptually):
  ```sql
  SELECT ...
  FROM emails
  WHERE user_id = %s
  ORDER BY embedding <=> %s::vector
  LIMIT %s;

## 5) Multi user implimentation approach 

**Metadata filtering:** 

- Every stored email row includes a user_id.
- Every retrieval query includes WHERE user_id = %s.
- This ensures cross-user vectors are never eligible for top-k retrieval.

### Demonstration (how it proves isolation)

- In the demo, user_1 asks: “What did Sarah say about the Q4 budget?”
   - Returns the relevant email and cites it.

- user_2 asks the same question:
  - Retrieval returns only user_2 emails (none contain that info)
  - LLM returns the safe fallback message

Additional isolation tests are demonstrated in the demo script by:
 - Deleting user-specific records
 - Querying Drive-only documents across users
 - Verifying no cross-user leakage occurs