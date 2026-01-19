# Evaluation Report  

## Personal Email Retrieval-Augmented Generation (RAG) System

This document evaluates the Personal Email RAG system with respect to performance, correctness, architectural tradeoffs, and future extensibility. The system is a fully local, multi-user RAG pipeline that ingests emails and documents, stores embeddings in a vector database, and answers natural-language queries using a locally hosted LLM.

---

## 1. Performance Benchmarks

Performance was evaluated on a local development machine using:
- PostgreSQL with pgvector for vector storage and retrieval
- SentenceTransformers for embedding generation
- A locally hosted LLM served via Ollama

All components run entirely offline using open-source tools.

### 1.1 Response Times

Observed average latencies during interactive demo queries:

| Stage | Average Latency |
|------|----------------|
| Embedding generation | 30–90 ms |
| Vector retrieval (cosine similarity search) | 3–10 ms |
| LLM answer generation | 2.5–7.5 s |
| End-to-end query latency | ~3–8 s |

**Observations:**
- Vector similarity search contributes negligible overhead compared to LLM inference.
- End-to-end latency is dominated by local LLM generation, which is expected for on-device inference.
- The system remains sufficiently responsive for an interactive personal assistant use case.

### 1.2 Accuracy Observations

Accuracy was evaluated qualitatively using controlled, synthetic email datasets across multiple users.

Observed behavior:
- Queries consistently retrieved semantically relevant emails when such emails existed for the querying user.
- Generated answers were grounded directly in retrieved content and reflected the underlying emails accurately.
- When no relevant emails existed, the system reliably returned the fallback response:

> “I don’t have enough information in your emails to answer that.”

This demonstrates:
- Correct retrieval grounding
- Effective hallucination avoidance
- Proper handling of missing or insufficient context
- Successful enforcement of user-level data isolation

---

## 2. Challenges Encountered and Solutions

This section documents the concrete engineering challenges encountered during implementation and how they were resolved.

### 2.1 Schema Drift and Ingestion Failures

**Challenge:**  
During development, repeated schema changes (adding Drive ingestion, introducing `message_id`, enforcing uniqueness, adding metadata fields) caused mismatches between the database schema and ingestion logic. This resulted in runtime failures such as:

- `psycopg2.errors.InvalidColumnReference`
- `ON CONFLICT specification does not match any constraint`
- Silent ingestion failures due to missing or nullable fields

**Solution:**  
The schema was stabilized with the following corrective actions:
- Introduced a **composite unique constraint** on `(user_id, message_id)`
- Enforced `message_id` as `NOT NULL`
- Ensured both email and Drive ingestions generate deterministic `message_id` values
- Aligned all `ON CONFLICT (user_id, message_id)` clauses with the actual database constraints

This ensured idempotent ingestion and eliminated duplicate rows across repeated runs.

---

### 2.2 Drive File Ingestion Conflicts

**Challenge:**  
Adding Google Drive file ingestion introduced new failure modes:
- `ON CONFLICT` errors due to missing unique constraints
- Conflicts between email and Drive rows sharing the same table
- Ambiguity around document identity across sources

**Solution:**  
- Unified emails and Drive documents into a **single `emails` table**
- Differentiated sources using a `source` field (`synthetic`, `drive`)
- Generated deterministic `message_id` values for Drive files based on file path and user
- Stored source-specific metadata in a JSONB `metadata` column

This allowed Drive documents to participate naturally in retrieval without breaking existing email logic.

---

### 2.3 Prompt Safety and Hallucination Control

**Challenge:**  
Early prompt versions allowed the LLM to answer questions using weakly related context or inferred knowledge.

**Solution:**  
The prompt was hardened with:
- Explicit “use ONLY provided documents” rules
- A fixed fallback response when context is insufficient
- Mandatory citation of document IDs used in the answer

This ensured deterministic failure behavior and prevented cross-user or out-of-context hallucinations.

---

### 2.4 Managing Incremental Changes Without Breaking the Pipeline

**Challenge:**  
Because ingestion, schema, retrieval, and generation are tightly coupled, fixing one issue (e.g., schema constraints) often surfaced another (e.g., ingestion assumptions).

**Solution:**  
- Adopted a strict order of operations:
  1. Stabilize schema
  2. Validate ingestion idempotency
  3. Verify retrieval correctness
  4. Only then tune indexing and prompts
- Used targeted database inspection queries (`COUNT`, `GROUP BY`, duplicate checks) after each change

This iterative validation approach prevented regressions and ensured correctness at each stage.

---

## 3. Limitations of the Current Approach

- **Single-machine execution**  
  The system is designed for local use and does not scale horizontally.

- **Cold-start latency**  
  Initial model loading introduces noticeable startup delay.

- **Coarse document granularity**  
  Emails and documents are embedded as full units rather than semantically chunked segments.

- **No reranking layer**  
  Retrieval relies purely on vector similarity without secondary re-ranking.

- **Synthetic evaluation data**  
  Accuracy evaluation was performed on controlled datasets rather than real-world inboxes.

These limitations were accepted to prioritize clarity, correctness, and ease of evaluation within the scope of the take-home assignment.

---

## 4. Potential Improvements and Production Considerations

### 4.1 Retrieval Quality
- Introduce semantic chunking for long emails and documents
- Add a re-ranking stage using a cross-encoder or lightweight LLM
- Apply similarity score thresholds to filter weak matches

### 4.2 Scalability
- Migrate PostgreSQL to a managed or distributed setup
- Add connection pooling and concurrency controls
- Separate ingestion and query services

### 4.3 LLM Serving
- Cache frequent query embeddings
- Support streaming responses for improved UX
- Enable model swapping via configuration

### 4.4 Security and Privacy
- Encrypt embeddings at rest
- Add authentication and authorization layers
- Implement audit logging for query access

--- 

This evaluation demonstrates that the system satisfies the technical and functional requirements of the take-home assignment while providing a strong foundation for a secure, extensible, multi-user RAG system.

