# Personal Email RAG (Local-Only)

A local, privacy-preserving Retrieval-Augmented Generation (RAG) system for querying personal emails and documents.
The system ingests emails and Google Drive files, stores embeddings in PostgreSQL using pgvector, and answers questions using a locally hosted LLM via Ollama — with strict grounding and per-user data isolation.

---

## Key Features

* **Fully local execution**

  * No OpenAI or cloud APIs
  * Local embeddings and local LLM inference via Ollama

* **Strict multi-user isolation**

  * Retrieval is filtered by `user_id` at the database layer
  * No cross-user vector leakage

* **Unified retrieval across sources**

  * Emails and Drive documents stored in a single schema
  * Source-aware prompting (`EMAIL` vs `DRIVE`)

* **Grounded, citation-based answers**

  * Answers generated strictly from retrieved documents
  * Explicit refusal when information is unavailable
  * Sources included with every answer

* **Reproducible local demo**

  * Dockerized Postgres + pgvector
  * Deterministic synthetic datasets
  * Clear demo and evaluation artifacts

---

## High-Level Architecture

**Ingest → Embed → Store → Retrieve → Generate**

1. **Ingestion**

   * Emails: JSON-based dataset (simulated Gmail export)
   * Drive files: `.txt`, `.md`, `.pdf`, `.docx`
   * All data normalized into a single table

2. **Embedding**

   * Model: `sentence-transformers/all-MiniLM-L6-v2`
   * Fast, CPU-friendly, normalized embeddings (384-dim)

3. **Storage**

   * PostgreSQL with pgvector
   * Metadata and embeddings stored together

4. **Retrieval**

   * Cosine similarity search via pgvector
   * SQL-level filtering using `WHERE user_id = ?`

5. **Generation**

   * Local LLM via Ollama (`mistral:7b-instruct`)
   * Strict prompt rules to prevent hallucination
   * Answers returned with latency metrics and sources

---

## Repository Structure

```text
personal-email-rag/
├─ src/
│  ├─ ingest_emails.py
│  ├─ ingest_drive_files.py
│  ├─ qa.py 
   └─ retrieve.py  
├─ db/
│  └─ schema.sql

├─ data/
|  ├─ drive/
|     ├─ user_1/ 
│     └─ user_2/  
│  ├─ user1_emails.json
│  └─ user2_emails.json
├─ docker-compose.yml
├─ requirements.txt
├─ README.md
├─ demo.md
├─ architecture.md
└─ evaluation.md
``` 

---

## Setup

### Start PostgreSQL

```
docker compose up -d
```

### Python Environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```
DATABASE_URL=postgresql://rag:rag@localhost:5432/ragdb
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct
```

### Start Ollama

```
ollama serve
ollama pull mistral:7b-instruct
```

---

## Running the System

### Ingest Emails

```
python src/ingest_emails.py
```

### Ingest Drive Files

```
python src/ingest_drive_files.py
```

### Run RAG Queries

```
python src/qa.py
```

The output includes:

* Embedding latency
* Retrieval latency
* LLM generation latency
* Answer with sources
* Top retrieved candidates

---

## Demo and Evaluation

* **Demo script:** `demo.md`
  Step-by-step walkthrough showing ingestion, querying, and multi-user isolation.

* **Architecture details:** `architecture.md`
  Design decisions, model choices, and tradeoffs.

* **Evaluation report:** `evaluation.md`
  Performance benchmarks, challenges, limitations, and production considerations.

---

## Design Principles

* Correctness over cleverness
* Isolation enforced at the database layer
* Grounded answers only (no hallucination)
* Local-first and privacy-preserving by default

---

## Scope Notes

This system is intentionally designed as a single-node, local prototype to demonstrate architectural correctness rather than production scale.
Scalability and production considerations are discussed in the evaluation report.

---



