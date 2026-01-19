# Demo Script: Personal Email RAG System (Local)

This document provides a complete, step-by-step demo of the Personal Email RAG system.  
It demonstrates:

1. Local Environment setup  
2. Email and Google Drive document ingestion  
3. End-to-end RAG queries with performance metrics
4. Switching between users and proving strict data isolation  

All components run **entirely locally** using open-source tools.

---

## 0. Prerequisites

Ensure the following are installed:

- Docker & Docker Compose  
- Python 3.10+  
- Ollama (local LLM runtime)

Verify installations:

```bash
docker --version
docker compose version
python --version
ollama --version
```

---

## 1. Environment Setup

### 1.1 Start PostgreSQL + pgvector 

Start the PostgreSQL container using Docker Compose:

```bash
docker compose up -d
```

Verify the container is running:

```bash
docker ps
```

This starts a PostgreSQL container with the pgvector extension enabled.

### 1.2 Create Python Virtual Environment 

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.3 Configure Environment Variables

Create a .env file in the repository root:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/personal_email_rag
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct
```

### 1.4 Start Ollama and Pull Model

In a separate terminal:

```bash
ollama serve
ollama pull mistral:7b-instruct
```

Optional Sanity Check:

```bash
ollama run mistral:7b-instruct "Say hello in one sentence."
```

---

## 2. Ingest Emails and Drive Files 

### 2.1 Initialize Database Schema

```bash
psql "$DATABASE_URL" -f db/schema.sql
```

This creates:
- The emails table (used for email rows and Drive doc rowws)
- The pgvector extension
- Required indexes for vector similarity search

### 2.2 Ingest Sample Emails(Multi-User)

```bash
python src/ingest_emails.py
```

Result:
 - user_1: 8 synthetic emails
 - user_2:  synthetic emails 
 - Embeddings generated locally and stored in Postgres

### 2.3 Ingest Local Drive Files 

Directory Structure: 

```bash
data/drive/user_1/
data/drive/user_2/
```

Supported formats: .txt, .md, .pdf, .docx 

Then ingest:

```bash
python src/ingest_drive_files.py
```

## 3. Run End-to-End RAG Queries

```bash
python src/qa.py
```

You should see:
 - Embedding latency
 - Retrieval latency
 - LLM latency
 - Answer + Sources
 - Top-k retrieved candidates (IDs + cosine distances)

## 4. Multi-User Isolation Demo 

### 4.1 Same Query, Different Users 

User 1: 

```bash
What did Sarah say about the Q4 budget?
```

→ Returns user_1 email + citations

User 2: 

```bash
What did Sarah say about the Q4 budget?
```

→ Returns a different answer grounded only in user_2 data
No shared Context. 

### 4.2 Explicit Cross-User Failure Case

Ask user_1 a question that only exists in user_2’s Drive docs:

```bash
python -c "from src.qa import answer_question; answer_question('user_1','What are the reliability targets for P1 incidents and MTTR?',k=5)"
```

Expected : 

```bash
I don't have enough information in your emails to answer that.
```

### 4.3 Database-Level Isolation Proof 

Delete a user_2 email directly:

```bash
docker exec -it personal_email_rag_db \
psql -U rag -d ragdb \
-c "DELETE FROM emails WHERE user_id='user_2' AND subject ILIKE '%budget%';"
```

Rerun the QA:

```bash
python src/qa.py
```

Result:
 - user_2 budget queries fail safely
 - user_1 queries unaffected


 






