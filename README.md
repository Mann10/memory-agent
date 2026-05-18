# memory-agent

A personal AI assistant that remembers conversations across sessions and searches the live web for current answers.

---

## Overview

This project is a conversational AI agent built with **LangGraph**, **FastAPI**, **PostgreSQL + pgvector**, **Redis**, and **NATS JetStream**. It handles real-time chat through a FastAPI layer, maintains short-term conversation state in Redis checkpoints, stores long-term personal memories in PostgreSQL, and uses Tavily for live web search.

The architecture splits into two runtime processes:

1. **Agent process** (`main.py` + `agent.py`): A FastAPI server that runs the LangGraph conversation graph.
2. **Worker process** (`worker.py`): A background NATS consumer that extracts personal facts from completed conversation turns and persists them to PostgreSQL.

---

## Architecture

```
+-----------+     HTTP      +------------------+     ainvoke      +------------------+
|  Client   | ------------> |  FastAPI         | ---------------> |  LangGraph       |
|           |               |  (main.py)       |                  |  (agent.py)      |
+-----------+               +------------------+                  +------------------+
                                                                      |
                                      +-------------------------------+-------------------+
                                      |                                   |               |
                                      v                                   v               v
                              +---------------+                  +----------------+  +---------+
                              |  Redis        |                  |  PostgreSQL    |  |  NATS   |
                              |  (checkpoints)|                  |  + pgvector     |  |JetStream|
                              +---------------+                  |  (memories)    |  +---------+
                                                               +----------------+      |
                                                                                      | publish
                                                                                      v
                                                                              +----------------+
                                                                              |  Worker        |
                                                                              |  (worker.py)   |
                                                                              +----------------+
                                                                                      |
                                                                                      v
                                                                              +----------------+
                                                                              |  PostgreSQL    |
                                                                              |  + pgvector     |
                                                                              |  (upsert facts) |
                                                                              +----------------+
```

### Components

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application. Exposes `GET /thread` and `POST /chat`. Manages lifespan startup/shutdown of Redis, Postgres, and the LangGraph graph. |
| `agent.py` | LangGraph graph definition. Contains `load_node`, `memory_node`, `agent_node`, `save_node`, and the `search_web` tool. Handles conversation summarization, memory retrieval, LLM invocation, and web search. |
| `worker.py` | NATS JetStream durable consumer. Regex-filters conversation turns, uses an LLM to extract facts, deduplicates against existing memories, and upserts into `PostgresStore`. |
| `nats_publisher.py` | Publishes conversation exchanges to the NATS `memory.extract` subject for async processing by the worker. |
| `docker-compose.yml` | Defines Redis, PostgreSQL (with pgvector), and NATS services with healthchecks. |
| `.example.env` | Template for required environment variables. |
| `requirements.txt` | Python package dependencies. |
| `.gitignore` | Git ignore rules. |

---

## Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI (main.py)
    participant G as LangGraph (agent.py)
    participant R as Redis (AsyncRedisSaver)
    participant PG as PostgreSQL + pgvector
    participant N as NATS JetStream
    participant W as Worker (worker.py)
    participant LLM as LLM (OpenAI/LiteLLM)
    participant T as Tavily API

    C->>API: POST /chat<br>{user_id, thread_id, message}
    API->>G: ainvoke(user_input, config)

    G->>R: load checkpoint<br>for thread_id
    R-->>G: restore prior state
    G->>G: load_node()<br>reads user_id from config

    G->>PG: store.search(("memories", user_id))<br>limit=100
    PG-->>G: return list of facts
    G->>G: memory_node()<br>builds memory_context string

    G->>LLM: ainvoke(messages + memory_context)<br>bind_tools(tools)
    alt LLM requests tool_calls
        G->>T: search_web(query)<br>AsyncTavilyClient
        T-->>G: search results
        G->>LLM: ainvoke(tool results)
    end
    LLM-->>G: assistant reply

    G->>R: save checkpoint<br>thread state
    G->>G: save_node()<br>publish_exchange() to NATS
    G-->>API: return {reply}
    API-->>C: 200 OK<br>{thread_id, reply}

    G->>N: publish "memory.extract"<br>{user_id, thread_id, user, assistant}
    N->>W: deliver message<br>durable="fact-extractor"
    W->>W: worth_extracting()<br>regex filter (no LLM cost)
    W->>LLM: extract_facts()<br>EXTRACT_PROMPT
    LLM-->>W: JSON array of facts
    W->>PG: upsert_facts()<br>dedup + store.put()
    PG-->>W: stored
    W->>N: msg.ack()
```

---

## Project Structure

```
memory-agent/
├── .example.env
├── .gitignore
├── agent.py
├── docker-compose.yml
├── main.py
├── nats_publisher.py
├── requirements.txt
└── worker.py
```

---

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- A LiteLLM-compatible API endpoint (or direct OpenAI API access)
- A Tavily API key

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mann10/memory-agent.git
cd memory-agent
```

### 2. Create a virtual environment

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .example.env .env
```

Edit `.env` and fill in your keys:

```env
LITELLM_API_KEY=your-api-key
LITELLM_API_BASE=https://api.openai.com/v1
MODEL=gpt-4o-mini

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
TAVILY_API_KEY=your-tavily-key
```

The app loads these automatically via `python-dotenv`.

### 5. Start infrastructure services

```bash
docker-compose up -d
```

This starts three containers:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| Redis | `redis:8.0-rc1` | `6379` | **Thread checkpoints** via `AsyncRedisSaver` |
| PostgreSQL | `pgvector/pgvector:pg16` | `5432` | Long-term memory via `PostgresStore` |
| NATS | `nats:2.10-alpine` | `4222` (clients), `8222` (monitoring) | Async messaging between agent and worker |

Check health:
```bash
docker-compose ps
```

NATS monitoring: http://localhost:8222

### 6. Run the FastAPI server

```bash
python -m main
```

Or:
```bash
python main.py
```

The server starts at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

### 7. Run the memory worker

Open a **second terminal**, activate the same virtual environment, then:

```bash
python -m worker
```

Or:
```bash
python worker.py
```

The worker listens on the `memory.extract` NATS subject. Without it, conversation turns are published but never extracted into long-term memory.

---

## API Endpoints

### `GET /thread`

Generate a fresh thread ID.

```bash
curl http://localhost:8000/thread
```

**Response:**
```json
{"thread_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
```

### `POST /chat`

Send a message to the agent.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "thread_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message": "My name is Alice and I work at a fintech startup in Ahmedabad."
  }'
```

**Response:**
```json
{"thread_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "reply": "Nice to meet you, Alice!"}
```

### Continuing a conversation

Reuse the same `thread_id` to keep short-term context. The `user_id` persists across threads for long-term memory.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "thread_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message": "What is the weather like today?"
  }'
```

If the LLM decides current information is needed, it calls `search_web` via Tavily automatically.

---

## How It Works

### Conversation Flow (agent.py)

The LangGraph graph has five nodes:

1. **`load_node`**: Reads `user_id` from `get_config()`, wraps the user's input in a `HumanMessage`.
2. **`memory_node`**: Searches `PostgresStore` under namespace `("memories", user_id)` for up to 100 facts. Injects them into state as `memory_context`.
3. **`agent_node`**: Builds the system prompt from `BASE_SYSTEM_PROMPT` + `memory_context`. If message history exceeds 10 messages, it summarizes the older ones and keeps the last 4 in full. Calls the LLM (bound with `search_web` tool). If `tool_calls` are present, routes to `tools`; otherwise routes to `save`.
4. **`tool_node`**: Executes `search_web` via `AsyncTavilyClient`.
5. **`save_node`**: Publishes the turn to NATS via `publish_exchange()` for background fact extraction.

### Thread Checkpoints (Redis)

LangGraph's `AsyncRedisSaver` (connected via `REDIS_URI`) automatically checkpoints the full graph state after every node execution. When you reuse a `thread_id`, the graph resumes from the last checkpoint stored in Redis. This gives you:

- **Short-term memory** within a conversation thread
- **Resumability** if the server restarts mid-conversation
- **Summarization** when history exceeds `SUMMARY_MSG_LIMIT` (10 messages); older messages are summarized and the last `SUMMARY_KEEP` (4) are kept in full context

### Memory Extraction Flow (worker.py)

1. **Filter**: `worth_extracting()` runs a cheap regex check against `FACT_PATTERNS` (e.g., "I am", "my name", "I work"). If no match, the message is dropped without LLM cost.
2. **Extract**: `extract_facts()` sends an `EXTRACT_PROMPT` to the LLM, asking for a JSON array of personal facts.
3. **Deduplicate**: `upsert_facts()` searches existing memories (limit 200) and skips exact duplicates.
4. **Store**: New facts are written to `PostgresStore` with a random UUID key under `("memories", user_id)`.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LITELLM_API_KEY` | Yes | - | API key for LLM provider |
| `LITELLM_API_BASE` | Yes | - | Base URL for LLM API |
| `MODEL` | Yes | - | Model name (e.g., `gpt-4o-mini`) |
| `TAVILY_API_KEY` | Yes | - | Tavily web search API key |
| `REDIS_URI` | No | `redis://localhost:6379` | Redis connection string for thread checkpoints |
| `DB_URI` | No | `postgresql://postgres:postgres@localhost:5432/agent_memory` | PostgreSQL connection string |
| `NATS_URI` | No | `nats://localhost:4222` | NATS connection string |
| `LANGCHAIN_TRACING_V2` | No | - | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | - | LangSmith API key |

---

## Dependencies

From `requirements.txt`:

- `langchain-openai`
- `langgraph`
- `langchain`
- `langgraph-checkpoint-redis`
- `langgraph-checkpoint-postgres`
- `fastapi`
- `nats-py`
- `tavily-python`
- `langchain-tavily`
- `langchain-community`
- `psycopg[binary,pool]`
- `python-dotenv`
- `pillow`
- `streamlit`

---

## Notes

- The graph is fully async. Every node is an `async def` function.
- `PostgresStore` is initialized once in `get_stores()` and held at module level for direct access by `memory_node`.
- `AsyncRedisSaver` is initialized in the same `get_stores()` call and passed to `build_graph()` as the checkpointer.
- The worker uses a durable NATS subscription (`durable="fact-extractor"`) so it resumes from the last acknowledged message after restarts.
- All Docker services include healthchecks.
- The `.gitignore` file is 21 bytes (likely excludes `__pycache__/` or similar).

---

## License

MIT
