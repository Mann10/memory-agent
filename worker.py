# worker.py  — subscribes NATS, cheap filter → LLM extractor → pgvector upsert
import asyncio, json, re, os
import nats
from langchain_openai import ChatOpenAI
from langgraph.store.postgres import PostgresStore
from dotenv import load_dotenv

DB_URI    = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5432/agent_memory")
NATS_URI  = os.getenv("NATS_URI", "nats://localhost:4222")

load_dotenv()

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")
MODEL       = os.getenv("MODEL")

# ── 1. Cheap filter (no LLM cost) ─────────────────────────────────────────
FACT_PATTERNS = [
    r"\bI (am|live|work|like|hate|prefer|use|have|want|need)\b",
    r"\bmy (name|job|hobby|goal|language|location|company|team)\b",
    r"\bI'm (a |an )?\w+",
    r"\bcall me\b",
]

def worth_extracting(user_msg: str, assistant_msg: str) -> bool:
    combined = f"{user_msg} {assistant_msg}"
    return any(re.search(p, combined, re.IGNORECASE) for p in FACT_PATTERNS)

# ── 2. LLM extractor ──────────────────────────────────────────────────────
llm = ChatOpenAI(
        openai_api_base=LITELLM_API_BASE,
        api_key=LITELLM_API_KEY,
        model=MODEL,
        temperature=0.5,
    )


EXTRACT_PROMPT = """\
Extract factual, reusable personal facts from this conversation turn.
Return ONLY a JSON array of short strings. Example:
["User prefers dark mode", "User works at a fintech startup in Ahmedabad"]
If nothing worth remembering, return [].

User: {user}
Assistant: {assistant}"""

def extract_facts(user: str, assistant: str) -> list[str]:
    response = llm.invoke(EXTRACT_PROMPT.format(user=user, assistant=assistant))
    text = response.content.strip().strip("```json").strip("```").strip()
    try:
        facts = json.loads(text)
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except json.JSONDecodeError:
        return []

# ── 3. Upsert into Postgres store ─────────────────────────────────────────
def upsert_facts(user_id: str, facts: list[str], store: PostgresStore):
    ns = ("memories", user_id)
    existing = {r.value["fact"] for r in store.search(ns, query="", limit=200)}
    new_count = 0
    for fact in facts:
        if fact not in existing:           # simple exact dedup for now
            import uuid
            store.put(ns, str(uuid.uuid4())[:8], {"fact": fact})
            new_count += 1
    if new_count:
        print(f"[worker] Saved {new_count} new facts for {user_id}")

# ── 4. NATS consumer loop ─────────────────────────────────────────────────
async def run_worker():
    nc  = await nats.connect(NATS_URI)
    js  = nc.jetstream()

    # Create stream if it doesn't exist
    try:
        await js.add_stream(name="MEMORY", subjects=["memory.extract"])
    except Exception:
        pass  # already exists

    with PostgresStore.from_conn_string(DB_URI) as store:
        store.setup()

        async def handler(msg):
            data = json.loads(msg.data.decode())
            user_id   = data["user_id"]
            user_msg  = data["user"]
            agent_msg = data["assistant"]

            if not worth_extracting(user_msg, agent_msg):
                print("[worker] Dropped — no facts detected")
                await msg.ack()
                return

            facts = extract_facts(user_msg, agent_msg)
            if facts:
                upsert_facts(user_id, facts, store)
            await msg.ack()

        await js.subscribe("memory.extract", cb=handler, durable="fact-extractor")
        print("[worker] Listening on memory.extract ...")
        await asyncio.Event().wait()  # run forever

if __name__ == "__main__":
    asyncio.run(run_worker())