# main.py  — thin API layer
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from agent import get_stores

stores = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    stores["instances"] = await get_stores()     # opens Redis + Postgres + builds graph once
    yield
    await stores["instances"]["stack"].aclose()   # clean shutdown

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    user_id:   str
    thread_id: Optional[str] = Field(default=None, description="Omit to start a new thread")
    message:   str

@app.get("/thread")
async def new_thread():
    """Generate a fresh thread ID the client can reuse across turns."""
    return {"thread_id": str(uuid.uuid4())}

@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    graph     = stores["instances"]["graph"]
    result    = await graph.ainvoke(
        {"user_input": req.message},
        config={"configurable": {
            "thread_id":thread_id,
            "user_id": req.user_id,      # ← passed via config, read by get_config() in nodes
        }},
    )
    return {"thread_id": thread_id, "reply": result["reply"]}