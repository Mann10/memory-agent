# agent.py  — LangGraph graph: Load → Memory → Agent → [Tools → Agent*] → Save
# Fully async, flat node functions - no factory closures.
import os, uuid, asyncio
from contextlib import AsyncExitStack
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.config import get_config
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import ToolNode
from tavily import AsyncTavilyClient

from nats_publisher import publish_exchange

load_dotenv()

LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")
MODEL            = os.getenv("MODEL")
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY")
REDIS_URI        = os.getenv("REDIS_URI", "redis://localhost:6379")
DB_URI           = os.getenv("DB_URI",    "postgresql://postgres:postgres@localhost:5432/agent_memory")

TTL_CFG = {"default_ttl": 60, "refresh_on_read": True}

SUMMARY_MSG_LIMIT = 10
SUMMARY_KEEP      = 4


# ── static system prompt (never changes) ─────────────────────────────────────
BASE_SYSTEM_PROMPT = (
    "You are a helpful personal assistant with long-term memory. "
    "Keep answers concise and practical. Greet the user by name when you know it. "
    "You have access to a web search tool for current information."
)


# ── tools ──────────────────────────────────────────────────────────────────
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

@tool
async def search_web(query: str) -> str:
    """Search the live web for current information. Use this for weather, news, facts, or anything time-sensitive."""
    if not tavily_client:
        return "Error: Tavily API key is not configured."
    try:
        result = await tavily_client.search(
            query=query,
            max_results=3,
            search_depth="basic",
            include_answer=True,
        )
        answer = result.get("answer", "")
        snippets = []
        for item in result.get("results", []):
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            if content:
                snippets.append(f"- {title}: {content} ({url})")
        
        output = ""
        if answer:
            output += f"Quick answer: {answer}\n\n"
        if snippets:
            output += "Sources:\n" + "\n".join(snippets)
        return output or "No relevant results found."
    except Exception as e:
        return f"Search failed: {e}"

tools = [search_web]
tool_node = ToolNode(tools)


# ── state ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_input:      str
    user_id:         str
    memory_context:  str
    reply:           str
    messages:        Annotated[list[BaseMessage], add_messages]


# ── helpers ────────────────────────────────────────────────────────────────
def _ns(user_id: str):
    return ("memories", user_id)

def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_base=LITELLM_API_BASE,
        api_key=LITELLM_API_KEY,
        model=MODEL,
        temperature=0.5,
    )


# ── nodes (flat, direct functions) ───────────────────────────────────────────
async def load_node(state: AgentState) -> dict:
    config  = get_config()
    user_id = config["configurable"]["user_id"]
    new_message = HumanMessage(content=state["user_input"])

    return {
        **state,
        "user_id":   user_id,
        "messages":  [new_message],
    }


# PostgresStore lives at module level; nodes reference it directly.
# Initialized in get_stores() before the graph is built.
store: PostgresStore | None = None

async def memory_node(state: AgentState) -> dict:
    results = await asyncio.to_thread(
        store.search,
        _ns(state["user_id"]),
        query="",
        limit=100,
    )
    lines = [r.value["fact"] for r in results if r.value.get("fact")]
    context = ""
    if lines:
        context = "\n\nWhat you know about this user:\n" + "\n".join(f"  - {l}" for l in lines)

    return {
        **state,
        "memory_context": context,
    }


async def agent_node(state: AgentState) -> dict:
    llm      = _make_llm()
    tool_llm = llm.bind_tools(tools)
    history  = state["messages"]

    # ── summarization ───────────────────────────────────────────────────────
    if len(history) > SUMMARY_MSG_LIMIT:
        old    = history[:-SUMMARY_KEEP]
        recent = history[-SUMMARY_KEEP:]

        summary_prompt = (
            "Summarize the following conversation in 2-3 sentences. "
            "Preserve key facts, user requests, and outcomes:\n\n"
            + "\n".join(f"{m.type}: {m.content}" for m in old if m.content)
        )
        summary_msg = await llm.ainvoke([HumanMessage(content=summary_prompt)])

        system_parts = [BASE_SYSTEM_PROMPT]
        if state["memory_context"]:
            system_parts.append(state["memory_context"])
        system_parts.append(f"Earlier conversation summary: {summary_msg.content}")
        system_content = "\n\n".join(system_parts)
        messages = [SystemMessage(content=system_content), *recent]
    else:
        system_parts = [BASE_SYSTEM_PROMPT]
        if state["memory_context"]:
            system_parts.append(state["memory_context"])
        system_content = "\n\n".join(system_parts)
        messages = [SystemMessage(content=system_content), *history]

    # ── call LLM ──────────────────────────────────────────────────────────────
    response = await tool_llm.ainvoke(messages)

    return {
        **state,
        "messages": [response],
        "reply":    response.content or "",
    }


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return "save"


async def save_node(state: AgentState) -> dict:
    config    = get_config()
    thread_id = config["configurable"]["thread_id"]
    await publish_exchange(
        state["user_id"],
        thread_id,
        state["user_input"],
        state["reply"],
    )
    return state


# ── graph factory ──────────────────────────────────────────────────────────
def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("load",   load_node)
    builder.add_node("memory", memory_node)
    builder.add_node("agent",  agent_node)
    builder.add_node("tools",  tool_node)
    builder.add_node("save",   save_node)

    builder.set_entry_point("load")
    builder.add_edge("load",   "memory")
    builder.add_edge("memory", "agent")

    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "save": "save"}
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("save", END)

    return builder.compile(checkpointer=checkpointer)


# ── store/checkpointer factory ─────────────────────────────────────────────
async def get_stores():
    global store

    stack        = AsyncExitStack()
    store        = stack.enter_context(PostgresStore.from_conn_string(DB_URI))
    checkpointer = await stack.enter_async_context(
        AsyncRedisSaver.from_conn_string(REDIS_URI, ttl=TTL_CFG)
    )
    store.setup()
    await checkpointer.asetup()

    # store is now module-level; nodes can reference it
    graph = build_graph(checkpointer)
    return {"stack": stack, "store": store, "checkpointer": checkpointer, "graph": graph}


# # ── REPL (dev only) ────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     user_id   = input("User ID: ").strip() or "anonymous"
#     thread_id = input("Thread ID (enter for random): ").strip() or str(uuid.uuid4())[:8]
#     config    = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

#     async def main():
#         s     = await get_stores()
#         graph = s["graph"]
#         print("\n=== Agent ready. Type 'quit' to exit. ===\n")
#         while True:
#             user_input = input("You: ").strip()
#             if not user_input or user_input.lower() in {"quit", "exit", "q"}:
#                 break
#             result = await graph.ainvoke({"user_input": user_input}, config=config)
#             print(f"\nAgent: {result['reply']}\n")

#     asyncio.run(main())