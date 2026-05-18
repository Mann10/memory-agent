# publisher.py
import nats, json, asyncio

async def publish_exchange(user_id: str, thread_id: str, user_msg: str, agent_reply: str):
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()
    payload = json.dumps({
        "user_id":   user_id,
        "thread_id": thread_id,
        "user":      user_msg,
        "assistant": agent_reply,
    }).encode()
    await js.publish("memory.extract", payload)
    await nc.drain()