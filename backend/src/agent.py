import json
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    RoomInputOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)

from livekit.plugins import murf, google, deepgram, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel


logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ---------- Helper: Path resolution ----------
def abs_path(rel):
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(backend_dir, rel)


# ---------- Load Catalog ----------
CATALOG_PATH = abs_path("json/day9_catalog.json")
ORDERS_PATH = abs_path("json/day9_orders.json")


def load_catalog():
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def load_orders():
    try:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_orders(orders):
    with open(ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2, ensure_ascii=False)


# ---------- Filtering Logic ----------
def filter_products(filters: dict):
    products = load_catalog()
    results = []

    for p in products:
        ok = True

        if "category" in filters:
            if p.get("category") != filters["category"]:
                ok = False

        if "max_price" in filters:
            if p.get("price", 999999) > filters["max_price"]:
                ok = False

        if "color" in filters:
            if p.get("color") != filters["color"]:
                ok = False

        if ok:
            results.append(p)

    return results


# ---------- Order Creation ----------
def create_order(items):
    products = load_catalog()
    orders = load_orders()

    order_items = []
    total = 0

    for item in items:
        pid = item["product_id"]
        qty = item["quantity"]

        product = next((p for p in products if p["id"] == pid), None)
        if not product:
            continue

        order_items.append({
            "product_id": pid,
            "name": product["name"],
            "price": product["price"],
            "quantity": qty
        })

        total += product["price"] * qty

    new_order = {
        "id": f"ORD-{len(orders) + 1}",
        "items": order_items,
        "total": total,
        "currency": "INR",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    orders.append(new_order)
    save_orders(orders)

    return new_order


# ---------- Agent ----------
class EcommerceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a helpful e-commerce shopping assistant.
You can filter products (category, price, color) and help place orders.
Always summarize results clearly.
""",
        )

    @function_tool
    async def browse_products(self, context: RunContext, filters: dict):
        """
        filters example:
        { "category": "tshirt", "max_price": 1000, "color": "black" }
        """
        results = filter_products(filters)

        if not results:
            return "No products found matching your filters."

        summary = []
        for p in results:
            summary.append(f"{p['name']} ({p['price']} INR) – ID: {p['id']}")

        return "Here are the matching products: " + " | ".join(summary)

    @function_tool
    async def order_product(self, context: RunContext, product_id: str, quantity: int):
        order = create_order([{"product_id": product_id, "quantity": quantity}])
        return (
            f"Order placed! Order ID {order['id']}. "
            f"Total: {order['total']} INR. "
        )

    @function_tool
    async def last_order(self, context: RunContext):
        orders = load_orders()
        if not orders:
            return "You haven't bought anything yet."
        last = orders[-1]
        items = ", ".join([f"{i['name']} x{i['quantity']}" for i in last["items"]])
        return f"Your last order was {last['id']}: {items}, total {last['total']} INR."


# ---------- Entry ----------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    agent = EcommerceAgent()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
