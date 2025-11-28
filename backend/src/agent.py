# backend/src/agent.py
import logging
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent_day7")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")

# -------------------------
# Paths (user-provided)
# -------------------------
JSON_DIR = r"C:\Users\Prince\Desktop\falcon-tdova-nov25-livekit\backend\json"
CATALOG_PATH = os.path.join(JSON_DIR, "catalog.json")
RECIPES_PATH = os.path.join(JSON_DIR, "recipes.json")
ORDERS_DIR = os.path.join(JSON_DIR, "orders")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(ORDERS_DIR, exist_ok=True)

# -------------------------
# Helpers to read/write JSON
# -------------------------
def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load json %s: %s", path, e)
        return None


def _write_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to write json %s: %s", path, e)


# -------------------------
# Default catalog/recipes if missing (small starter)
# -------------------------
DEFAULT_CATALOG = [
    {"id": "bread", "name": "Whole Wheat Bread", "category": "groceries", "price": 40},
    {"id": "peanut_butter", "name": "Peanut Butter", "category": "groceries", "price": 150},
    {"id": "milk", "name": "Full Cream Milk 1L", "category": "groceries", "price": 60},
    {"id": "eggs", "name": "Eggs Pack of 6", "category": "groceries", "price": 55},
    {"id": "chips", "name": "Potato Chips", "category": "snacks", "price": 30},
    {"id": "noodles", "name": "Instant Noodles", "category": "snacks", "price": 25},
    {"id": "pizza_margherita", "name": "Margherita Pizza", "category": "prepared_food", "price": 250},
    {"id": "pasta", "name": "Pasta 500g", "category": "groceries", "price": 80},
    {"id": "pasta_sauce", "name": "Pasta Sauce", "category": "groceries", "price": 120},
    {"id": "butter", "name": "Butter 200g", "category": "groceries", "price": 55}
]

DEFAULT_RECIPES = {
    "peanut butter sandwich": ["bread", "peanut_butter"],
    "pasta for two": ["pasta", "pasta_sauce", "butter"]
}

# Ensure minimal files exist
if not os.path.exists(CATALOG_PATH):
    _write_json(CATALOG_PATH, DEFAULT_CATALOG)
if not os.path.exists(RECIPES_PATH):
    _write_json(RECIPES_PATH, DEFAULT_RECIPES)


# -------------------------
# Catalog helper
# -------------------------
class Catalog:
    def __init__(self, path):
        self.path = path
        self.items = self._load()

    def _load(self):
        data = _load_json(self.path)
        if isinstance(data, list):
            return data
        return []

    def find_by_id_or_name(self, text):
        t = (text or "").strip().lower()
        if not t:
            return None
        # try id exact
        for it in self.items:
            if it.get("id", "").lower() == t:
                return it
        # try exact name
        for it in self.items:
            if it.get("name", "").lower() == t:
                return it
        # try partial name
        for it in self.items:
            if t in it.get("name", "").lower():
                return it
        # try fuzzy token match: match any word token
        tokens = set(t.split())
        best = None
        best_score = 0
        for it in self.items:
            name_tokens = set(it.get("name", "").lower().split())
            score = len(tokens & name_tokens)
            if score > best_score:
                best_score = score
                best = it
        return best if best_score > 0 else None

    def price_of(self, item_id):
        for it in self.items:
            if it.get("id") == item_id:
                return it.get("price", 0)
        return 0


# -------------------------
# Cart object (session-level)
# -------------------------
class Cart:
    def __init__(self):
        # { item_id: {item: <catalog item>, qty: int, note: str} }
        self._items = {}

    def add(self, item, quantity=1, note=None):
        item_id = item.get("id")
        if item_id in self._items:
            self._items[item_id]["qty"] += quantity
            if note:
                self._items[item_id]["note"] = note
        else:
            self._items[item_id] = {"item": item, "qty": quantity, "note": note}

    def remove(self, item_id):
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def update_qty(self, item_id, quantity):
        if item_id in self._items:
            if quantity <= 0:
                del self._items[item_id]
            else:
                self._items[item_id]["qty"] = quantity
            return True
        return False

    def list_items(self):
        lines = []
        total = 0
        for iid, info in self._items.items():
            name = info["item"].get("name")
            qty = info["qty"]
            price = info["item"].get("price", 0)
            subtotal = qty * price
            total += subtotal
            note = info.get("note")
            lines.append({"id": iid, "name": name, "qty": qty, "unit_price": price, "subtotal": subtotal, "note": note})
        return lines, total

    def is_empty(self):
        return len(self._items) == 0

    def clear(self):
        self._items = {}

    def to_order_items(self):
        lines, total = self.list_items()
        return lines, total


# -------------------------
# NLP helpers (very simple)
# -------------------------
def parse_add_command(text):
    """
    Attempts to parse "add 2 breads" / "add bread" / "put 3 milk" etc.
    Returns (qty:int or None, item_text)
    """
    text = (text or "").lower()
    # common patterns: add 2 bread, add bread, add 2 packs of bread
    m = re.search(r"(?:add|put|please add|i want|i need|order)\s+(\d+)\s+([a-z0-9\s\-_]+)", text)
    if m:
        qty = int(m.group(1))
        item = m.group(2).strip()
        return qty, item
    # "add bread" / "add peanut butter"
    m2 = re.search(r"(?:add|put|please add|i want|i need|order)\s+([a-z0-9\s\-_]+)", text)
    if m2:
        return 1, m2.group(1).strip()
    return None, None


def parse_remove_command(text):
    # remove bread / remove peanut butter
    m = re.search(r"(?:remove|delete|cancel|remove item)\s+([a-z0-9\s\-_]+)", (text or "").lower())
    if m:
        return m.group(1).strip()
    return None


def parse_update_command(text):
    # update bread to 3 / change bread to 2 / set eggs to 6
    m = re.search(r"(?:update|change|set)\s+([a-z0-9\s\-_]+)\s+(?:to|=)\s*(\d+)", (text or "").lower())
    if m:
        return m.group(1).strip(), int(m.group(2))
    # "make it 3" (requires context) -> not handled here
    return None, None


def detect_place_order(text):
    return bool(re.search(r"\b(place order|place my order|checkout|i'm done|i am done|that's all|confirm order|order now)\b", (text or "").lower()))


def detect_show_cart(text):
    return bool(re.search(r"\b(what's in my cart|what is in my cart|show cart|show me my cart|cart|my cart|list cart)\b", (text or "").lower()))


def detect_recipe_request(text):
    # "ingredients for peanut butter sandwich" or "ingredients for X" or "get me ingredients for X"
    m = re.search(r"(?:ingredients for|ingredients to make|ingredients needed for|get me ingredients for|get ingredients for|ingredients for a|ingredients for the)\s+([a-z0-9\s\-_]+)", (text or "").lower())
    if m:
        dish = m.group(1).strip()
        return dish
    return None


# -------------------------
# The Agent
# -------------------------
class OrderingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a friendly food & grocery ordering assistant. Greet user, explain capability, take items into cart, confirm changes, and place orders saving them to JSON.
            """
        )
        self.catalog = Catalog(CATALOG_PATH)
        self.recipes = _load_json(RECIPES_PATH) or {}
        self.cart = Cart()

    async def on_enter(self) -> None:
        titles = ", ".join([it.get("name") for it in self.catalog.items[:6]])
        await self.session.generate_reply(instructions=f"Hello! I can help you order groceries and simple meals. Try saying 'Add 2 bread' or 'ingredients for peanut butter sandwich'. Example items: {titles}.")

    @function_tool(name="process_order")
    async def process_order(self, context: RunContext, user_input: str) -> str:
        text = (user_input or "").strip()
        if not text:
            return "Say what you'd like to do — add items, remove items, show cart, or place order."

        # 1) detect recipe/ingredients request
        dish = detect_recipe_request(text)
        if dish:
            # try direct match in recipes
            dish_key = dish.lower()
            matched = None
            # try direct match
            if dish_key in self.recipes:
                matched = self.recipes[dish_key]
            else:
                # try partial title match keys
                for k in self.recipes:
                    if dish_key in k.lower():
                        matched = self.recipes[k]
                        break
            if not matched:
                return f"Sorry, I couldn't find a recipe mapping for '{dish}'. You can try a simpler name."
            # add each mapped item (default qty 1)
            added_names = []
            for item_id in matched:
                # find item in catalog
                item = None
                # item_id may be id or name - try id first
                for it in self.catalog.items:
                    if it.get("id") == item_id or it.get("id") == item_id.lower():
                        item = it
                        break
                if not item:
                    # try name search
                    item = self.catalog.find_by_id_or_name(item_id)
                if item:
                    self.cart.add(item, quantity=1)
                    added_names.append(item.get("name"))
            if added_names:
                return f"I added {', '.join(added_names)} to your cart for '{dish}'."
            return f"Couldn't add items for '{dish}'. Check the recipe mapping."

        # 2) add command
        qty, item_text = parse_add_command(text)
        if item_text:
            item = self.catalog.find_by_id_or_name(item_text)
            if not item:
                return f"I couldn't find '{item_text}' in the catalog. Try another item."
            q = qty if qty and qty > 0 else 1
            self.cart.add(item, quantity=q)
            return f"Added {q} x {item.get('name')} to your cart."

        # 3) remove command
        rem = parse_remove_command(text)
        if rem:
            # find item
            item = self.catalog.find_by_id_or_name(rem)
            if not item:
                # maybe user gave id name or partial
                return f"Couldn't find item '{rem}' to remove."
            ok = self.cart.remove(item.get("id"))
            if ok:
                return f"Removed {item.get('name')} from your cart."
            return f"{item.get('name')} was not in your cart."

        # 4) update command
        upd_item_text, upd_qty = parse_update_command(text)
        if upd_item_text and upd_qty is not None:
            item = self.catalog.find_by_id_or_name(upd_item_text)
            if not item:
                return f"Couldn't find '{upd_item_text}' to update."
            ok = self.cart.update_qty(item.get("id"), upd_qty)
            if ok:
                return f"Updated {item.get('name')} to quantity {upd_qty}."
            return f"{item.get('name')} is not in your cart."

        # 5) show cart
        if detect_show_cart(text):
            if self.cart.is_empty():
                return "Your cart is empty."
            lines, total = self.cart.list_items()
            text_lines = []
            for it in lines:
                note = f" ({it['note']})" if it.get("note") else ""
                text_lines.append(f"{it['qty']} x {it['name']}{note} — ₹{it['subtotal']}")
            return "Cart:\n" + "\n".join(text_lines) + f"\nTotal: ₹{total}"

        # 6) place order
        if detect_place_order(text):
            if self.cart.is_empty():
                return "Your cart is empty—nothing to place."
            lines, total = self.cart.to_order_items()
            order = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "items": lines,
                "total": total,
                "notes": None
            }
            # simple filename
            fname = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = os.path.join(ORDERS_DIR, fname)
            _write_json(path, order)
            # clear cart
            self.cart.clear()
            return f"Order placed and saved as {fname}. Total: ₹{total}. Thank you!"
        
        # 7) quick intents: list catalog sample
        if re.search(r"\b(menu|catalog|items|list items|what can i order)\b", text.lower()):
            titles = [f"{it.get('name')} (₹{it.get('price')})" for it in self.catalog.items[:20]]
            return "Catalog sample: " + ", ".join(titles)

        # 8) fallback - ask clarifying question
        return "I didn't understand that. You can say 'Add 2 bread', 'Remove peanut butter', 'Show cart', 'Ingredients for peanut butter sandwich', or 'Place my order'."

# -------------------------
# Prewarm and Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.warning("VAD prewarm failed: %s", e)


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Start AgentSession with reliable defaults
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
        vad=ctx.proc.userdata.get("vad"),
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=OrderingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
