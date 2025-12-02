# backend/src/agent.py
import logging
import os
import random
import json
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

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")


def _abs_backend_path(rel_path: str) -> str:
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.normpath(os.path.join(backend_dir, rel_path))


# Optional: if you want scenarios in a file, place at backend/json/day10_scenarios.json
SCENARIO_JSON = _abs_backend_path("json/day10_scenarios.json")


DEFAULT_SCENARIOS = [
    "You are a barista who must tell a customer that their latte is actually a portal to another dimension.",
    "You are a time-traveling tour guide explaining smartphones to someone from the 1800s.",
    "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
    "You are a superhero who forgot their superpower today and must improvise a heroic moment.",
    "You are a movie director who cast a goat in the lead role and must justify it to critics.",
    "You are a Bollywood actor delivering the most melodramatic breakup speech on a crowded train."
]


def load_scenarios():
    if os.path.exists(SCENARIO_JSON):
        try:
            with open(SCENARIO_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data
        except Exception as e:
            logger.warning("Failed to load scenarios json: %s", e)
    return DEFAULT_SCENARIOS


class ImprovState:
    def __init__(self, player_name=None, max_rounds=3):
        self.player_name = player_name
        self.current_round = 0
        self.max_rounds = max_rounds
        self.rounds = []  # each is dict {scenario, host_reaction, player_lines}
        self.phase = "intro"  # intro | awaiting_improv | reacting | done
        self.current_scenario = None
        self.current_player_lines = []
        self.turns_in_round = 0


class ImprovHostAgent(Agent):
    def __init__(self, scenarios=None) -> None:
        super().__init__(
            instructions="""
            You are the host of a TV improv show called "Improv Battle".
            Persona: high-energy, witty, clear about rules, sometimes teasing, sometimes praising.
            Behavior:
            - Introduce the show and rules.
            - For each round: announce scenario, prompt player "Start improvising — go!"
            - After the player indicates end of scene or after a small heuristic, react: mix praise/critique/tease.
            - Maintain basic state: player name, current round, rounds list, phase.
            - When max rounds reached, give a closing summary referencing moments from rounds.
            - Stay respectful and constructive.
            """,
        )
        self.scenarios = scenarios or load_scenarios()
        self.improv_state = ImprovState(max_rounds=3)

    async def on_enter(self) -> None:
        # Called when agent enters session — start show intro
        intro = (
            "Welcome to Improv Battle! I'm your host — high energy, little bit cheeky, "
            "and here to push your creative limits. "
            "Rules: I'll give a scenario each round. Improv in character for a bit, then say 'End scene' or 'Okay' when done. "
            "We'll do three rounds. What's your name, contestant?"
        )
        await self.session.generate_reply(instructions=intro)

    def _choose_scenario(self):
        # choose next scenario (rotate/random)
        remaining = list(self.scenarios)
        return random.choice(remaining)

    def _is_end_of_scene(self, text: str):
        if not text:
            return False
        t = text.strip().lower()
        # explicit phrases
        if any(kw in t for kw in ("end scene", "end the scene", "end show", "end scene.", "done", "okay", "i'm done", "that’s it", "that's it")):
            return True
        # short heuristic: player used 2+ turns in this round
        if self.improv_state.turns_in_round >= 2:
            # if they speak a short confirmation word treat as end; otherwise we will wait for explicit
            if len(t.split()) <= 3:
                return True
        return False

    def _generate_reaction(self, player_lines: list[str]) -> str:
        """
        Create a varied realistic host reaction using templates.
        For demo stability we use templates; you can replace with LLM call if desired.
        """
        recent = player_lines[-1] if player_lines else ""
        tone = random.choice(["positive", "tease", "constructive", "amused"])
        examples = {
            "positive": [
                "That was brilliant — loved the details and your commitment to the character!",
                "Amazing energy — the audience would be in stitches right now.",
                "Solid choice — you owned the character and made clear choices."
            ],
            "tease": [
                "Ha! That was delightfully absurd — I think the prop department is still recovering.",
                "Bold move! I didn't see that coming — you cheeky improviser.",
                "I loved that, even if the script committee may disagree."
            ],
            "constructive": [
                "Good start — try leaning more into the character's objectives next time.",
                "Nice premise. You could have taken a pause there to heighten the tension.",
                "Solid idea — to make it stronger, pick one clear goal for your character."
            ],
            "amused": [
                "I laughed out loud — excellent use of the space!",
                "That twist was unexpected and delightful.",
                "Very funny — that line about the goat will be replayed in my head all day."
            ]
        }
        pick = random.choice(examples[tone])
        # reference player's last line if short
        ref = ""
        if recent and len(recent.split()) <= 8:
            ref = f" I particularly liked when you said: \"{recent}\"."
        return f"{pick}{ref}"

    @function_tool(name="improv_round")
    async def improv_round(self, context: RunContext, user_input: str) -> str:
        """
        Single tool that drives the improv flow.
        It handles:
         - intro (get player name)
         - starting rounds (announce scenario)
         - collecting player lines
         - end-of-scene detection and reactions
         - closing summary
        """
        state = self.improv_state
        text = (user_input or "").strip()

        # INTRO: capture player name
        if state.phase == "intro":
            if not state.player_name:
                # set name from user's spoken input (if they said a name)
                if text:
                    state.player_name = text.split()[0].capitalize()
                    # start first round
                    state.current_round = 1
                    state.current_scenario = self._choose_scenario()
                    state.phase = "awaiting_improv"
                    state.turns_in_round = 0
                    state.current_player_lines = []
                    return (
                        f"Great to meet you, {state.player_name}! Round {state.current_round} of {state.max_rounds}. "
                        f"Scenario: {state.current_scenario} Start improvising — go!"
                    )
                else:
                    return "I didn't catch your name. What's your name, contestant?"
            else:
                # shouldn't happen, but move on
                state.phase = "awaiting_improv"

        # AWAITING_IMPROV: collect user lines until end-of-scene
        if state.phase == "awaiting_improv":
            if not text:
                return "Whenever you're ready, start the improv. Say something in character!"
            # store player line
            state.current_player_lines.append(text)
            state.turns_in_round += 1

            # check for explicit end
            if self._is_end_of_scene(text):
                # finalize reaction
                reaction = self._generate_reaction(state.current_player_lines)
                # save round
                state.rounds.append({
                    "scenario": state.current_scenario,
                    "player_lines": state.current_player_lines.copy(),
                    "host_reaction": reaction,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                # prepare next
                if state.current_round >= state.max_rounds:
                    state.phase = "done"
                    # closing summary
                    summary = self._closing_summary()
                    return f"{reaction} \n\nThat's the final round. {summary}"
                else:
                    state.current_round += 1
                    # pick next
                    state.current_scenario = self._choose_scenario()
                    state.phase = "awaiting_improv"
                    state.current_player_lines = []
                    state.turns_in_round = 0
                    return f"{reaction} \n\nNice one. Next — Round {state.current_round}. Scenario: {state.current_scenario} Start improvising — go!"
            else:
                # not ended yet — encourage more
                return "Nice — continue! Say 'End scene' or 'Okay' when you're done."

        # DONE: game finished
        if state.phase == "done":
            return "Thanks for playing Improv Battle! If you'd like to play again, say 'restart'."

        # restart handling
        if text.lower().strip() in ("restart", "play again", "start over"):
            self.improv_state = ImprovState(player_name=state.player_name, max_rounds=state.max_rounds)
            return "Restarting the show. What's your name?"

        # fallback
        return "I didn't catch that — say your name to start, or say 'restart' to start over."

    def _closing_summary(self):
        state = self.improv_state
        if not state.rounds:
            return "We didn't complete any rounds. Say 'restart' to try again."
        # simple analysis: count which rounds had 'amusing' or 'tease' etc via host_reaction keywords
        highlights = []
        for i, r in enumerate(state.rounds, start=1):
            snippet = r["player_lines"][-1] if r["player_lines"] else ""
            highlights.append(f"Round {i}: memorable line — \"{snippet}\"")
        style = random.choice([
            "You're an improviser who leans into absurdity and surprise.",
            "You favor character commitment and clear choices.",
            "You created great twists and showed good comedic timing."
        ])
        return f"{style} Highlights: " + " | ".join(highlights) + " Thanks for playing! Say 'restart' to go again."


# --- Prewarm and entrypoint ---
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    agent = ImprovHostAgent()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
