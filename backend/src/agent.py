# Day 8 – Voice Game Master (D&D-Style Adventure)
import logging
import os
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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ------------------------------------------------------------
# GAME MASTER AGENT (NO TOOLS NEEDED — primary goal)
# ------------------------------------------------------------
class GameMaster(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a D&D-style Game Master running an interactive fantasy adventure.

            • Universe: A magical fantasy world filled with forests, ruins, dragons, and ancient secrets.  
            • Tone: Dramatic, immersive, atmospheric.  
            • Role: Describe scenes, react to the player's actions, and keep the story flowing.  
            • Always end your message with a clear prompt: “What do you do next?”  

            Rules:
            - Continue the story logically based on previous player choices.
            - Introduce characters, locations, items, danger, allies, mysteries.
            - Keep responses short, rich, exciting, and cinematic.
            - Do NOT make the player's decisions for them.
            - Always push the story forward.
            """
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="""
            Start the adventure immediately.
            Describe the opening scene of the fantasy world.
            End with: 'What do you do next?'
            """
        )


# ------------------------------------------------------------
# Prewarm (VAD)
# ------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=GameMaster(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
