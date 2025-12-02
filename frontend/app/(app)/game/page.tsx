"use client";

import { useSearchParams } from "next/navigation";
import VoiceAgent from "C:/Users/Prince/Desktop/falcon-tdova-nov25-livekit/frontend/components/VoiceAgent";

export default function GamePage() {
  const params = useSearchParams();
  const name = params.get("name") || "Player";

  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-purple-950 to-black text-white flex flex-col">

      {/* Header */}
      <header className="p-5 bg-purple-800 bg-opacity-60 backdrop-blur-md shadow-lg text-center">
        <h1 className="text-3xl font-bold tracking-wide">
          🎤 Improv Battle - Welcome, {name}!
        </h1>
        <p className="text-purple-300 mt-1">
          Let the AI Game Master challenge your creativity!
        </p>
      </header>

      {/* Game Area */}
      <div className="flex-1 p-4 md:p-8">
        <div
          className="max-w-3xl mx-auto bg-black bg-opacity-40 border border-purple-700
                     rounded-2xl p-6 shadow-[0_0_30px_rgba(168,85,247,0.4)]
                     backdrop-blur-lg"
        >
          <VoiceAgent playerName={name} />
        </div>
      </div>
    </div>
  );
}
