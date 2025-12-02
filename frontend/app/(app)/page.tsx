"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const [name, setName] = useState("");
  const router = useRouter();

  const startGame = () => {
    if (!name.trim()) return alert("Please enter your name!");
    router.push(`/game?name=${encodeURIComponent(name)}`);
  };

  return (
    <div className="h-screen w-full flex flex-col items-center justify-center bg-gradient-to-br from-black via-purple-900 to-black text-white px-4">

      {/* Title */}
      <h1 className="text-5xl md:text-6xl font-extrabold mb-6 tracking-wide text-center drop-shadow-lg">
        🎭 Improv Battle
      </h1>

      <p className="text-gray-300 text-lg mb-8 text-center max-w-md">
        Enter your name to join the stage and face the AI Game Master using
        <span className="text-purple-400 font-semibold"> Murf Falcon TTS</span>.
      </p>

      {/* Input Box */}
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Enter your name..."
        className="w-72 p-3 rounded-xl bg-gray-800 border border-gray-700
                   text-white mb-4 focus:border-purple-500 focus:ring-2 
                   focus:ring-purple-600 outline-none transition"
      />

      {/* Start Button */}
      <button
        onClick={startGame}
        className="px-8 py-3 rounded-xl text-lg font-semibold bg-purple-600 
                   hover:bg-purple-700 active:bg-purple-800 transition
                   shadow-[0_0_20px_rgba(168,85,247,0.6)] hover:shadow-[0_0_35px_rgba(168,85,247,0.9)]"
      >
        Start Improv Battle 🎤
      </button>
    </div>
  );
}
