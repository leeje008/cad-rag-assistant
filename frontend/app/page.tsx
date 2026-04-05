"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/header";
import { ChatInterface } from "@/components/chat-interface";

const DEFAULT_MODEL = "qwen3.5:27b";

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<
    "connected" | "disconnected" | "checking"
  >("checking");
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        setBackendStatus(data.ollama ? "connected" : "disconnected");
        if (data.models?.length > 0) {
          setModels(data.models);
        }
      } catch {
        setBackendStatus("disconnected");
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col h-screen">
      <Header
        backendStatus={backendStatus}
        models={models}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
      />
      <ChatInterface selectedModel={selectedModel} />
    </div>
  );
}
