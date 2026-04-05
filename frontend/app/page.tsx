"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/header";
import { ChatInterface } from "@/components/chat-interface";

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<
    "connected" | "disconnected" | "checking"
  >("checking");

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        setBackendStatus(data.ollama ? "connected" : "disconnected");
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
      <Header backendStatus={backendStatus} />
      <ChatInterface />
    </div>
  );
}
