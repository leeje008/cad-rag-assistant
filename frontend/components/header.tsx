"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const EMBED_MODELS = ["mxbai-embed-large", "nomic-embed-text"];

interface HeaderProps {
  backendStatus: "connected" | "disconnected" | "checking";
  models: string[];
  selectedModel: string;
  onModelChange: (model: string) => void;
}

export function Header({
  backendStatus,
  models,
  selectedModel,
  onModelChange,
}: HeaderProps) {
  const chatModels = models.filter(
    (m) => !EMBED_MODELS.some((em) => m.startsWith(em)),
  );

  return (
    <header className="border-b border-border bg-card px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <svg
            className="h-6 w-6 text-primary"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          <h1 className="text-lg font-semibold tracking-tight">
            CAD RAG Assistant
          </h1>
        </div>
        <Select value={selectedModel} onValueChange={(v) => v && onModelChange(v)}>
          <SelectTrigger className="w-[200px] h-8 text-xs font-mono">
            <SelectValue placeholder="모델 선택" />
          </SelectTrigger>
          <SelectContent>
            {chatModels.map((model) => (
              <SelectItem key={model} value={model} className="text-xs font-mono">
                {model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex items-center gap-2">
        <div
          className={`h-2 w-2 rounded-full ${
            backendStatus === "connected"
              ? "bg-green-500"
              : backendStatus === "checking"
                ? "bg-yellow-500 animate-pulse"
                : "bg-red-500"
          }`}
        />
        <span className="text-xs text-muted-foreground">
          {backendStatus === "connected"
            ? "백엔드 연결됨"
            : backendStatus === "checking"
              ? "연결 확인 중..."
              : "백엔드 연결 안됨"}
        </span>
      </div>
    </header>
  );
}
