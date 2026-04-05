import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { SourceCards, type Source } from "@/components/source-card";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
}

export function MessageBubble({
  role,
  content,
  sources = [],
  isStreaming = false,
}: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <Avatar className="h-8 w-8 shrink-0">
        <AvatarFallback
          className={
            isUser
              ? "bg-primary text-primary-foreground text-xs"
              : "bg-muted text-muted-foreground text-xs"
          }
        >
          {isUser ? "U" : "AI"}
        </AvatarFallback>
      </Avatar>
      <div
        className={`flex flex-col max-w-[80%] ${isUser ? "items-end" : "items-start"}`}
      >
        <div
          className={`rounded-lg px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-card border border-border"
          }`}
        >
          {content}
          {isStreaming && (
            <span className="inline-block w-1.5 h-4 ml-0.5 bg-foreground/70 animate-pulse" />
          )}
        </div>
        {!isUser && sources.length > 0 && <SourceCards sources={sources} />}
      </div>
    </div>
  );
}
