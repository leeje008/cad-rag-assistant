import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export interface Source {
  document: string;
  page?: number;
  section?: string;
  relevance?: number;
}

interface SourceCardProps {
  sources: Source[];
}

export function SourceCards({ sources }: SourceCardProps) {
  if (sources.length === 0) return null;

  return (
    <div className="flex flex-col gap-2 mt-3">
      <span className="text-xs font-medium text-muted-foreground">
        참고 문서
      </span>
      <div className="flex gap-2 flex-wrap">
        {sources.map((source, i) => (
          <Card
            key={i}
            className="w-fit max-w-[280px] bg-muted/50 border-border/50"
          >
            <CardHeader className="p-3">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-[10px] font-mono">
                  [{i + 1}]
                </Badge>
                {source.relevance !== undefined && (
                  <Badge variant="secondary" className="text-[10px]">
                    {Math.round(source.relevance * 100)}%
                  </Badge>
                )}
              </div>
              <CardTitle className="text-xs font-medium leading-tight mt-1 truncate">
                {source.document}
              </CardTitle>
              {(source.page || source.section) && (
                <CardDescription className="text-[10px]">
                  {source.page && `p.${source.page}`}
                  {source.page && source.section && " · "}
                  {source.section}
                </CardDescription>
              )}
            </CardHeader>
          </Card>
        ))}
      </div>
    </div>
  );
}
