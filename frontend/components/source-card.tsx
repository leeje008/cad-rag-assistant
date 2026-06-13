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
  text?: string;
  source_path?: string;
  chunk_id?: string;
  chunk_type?: string;
  table_id?: string | null;
  parent_id?: string | null;
  figure_id?: string | null;
  bbox?: number[] | null;
  image_url?: string | null;
  table_html?: string | null;
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
                {source.chunk_type === "image" && (
                  <Badge variant="secondary" className="text-[10px]">
                    그림
                  </Badge>
                )}
                {(source.chunk_type === "table" ||
                  source.chunk_type === "table_summary") && (
                  <Badge variant="secondary" className="text-[10px]">
                    표
                  </Badge>
                )}
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
              {source.chunk_type === "image" && source.image_url && (
                <a
                  href={source.image_url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-2 block"
                  title="원본 이미지 열기"
                >
                  {/* Proxied local asset with dynamic dimensions — next/image
                      adds nothing here but remotePatterns config. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={source.image_url}
                    alt={source.section ?? source.document}
                    loading="lazy"
                    className="max-h-32 w-auto rounded border border-border/50"
                  />
                </a>
              )}
              {source.table_html && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-[10px] text-muted-foreground select-none">
                    표 보기
                  </summary>
                  <div
                    className="mt-1 max-h-48 overflow-auto text-[10px] [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-border [&_th]:bg-muted [&_th]:px-1 [&_th]:py-0.5 [&_td]:border [&_td]:border-border [&_td]:px-1 [&_td]:py-0.5"
                    // Docling output from on-prem documents — inside the trust
                    // boundary. Add sanitization if external uploads appear.
                    dangerouslySetInnerHTML={{ __html: source.table_html }}
                  />
                </details>
              )}
            </CardHeader>
          </Card>
        ))}
      </div>
    </div>
  );
}
