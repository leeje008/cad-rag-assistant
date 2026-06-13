"use client";

import { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const remarkPlugins = [remarkGfm];

const components: Components = {
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-muted px-2 py-1 text-left font-medium">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-border px-2 py-1 align-top">{children}</td>
  ),
  p: ({ children }) => <p className="my-1.5 first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => (
    <ul className="my-1.5 list-disc pl-5 space-y-0.5">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-1.5 list-decimal pl-5 space-y-0.5">{children}</ol>
  ),
  h1: ({ children }) => <h1 className="mt-3 mb-1.5 font-semibold">{children}</h1>,
  h2: ({ children }) => <h2 className="mt-3 mb-1.5 font-semibold">{children}</h2>,
  h3: ({ children }) => <h3 className="mt-2 mb-1 font-semibold">{children}</h3>,
  code: ({ children, className }) =>
    className ? (
      <code className={`${className} block overflow-x-auto rounded bg-muted p-2 text-xs`}>
        {children}
      </code>
    ) : (
      <code className="rounded bg-muted px-1 py-0.5 text-xs">{children}</code>
    ),
  pre: ({ children }) => <pre className="my-2">{children}</pre>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2">
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-1.5 border-l-2 border-border pl-3 text-muted-foreground">
      {children}
    </blockquote>
  ),
};

// Memoized on `content`: during streaming the parent re-renders per token,
// but sibling state changes (sources, scroll) must not re-parse the markdown.
export const MarkdownContent = memo(function MarkdownContent({
  content,
}: {
  content: string;
}) {
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} components={components}>
      {content}
    </ReactMarkdown>
  );
});
