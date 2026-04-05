export const maxDuration = 60;

export async function POST(request: Request) {
  const body = await request.json();
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";

  const response = await fetch(`${backendUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    return new Response("Backend error", { status: response.status });
  }

  return new Response(response.body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "X-Vercel-AI-Data-Stream": "v1",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
