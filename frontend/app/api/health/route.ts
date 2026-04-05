export async function GET() {
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";

  try {
    const response = await fetch(`${backendUrl}/api/health`, {
      next: { revalidate: 0 },
    });
    const data = await response.json();
    return Response.json(data);
  } catch {
    return Response.json(
      { status: "error", ollama: false, models: [] },
      { status: 503 },
    );
  }
}
