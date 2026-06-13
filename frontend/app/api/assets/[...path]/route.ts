export async function GET(
  _request: Request,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return new Response("BACKEND_URL env var is not configured", { status: 500 });
  }

  const { path } = await params;
  const safePath = path.map(encodeURIComponent).join("/");

  const response = await fetch(`${backendUrl}/api/assets/${safePath}`);
  if (!response.ok) {
    return new Response("Asset not found", { status: response.status });
  }

  return new Response(response.body, {
    headers: {
      "Content-Type": response.headers.get("Content-Type") ?? "image/png",
      "Cache-Control":
        response.headers.get("Cache-Control") ??
        "public, max-age=31536000, immutable",
    },
  });
}
