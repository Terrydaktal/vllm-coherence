// Optional transport for the Whonix client; ordinary host Pi keeps its SSH path.
export function radianceBridgeUrl() {
  const configured = process.env.QWEN_RADIANCE_BRIDGE_URL;
  if (!configured) return undefined;
  const url = new URL(configured);
  if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" ||
      url.username || url.password || url.search || url.hash || url.pathname !== "/qwen-radiance/control") {
    throw new Error("invalid local Radiance bridge URL");
  }
  return url.href;
}

export async function radianceBridgeRequest(value, { timeout = 130000, fetcher = fetch } = {}) {
  const url = radianceBridgeUrl();
  if (!url) throw new Error("Radiance bridge is not configured");
  const response = await fetcher(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value), signal: AbortSignal.timeout(timeout), redirect: "error",
  });
  const reader = response.body.getReader();
  const chunks = [];
  let bytes = 0;
  try {
    while (true) {
      const { value: chunk, done } = await reader.read();
      if (done) break;
      bytes += chunk.byteLength;
      if (bytes > 8 * 1024 * 1024) {
        await reader.cancel();
        throw new Error("Radiance bridge response is too large");
      }
      chunks.push(Buffer.from(chunk));
    }
  } finally { reader.releaseLock(); }
  const result = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  if (!response.ok) throw new Error(result.error || "Radiance bridge request failed");
  return result;
}
