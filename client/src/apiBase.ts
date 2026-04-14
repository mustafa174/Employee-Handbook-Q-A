/**
 * API base for fetch().
 * - Empty: use same-origin `/api/...` (Vite dev proxy when RAG_API_URL is set in vite.config).
 * - Set VITE_API_BASE_URL to hit FastAPI directly (e.g. production static site + API on another host).
 */
export function getApiBase(): string {
  const v = import.meta.env.VITE_API_BASE_URL;
  if (typeof v === "string" && v.trim()) {
    return v.replace(/\/+$/, "");
  }
  return "";
}

export function apiUrl(path: string): string {
  const base = getApiBase();
  const p = path.startsWith("/") ? path : `/${path}`;
  return base ? `${base}${p}` : p;
}
