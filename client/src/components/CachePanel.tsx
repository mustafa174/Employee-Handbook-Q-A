import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, RefreshCw, Trash2 } from "lucide-react";
import type { Config, Data, Layout } from "plotly.js";
import Plot from "react-plotly.js";
import { apiUrl } from "../apiBase";
import { useTheme } from "../theme";

type CacheStatsResponse = {
  total_cached_queries?: number;
  totalCachedQueries?: number;
  count?: number;
  total_queries?: number;
  totalQueries?: number;
  cached_queries?: number;
  cachedQueries?: number;
};

type VectorPointResponse = {
  x?: number;
  y?: number;
  text?: string;
  chunk_text?: string;
  label?: string;
  source?: string;
  kind?: string;
  category?: string;
  embedding?: number[];
};

type CacheVizResponse = {
  points?: VectorPointResponse[];
  data?: VectorPointResponse[];
  vectors?: VectorPointResponse[];
  items?: VectorPointResponse[];
};

type Category = "PTO" | "VPN" | "Personal" | "General";

type VizPoint = {
  x: number;
  y: number;
  label: string;
  text: string;
  category: Category;
};

const CATEGORY_COLORS: Record<Category, string> = {
  PTO: "#3b82f6",
  VPN: "#22c55e",
  Personal: "#ef4444",
  General: "#a1a1aa",
};

const safeNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const getErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : "Unexpected error");

const isNotFoundResponse = (error: unknown): boolean =>
  error instanceof Error && error.message.toLowerCase().includes("not found");

const guessCategory = (raw: VectorPointResponse): Category => {
  const fromApi = raw.category?.toLowerCase();
  if (fromApi?.includes("pto") || fromApi?.includes("leave")) return "PTO";
  if (fromApi?.includes("vpn") || fromApi?.includes("it")) return "VPN";
  if (fromApi?.includes("personal") || fromApi?.includes("employee")) return "Personal";

  const combined = `${raw.text ?? ""} ${raw.chunk_text ?? ""} ${raw.label ?? ""} ${raw.source ?? ""}`.toLowerCase();
  if (combined.includes("pto") || combined.includes("leave") || combined.includes("vacation")) return "PTO";
  if (combined.includes("vpn") || combined.includes("globalprotect") || combined.includes("gateway")) return "VPN";
  if (combined.includes("employee") || combined.includes("balance") || combined.includes("sick")) return "Personal";
  return "General";
};

const normalizePoint = (raw: VectorPointResponse, index: number): VizPoint | null => {
  const x = safeNumber(raw.x) ?? safeNumber(raw.embedding?.[0]) ?? null;
  const y = safeNumber(raw.y) ?? safeNumber(raw.embedding?.[1]) ?? null;
  if (x === null || y === null) return null;
  const text = (raw.text ?? raw.chunk_text ?? "No text available").trim();
  const label = (raw.label ?? raw.kind ?? `Point ${index + 1}`).trim();
  return {
    x,
    y,
    label: label || `Point ${index + 1}`,
    text: text || "No text available",
    category: guessCategory(raw),
  };
};

const fetchCacheStats = async (): Promise<number> => {
  const res = await fetch(apiUrl("/api/cache/stats"));
  if (!res.ok) throw new Error((await res.text()) || "Failed to fetch cache stats");
  const body = (await res.json()) as CacheStatsResponse;
  return body.total_cached_queries ?? body.totalCachedQueries ?? body.total_queries ?? body.totalQueries ?? body.cached_queries ?? body.cachedQueries ?? body.count ?? 0;
};

const fetchVizData = async (): Promise<VizPoint[]> => {
  const res = await fetch(apiUrl("/api/cache/viz"));
  if (!res.ok) throw new Error((await res.text()) || "Failed to fetch visualization data");
  const body = (await res.json()) as CacheVizResponse | VectorPointResponse[];
  const source = Array.isArray(body) ? body : (body.points ?? body.data ?? body.vectors ?? body.items ?? []);
  return source.map((point, index) => normalizePoint(point, index)).filter((point): point is VizPoint => point !== null);
};

const purgeCache = async (): Promise<void> => {
  const res = await fetch(apiUrl("/api/cache/purge"), { method: "DELETE" });
  if (!res.ok) throw new Error((await res.text()) || "Failed to purge cache");
};

const buildPlotTraces = (points: VizPoint[]): Data[] => {
  const categories: Category[] = ["PTO", "VPN", "Personal", "General"];
  return categories
    .map((category) => {
      const categoryPoints = points.filter((point) => point.category === category);
      if (categoryPoints.length === 0) return null;
      return {
        type: "scattergl",
        mode: "markers",
        name: category,
        x: categoryPoints.map((point) => point.x),
        y: categoryPoints.map((point) => point.y),
        text: categoryPoints.map((point) => `<b>${point.label}</b><br>${point.text}`),
        marker: {
          color: CATEGORY_COLORS[category],
          size: 9,
          line: { color: "#18181b", width: 0.5 },
          opacity: 0.9,
        },
        hovertemplate: "%{text}<extra></extra>",
      } as Data;
    })
    .filter((trace): trace is Data => trace !== null);
};

type CachePanelProps = {
  fallbackCount?: number;
};

export const CachePanel = ({ fallbackCount = 0 }: CachePanelProps) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const queryClient = useQueryClient();

  const statsQuery = useQuery({
    queryKey: ["cache", "stats"],
    queryFn: fetchCacheStats,
    staleTime: 30_000,
  });
  const vizQuery = useQuery({
    queryKey: ["cache", "viz"],
    queryFn: fetchVizData,
    staleTime: 30_000,
  });

  const purgeMutation = useMutation({
    mutationFn: purgeCache,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["cache", "stats"] }),
        queryClient.invalidateQueries({ queryKey: ["cache", "viz"] }),
      ]);
    },
  });

  const traces = buildPlotTraces(vizQuery.data ?? []);
  const statsUnavailable = isNotFoundResponse(statsQuery.error);
  const vizUnavailable = isNotFoundResponse(vizQuery.error);

  const plotLayout: Partial<Layout> = {
    autosize: true,
    paper_bgcolor: "transparent",
    plot_bgcolor: isDark ? "#0f172a" : "#fafafa",
    margin: { l: 36, r: 24, t: 16, b: 38 },
    hoverlabel: {
      bgcolor: isDark ? "#18181b" : "#ffffff",
      bordercolor: isDark ? "#3f3f46" : "#d4d4d8",
      font: { color: isDark ? "#fafafa" : "#18181b" },
    },
    xaxis: { title: { text: "Embedding X" }, color: isDark ? "#d4d4d8" : "#3f3f46" },
    yaxis: { title: { text: "Embedding Y" }, color: isDark ? "#d4d4d8" : "#3f3f46" },
    legend: { orientation: "h", y: 1.08, x: 0, font: { color: isDark ? "#d4d4d8" : "#3f3f46" } },
  };

  const plotConfig: Partial<Config> = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition-all duration-300 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">Cache Management</h2>
            <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">Monitor semantic cache usage and purge stale query embeddings.</p>
          </div>
          <button
            type="button"
            onClick={() => purgeMutation.mutate()}
            disabled={purgeMutation.isPending}
            className="inline-flex items-center gap-2 rounded-lg border border-rose-300 bg-rose-50 px-4 py-2 text-sm font-medium text-rose-700 transition-all duration-200 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-70 dark:border-rose-700 dark:bg-rose-900/25 dark:text-rose-200 dark:hover:bg-rose-900/45"
          >
            {purgeMutation.isPending ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
            Purge Semantic Cache
          </button>
        </div>

        <div className="mt-5 rounded-xl border border-zinc-200 bg-zinc-50/85 p-4 transition-all duration-300 dark:border-zinc-700 dark:bg-zinc-800/55">
          <p className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">Total Cached Queries</p>
          <div className="mt-1 text-3xl font-semibold text-zinc-900 dark:text-zinc-100">
            {statsQuery.isLoading ? <Loader2 className="size-6 animate-spin" /> : (statsUnavailable ? fallbackCount : (statsQuery.data ?? 0))}
          </div>
          {statsQuery.error ? <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">{getErrorMessage(statsQuery.error)}</p> : null}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition-all duration-300 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">Vector Space Visualizer</h2>
            <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">2D semantic projection of cached queries and indexed chunks.</p>
          </div>
          <button
            type="button"
            onClick={() => void vizQuery.refetch()}
            disabled={vizQuery.isFetching}
            className="inline-flex items-center gap-2 rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-700 transition-all duration-200 hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-70 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
          >
            <RefreshCw className={["size-4", vizQuery.isFetching ? "animate-spin" : ""].join(" ")} />
            Refresh Visualizer
          </button>
        </div>

        <div className="mt-5 h-[460px] overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50/80 p-2 transition-all duration-300 dark:border-zinc-700 dark:bg-zinc-950/70">
          {vizUnavailable ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-sm text-amber-600 dark:text-amber-400">
              Vector visualizer endpoint is unavailable (`/api/cache/viz`).
            </div>
          ) : vizQuery.isLoading ? (
            <div className="flex h-full items-center justify-center text-zinc-500 dark:text-zinc-400"><Loader2 className="size-6 animate-spin" /></div>
          ) : vizQuery.isError ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-sm text-rose-600 dark:text-rose-400">{getErrorMessage(vizQuery.error)}</div>
          ) : traces.length === 0 ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-sm text-zinc-500 dark:text-zinc-400">No vector points available yet. Run a few queries, then refresh.</div>
          ) : (
            <Plot data={traces} layout={plotLayout} config={plotConfig} style={{ width: "100%", height: "100%" }} useResizeHandler />
          )}
        </div>
      </section>
    </div>
  );
};
