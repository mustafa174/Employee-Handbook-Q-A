import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  AskResponse,
  HealthResponse,
  IngestPathRequest,
  IngestResponse,
} from "@employee-handbook/shared";
import { BookOpen, Loader2, RefreshCw, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { apiUrl } from "../apiBase";
import { useChatHistory } from "../state/ChatHistoryContext";

const getErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : "Unexpected error");

const fetchHealth = async (): Promise<HealthResponse> => {
  const res = await fetch(apiUrl("/api/health"));
  if (!res.ok) throw new Error("API unreachable");
  return res.json() as Promise<HealthResponse>;
};

const postBootstrap = async (): Promise<IngestResponse> => {
  const res = await fetch(apiUrl("/api/bootstrap"), { method: "POST" });
  if (!res.ok) throw new Error((await res.text()) || "Bootstrap failed");
  return res.json() as Promise<IngestResponse>;
};

const postIngest = async (body: IngestPathRequest): Promise<IngestResponse> => {
  const res = await fetch(apiUrl("/api/ingest"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      handbook_path: body.handbook_path,
      replace: body.replace,
    }),
  });
  if (!res.ok) throw new Error((await res.text()) || "Ingest failed");
  return res.json() as Promise<IngestResponse>;
};

const purgeSemanticCache = async (): Promise<void> => {
  const res = await fetch(apiUrl("/api/cache/purge"), { method: "DELETE" });
  if (!res.ok) throw new Error((await res.text()) || "Cache purge failed");
};

const fetchCacheStatsProbe = async (): Promise<boolean> => {
  const res = await fetch(apiUrl("/api/cache/stats"));
  if (!res.ok) throw new Error("Cache DB unavailable");
  return true;
};

const fetchAskProbe = async (): Promise<AskResponse> => {
  // #region agent log
  fetch("http://127.0.0.1:7340/ingest/caf36bdf-b3aa-457a-8a14-e9c51eb4cc1d", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "0b08b0" },
    body: JSON.stringify({
      sessionId: "0b08b0",
      runId: "pre-fix",
      hypothesisId: "H1",
      location: "Settings.tsx:fetchAskProbe",
      message: "Settings ask probe fired",
      data: { question: "How many PTO days do I have?" },
      timestamp: Date.now(),
    }),
  }).catch(() => {});
  // #endregion
  const res = await fetch(apiUrl("/api/ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: "How many PTO days do I have?",
      employee_id: "E001",
      use_rag: true,
      skip_cache: true,
    }),
  });
  if (!res.ok) throw new Error("RAG pipeline probe failed");
  return res.json() as Promise<AskResponse>;
};

type StatusTone = "ok" | "warn" | "down";

const toneClass: Record<StatusTone, string> = {
  ok: "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-200",
  warn: "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200",
  down: "border-rose-300 bg-rose-50 text-rose-800 dark:border-rose-700 dark:bg-rose-950/40 dark:text-rose-200",
};

const CACHE_PREF_STORAGE_KEY = "handbook-enable-semantic-cache-v1";
const CACHE_PREF_EVENT = "handbook:cache-preference-changed";

const clearBrowserStorage = async (): Promise<void> => {
  try {
    globalThis.localStorage.clear();
  } catch {
    // Ignore storage failures; other clearing steps may still succeed.
  }

  try {
    globalThis.sessionStorage.clear();
  } catch {
    // Ignore storage failures; other clearing steps may still succeed.
  }

  if ("caches" in globalThis) {
    try {
      const keys = await globalThis.caches.keys();
      await Promise.all(keys.map((key) => globalThis.caches.delete(key)));
    } catch {
      // Ignore Cache Storage failures.
    }
  }

  const indexedDbApi = globalThis.indexedDB;
  if (!indexedDbApi) return;

  const deleteDatabase = (name: string): Promise<void> =>
    new Promise((resolve) => {
      try {
        const request = indexedDbApi.deleteDatabase(name);
        request.onsuccess = () => resolve();
        request.onerror = () => resolve();
        request.onblocked = () => resolve();
      } catch {
        resolve();
      }
    });

  const factoryWithDatabases = indexedDbApi as IDBFactory & {
    databases?: () => Promise<Array<{ name?: string }>>;
  };

  if (typeof factoryWithDatabases.databases === "function") {
    try {
      const databases = await factoryWithDatabases.databases();
      const names = databases
        .map((entry) => entry.name)
        .filter((name): name is string => typeof name === "string" && name.length > 0);
      await Promise.all(names.map((name) => deleteDatabase(name)));
      return;
    } catch {
      // Fall back to deleting known app database names below.
    }
  }

  await Promise.all([
    deleteDatabase("handbook-chat-history-v1"),
    deleteDatabase("handbook-naive-turns-v1"),
    deleteDatabase("handbook-enable-semantic-cache-v1"),
  ]);
};

export const SettingsPage = () => {
  const queryClient = useQueryClient();
  const { setChatHistory, setSelectedEmployeeId } = useChatHistory();
  const [handbookPath, setHandbookPath] = useState("fixtures/handbook.md");
  const [enableSemanticCache, setEnableSemanticCache] = useState<boolean>(() => {
    try {
      const raw = globalThis.localStorage.getItem(CACHE_PREF_STORAGE_KEY);
      if (raw === null) return true;
      return raw !== "false";
    } catch {
      return true;
    }
  });

  const onToggleSemanticCache = (next: boolean) => {
    setEnableSemanticCache(next);
    try {
      globalThis.localStorage.setItem(CACHE_PREF_STORAGE_KEY, String(next));
      globalThis.dispatchEvent(new Event(CACHE_PREF_EVENT));
    } catch {
      // Ignore storage failures; session state still applies.
    }
  };
  const health = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: 6,
    retryDelay: (i) => Math.min(500 * 2 ** i, 8000),
  });
  const bootstrap = useMutation({ mutationFn: postBootstrap });
  const ingest = useMutation({ mutationFn: postIngest });
  const clearSiteStorage = useMutation({
    mutationFn: async () => {
      await purgeSemanticCache();
      globalThis.localStorage.removeItem("handbook-chat-history-v1");
      globalThis.localStorage.removeItem("handbook-naive-turns-v1");
      setChatHistory([]);
      setSelectedEmployeeId("E001");
      globalThis.dispatchEvent(new Event("handbook:clear-site-storage"));
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["status", "cache-db"] });
      void queryClient.invalidateQueries({ queryKey: ["status", "rag-probe"] });
      void queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });
  const atomicClear = useMutation({
    mutationFn: async () => {
      // #region agent log
      fetch("http://127.0.0.1:7340/ingest/caf36bdf-b3aa-457a-8a14-e9c51eb4cc1d", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "0b08b0" },
        body: JSON.stringify({
          sessionId: "0b08b0",
          runId: "pre-fix",
          hypothesisId: "H4",
          location: "Settings.tsx:atomicClear",
          message: "Atomic clear started",
          data: {},
          timestamp: Date.now(),
        }),
      }).catch(() => {});
      // #endregion
      await purgeSemanticCache();
      await clearBrowserStorage();
      await queryClient.cancelQueries();
      queryClient.clear();
      setChatHistory([]);
      setSelectedEmployeeId("E001");
      globalThis.dispatchEvent(new Event(CACHE_PREF_EVENT));
      globalThis.dispatchEvent(new Event("handbook:clear-site-storage"));
    },
    onSuccess: () => {
      globalThis.location.assign("/assistant");
    },
  });
  const purgeCacheOnly = useMutation({
    mutationFn: purgeSemanticCache,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["status", "cache-db"] });
      void queryClient.invalidateQueries({ queryKey: ["status", "rag-probe"] });
      void queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });
  const cacheProbe = useQuery({
    queryKey: ["status", "cache-db"],
    queryFn: fetchCacheStatsProbe,
    staleTime: 30_000,
  });
  const askProbe = useQuery({
    queryKey: ["status", "rag-probe"],
    queryFn: fetchAskProbe,
    enabled: health.isSuccess,
    staleTime: 45_000,
  });

  const systemStatuses = useMemo(() => {
    const modelTone: StatusTone = health.isSuccess ? "ok" : health.isError ? "down" : "warn";
    const modelText = health.isSuccess
      ? `Connected (${health.data.chat_model ?? "model detected"})`
      : health.isError
        ? "Disconnected"
        : "Checking";

    const dbTone: StatusTone = cacheProbe.isSuccess ? "ok" : cacheProbe.isError ? "down" : "warn";
    const dbText = cacheProbe.isSuccess ? "Connected (cache store reachable)" : cacheProbe.isError ? "Unavailable" : "Checking";

    const internetOnline = typeof navigator !== "undefined" ? navigator.onLine : true;
    const internetTone: StatusTone = internetOnline ? "ok" : "down";
    const internetText = internetOnline ? "Online" : "Offline";

    const mcpStep = askProbe.data?.pipeline_steps?.find((step) => step.id === "mcp_hr");
    const mcpTone: StatusTone =
      askProbe.isError ? "down" : mcpStep?.status === "ok" ? "ok" : mcpStep?.status === "skipped" ? "warn" : "warn";
    const mcpText =
      askProbe.isError
        ? "Probe failed"
        : mcpStep?.status === "ok"
          ? "Connected"
          : mcpStep?.status === "skipped"
            ? "Reachable (not triggered in probe)"
            : "Checking";

    const ragReady = Boolean(askProbe.data?.pipeline_steps?.some((step) => step.id === "llm"));
    const ragTone: StatusTone = askProbe.isError ? "down" : ragReady ? "ok" : "warn";
    const ragText = askProbe.isError ? "Degraded" : ragReady ? "Operational" : "Checking";

    return [
      { label: "Model Connectivity", tone: modelTone, text: modelText },
      { label: "DB Connectivity", tone: dbTone, text: dbText },
      { label: "Internet", tone: internetTone, text: internetText },
      { label: "MCP Status", tone: mcpTone, text: mcpText },
      { label: "RAG Pipeline", tone: ragTone, text: ragText },
    ];
  }, [askProbe.data, askProbe.isError, cacheProbe.isError, cacheProbe.isSuccess, health.data, health.isError, health.isSuccess]);

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition-all duration-300 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">System Status</h2>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Live runtime checks for model, cache DB, internet, MCP, and RAG pipeline health.
        </p>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {systemStatuses.map((status) => (
            <div key={status.label} className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-700 dark:bg-zinc-800/50">
              <p className="text-xs text-zinc-500 dark:text-zinc-400">{status.label}</p>
              <span className={["mt-2 inline-flex rounded-full border px-2 py-0.5 text-xs font-medium", toneClass[status.tone]].join(" ")}>
                {status.text}
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition-all duration-300 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">Model Context</h2>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          OpenAI chat model:{" "}
          <span className="font-mono text-xs">{health.data?.chat_model ?? "gpt-4o-mini"}</span>
        </p>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition-all duration-300 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex items-center gap-3">
          <BookOpen className="size-7 text-emerald-600 dark:text-emerald-400" aria-hidden />
          <div>
            <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">RAG API status</h2>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              {health.isPending && "Checking..."}
              {health.isError && "Run `npm run dev` from repo root (starts rag-api on :3001)."}
              {health.isSuccess && (
                <>
                  {health.data.service} - {health.data.status}
                  {health.data.chat_model ? (
                    <>
                      {" "}
                      · LLM <span className="font-mono text-xs">{health.data.chat_model}</span>
                    </>
                  ) : null}
                </>
              )}
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition-all duration-300 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-5 rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/40">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Semantic Cache</h3>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            Toggle whether RAG chat requests read/write semantic cache records.
          </p>
          <label className="mt-3 inline-flex cursor-pointer items-center gap-3 text-sm text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={enableSemanticCache}
              onChange={(e) => onToggleSemanticCache(e.target.checked)}
              className="size-4 rounded border-zinc-300 text-emerald-600 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-900"
            />
            Enable semantic cache for chat requests
          </label>
          <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
            Current mode: {enableSemanticCache ? "Cache ON" : "Cache OFF (skip_cache=true)"}
          </p>
        </div>
        <div className="mb-5 rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/40">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Site Storage</h3>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            Clears persisted chat history and semantic cache.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => atomicClear.mutate()}
              disabled={atomicClear.isPending}
              className="inline-flex items-center gap-2 rounded-lg border border-zinc-900 bg-zinc-900 px-3 py-2 text-sm text-white transition-colors hover:bg-zinc-800 disabled:opacity-50 dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              {atomicClear.isPending ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <RefreshCw className="size-4" aria-hidden />
              )}
              Atomic Clear and Restart
            </button>
            <button
              type="button"
              onClick={() => clearSiteStorage.mutate()}
              disabled={clearSiteStorage.isPending}
              className="inline-flex items-center gap-2 rounded-lg border border-rose-300 px-3 py-2 text-sm text-rose-700 transition-colors hover:bg-rose-50 disabled:opacity-50 dark:border-rose-700 dark:text-rose-300 dark:hover:bg-rose-950/40"
            >
              {clearSiteStorage.isPending ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <RefreshCw className="size-4" aria-hidden />
              )}
              Clear Site Storage
            </button>
            <button
              type="button"
              onClick={() => purgeCacheOnly.mutate()}
              disabled={purgeCacheOnly.isPending}
              className="inline-flex items-center gap-2 rounded-lg border border-amber-300 px-3 py-2 text-sm text-amber-700 transition-colors hover:bg-amber-50 disabled:opacity-50 dark:border-amber-700 dark:text-amber-300 dark:hover:bg-amber-950/40"
            >
              {purgeCacheOnly.isPending ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <RefreshCw className="size-4" aria-hidden />
              )}
              Purge Cache
            </button>
            {clearSiteStorage.isError ? (
              <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">{getErrorMessage(clearSiteStorage.error)}</p>
            ) : null}
            {atomicClear.isError ? (
              <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">{getErrorMessage(atomicClear.error)}</p>
            ) : null}
            {purgeCacheOnly.isError ? (
              <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">{getErrorMessage(purgeCacheOnly.error)}</p>
            ) : null}
          </div>
        </div>
        <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">Employee Handbook Knowledge Base</h2>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Ingest all supported files in `fixtures/` (.md/.txt/.pdf) into one Chroma collection.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => bootstrap.mutate()}
            disabled={bootstrap.isPending}
            className="inline-flex items-center gap-2 rounded-lg bg-emerald-700 px-3 py-2 text-sm text-white transition-colors hover:bg-emerald-600 disabled:opacity-50"
          >
            {bootstrap.isPending ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <RefreshCw className="size-4" aria-hidden />}
            Load default fixture
          </button>
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-2">
          <label className="text-sm text-zinc-600 dark:text-zinc-400">
            Path
            <input
              className="ml-2 rounded border border-zinc-300 bg-white px-2 py-1 text-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
              value={handbookPath}
              onChange={(e) => setHandbookPath(e.target.value)}
            />
          </label>
          <button
            type="button"
            onClick={() => ingest.mutate({ handbook_path: handbookPath, replace: true })}
            disabled={ingest.isPending}
            className="inline-flex items-center gap-2 rounded-lg border border-zinc-300 px-3 py-2 text-sm text-zinc-800 transition-colors hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-600 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            {ingest.isPending ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Upload className="size-4" aria-hidden />}
            Re-index path
          </button>
        </div>
        {(bootstrap.error || ingest.error) ? (
          <p className="mt-3 text-sm text-rose-600 dark:text-rose-400">{getErrorMessage(bootstrap.error ?? ingest.error)}</p>
        ) : null}
      </section>

    </div>
  );
};
