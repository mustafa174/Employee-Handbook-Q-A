import type {
  AskRequest,
  AskResponse,
  PipelineStep,
} from "@employee-handbook/shared";
import { AlertTriangle, Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { useEffect, useMemo, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { apiUrl } from "../apiBase";
import { CachePanel } from "./CachePanel";
import { RAGPipelineVisualizer } from "./RAGPipelineVisualizer";
import { useTheme } from "../theme";
import { HarassmentReportModal } from "./HarassmentReportModal";

type AskRequestWithHistory = AskRequest & {
  chat_history?: ChatHistoryItem[];
};

type AskStreamEvent =
  | { type: "run_start"; run_id: string }
  | { type: "node_start"; run_id: string; node: string }
  | { type: "node_end"; run_id: string; node: string; status?: string }
  | { type: "text"; run_id: string; content: string }
  | { type: "done"; run_id: string; final: AskResponse }
  | { type: "error"; run_id: string; message: string };

const postAsk = async (body: AskRequestWithHistory): Promise<AskResponse> => {
  const res = await fetch(apiUrl("/api/ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: body.question,
      employee_id: body.employee_id || undefined,
      chat_history: body.chat_history,
      use_rag: body.use_rag ?? true,
      skip_cache: body.skip_cache ?? false,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    try {
      const j = JSON.parse(text) as { detail?: unknown };
      if (typeof j.detail === "string") throw new Error(j.detail);
    } catch (e) {
      if (e instanceof Error && !(e instanceof SyntaxError)) throw e;
    }
    throw new Error(text || "Ask failed");
  }
  return res.json() as Promise<AskResponse>;
};

const postAskStream = async (
  body: AskRequestWithHistory,
  handlers: {
    onNodeStart: (node: string) => void;
    onNodeEnd: (node: string, status?: string) => void;
    onText: (chunk: string) => void;
  },
): Promise<AskResponse> => {
  const res = await fetch(apiUrl("/api/ask/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: body.question,
      employee_id: body.employee_id || undefined,
      chat_history: body.chat_history,
      use_rag: body.use_rag ?? true,
      skip_cache: body.skip_cache ?? false,
    }),
  });
  if (!res.ok || !res.body) {
    const text = await res.text();
    throw new Error(text || "Ask stream failed");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: AskResponse | null = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx = buffer.indexOf("\n\n");
    while (idx >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const dataLines = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim());
      const payloadText = dataLines.join("");
      if (!payloadText) {
        idx = buffer.indexOf("\n\n");
        continue;
      }
      let event: AskStreamEvent;
      try {
        event = JSON.parse(payloadText) as AskStreamEvent;
      } catch {
        idx = buffer.indexOf("\n\n");
        continue;
      }
      if (event.type === "node_start") handlers.onNodeStart(event.node);
      if (event.type === "node_end") handlers.onNodeEnd(event.node, event.status);
      if (event.type === "text") handlers.onText(event.content);
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "done") finalResponse = event.final;
      idx = buffer.indexOf("\n\n");
    }
  }
  if (!finalResponse) throw new Error("Stream completed without final payload");
  return finalResponse;
};

type ChatTurn = {
  id: string;
  question: string;
  questionAt: string;
  employeeId?: string;
  employeeName?: string;
  answer?: string;
  answerAt?: string;
  error?: string;
  agentAction?: NonNullable<AskResponse["agent_action"]>;
};

type ChatHistoryItem = {
  role: "user" | "assistant";
  content: string;
};

type HandbookQAProps = {
  chatHistory?: ChatHistoryItem[];
  onChatHistoryChange?: (history: ChatHistoryItem[]) => void;
  selectedEmployeeId: string;
  selectedEmployeeName?: string;
};

type HarassmentReportPayload = NonNullable<AskResponse["agent_action"]>["payload"];

type TokenVector = {
  token: string;
  values: number[];
};

type BarPoint = {
  name: string;
  score: number;
  chunkIndex: number;
  attempt: number;
};

type RetrievalAttemptView = {
  attempt: number;
  query: string;
  top_score: number;
  verdict: "answerable" | "re-search";
  reason?: string;
  citations: AskResponse["citations"];
};

const tokenize = (text: string): string[] => {
  const all = text.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];
  return all.filter((t) => t.length >= 2).slice(0, 12);
};

const hashWord = (word: string): number => {
  let h = 0;
  for (let i = 0; i < word.length; i += 1) {
    h = (h * 31 + word.charCodeAt(i)) >>> 0;
  }
  return h;
};

const tokenVectors = (query: string): TokenVector[] =>
  tokenize(query).map((token) => {
    const h = hashWord(token);
    return {
      token,
      values: [
        ((h >>> 0) & 255) / 255,
        ((h >>> 8) & 255) / 255,
        ((h >>> 16) & 255) / 255,
        ((h >>> 24) & 255) / 255,
      ],
    };
  });

const mcpStatusMeta = (step: PipelineStep | undefined) => {
  if (!step) return { label: "Unknown", dot: "bg-zinc-400" };
  if (step.detail?.startsWith("✅ Intent:")) return { label: step.detail, dot: "bg-emerald-500" };
  if (step.status === "ok") return { label: "Fetched", dot: "bg-emerald-500" };
  if (step.status === "skipped") return { label: "Skipped", dot: "bg-zinc-400" };
  if (step.status === "triggered") return { label: "Triggered", dot: "bg-amber-500" };
  return { label: "Empty", dot: "bg-amber-500" };
};

const pipelineStepTone = (
  status: PipelineStep["status"],
): "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-200"
  | "border-zinc-300 bg-zinc-100 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
  | "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200" => {
  if (status === "ok") {
    return "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-200";
  }
  if (status === "skipped") {
    return "border-zinc-300 bg-zinc-100 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200";
  }
  return "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200";
};

const escapeRegExp = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const chunkTerms = (chunkText: string): Set<string> => {
  const words = chunkText.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];
  return new Set(words.filter((w) => w.length >= 4).slice(0, 80));
};

const highlightAnswerByChunk = (answer: string, chunkText: string): ReactNode => {
  const terms = [...chunkTerms(chunkText)];
  if (terms.length === 0) return answer;
  const re = new RegExp(`\\b(${terms.map(escapeRegExp).join("|")})\\b`, "giu");
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  for (;;) {
    m = re.exec(answer);
    if (m === null) break;
    if (m.index > last) out.push(answer.slice(last, m.index));
    out.push(
      <mark key={`attn-${key++}`} className="rounded-sm bg-amber-200/85 px-0.5 text-inherit dark:bg-amber-500/35">
        {m[0]}
      </mark>,
    );
    last = m.index + m[0].length;
  }
  if (last < answer.length) out.push(answer.slice(last));
  return out.length === 0 ? answer : out;
};

const splitSourceLine = (answer: string): { body: string; sourceLine: string | null } => {
  const lines = answer.split("\n");
  let i = lines.length - 1;
  while (i >= 0 && lines[i].trim() === "") i -= 1;
  if (i >= 0 && lines[i].trim().toLowerCase().startsWith("source:")) {
    const sourceLine = lines[i].trim();
    const body = lines.slice(0, i).join("\n").trimEnd();
    return { body, sourceLine };
  }
  return { body: answer, sourceLine: null };
};

const renderAnswerMarkdown = (answer: string): ReactNode => {
  const { body, sourceLine } = splitSourceLine(answer);
  return (
    <div className="text-sm text-zinc-800 dark:text-zinc-200">
      <ReactMarkdown
        components={{
          p: ({ children }) => <p className="my-1 leading-6">{children}</p>,
          ul: ({ children }) => <ul className="my-1 list-disc pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="my-1 list-decimal pl-5">{children}</ol>,
          li: ({ children }) => <li className="my-0.5">{children}</li>,
          h1: ({ children }) => <h1 className="my-1 text-base font-semibold">{children}</h1>,
          h2: ({ children }) => <h2 className="my-1 text-sm font-semibold">{children}</h2>,
          h3: ({ children }) => <h3 className="my-1 text-sm font-medium">{children}</h3>,
          pre: ({ children }) => (
            <pre className="my-1 overflow-x-auto rounded bg-zinc-100 px-2 py-1 dark:bg-zinc-900">{children}</pre>
          ),
        }}
      >
        {body}
      </ReactMarkdown>
      {sourceLine ? <p className="mt-2 text-xs italic opacity-60">{sourceLine}</p> : null}
    </div>
  );
};

const buildHistoryFromTurns = (turns: ChatTurn[]): ChatHistoryItem[] => {
  const history: ChatHistoryItem[] = [];
  for (const turn of turns) {
    if (turn.question.trim()) {
      history.push({ role: "user", content: turn.question.trim() });
    }
    if (turn.answer && !turn.error && turn.answer.trim()) {
      history.push({ role: "assistant", content: turn.answer.trim() });
    }
  }
  return history;
};

const rebuildTurnsFromHistory = (history: ChatHistoryItem[]): ChatTurn[] => {
  const turns: ChatTurn[] = [];
  let pendingUser: ChatHistoryItem | null = null;
  for (const item of history) {
    if (item.role === "user") {
      pendingUser = item;
      continue;
    }
    if (!pendingUser) continue;
    const id =
      typeof globalThis.crypto !== "undefined" && typeof globalThis.crypto.randomUUID === "function"
        ? globalThis.crypto.randomUUID()
        : `turn-${Date.now()}-${turns.length}`;
    turns.push({
      id,
      question: pendingUser.content,
      questionAt: new Date().toISOString(),
      answer: item.content,
      answerAt: new Date().toISOString(),
    });
    pendingUser = null;
  }
  return turns;
};

const limitToWords = (text: string, maxWords: number): string => {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length <= maxWords) return text;
  return words.slice(0, maxWords).join(" ");
};

const latestCompletedTurnKey = (turns: ChatTurn[]): string | null => {
  const last = turns.at(-1);
  if (!last) return null;
  if (!last.answerAt && !last.error) return null;
  return `${last.id}::${last.answerAt ?? ""}::${last.error ?? ""}`;
};

const NAIVE_TURNS_STORAGE_KEY = "handbook-naive-turns-v1";
const NAIVE_TURNS_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const CLEAR_SITE_STORAGE_EVENT = "handbook:clear-site-storage";
const CACHE_PREF_STORAGE_KEY = "handbook-enable-semantic-cache-v1";
const CACHE_PREF_EVENT = "handbook:cache-preference-changed";

type StoredNaiveTurns = {
  savedAt: number;
  turns: ChatTurn[];
};

const isChatTurn = (value: unknown): value is ChatTurn => {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return typeof v.id === "string" && typeof v.question === "string" && typeof v.questionAt === "string";
};

const readStoredNaiveTurns = (): ChatTurn[] => {
  try {
    const raw = globalThis.localStorage.getItem(NAIVE_TURNS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as StoredNaiveTurns;
    if (!parsed || typeof parsed !== "object") return [];
    if (typeof parsed.savedAt !== "number" || Date.now() - parsed.savedAt > NAIVE_TURNS_MAX_AGE_MS) {
      globalThis.localStorage.removeItem(NAIVE_TURNS_STORAGE_KEY);
      return [];
    }
    if (!Array.isArray(parsed.turns)) return [];
    return parsed.turns.filter(isChatTurn);
  } catch {
    return [];
  }
};

const scrollPaneToBottom = (paneRef: React.RefObject<HTMLDivElement | null>) => {
  const el = paneRef.current;
  if (!el) return;
  el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
};

export const HandbookQA = ({
  chatHistory,
  onChatHistoryChange,
  selectedEmployeeId,
  selectedEmployeeName,
}: HandbookQAProps) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const [question, setQuestion] = useState("How far in advance should I request PTO?");
  const [lastAnswer, setLastAnswer] = useState<AskResponse | null>(null);
  const [lastQuery, setLastQuery] = useState("");
  const [naiveTurns, setNaiveTurns] = useState<ChatTurn[]>(readStoredNaiveTurns);
  const [ragTurns, setRagTurns] = useState<ChatTurn[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [enableSemanticCache, setEnableSemanticCache] = useState<boolean>(() => {
    try {
      const raw = globalThis.localStorage.getItem(CACHE_PREF_STORAGE_KEY);
      if (raw === null) return true;
      return raw !== "false";
    } catch {
      return true;
    }
  });
  const [hoveredChunkIndex, setHoveredChunkIndex] = useState<number | null>(null);
  const [lastRagTurnId, setLastRagTurnId] = useState<string | null>(null);
  const [activeNodes, setActiveNodes] = useState<string[]>([]);
  const [doneNodes, setDoneNodes] = useState<string[]>([]);
  const [reportModalPayload, setReportModalPayload] = useState<HarassmentReportPayload | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const naivePaneRef = useRef<HTMLDivElement | null>(null);
  const ragPaneRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!chatHistory || chatHistory.length === 0 || ragTurns.length > 0) return;
    setRagTurns(rebuildTurnsFromHistory(chatHistory));
  }, [chatHistory, ragTurns.length]);

  useEffect(() => {
    try {
      const payload: StoredNaiveTurns = {
        savedAt: Date.now(),
        turns: naiveTurns,
      };
      globalThis.localStorage.setItem(NAIVE_TURNS_STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Ignore storage write failures.
    }
  }, [naiveTurns]);

  useEffect(() => {
    const onClearSiteStorage = () => {
      setNaiveTurns([]);
      setRagTurns([]);
      setLastAnswer(null);
      setLastRagTurnId(null);
      setSendError(null);
      setLastQuery("");
      setQuestion("");
      setActiveNodes([]);
      setDoneNodes([]);
    };
    globalThis.addEventListener(CLEAR_SITE_STORAGE_EVENT, onClearSiteStorage);
    return () => {
      globalThis.removeEventListener(CLEAR_SITE_STORAGE_EVENT, onClearSiteStorage);
    };
  }, []);

  useEffect(() => {
    const onCachePreferenceChanged = () => {
      try {
        const raw = globalThis.localStorage.getItem(CACHE_PREF_STORAGE_KEY);
        setEnableSemanticCache(raw !== "false");
      } catch {
        setEnableSemanticCache(true);
      }
    };
    globalThis.addEventListener(CACHE_PREF_EVENT, onCachePreferenceChanged);
    return () => {
      globalThis.removeEventListener(CACHE_PREF_EVENT, onCachePreferenceChanged);
    };
  }, []);

  const runAsk = () => {
    const q = question.trim();
    const contextValue = selectedEmployeeId.trim() || undefined;
    const selectedEmployeeNameForTurn = selectedEmployeeName?.trim() || undefined;
    if (!q || isSending) return;
    // #region agent log
    fetch("http://127.0.0.1:7340/ingest/caf36bdf-b3aa-457a-8a14-e9c51eb4cc1d", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "0b08b0" },
      body: JSON.stringify({
        sessionId: "0b08b0",
        runId: "pre-fix",
        hypothesisId: "H2",
        location: "HandbookQA.tsx:runAsk",
        message: "Assistant runAsk invoked",
        data: { question: q, selectedEmployeeId: contextValue ?? null },
        timestamp: Date.now(),
      }),
    }).catch(() => {});
    // #endregion
    setQuestion("");
    const outgoingHistory = chatHistory ?? buildHistoryFromTurns(ragTurns);

    const id =
      typeof globalThis.crypto !== "undefined" && typeof globalThis.crypto.randomUUID === "function"
        ? globalThis.crypto.randomUUID()
        : `turn-${Date.now()}`;

    setSendError(null);
    setIsSending(true);
    setActiveNodes([]);
    setDoneNodes([]);
    setLastQuery(q);
    const questionAt = new Date().toISOString();
    setNaiveTurns((prev) => [
      ...prev,
      { id, question: q, questionAt, employeeId: contextValue, employeeName: selectedEmployeeNameForTurn },
    ]);
    setRagTurns((prev) => [
      ...prev,
      { id, question: q, questionAt, employeeId: contextValue, employeeName: selectedEmployeeNameForTurn },
    ]);
    requestAnimationFrame(() => {
      scrollPaneToBottom(naivePaneRef);
      scrollPaneToBottom(ragPaneRef);
    });

    const base = {
      question: q,
      employee_id: contextValue,
      chat_history: outgoingHistory,
      skip_cache: !enableSemanticCache,
    } satisfies AskRequestWithHistory;

    void (async () => {
      const [naiveSettled, ragSettled] = await Promise.allSettled([
        postAsk({ ...base, use_rag: false }),
        postAskStream(
          { ...base, use_rag: true },
          {
            onNodeStart: (node) => {
              setActiveNodes((prev) => (prev.includes(node) ? prev : [...prev, node]));
            },
            onNodeEnd: (node, status) => {
              setActiveNodes((prev) => prev.filter((id2) => id2 !== node));
              if (status !== "skipped") {
                setDoneNodes((prev) => (prev.includes(node) ? prev : [...prev, node]));
              }
            },
            onText: (chunk) => {
              setRagTurns((prev) =>
                prev.map((t) => (t.id === id ? { ...t, answer: `${t.answer ?? ""}${chunk}` } : t)),
              );
              requestAnimationFrame(() => {
                scrollPaneToBottom(ragPaneRef);
              });
            },
          },
        ),
      ]);

      setNaiveTurns((prev) =>
        prev.map((t) => {
          if (t.id !== id) return t;
          if (naiveSettled.status === "fulfilled") {
            return { ...t, answer: naiveSettled.value.answer, answerAt: new Date().toISOString() };
          }
          return {
            ...t,
            answerAt: new Date().toISOString(),
            error:
              naiveSettled.reason instanceof Error
                ? naiveSettled.reason.message
                : "Standard chatbot request failed",
          };
        }),
      );

      setRagTurns((prev) =>
        prev.map((t) => {
          if (t.id !== id) return t;
          if (ragSettled.status === "fulfilled") {
            const action = ragSettled.value.agent_action;
            const mergedAction =
              action?.type === "HARASSMENT_REPORT"
                ? {
                    ...action,
                    payload: {
                      ...action.payload,
                      employee_id: contextValue ?? action.payload.employee_id,
                      employee_name: selectedEmployeeNameForTurn ?? action.payload.employee_name,
                    },
                  }
                : action;
            return {
              ...t,
              answer: ragSettled.value.answer,
              answerAt: new Date().toISOString(),
              agentAction: mergedAction ?? undefined,
            };
          }
          return {
            ...t,
            answerAt: new Date().toISOString(),
            error: ragSettled.reason instanceof Error ? ragSettled.reason.message : "RAG request failed",
          };
        }),
      );

      if (ragSettled.status === "fulfilled") {
        setLastAnswer(ragSettled.value);
        setLastRagTurnId(id);
        setActiveNodes([]);
        const nextHistory: ChatHistoryItem[] = [
          ...outgoingHistory,
          { role: "user", content: q },
          { role: "assistant", content: ragSettled.value.answer },
        ];
        onChatHistoryChange?.(nextHistory);
      } else {
        setSendError(ragSettled.reason instanceof Error ? ragSettled.reason.message : "RAG request failed");
        setActiveNodes([]);
      }
      requestAnimationFrame(() => {
        scrollPaneToBottom(naivePaneRef);
        scrollPaneToBottom(ragPaneRef);
      });
      setIsSending(false);
    })();
  };

  const vectors = useMemo(() => tokenVectors(lastQuery), [lastQuery]);
  const barData: BarPoint[] = useMemo(() => {
    if (!lastAnswer) return [];
    const attempts = (lastAnswer as AskResponse & { retrieval_attempts?: RetrievalAttemptView[] })
      .retrieval_attempts ?? [];
    if (attempts.length > 0) {
      const latestAttempt = attempts[attempts.length - 1]?.attempt;
      return attempts.flatMap((attemptData: RetrievalAttemptView) =>
        attemptData.citations.map((c, i: number) => ({
          score: c.score,
          chunkIndex: attemptData.attempt === latestAttempt ? i : -1,
          attempt: attemptData.attempt,
          name:
            "source" in c && typeof c.source === "string" && c.source.trim().length > 0
              ? `A${attemptData.attempt} · ${c.source} #${i + 1}`
              : `A${attemptData.attempt} · Chunk ${i + 1}`,
        })),
      );
    }
    return (
      lastAnswer.citations.map((c, i) => ({
        score: c.score,
        chunkIndex: i,
        attempt: 1,
        name:
          "source" in c && typeof c.source === "string" && c.source.trim().length > 0
            ? `${c.source} · #${i + 1}`
            : `Chunk ${i + 1}`,
      })) ?? []
    );
  }, [lastAnswer]);
  const mcpStep = lastAnswer?.pipeline_steps.find((s) => s.id === "mcp_hr");
  const retrieveStep = lastAnswer?.pipeline_steps.find((s) => s.id === "retrieve");
  const mcpMeta = mcpStatusMeta(mcpStep);
  const llmNodeLabel = useMemo(
    () => `OpenAI\n${lastAnswer?.chat_model ?? "gpt-4o-mini"}`,
    [lastAnswer?.chat_model],
  );
  const pipelineComplete = !isSending && (lastAnswer !== null || sendError !== null);
  const chartGrid = isDark ? "#3f3f46" : "#d4d4d8";
  const chartTick = isDark ? "#a1a1aa" : "#52525b";
  const chartAxisStroke = isDark ? "#52525b" : "#a1a1aa";
  const hoveredChunkText =
    hoveredChunkIndex !== null ? (lastAnswer?.citations[hoveredChunkIndex]?.text ?? "") : "";
  const hiddenPrompt = useMemo(() => {
    const history = (chatHistory ?? buildHistoryFromTurns(ragTurns))
      .slice(-10)
      .map((h) => `${h.role === "user" ? "User" : "Assistant"}: ${h.content}`)
      .join("\n");
    const context = (lastAnswer?.citations ?? [])
      .map((c, i) => `[Chunk ${i + 1}] ${c.text}`)
      .join("\n\n---\n\n");
    const mcp = mcpStep?.detail ? `\n\nMCP context:\n${mcpStep.detail}` : "";
    return `Conversation history:\n${history || "<none>"}\n\nCurrent question:\n${lastQuery || "<none>"}\n\nHandbook excerpts:\n${context || "<none>"}${mcp}`;
  }, [chatHistory, lastAnswer?.citations, lastQuery, mcpStep?.detail, ragTurns]);
  const shouldWarnNoPolicySource = useMemo(() => {
    if (!lastAnswer) return false;
    if ((lastAnswer.citations?.length ?? 0) > 0) return false;
    if (!lastAnswer.use_rag) return false;
    // Show "no policy source" only when retrieval actually attempted policy grounding.
    return retrieveStep?.status === "empty";
  }, [lastAnswer, retrieveStep?.status]);

  useEffect(() => {
    if (!lastAnswer) return;
    // #region agent log
    fetch("http://127.0.0.1:7340/ingest/caf36bdf-b3aa-457a-8a14-e9c51eb4cc1d", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "0b08b0" },
      body: JSON.stringify({
        sessionId: "0b08b0",
        runId: "repro-warning-1",
        hypothesisId: "H1",
        location: "HandbookQA.tsx:source-warning",
        message: "No-policy warning decision snapshot",
        data: {
          citations: lastAnswer.citations?.length ?? 0,
          use_rag: Boolean(lastAnswer.use_rag),
          retrieve_status: retrieveStep?.status ?? null,
          should_warn: shouldWarnNoPolicySource,
          last_query: lastQuery,
        },
        timestamp: Date.now(),
      }),
    }).catch(() => {});
    // #endregion
  }, [lastAnswer, lastQuery, retrieveStep?.status, shouldWarnNoPolicySource]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.max(el.scrollHeight, 52)}px`;
  }, [question]);

  const latestNaiveCompletedKey = useMemo(() => latestCompletedTurnKey(naiveTurns), [naiveTurns]);
  const latestRagCompletedKey = useMemo(() => latestCompletedTurnKey(ragTurns), [ragTurns]);

  useEffect(() => {
    if (!latestNaiveCompletedKey) return;
    scrollPaneToBottom(naivePaneRef);
  }, [latestNaiveCompletedKey]);

  useEffect(() => {
    if (!latestRagCompletedKey) return;
    scrollPaneToBottom(ragPaneRef);
  }, [latestRagCompletedKey]);

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/50 lg:p-6">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:gap-6">
          <section className="flex min-h-0 flex-col rounded-xl border border-zinc-200 bg-white/80 p-5 dark:border-zinc-800 dark:bg-zinc-900/40 lg:h-[36rem] lg:max-h-[36rem]">
          <h2 className="mb-2 text-base font-medium text-zinc-900 dark:text-zinc-100">Naive Chatbot Pane</h2>
          <p className="mb-4 text-xs text-zinc-500 dark:text-zinc-500">Standard LLM without retrieval.</p>
          <div ref={naivePaneRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
            {naiveTurns.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-500">No messages yet.</p>
            ) : (
              naiveTurns.map((turn) => (
                <article key={turn.id} className="rounded-lg bg-sky-50/80 p-3 shadow-sm shadow-sky-200/55 dark:bg-sky-950/20 dark:shadow-sky-950/30">
                  <p className="text-xs uppercase text-zinc-500 dark:text-zinc-500">Q</p>
                  <p className="mb-2 whitespace-pre-wrap text-sm text-zinc-900 dark:text-zinc-100">{turn.question}</p>
                  {/* <p className="mb-2 text-right text-[11px] text-zinc-500 dark:text-zinc-500">
                    {formatTurnTime(turn.questionAt)}
                  </p> */}
                  {turn.answer || turn.error ? (
                    <p className="text-xs uppercase text-zinc-500 dark:text-zinc-500">A</p>
                  ) : null}
                  {turn.error ? (
                    <p className="whitespace-pre-wrap text-sm text-zinc-800 dark:text-zinc-200">{turn.error}</p>
                  ) : turn.answer ? (
                    renderAnswerMarkdown(turn.answer)
                  ) : (
                    null
                  )}
                  {/* {turn.answerAt ? (
                    <p className="mt-2 text-right text-[11px] text-zinc-500 dark:text-zinc-500">
                      {formatTurnTime(turn.answerAt)}
                    </p>
                  ) : null} */}
                </article>
              ))
            )}
            {isSending ? (
              <article className="rounded-lg border border-sky-200 bg-sky-50/80 p-3 shadow-sm shadow-sky-200/40 dark:border-sky-900/60 dark:bg-sky-950/20 dark:shadow-sky-950/30">
                <p className="text-sm text-sky-700 dark:text-sky-300">Generating...</p>
              </article>
            ) : null}
          </div>
          </section>

          <section className="flex min-h-0 flex-col rounded-xl border border-zinc-200 bg-white/80 p-5 dark:border-zinc-800 dark:bg-zinc-900/40 lg:h-[36rem] lg:max-h-[36rem]">
            <div className="mb-2 flex items-center gap-2">
              <h2 className="text-base font-medium text-zinc-900 dark:text-zinc-100">RAG Chatbot Pane</h2>
            </div>
          <p className="mb-4 text-xs text-zinc-500 dark:text-zinc-500">Grounded LLM with handbook retrieval.</p>
          <div ref={ragPaneRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
            {ragTurns.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-500">No messages yet.</p>
            ) : (
              ragTurns.map((turn) => (
                <article key={turn.id} className="rounded-lg bg-sky-50/80 p-3 shadow-sm shadow-sky-200/55 dark:bg-sky-950/20 dark:shadow-sky-950/30">
                  <p className="text-xs uppercase text-zinc-500 dark:text-zinc-500">Q</p>
                  <p className="mb-2 whitespace-pre-wrap text-sm text-zinc-900 dark:text-zinc-100">{turn.question}</p>
                  {/* <p className="mb-2 text-right text-[11px] text-zinc-500 dark:text-zinc-500">
                    {formatTurnTime(turn.questionAt)}
                  </p> */}
                  {turn.answer || turn.error ? (
                    <p className="text-xs uppercase text-zinc-500 dark:text-zinc-500">A</p>
                  ) : null}
                  {turn.error ? (
                    <p className="whitespace-pre-wrap text-sm text-zinc-800 dark:text-zinc-200">{turn.error}</p>
                  ) : turn.answer ? (
                    turn.id === lastRagTurnId && hoveredChunkText ? (
                      <p className="whitespace-pre-wrap text-sm text-zinc-800 dark:text-zinc-200">
                        {highlightAnswerByChunk(turn.answer, hoveredChunkText)}
                      </p>
                    ) : (
                      renderAnswerMarkdown(turn.answer)
                    )
                  ) : (
                    null
                  )}
                  {/* {turn.answerAt ? (
                    <p className="mt-2 text-right text-[11px] text-zinc-500 dark:text-zinc-500">
                      {formatTurnTime(turn.answerAt)}
                    </p>
                  ) : null} */}
                  {turn.id === lastRagTurnId && shouldWarnNoPolicySource ? (
                    <p className="mt-2 inline-block rounded border border-amber-300 bg-amber-50 px-2 py-1 text-xs text-amber-800 dark:border-amber-700 dark:bg-amber-950/50 dark:text-amber-200">
                      ⚠️ No policy source found
                    </p>
                  ) : null}
                  {turn.agentAction?.type === "HARASSMENT_REPORT" ? (
                    <button
                      type="button"
                      onClick={() =>
                        setReportModalPayload({
                          ...turn.agentAction.payload,
                          employee_id: turn.employeeId ?? turn.agentAction.payload.employee_id,
                          employee_name: turn.employeeName ?? turn.agentAction.payload.employee_name,
                        })
                      }
                      className="mt-3 rounded-lg border border-sky-300 bg-sky-50 px-3 py-1.5 text-xs font-medium text-sky-700 hover:bg-sky-100 dark:border-sky-700 dark:bg-sky-950/40 dark:text-sky-200 dark:hover:bg-sky-900/50"
                    >
                      Prepare HR Report
                    </button>
                  ) : null}
                </article>
              ))
            )}
            {isSending ? (
              <article className="rounded-lg border border-sky-200 bg-sky-50/80 p-3 shadow-sm shadow-sky-200/40 dark:border-sky-900/60 dark:bg-sky-950/20 dark:shadow-sky-950/30">
                <p className="text-sm text-sky-700 dark:text-sky-300">Retrieving...</p>
              </article>
            ) : null}
          </div>
          </section>
        </div>

        <section className="mt-4 rounded-xl  border-zinc-200 bg-white/80 p-5 dark:border-zinc-800 dark:bg-zinc-900/40">
          <div className="flex min-h-0 flex-1 flex-col gap-4">
            <label className="relative block">
              <textarea
                ref={inputRef}
                className="mt-2 min-h-[3.35rem] w-full resize-none overflow-hidden rounded-[1.75rem] border
                 border-zinc-300 bg-zinc-50 px-5 py-3 pr-16 text-sm leading-6
                  text-zinc-900 shadow-sm outline-none transition-all focus:border-sky-400
                   focus:bg-white focus:ring-2 focus:ring-sky-400/30 
                   dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:focus:border-sky-500 dark:focus:bg-zinc-950"
                rows={1}
                value={question}
                onChange={(e) => setQuestion(limitToWords(e.target.value, 200))}
                onKeyDown={(e) => {
                  if (e.key !== "Enter") return;
                  if (e.shiftKey) return;
                  e.preventDefault();
                  runAsk();
                }}
                placeholder="Ask anything from the handbook... Press Enter to send, Shift+Enter for new line."
              />
              <button
                type="button"
                onClick={runAsk}
                disabled={isSending || !question.trim()}
                className="absolute right-3 top-1/2 inline-flex size-10 -translate-y-1/2 mt-0.5 items-center justify-center rounded-full bg-sky-600 text-white shadow-sm transition-all hover:scale-[1.03] hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
                aria-label="Send message"
              >
                {isSending ? (
                  <Loader2 className="size-5 animate-spin" aria-hidden />
                ) : (
                  <svg viewBox="0 0 24 24" className="size-5" fill="none" aria-hidden>
                    <path d="M4 12L20 4L13 20L11 13L4 12Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
                  </svg>
                )}
              </button>
            </label>
            {sendError ? <p className="text-sm text-red-600 dark:text-red-400">{sendError}</p> : null}
          </div>
        </section>
      </div>

      <section className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/50">
        <RAGPipelineVisualizer
          isLoading={isSending}
          pipelineComplete={pipelineComplete}
          isDark={isDark}
          llmNodeLabel={llmNodeLabel}
          activeNodes={activeNodes}
          doneNodes={doneNodes}
        />
        {lastAnswer?.pipeline_steps && lastAnswer.pipeline_steps.length > 0 ? (
          <div className="mt-3 flex flex-wrap gap-2">
            {lastAnswer.pipeline_steps.map((step) => (
              <div key={step.id} className={["rounded-full border px-2 py-1 text-xs font-medium", pipelineStepTone(step.status)].join(" ")}>
                {step.label}
              </div>
            ))}
          </div>
        ) : null}
      </section>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:gap-6">
        <section className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/50 lg:min-h-[24rem]">
            <h3 className="mb-2 text-sm font-medium text-zinc-900 dark:text-zinc-100">Card A: Token &amp; Vector Explorer</h3>
            <div className="max-h-[18rem] space-y-2 overflow-y-auto">
              {vectors.length === 0 ? (
                <p className="text-sm text-zinc-500 dark:text-zinc-500">No query sent yet.</p>
              ) : (
                vectors.map((row) => (
                  <div key={row.token} className="rounded-md border border-zinc-200 p-2 dark:border-zinc-700">
                    <div className="mb-1 text-xs font-medium text-zinc-700 dark:text-zinc-300">{row.token}</div>
                    <div className="grid grid-cols-4 gap-1">
                      {row.values.map((v, i) => (
                        <div key={`${row.token}-${i}`} className="rounded px-1 py-0.5 text-[10px] text-zinc-800 dark:text-zinc-100" title={`Dimension ${i + 1} of 768`} style={{ backgroundColor: isDark ? `rgba(14,165,233,${0.18 + v * 0.5})` : `rgba(14,165,233,${0.08 + v * 0.4})` }}>
                          {v.toFixed(2)}
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

        <section className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/50 lg:min-h-[24rem]">
            <h3 className="mb-2 text-sm font-medium text-zinc-900 dark:text-zinc-100">Card B: Retrieval Map</h3>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={barData}
                  margin={{ top: 8, right: 8, left: 0, bottom: 8 }}
                  onMouseMove={(state) => {
                    const idx = state.activeTooltipIndex;
                    if (typeof idx === "number" && barData[idx]) {
                      setHoveredChunkIndex(barData[idx].chunkIndex >= 0 ? barData[idx].chunkIndex : null);
                    }
                  }}
                  onMouseLeave={() => setHoveredChunkIndex(null)}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={chartGrid} />
                  <XAxis dataKey="name" tick={{ fill: chartTick, fontSize: 11 }} stroke={chartAxisStroke} />
                  <YAxis tick={{ fill: chartTick, fontSize: 11 }} stroke={chartAxisStroke} domain={[0, "auto"]} />
                  <Tooltip contentStyle={{ backgroundColor: isDark ? "#18181b" : "#ffffff", border: `1px solid ${isDark ? "#3f3f46" : "#e4e4e7"}`, borderRadius: "8px", color: isDark ? "#f4f4f5" : "#18181b" }} />
                  <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                    {barData.map((row) => (
                      <Cell
                        key={row.name}
                        fill={hoveredChunkIndex === row.chunkIndex && row.chunkIndex >= 0 ? "#f59e0b" : row.attempt === 1 ? "#0ea5e9" : "#22c55e"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

      </div>

      <section className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/50 lg:min-h-[26rem]">
        <h3 className="mb-2 text-sm font-medium text-zinc-900 dark:text-zinc-100">Card C: MCP Tool Status</h3>
        <div className="flex items-center gap-2">
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${mcpMeta.dot}`} />
          <span className="text-sm text-zinc-700 dark:text-zinc-300">{mcpMeta.label}</span>
        </div>
        <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-500">{mcpStep?.detail ?? "No run yet."}</p>
        <div className="mt-3 rounded-md border border-zinc-200 bg-zinc-50 p-2 dark:border-zinc-700 dark:bg-zinc-950/60">
          <p className="mb-2 text-xs font-medium text-zinc-700 dark:text-zinc-300">Hidden Prompt (Augmentation)</p>
          <textarea
            value={hiddenPrompt}
            disabled
            readOnly
            className="h-64 w-full resize-y rounded border border-zinc-300 bg-zinc-100/80 p-2 font-mono text-[11px] text-zinc-600 outline-none dark:border-zinc-700 dark:bg-zinc-900/80 dark:text-zinc-400"
          />
        </div>
        {lastAnswer?.isEscalated ? (
          <div className="mt-3 rounded-md border border-red-300 bg-red-50 p-2 text-xs text-red-800 dark:border-red-800 dark:bg-red-950/70 dark:text-red-200">
            <div className="flex items-start gap-2">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              <span>Escalated to HR flow for safety-sensitive topic.</span>
            </div>
          </div>
        ) : null}
      </section>

      <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/50">
        <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">Vector Space Visualizer</h2>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          2D semantic projection of cached queries and indexed chunks.
        </p>
      </section>

      <CachePanel fallbackCount={(chatHistory ?? []).filter((item) => item.role === "user").length} />
      <HarassmentReportModal payload={reportModalPayload} onClose={() => setReportModalPayload(null)} />
    </div>
  );
};
