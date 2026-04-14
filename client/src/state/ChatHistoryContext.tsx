import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type ChatHistoryItem = {
  role: "user" | "assistant";
  content: string;
};

type ChatHistoryContextValue = {
  chatHistory: ChatHistoryItem[];
  setChatHistory: (history: ChatHistoryItem[]) => void;
  selectedEmployeeId: string;
  setSelectedEmployeeId: (employeeId: string) => void;
  sessionQueryCount: number;
};

const ChatHistoryContext = createContext<ChatHistoryContextValue | null>(null);
const STORAGE_KEY = "handbook-chat-history-v1";
const MAX_AGE_MS = 24 * 60 * 60 * 1000;

type StoredPayload = {
  savedAt: number;
  history: ChatHistoryItem[];
  selectedEmployeeId?: string;
};

const isChatHistoryItem = (item: unknown): item is ChatHistoryItem => {
  if (!item || typeof item !== "object") return false;
  const role = (item as { role?: unknown }).role;
  const content = (item as { content?: unknown }).content;
  return (role === "user" || role === "assistant") && typeof content === "string";
};

const readStoredHistory = (): ChatHistoryItem[] => {
  try {
    const raw = globalThis.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as StoredPayload;
    if (!parsed || typeof parsed !== "object") return [];
    if (typeof parsed.savedAt !== "number" || Date.now() - parsed.savedAt > MAX_AGE_MS) {
      globalThis.localStorage.removeItem(STORAGE_KEY);
      return [];
    }
    const history = Array.isArray(parsed.history) ? parsed.history.filter(isChatHistoryItem) : [];
    return history;
  } catch {
    return [];
  }
};

export const ChatHistoryProvider = ({ children }: { children: ReactNode }) => {
  const initialHistory = readStoredHistory();
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>(initialHistory);
  const [selectedEmployeeId, setSelectedEmployeeId] = useState<string>(() => {
    try {
      const raw = globalThis.localStorage.getItem(STORAGE_KEY);
      if (!raw) return "E001";
      const parsed = JSON.parse(raw) as StoredPayload;
      const value = parsed?.selectedEmployeeId;
      return typeof value === "string" && value.trim() ? value : "E001";
    } catch {
      return "E001";
    }
  });

  useEffect(() => {
    try {
      const payload: StoredPayload = {
        savedAt: Date.now(),
        history: chatHistory,
        selectedEmployeeId,
      };
      globalThis.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Ignore storage errors; in-memory state still works.
    }
  }, [chatHistory, selectedEmployeeId]);

  const value = useMemo<ChatHistoryContextValue>(
    () => ({
      chatHistory,
      setChatHistory,
      selectedEmployeeId,
      setSelectedEmployeeId,
      sessionQueryCount: chatHistory.filter((item) => item.role === "user").length,
    }),
    [chatHistory, selectedEmployeeId],
  );

  return <ChatHistoryContext.Provider value={value}>{children}</ChatHistoryContext.Provider>;
};

export const useChatHistory = (): ChatHistoryContextValue => {
  const ctx = useContext(ChatHistoryContext);
  if (!ctx) {
    throw new Error("useChatHistory must be used within ChatHistoryProvider");
  }
  return ctx;
};
