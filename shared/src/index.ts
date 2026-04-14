import { z } from "zod";

export const citationSchema = z.object({
  text: z.string(),
  score: z.number(),
  source: z.string().optional(),
  section_title: z.string().optional(),
});

export type Citation = z.infer<typeof citationSchema>;

export const pipelineStepSchema = z.object({
  id: z.string(),
  label: z.string(),
  status: z.enum(["ok", "skipped", "empty", "triggered"]),
  detail: z.string().optional(),
});

export type PipelineStep = z.infer<typeof pipelineStepSchema>;

export const retrievalAttemptSchema = z.object({
  attempt: z.number().int().min(1),
  query: z.string(),
  top_score: z.number(),
  verdict: z.enum(["answerable", "re-search"]),
  reason: z.string().optional(),
  citations: z.array(citationSchema),
});

export type RetrievalAttempt = z.infer<typeof retrievalAttemptSchema>;

export const askRequestSchema = z.object({
  question: z.string().min(1),
  employee_id: z.string().optional(),
  chat_history: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().min(1),
      }),
    )
    .optional(),
  /** When false, skip Chroma retrieval and HR context — pure LLM for comparison UI. Default true on server. */
  use_rag: z.boolean().optional(),
});

export type AskRequest = z.infer<typeof askRequestSchema>;

export const askResponseSchema = z.object({
  answer: z.string(),
  citations: z.array(citationSchema),
  retrieval_attempts: z.array(retrievalAttemptSchema).optional(),
  isEscalated: z.boolean(),
  escalation_reason: z.string().nullable().optional(),
  pipeline_steps: z.array(pipelineStepSchema),
  use_rag: z.boolean(),
  /** OpenAI chat model id from rag-api (`OPENAI_CHAT_MODEL`). */
  chat_model: z.string(),
  cache_hit: z.boolean().optional(),
  cache_reason: z.enum(["hit", "miss", "kb_changed"]).optional(),
  cache_kb_signature: z.string().optional(),
});

export type AskResponse = z.infer<typeof askResponseSchema>;

export const ingestPathRequestSchema = z.object({
  handbook_path: z.string().default("fixtures/handbook.md"),
  replace: z.boolean().default(true),
});

export type IngestPathRequest = z.infer<typeof ingestPathRequestSchema>;

export const ingestResponseSchema = z.object({
  chunks_indexed: z.number(),
  source_path: z.string(),
});

export type IngestResponse = z.infer<typeof ingestResponseSchema>;

export const healthResponseSchema = z.object({
  status: z.literal("ok"),
  service: z.string(),
  chat_model: z.string().optional(),
  embedding_model: z.string().optional(),
});

export type HealthResponse = z.infer<typeof healthResponseSchema>;
