import { describe, expect, it } from "vitest";
import { askRequestSchema, askResponseSchema } from "./index";

describe("handbook schemas", () => {
  it("parses ask request", () => {
    const q = askRequestSchema.parse({
      question: "What is PTO?",
      employee_id: "E001",
    });
    expect(q.question).toBe("What is PTO?");
  });

  it("parses ask response", () => {
    const r = askResponseSchema.parse({
      answer: "PTO is paid time off.",
      citations: [{ text: "Full-time employees accrue", score: 0.92 }],
      isEscalated: false,
      pipeline_steps: [
        { id: "guardrail", label: "Guardrail", status: "ok" as const },
        { id: "llm", label: "OpenAI", status: "ok" as const },
      ],
      use_rag: true,
      chat_model: "gpt-4o-mini",
    });
    expect(r.citations).toHaveLength(1);
    expect(r.citations[0].score).toBeCloseTo(0.92);
  });
});
