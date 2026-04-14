import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeProvider } from "../theme";
import { HandbookQA } from "./HandbookQA";

vi.mock("./RAGPipelineVisualizer", () => ({
  RAGPipelineVisualizer: () => (
    <div data-testid="rag-pipeline-visualizer-stub" aria-hidden>
      pipeline
    </div>
  ),
}));

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (url.includes("/api/health")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ status: "ok", service: "handbook-rag-api" }),
        } as Response);
      }
      return Promise.resolve({ ok: false } as Response);
    }),
  );
});

const wrap = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <HandbookQA selectedEmployeeId="E001" />
      </ThemeProvider>
    </QueryClientProvider>,
  );
};

describe("HandbookQA", () => {
  it("renders handbook section", () => {
    wrap();
    expect(screen.getByRole("heading", { name: /enterprise ai knowledge base/i })).toBeDefined();
  });
});
