import type { ReactNode } from "react";

const ArrowDown = () => (
  <div className="flex justify-center py-1" aria-hidden>
    <span className="text-lg font-light text-zinc-400 dark:text-zinc-500">↓</span>
  </div>
);

function LayerCard({
  title,
  subtitle,
  children,
  tone = "neutral",
}: Readonly<{
  title: string;
  subtitle?: string;
  children?: ReactNode;
  tone?: "neutral" | "sky" | "violet" | "amber" | "emerald";
}>) {
  const tones = {
    neutral: "border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-900/80",
    sky: "border-sky-200 bg-sky-50/90 dark:border-sky-900/60 dark:bg-sky-950/40",
    violet: "border-violet-200 bg-violet-50/90 dark:border-violet-900/60 dark:bg-violet-950/40",
    amber: "border-amber-200 bg-amber-50/90 dark:border-amber-900/50 dark:bg-amber-950/35",
    emerald: "border-emerald-200 bg-emerald-50/90 dark:border-emerald-900/50 dark:bg-emerald-950/35",
  } as const;
  return (
    <div className={`rounded-2xl border px-5 py-4 shadow-sm ${tones[tone]}`}>
      <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</h3>
      {subtitle ? <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">{subtitle}</p> : null}
      {children ? <div className="mt-3 text-xs text-zinc-700 dark:text-zinc-300">{children}</div> : null}
    </div>
  );
}

function BranchPill({ label, className }: Readonly<{ label: string; className: string }>) {
  return (
    <span
      className={`rounded-lg border px-2.5 py-1.5 text-center text-[10px] font-semibold uppercase tracking-wide sm:text-xs ${className}`}
    >
      {label}
    </span>
  );
}

export function DocumentPage() {
  return (
    <article className="mx-auto max-w-3xl space-y-10 py-8 pb-16 text-zinc-800 dark:text-zinc-200">
      <header className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-sky-600 dark:text-sky-400">
          Employee Handbook Q&amp;A
        </p>
        <h1 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-3xl">
          System architecture
        </h1>
        <p className="text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
          Layered hybrid design for accurate policy answers, safe handling of personal data, and scalable AI
          workflows. The diagram below matches how the app is wired today: React and Vite at the top, FastAPI
          orchestration, semantic cache, LangGraph with route-specific branches, Chroma for handbook RAG, and
          MCP-shaped tools for employee data.
        </p>
      </header>

      <section aria-labelledby="arch-diagram-heading" className="space-y-3">
        <h2 id="arch-diagram-heading" className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
          Architecture diagram
        </h2>
        <p className="text-xs text-zinc-500 dark:text-zinc-500">
          Read top to bottom. Dashed boxes are conditional paths chosen by the router after guardrails.
        </p>
        <div className="rounded-2xl border border-zinc-200 bg-gradient-to-b from-zinc-50 to-white p-6 dark:border-zinc-800 dark:from-zinc-950 dark:to-zinc-900/80">
          <LayerCard
            title="Presentation layer"
            subtitle="React and Vite"
            tone="sky"
          >
            <ul className="list-inside list-disc space-y-1">
              <li>Reusable UI (assistant, settings, documentation)</li>
              <li>Fast local dev and production builds via Vite</li>
              <li>Calls FastAPI over HTTP; RAG pane uses SSE for stream + node progress</li>
            </ul>
          </LayerCard>
          <ArrowDown />
          <LayerCard
            title="API layer"
            subtitle="FastAPI (rag-api)"
            tone="violet"
          >
            <ul className="list-inside list-disc space-y-1">
              <li>
                <code className="text-[11px]">POST /api/ask</code> and{" "}
                <code className="text-[11px]">POST /api/ask/stream</code>
              </li>
              <li>Async-friendly, typed request and response models</li>
              <li>Post-graph response contract: policy citations, profile leakage checks, recovery text</li>
            </ul>
          </LayerCard>
          <ArrowDown />
          <LayerCard title="Semantic cache" subtitle="Optional read-through / write-through" tone="amber">
            <p>
              If a similar question was answered and cached (route- and KB-aware), the API can return immediately,
              cutting latency and token cost. Sensitive prompts can bypass cache by policy.
            </p>
          </LayerCard>
          <ArrowDown />
          <LayerCard title="Orchestration layer" subtitle="LangGraph (compiled graph)" tone="emerald">
            <p className="mb-3 font-medium text-emerald-900/90 dark:text-emerald-100/90">LangGraph nodes (in order)</p>
            <ol className="list-inside list-decimal space-y-1 text-[11px] sm:text-xs">
              <li>Query refinement — normalize, optional multi-question split, retrieval query shaping</li>
              <li>Guardrails — sensitive-pattern handling and escalation signals</li>
              <li>Router — intent and execution route (policy, personal, mixed, clarify, general)</li>
              <li>Retrieval — Chroma vector search and grading / re-search on policy and mixed paths</li>
              <li>Employee tools — profile and balance context on personal and mixed paths (MCP-shaped)</li>
              <li>Generation — grounded LLM answer with structured output where used</li>
            </ol>
            <p className="mt-3 text-[11px] text-emerald-900/80 dark:text-emerald-200/80">
              LangChain is used inside nodes (models, embeddings, structured output, retriever wiring), not as the
              outer orchestration shell. LangGraph decides which path runs; LangChain helps execute each node.
            </p>
          </LayerCard>
          <ArrowDown />
          <div className="rounded-2xl border border-dashed border-zinc-300 bg-zinc-50/80 px-4 py-4 dark:border-zinc-600 dark:bg-zinc-900/50">
            <p className="mb-3 text-center text-xs font-semibold text-zinc-700 dark:text-zinc-300">
              Router branches (after guardrail)
            </p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <BranchPill
                label="Policy"
                className="border-sky-300 bg-sky-100/80 text-sky-900 dark:border-sky-800 dark:bg-sky-950/50 dark:text-sky-100"
              />
              <BranchPill
                label="Personal"
                className="border-violet-300 bg-violet-100/80 text-violet-900 dark:border-violet-800 dark:bg-violet-950/50 dark:text-violet-100"
              />
              <BranchPill
                label="Mixed"
                className="border-fuchsia-300 bg-fuchsia-100/80 text-fuchsia-900 dark:border-fuchsia-800 dark:bg-fuchsia-950/50 dark:text-fuchsia-100"
              />
              <BranchPill
                label="Clarify / General"
                className="border-amber-300 bg-amber-100/80 text-amber-950 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-100"
              />
            </div>
            <ul className="mt-3 space-y-1.5 text-[11px] leading-snug text-zinc-600 dark:text-zinc-400">
              <li>
                <strong className="text-zinc-800 dark:text-zinc-300">Policy</strong> — RAG over handbook chunks in
                Chroma (lightweight vector store; alternatives include Pinecone, Weaviate, Milvus, pgvector).
              </li>
              <li>
                <strong className="text-zinc-800 dark:text-zinc-300">Personal</strong> — structured employee data
                (fixtures such as <code className="text-[10px]">employees.json</code>), callable in an MCP-style shape
                for future HRIS, payroll, or internal APIs.
              </li>
              <li>
                <strong className="text-zinc-800 dark:text-zinc-300">Mixed</strong> — both handbook evidence and
                employee context (for example PTO balance plus carryover policy in one turn).
              </li>
              <li>
                <strong className="text-zinc-800 dark:text-zinc-300">Clarify / general</strong> — clarification
                prompts or scoped general responses when routing demands it.
              </li>
            </ul>
          </div>
          <ArrowDown />
          <LayerCard title="Data and models" subtitle="Runtime dependencies" tone="neutral">
            <ul className="list-inside list-disc space-y-1">
              <li>Chroma — embeddings and similarity search over ingested handbook and IT guides</li>
              <li>OpenAI — chat and embedding models (configured via environment)</li>
              <li>Semantic cache store — JSON on disk under rag-api data directory</li>
            </ul>
          </LayerCard>
          <ArrowDown />
          <LayerCard title="Response to client" subtitle="JSON or SSE + final payload" tone="sky">
            Streaming sends staged node events for the RAG logic flow visualizer, then tokenized answer chunks, then a
            final object aligned with the non-stream contract. Both paths apply the same response contract rules
            before return.
          </LayerCard>
        </div>
      </section>

      <section className="space-y-4 text-sm leading-relaxed">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Why these choices</h2>
        <p>
          <strong className="text-zinc-900 dark:text-zinc-100">React and Vite</strong> keep the UI composable and the
          edit-build loop fast. <strong className="text-zinc-900 dark:text-zinc-100">FastAPI</strong> fits the
          Python-first LLM ecosystem and supports async streaming. <strong className="text-zinc-900 dark:text-zinc-100">
            LangGraph
          </strong>{" "}
          models non-linear flows: deterministic routing, retries, traceability, and separate execution paths instead
          of a single linear chain.
        </p>
        <p>
          <strong className="text-zinc-900 dark:text-zinc-100">RAG</strong> is used where policy is unstructured
          narrative (leave rules, benefits, procedures). <strong className="text-zinc-900 dark:text-zinc-100">
            Structured tools
          </strong>{" "}
          are better for exact values such as PTO balance, department, or manager relationships, with a connector style
          that can later swap fixtures for Workday, SAP, PostgreSQL, or internal HR APIs without rewriting the graph.
        </p>
        <p className="rounded-xl border border-zinc-200 bg-zinc-50/80 px-4 py-3 text-xs italic text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-400">
          In one line: LangGraph orchestration, LangChain primitives inside nodes, Chroma RAG for handbook knowledge,
          MCP-style employee tools for personal data, semantic cache for repeat traffic, and a final API contract for
          safety and shape.
        </p>
      </section>

      <footer className="border-t border-zinc-200 pt-6 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-500">
        For file-level module detail, see <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">docs/ARCHITECTURE.md</code>{" "}
        in the repository. The LangChain-focused walkthrough remains under{" "}
        <strong className="text-zinc-700 dark:text-zinc-400">Documentation</strong> in the app nav.
      </footer>
    </article>
  );
}
