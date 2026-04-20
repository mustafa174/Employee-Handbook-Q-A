import { useEffect, useMemo, useState } from "react";

type DocSection = {
  id: string;
  title: string;
};

const sections: DocSection[] = [
  { id: "simple-truth", title: "Simple Truth" },
  { id: "langchain-section", title: "LangChain Section" },
  { id: "project-usage", title: "In Your Project Likely Usage" },
  { id: "langgraph-adds", title: "Then What LangGraph Adds" },
  { id: "workflow-example", title: "Example Workflow" },
  { id: "why-stack-choice", title: "Why This Stack Was Needed" },
  { id: "lead-explanation", title: "Best Way to Explain to Lead" },
  { id: "analogy", title: "Analogy" },
  { id: "why-notice", title: "Why You Did Not Notice It" },
  { id: "code-documentation", title: "Code Documentation: What Is Used and How" },
  { id: "alternatives", title: "Alternatives and Tradeoffs" },
];

const PersonalDocumentationPage = () => {
  const [activeSection, setActiveSection] = useState<string>(sections[0].id);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          setActiveSection(visible[0].target.id);
        }
      },
      { rootMargin: "-15% 0px -60% 0px", threshold: [0.1, 0.3, 0.6] },
    );

    sections.forEach((section) => {
      const node = document.getElementById(section.id);
      if (node) observer.observe(node);
    });

    return () => observer.disconnect();
  }, []);

  const tabButtons = useMemo(
    () =>
      sections.map((section) => (
        <button
          key={section.id}
          type="button"
          onClick={() => {
            const el = document.getElementById(section.id);
            el?.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
          className={[
            "rounded-md px-3 py-1.5 text-xs font-medium transition-all duration-200",
            activeSection === section.id
              ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-zinc-50"
              : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-zinc-100",
          ].join(" ")}
          aria-current={activeSection === section.id ? "true" : "false"}
        >
          {section.title}
        </button>
      )),
    [activeSection],
  );

  return (
    <article className="mx-auto max-w-5xl space-y-6 pb-12 pt-4">
      <header className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">LangGraph + LangChain Study Notes</h2>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          Personal documentation page focused on how LangChain works underneath LangGraph in this project.
        </p>
      </header>

      <section className="sticky top-3 z-10 rounded-xl border border-zinc-200 bg-zinc-100/90 p-2 backdrop-blur dark:border-zinc-700 dark:bg-zinc-800/90">
        <div className="flex flex-wrap gap-1">{tabButtons}</div>
      </section>

      <section id="simple-truth" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Simple Truth</h3>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          LangGraph is built within the LangChain ecosystem. Most LangGraph projects still use LangChain components such as:
        </p>
        <ul className="mt-2 list-disc space-y-1 pl-6 text-sm text-zinc-700 dark:text-zinc-300">
          <li>chat model wrappers</li>
          <li>prompt templates</li>
          <li>retrievers</li>
          <li>vector store integrations</li>
          <li>document loaders</li>
          <li>output parsers</li>
          <li>tools</li>
        </ul>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          So even if your project says “LangGraph,” LangChain is usually underneath.
        </p>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          This is important because teams often think they are making a pure orchestration decision, but in practice they are
          also making integration decisions for model access, retrieval patterns, prompt typing, and schema enforcement. That is
          why understanding both layers (LangGraph + LangChain) is necessary for debugging and architecture decisions.
        </p>
      </section>

      <section id="langchain-section" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">LangChain Section</h3>

        <h4 className="mt-4 text-base font-semibold text-zinc-900 dark:text-zinc-100">Core Concept</h4>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          <span className="font-semibold">LangGraph = Flow Orchestrator</span>
        </p>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">Defines:</p>
        <ul className="mt-1 list-disc space-y-1 pl-6 text-sm text-zinc-700 dark:text-zinc-300">
          <li>nodes</li>
          <li>edges</li>
          <li>routing decisions</li>
          <li>shared state</li>
        </ul>

        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          <span className="font-semibold">LangChain = What Happens Inside Nodes</span>
        </p>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">Each node can use:</p>
        <ul className="mt-1 list-disc space-y-1 pl-6 text-sm text-zinc-700 dark:text-zinc-300">
          <li>prompts</li>
          <li>LLM calls</li>
          <li>retrievers</li>
          <li>embeddings</li>
          <li>tools</li>
          <li>parsers</li>
        </ul>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          So LangGraph controls where to go next, LangChain handles what each step does.
        </p>

        <h4 className="mt-5 text-base font-semibold text-zinc-900 dark:text-zinc-100">What Your Project Nodes Likely Look Like</h4>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          Based on your HR assistant architecture, probable graph:
        </p>
        <pre className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100">
{`START
  ↓
Refine Query Node
  ↓
Guardrail Node
  ↓
Intent Router Node
  ├── Policy Node
  ├── Personal Data Node
  ├── Mixed Query Node
  └── Clarification Node
  ↓
Answer Validation Node
  ↓
Cache Save Node
  ↓
END`}
        </pre>

        <h4 className="mt-5 text-base font-semibold text-zinc-900 dark:text-zinc-100">How LangChain Is Used Inside Each Node</h4>
        <ol className="mt-2 list-decimal space-y-3 pl-6 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          <li>
            <span className="font-semibold">Refine Query Node</span>
            <p>Uses LangChain prompt + model.</p>
            <p>
              Input: “My pto?”<br />
              Output: “What is my PTO balance?”
            </p>
          </li>
          <li>
            <span className="font-semibold">Guardrail Node</span>
            <p>Uses prompt classifier or moderation logic.</p>
            <p>Detects: jailbreak, abuse, irrelevant request.</p>
          </li>
          <li>
            <span className="font-semibold">Intent Router Node</span>
            <p>Likely LLM classification node.</p>
            <p>Returns: policy, personal, mixed.</p>
            <p>This can use LangChain structured output parser.</p>
          </li>
          <li>
            <span className="font-semibold">Policy Node</span>
            <p>Uses LangChain retrieval stack: embeddings, ChromaDB retriever, prompt with context, and LLM answer.</p>
          </li>
          <li>
            <span className="font-semibold">Personal Data Node</span>
            <p>Uses tool/function/data source.</p>
            <p>Reads employee record using `employee_id`.</p>
          </li>
          <li>
            <span className="font-semibold">Mixed Node</span>
            <p>Combines policy retrieval, employee data lookup, and final synthesized answer.</p>
          </li>
          <li>
            <span className="font-semibold">Validation Node</span>
            <p>Uses LLM or rules to verify grounded answer, no hallucination, and proper formatting.</p>
          </li>
          <li>
            <span className="font-semibold">Cache Node</span>
            <p>Stores reusable answers.</p>
          </li>
        </ol>

        <h4 className="mt-5 text-base font-semibold text-zinc-900 dark:text-zinc-100">What State Moves Between Nodes</h4>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">LangGraph state likely carries:</p>
        <pre className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100">
{`{
  "question": "...",
  "employee_id": "E001",
  "chat_history": [...],
  "intent": "policy",
  "retrieved_docs": [...],
  "answer": "...",
  "cache_hit": false
}`}
        </pre>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">Each node reads/writes state.</p>

        <h4 className="mt-5 text-base font-semibold text-zinc-900 dark:text-zinc-100">Why This Architecture Is Good</h4>
        <ul className="mt-2 list-disc space-y-2 pl-6 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          <li>
            <span className="font-semibold">Separation of Concerns:</span> each node has one responsibility.
          </li>
          <li>
            <span className="font-semibold">Easy Maintenance:</span> you can modify router behavior without touching retrieval.
          </li>
          <li>
            <span className="font-semibold">Scalable:</span> easy to add new nodes later (payroll node, benefits node, escalation node).
          </li>
          <li>
            <span className="font-semibold">Production Friendly:</span> easier debugging because you can identify where failure occurred.
          </li>
        </ul>

        <h4 className="mt-5 text-base font-semibold text-zinc-900 dark:text-zinc-100">What I Think of Your Project Maturity</h4>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          This is not beginner-level chaining. This is closer to enterprise AI workflow design because it uses routing, stateful
          flow, caching, hybrid data sources, and validation. That is exactly where LangGraph is valuable.
        </p>

        <h4 className="mt-5 text-base font-semibold text-zinc-900 dark:text-zinc-100">What to Tell Lead</h4>
        <blockquote className="mt-2 rounded-lg border-l-4 border-sky-500 bg-sky-50 px-4 py-3 text-sm text-sky-900 dark:bg-sky-950/40 dark:text-sky-100">
          “We used LangGraph to orchestrate the assistant through modular nodes. Each node performs a specific responsibility such
          as classification, retrieval, personal data lookup, validation, or caching. Inside those nodes, LangChain components
          handle prompts, LLM calls, and retrieval integrations.”
        </blockquote>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">Excellent answer.</p>

        <h4 className="mt-5 text-base font-semibold text-zinc-900 dark:text-zinc-100">My Honest Guess on Why LangGraph Was Chosen</h4>
        <p className="mt-2 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          Because a simple LangChain chain would become messy for policy vs personal branching, retries, guardrails, multi-source
          answers, and cache checkpoints. Graph structure solves this cleanly.
        </p>
      </section>

      <section id="project-usage" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">In Your Project Likely Usage</h3>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          Based on your HR assistant architecture, LangChain is probably used for:
        </p>
        <ol className="mt-3 list-decimal space-y-3 pl-6 text-sm text-zinc-700 dark:text-zinc-300">
          <li>
            <span className="font-semibold">LLM Connection</span>
            <p className="mt-1">
              Example: OpenAI model wrapper. This centralizes auth, retry handling, model parameters, and JSON-structured output
              in one consistent interface.
            </p>
          </li>
          <li>
            <span className="font-semibold">ChromaDB Integration</span>
            <p className="mt-1">
              Using LangChain vector store adapters for ChromaDB keeps retrieval API consistent even if you later change the
              underlying store.
            </p>
          </li>
          <li>
            <span className="font-semibold">Embeddings</span>
            <p className="mt-1">
              Text converted into vectors via LangChain embedding classes, so ingestion and query-time similarity use the same
              abstraction.
            </p>
          </li>
          <li>
            <span className="font-semibold">Prompt Templates</span>
            <p className="mt-1">
              System prompts / router prompts / answer prompts are grouped into repeatable prompt contracts instead of one-off
              string building.
            </p>
          </li>
          <li>
            <span className="font-semibold">Retrievers</span>
            <p className="mt-1">
              Fetching relevant handbook chunks with scoring to support grounded policy answers and citation behavior.
            </p>
          </li>
          <li>
            <span className="font-semibold">Message Objects</span>
            <p className="mt-1">
              Managing HumanMessage / AIMessage state so multi-turn context remains explicit, typed, and debuggable.
            </p>
          </li>
        </ol>
      </section>

      <section id="langgraph-adds" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Then What LangGraph Adds</h3>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">LangGraph adds:</p>
        <ul className="mt-2 list-disc space-y-1 pl-6 text-sm text-zinc-700 dark:text-zinc-300">
          <li>nodes</li>
          <li>edges</li>
          <li>branching logic</li>
          <li>state management</li>
          <li>retries</li>
          <li>loops</li>
          <li>multi-step workflows</li>
        </ul>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          The key value is control. Instead of one giant prompt trying to do everything, the system becomes a staged workflow
          where each node has one responsibility. That makes behavior easier to test, reason about, and fix when failures happen.
        </p>
      </section>

      <section id="workflow-example" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Example Workflow</h3>
        <p className="mt-3 rounded-lg border border-zinc-200 bg-zinc-50 px-4 py-3 font-mono text-sm text-zinc-800 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100">
          Question -&gt; classify -&gt; route -&gt; retrieve -&gt; answer -&gt; validate
        </p>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          Why this matters: if classification is wrong, retrieval is skipped or misused; if retrieval quality is weak, generation
          can be politely wrong; if validation is missing, policy answers may leak personal data. Breaking the flow into stages
          gives measurable checkpoints instead of hidden failure.
        </p>
      </section>

      <section id="why-stack-choice" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Why This Stack Was Needed</h3>
        <div className="mt-3 space-y-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          <p>
            This HR assistant has mixed requirements: it must answer handbook policy questions, personal profile questions, and
            mixed questions in one turn. A single-prompt chatbot is usually not reliable enough for that because policy grounding,
            privacy boundaries, and route-specific behavior need deterministic control.
          </p>
          <p>
            LangChain components solve integration and foundation concerns (model wrappers, embeddings, vector retrieval,
            structured outputs). LangGraph solves orchestration concerns (which step runs next, when to retry retrieval, when to
            escalate, and when to skip policy retrieval for pure personal asks).
          </p>
          <p>
            The practical need is trust and auditability: leadership and HR teams need to know why an answer was produced and what
            evidence was used. This stack supports that by combining citations, route traces, and post-graph response contracts.
          </p>
        </div>
      </section>

      <section id="lead-explanation" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Best Way to Explain to Lead</h3>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">Say:</p>
        <blockquote className="mt-2 rounded-lg border-l-4 border-sky-500 bg-sky-50 px-4 py-3 text-sm text-sky-900 dark:bg-sky-950/40 dark:text-sky-100">
          “We use LangGraph for workflow orchestration, and LangChain components for model access, retrieval, embeddings, and prompting.”
        </blockquote>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">That sounds technically correct and senior.</p>
      </section>

      <section id="analogy" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Analogy</h3>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          LangChain = engine parts
          <br />
          LangGraph = vehicle navigation system
        </p>
      </section>

      <section id="why-notice" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Why You Did Not Notice It</h3>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">Many repos import things like:</p>
        <pre className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100">
{`from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate`}
        </pre>
        <p className="mt-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          People say “we use LangGraph,” but those are LangChain ecosystem packages.
        </p>
      </section>

      <section id="code-documentation" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Code Documentation: What Is Used and How</h3>
        <div className="mt-4 space-y-4 text-sm text-zinc-700 dark:text-zinc-300">
          <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">1) Graph orchestration - `rag-api/app/rag_graph.py`</p>
            <p className="mt-1">
              Uses `StateGraph` from LangGraph to define nodes (`query_refiner`, `guardrail`, `router`, `retrieve`,
              `grade_documents`, `balance`, `generate`) and conditional edges.
            </p>
            <p className="mt-1">
              <span className="font-semibold">Need:</span>{" "}
              predictable control flow for policy/personal/mixed routes.{" "}
              <span className="ml-2 font-semibold">Alternative:</span>{" "}
              one monolithic LLM prompt or ad-hoc `if/else` controller outside a graph.{" "}
              <span className="ml-2 font-semibold">Why chosen:</span>{" "}
              graph form makes route behavior explicit and testable.
            </p>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">2) LLM wrappers - `ChatOpenAI`</p>
            <p className="mt-1">
              The code uses `ChatOpenAI` for query refinement, retrieval grading, query rewrite, and final answer generation.
            </p>
            <p className="mt-1">
              It also uses structured outputs (`with_structured_output`) so each model response matches a typed schema.
            </p>
            <p className="mt-1">
              <span className="font-semibold">Need:</span>{" "}
              stable model access + schema-safe JSON outputs.{" "}
              <span className="ml-2 font-semibold">Alternative:</span>{" "}
              raw OpenAI SDK calls with manual parsing.{" "}
              <span className="ml-2 font-semibold">Why chosen:</span>{" "}
              less boilerplate, fewer parsing bugs, easier typed outputs.
            </p>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">3) Message objects - `HumanMessage`, `AIMessage`, `SystemMessage`</p>
            <p className="mt-1">
              Conversation turns are stored in graph state with LangChain message classes. This gives predictable state handling
              for history-aware prompts.
            </p>
            <p className="mt-1">
              <span className="font-semibold">Need:</span>{" "}
              clean multi-turn context management.{" "}
              <span className="ml-2 font-semibold">Alternative:</span>{" "}
              plain dict arrays with custom role strings.{" "}
              <span className="ml-2 font-semibold">Why chosen:</span>{" "}
              better interoperability with LangChain/LangGraph APIs.
            </p>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">4) Embeddings + vector store - `rag-api/app/vectorstore.py`</p>
            <p className="mt-1">
              Uses `OpenAIEmbeddings` for vector creation and LangChain `Chroma` adapter for persistence/querying. Retrieval is
              done through `similarity_search_with_score`.
            </p>
            <p className="mt-1">
              <span className="font-semibold">Need:</span>{" "}
              semantic retrieval of handbook policy text.{" "}
              <span className="ml-2 font-semibold">Alternative:</span>{" "}
              keyword/BM25 only retrieval or another store (Pinecone, Weaviate, FAISS).{" "}
              <span className="ml-2 font-semibold">Why chosen:</span>{" "}
              Chroma is simple for local development; adapter keeps switch cost lower later.
            </p>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">5) Ingestion/chunking - `rag-api/app/handbook_ingest.py`</p>
            <p className="mt-1">
              Uses `RecursiveCharacterTextSplitter` and LangChain `Document` objects to chunk handbook files and attach metadata
              before writing into Chroma.
            </p>
            <p className="mt-1">
              <span className="font-semibold">Need:</span>{" "}
              consistent chunk sizes with usable metadata for citations.{" "}
              <span className="ml-2 font-semibold">Alternative:</span>{" "}
              custom split code, token splitters, or semantic splitters.{" "}
              <span className="ml-2 font-semibold">Why chosen:</span>{" "}
              recursive splitting is robust across markdown/text/PDF with minimal complexity.
            </p>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">6) API wrapper over graph - `rag-api/app/main.py`</p>
            <p className="mt-1">
              FastAPI endpoints call the compiled graph, apply response contracts, and stream node events. This is the execution
              layer around LangGraph runtime.
            </p>
            <p className="mt-1">
              <span className="font-semibold">Need:</span>{" "}
              production API shape with validation and streaming.{" "}
              <span className="ml-2 font-semibold">Alternative:</span>{" "}
              direct graph invocation from UI or CLI-only flow.{" "}
              <span className="ml-2 font-semibold">Why chosen:</span>{" "}
              clear backend boundary, typed responses, and easier frontend integration.
            </p>
          </div>

          <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">7) Additional routing intelligence</p>
            <p className="mt-1">
              `intent_policy.py` provides deterministic route rules, while `semantic_router.py` adds optional semantic rescue
              with sentence-transformers when lexical routing is uncertain.
            </p>
            <p className="mt-1">
              <span className="font-semibold">Need:</span>{" "}
              high routing precision with graceful fallback.{" "}
              <span className="ml-2 font-semibold">Alternative:</span>{" "}
              pure LLM intent classification every request.{" "}
              <span className="ml-2 font-semibold">Why chosen:</span>{" "}
              deterministic rules are cheaper and more predictable; rescue adds flexibility only where needed.
            </p>
          </div>
        </div>
      </section>

      <section id="alternatives" className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Alternatives and Tradeoffs</h3>
        <div className="mt-3 space-y-3 text-sm leading-7 text-zinc-700 dark:text-zinc-300">
          <p>
            <span className="font-semibold">Alternative 1: Pure LangChain chains without LangGraph.</span> This can work for
            simple linear flows, but it becomes harder to express route branching and retries cleanly once you mix policy,
            personal, and escalation logic.
          </p>
          <p>
            <span className="font-semibold">Alternative 2: Agent-only design with tool-calling.</span> More flexible but less
            deterministic for compliance-sensitive HR policy behavior. Great for exploration, weaker for strict routing contracts.
          </p>
          <p>
            <span className="font-semibold">Alternative 3: Traditional rule engine + no LLM in control path.</span> Very
            predictable but poor language understanding and weak user experience for natural language follow-ups.
          </p>
          <p>
            The current design is a balanced architecture: deterministic where safety and policy boundaries matter, model-driven
            where language understanding and synthesis add value.
          </p>
        </div>
      </section>
    </article>
  );
};

export { PersonalDocumentationPage };
