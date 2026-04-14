import {
  Background,
  BackgroundVariant,
  Handle,
  Position,
  ReactFlow,
  ReactFlowProvider,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { memo, useMemo } from "react";

type StageState = "idle" | "active" | "done";

type StageData = { label: string; state: StageState };

type RagStageType = Node<StageData, "ragStage">;
type QueryOnlyType = Node<StageData, "queryOnly">;
type LlmOnlyType = Node<StageData, "llmOnly">;

const RAGStageNode = memo(function RAGStageNode({ data }: NodeProps<RagStageType>) {
  const { label, state } = data;
  return (
    <div
      className={[
        "min-w-[132px] max-w-[168px] rounded-xl border px-2.5 py-2 text-center text-[10px] font-semibold leading-tight shadow-sm transition-all duration-300 dark:shadow-none sm:text-[11px]",
        state === "active" &&
          "scale-[1.02] border-sky-500 bg-sky-50 ring-2 ring-sky-400/70 dark:border-sky-500 dark:bg-sky-950/50 dark:ring-sky-400/40",
        state === "done" &&
          "border-emerald-500/70 bg-emerald-50/90 dark:border-emerald-600/60 dark:bg-emerald-950/35",
        state === "idle" && "border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-900/80",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2 !w-2 !border-0 !bg-zinc-400 dark:!bg-zinc-500"
        isConnectable={false}
      />
      {label}
      <Handle
        type="source"
        position={Position.Right}
        className="!h-2 !w-2 !border-0 !bg-zinc-400 dark:!bg-zinc-500"
        isConnectable={false}
      />
    </div>
  );
});

function QueryOnlyNode({ data }: NodeProps<QueryOnlyType>) {
  return (
    <div
      className={[
        "min-w-[132px] max-w-[168px] rounded-xl border px-2.5 py-2 text-center text-[10px] font-semibold leading-tight shadow-sm transition-all duration-300 dark:shadow-none sm:text-[11px]",
        data.state === "active" &&
          "scale-[1.02] border-sky-500 bg-sky-50 ring-2 ring-sky-400/70 dark:border-sky-500 dark:bg-sky-950/50",
        data.state === "done" &&
          "border-emerald-500/70 bg-emerald-50/90 dark:border-emerald-600/60 dark:bg-emerald-950/35",
        data.state === "idle" && "border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-900/80",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {data.label}
      <Handle
        type="source"
        position={Position.Right}
        className="!h-2 !w-2 !border-0 !bg-zinc-400 dark:!bg-zinc-500"
        isConnectable={false}
      />
    </div>
  );
}

function LLMOnlyNode({ data }: NodeProps<LlmOnlyType>) {
  return (
    <div
      className={[
        "min-w-[132px] max-w-[168px] rounded-xl border px-2.5 py-2 text-center text-[10px] font-semibold leading-tight shadow-sm transition-all duration-300 dark:shadow-none sm:text-[11px]",
        data.state === "active" &&
          "scale-[1.02] border-sky-500 bg-sky-50 ring-2 ring-sky-400/70 dark:border-sky-500 dark:bg-sky-950/50",
        data.state === "done" &&
          "border-emerald-500/70 bg-emerald-50/90 dark:border-emerald-600/60 dark:bg-emerald-950/35",
        data.state === "idle" && "border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-900/80",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2 !w-2 !border-0 !bg-zinc-400 dark:!bg-zinc-500"
        isConnectable={false}
      />
      <span className="whitespace-pre-line">{data.label}</span>
    </div>
  );
}

const nodeTypesFull = {
  ragStage: RAGStageNode,
  queryOnly: QueryOnlyNode,
  llmOnly: LLMOnlyNode,
} satisfies NodeTypes;

const stateForNode = (
  nodeId: string,
  pipelineComplete: boolean,
  activeNodeIds: Set<string>,
  doneNodeIds: Set<string>,
): StageState => {
  if (activeNodeIds.has(nodeId)) return "active";
  if (doneNodeIds.has(nodeId) || pipelineComplete) return "done";
  return "idle";
};

function buildNodes(
  pipelineComplete: boolean,
  llmNodeLabel: string,
  activeNodeIds: Set<string>,
  doneNodeIds: Set<string>,
): Node[] {
  const s = (id: string) => stateForNode(id, pipelineComplete, activeNodeIds, doneNodeIds);
  return [
    {
      id: "query",
      type: "queryOnly",
      position: { x: 0, y: 100 },
      data: { label: "Query", state: s("query") },
    },
    {
      id: "guardrail",
      type: "ragStage",
      position: { x: 150, y: 100 },
      data: { label: "Safety Guardrail", state: s("guardrail") },
    },
    {
      id: "router",
      type: "ragStage",
      position: { x: 300, y: 100 },
      data: { label: "Intent Router", state: s("router") },
    },
    {
      id: "chroma",
      type: "ragStage",
      position: { x: 500, y: 40 },
      data: { label: "ChromaDB (Policy)", state: s("chroma") },
    },
    {
      id: "mcp",
      type: "ragStage",
      position: { x: 500, y: 160 },
      data: { label: "Employee Tool (MCP)", state: s("mcp") },
    },
    {
      id: "synthesis",
      type: "ragStage",
      position: { x: 700, y: 100 },
      data: { label: "Synthesis & Grounding", state: s("synthesis") },
    },
    {
      id: "judge",
      type: "ragStage",
      position: { x: 850, y: 100 },
      data: { label: "Atomic Judge", state: s("judge") },
    },
    {
      id: "output",
      type: "llmOnly",
      position: { x: 1000, y: 100 },
      data: { label: `Verified Response\n${llmNodeLabel}`, state: s("output") },
    },
  ];
}

function buildEdges(isLoading: boolean, isDark: boolean): Edge[] {
  const stroke = isDark ? "#71717a" : "#a1a1aa";
  const blueStroke = isDark ? "#38bdf8" : "#0284c7";
  const greenStroke = isDark ? "#4ade80" : "#16a34a";
  const animated = isLoading;
  return [
    { id: "e-q-g", source: "query", target: "guardrail", type: "smoothstep", animated, style: { stroke } },
    { id: "e-g-r", source: "guardrail", target: "router", type: "smoothstep", animated, style: { stroke } },
    { id: "e-r-c", source: "router", target: "chroma", type: "smoothstep", animated, style: { stroke: blueStroke } },
    { id: "e-r-m", source: "router", target: "mcp", type: "smoothstep", animated, style: { stroke: greenStroke } },
    { id: "e-c-s", source: "chroma", target: "synthesis", type: "smoothstep", animated, style: { stroke: blueStroke } },
    { id: "e-m-s", source: "mcp", target: "synthesis", type: "smoothstep", animated, style: { stroke: greenStroke } },
    { id: "e-s-j", source: "synthesis", target: "judge", type: "smoothstep", animated, style: { stroke } },
    { id: "e-j-o", source: "judge", target: "output", type: "smoothstep", animated, style: { stroke } },
  ];
}

type InnerProps = {
  isLoading: boolean;
  pipelineComplete: boolean;
  isDark: boolean;
  /** Shown on the final node; use `\n` for a second line (e.g. OpenAI + model id). */
  llmNodeLabel: string;
  activeNodes?: string[];
  doneNodes?: string[];
};

function RAGPipelineFlow({
  isLoading,
  pipelineComplete,
  isDark,
  llmNodeLabel,
  activeNodes,
  doneNodes,
}: Readonly<InnerProps>) {
  const activeNodeIds = useMemo(() => new Set(activeNodes ?? []), [activeNodes]);
  const doneNodeIds = useMemo(() => new Set(doneNodes ?? []), [doneNodes]);

  const nodes = useMemo(
    () => buildNodes(pipelineComplete, llmNodeLabel, activeNodeIds, doneNodeIds),
    [pipelineComplete, llmNodeLabel, activeNodeIds, doneNodeIds],
  );

  const edges = useMemo(() => buildEdges(isLoading, isDark), [isLoading, isDark]);

  return (
    <div className="h-[300px] w-full overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950/50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypesFull}
        fitView
        fitViewOptions={{ padding: 0.16, maxZoom: 1.05, minZoom: 0.4 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        preventScrolling
        colorMode={isDark ? "dark" : "light"}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={14} size={1} className="opacity-40" />
      </ReactFlow>
    </div>
  );
}

export function RAGPipelineVisualizer(props: Readonly<InnerProps>) {
  return (
    <ReactFlowProvider>
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-500">
          RAG logic flow
        </p>
        <RAGPipelineFlow {...props} />
      </div>
    </ReactFlowProvider>
  );
}
