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
type RouteBucket = "POLICY" | "IT" | "OUT_OF_SCOPE";

type StageData = { label: string; state: StageState };

type RagStageType = Node<StageData, "ragStage">;
type QueryOnlyType = Node<StageData, "queryOnly">;
type LlmOnlyType = Node<StageData, "llmOnly">;

const RAGStageNode = memo(function RAGStageNode({ data }: NodeProps<RagStageType>) {
  const { label, state } = data;
  return (
    <div
      className={[
        "min-w-[156px] max-w-[196px] rounded-2xl border px-3 py-2.5 text-center text-[10px] font-semibold leading-tight shadow-sm transition-all duration-300 dark:shadow-none sm:text-[11px]",
        state === "active" &&
          "scale-[1.02] border-sky-500 bg-sky-50 ring-2 ring-sky-400/70 dark:border-sky-500 dark:bg-sky-950/50 dark:ring-sky-400/40",
        state === "done" &&
          "border-emerald-500/70 bg-emerald-50/90 dark:border-emerald-600/60 dark:bg-emerald-950/35",
        state === "idle" && "border-zinc-200 bg-white/95 dark:border-zinc-700 dark:bg-zinc-900/85",
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
      <Handle
        id="top"
        type="target"
        position={Position.Top}
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
      <Handle
        id="bottom"
        type="source"
        position={Position.Bottom}
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
        "min-w-[156px] max-w-[196px] rounded-2xl border px-3 py-2.5 text-center text-[10px] font-semibold leading-tight shadow-sm transition-all duration-300 dark:shadow-none sm:text-[11px]",
        data.state === "active" &&
          "scale-[1.02] border-sky-500 bg-sky-50 ring-2 ring-sky-400/70 dark:border-sky-500 dark:bg-sky-950/50",
        data.state === "done" &&
          "border-emerald-500/70 bg-emerald-50/90 dark:border-emerald-600/60 dark:bg-emerald-950/35",
        data.state === "idle" && "border-zinc-200 bg-white/95 dark:border-zinc-700 dark:bg-zinc-900/85",
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
        "min-w-[170px] max-w-[210px] rounded-2xl border px-3 py-2.5 text-center text-[10px] font-semibold leading-tight shadow-sm transition-all duration-300 dark:shadow-none sm:text-[11px]",
        data.state === "active" &&
          "scale-[1.02] border-sky-500 bg-sky-50 ring-2 ring-sky-400/70 dark:border-sky-500 dark:bg-sky-950/50",
        data.state === "done" &&
          "border-emerald-500/70 bg-emerald-50/90 dark:border-emerald-600/60 dark:bg-emerald-950/35",
        data.state === "idle" && "border-zinc-200 bg-white/95 dark:border-zinc-700 dark:bg-zinc-900/85",
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
      <Handle
        id="bottom"
        type="target"
        position={Position.Bottom}
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
  _pipelineComplete: boolean,
  activeNodeIds: Set<string>,
  doneNodeIds: Set<string>,
): StageState => {
  if (activeNodeIds.has(nodeId)) return "active";
  if (doneNodeIds.has(nodeId)) return "done";
  return "idle";
};

function buildNodes(
  pipelineComplete: boolean,
  llmNodeLabel: string,
  routeBuckets: readonly RouteBucket[],
  activeNodeIds: Set<string>,
  doneNodeIds: Set<string>,
): Node[] {
  const s = (id: string) => stateForNode(id, pipelineComplete, activeNodeIds, doneNodeIds);
  const selectedBuckets = new Set<RouteBucket>(routeBuckets);
  const bucketState = (name: RouteBucket): StageState => {
    if (!selectedBuckets.has(name)) return "idle";
    return pipelineComplete ? "done" : "active";
  };
  return [
    {
      id: "query",
      type: "queryOnly",
      position: { x: 0, y: 120 },
      data: { label: "User Query", state: s("query") },
    },
    {
      id: "guardrail",
      type: "ragStage",
      position: { x: 185, y: 120 },
      data: { label: "Safety Guardrail\nHard Override", state: s("guardrail") },
    },
    {
      id: "router",
      type: "ragStage",
      position: { x: 410, y: 120 },
      data: { label: "Intent Router\nHard Route Lock", state: s("router") },
    },
    {
      id: "policy",
      type: "ragStage",
      position: { x: 640, y: 18 },
      data: { label: "POLICY\nHandbook / Process", state: bucketState("POLICY") },
    },
    {
      id: "it",
      type: "ragStage",
      position: { x: 640, y: 120 },
      data: { label: "IT\nSupport & IT Policies", state: bucketState("IT") },
    },
    {
      id: "out_of_scope",
      type: "ragStage",
      position: { x: 640, y: 222 },
      data: { label: "OUT OF SCOPE\nFallback / Escalation", state: bucketState("OUT_OF_SCOPE") },
    },
    {
      id: "output",
      type: "llmOnly",
      position: { x: 930, y: 120 },
      data: { label: `Final Response\n${llmNodeLabel}`, state: s("output") },
    },
  ];
}

function buildEdges(isLoading: boolean, isDark: boolean): Edge[] {
  const stroke = isDark ? "#71717a" : "#94a3b8";
  const blueStroke = isDark ? "#38bdf8" : "#2563eb";
  const greenStroke = isDark ? "#4ade80" : "#16a34a";
  const amberStroke = isDark ? "#fbbf24" : "#d97706";
  const animated = isLoading;
  return [
    { id: "e-q-g", source: "query", target: "guardrail", type: "smoothstep", animated, style: { stroke, strokeWidth: 2 } },
    { id: "e-g-r", source: "guardrail", target: "router", type: "smoothstep", animated, style: { stroke, strokeWidth: 2 } },
    { id: "e-r-p", source: "router", target: "policy", type: "smoothstep", animated, style: { stroke: blueStroke, strokeWidth: 2.4 } },
    { id: "e-r-i", source: "router", target: "it", type: "smoothstep", animated, style: { stroke: greenStroke, strokeWidth: 2.4 } },
    { id: "e-r-o", source: "router", target: "out_of_scope", type: "smoothstep", animated, style: { stroke: amberStroke, strokeWidth: 2.2 } },
    { id: "e-p-out", source: "policy", target: "output", type: "smoothstep", animated, style: { stroke: blueStroke, strokeWidth: 2 } },
    { id: "e-i-out", source: "it", target: "output", type: "smoothstep", animated, style: { stroke: greenStroke, strokeWidth: 2 } },
    { id: "e-o-out", source: "out_of_scope", target: "output", type: "smoothstep", animated, style: { stroke: amberStroke, strokeWidth: 2 } },
  ];
}

type InnerProps = {
  isLoading: boolean;
  pipelineComplete: boolean;
  isDark: boolean;
  /** Shown on the final node; use `\n` for a second line (e.g. OpenAI + model id). */
  llmNodeLabel: string;
  routeBuckets?: readonly RouteBucket[];
  activeNodes?: string[];
  doneNodes?: string[];
};

function RAGPipelineFlow({
  isLoading,
  pipelineComplete,
  isDark,
  llmNodeLabel,
  routeBuckets = [],
  activeNodes,
  doneNodes,
}: Readonly<InnerProps>) {
  const activeNodeIds = useMemo(() => new Set(activeNodes ?? []), [activeNodes]);
  const doneNodeIds = useMemo(() => new Set(doneNodes ?? []), [doneNodes]);
  const normalizedRouteBuckets = useMemo(
    () => [...new Set(routeBuckets)],
    [routeBuckets],
  );

  const nodes = useMemo(
    () => buildNodes(pipelineComplete, llmNodeLabel, normalizedRouteBuckets, activeNodeIds, doneNodeIds),
    [pipelineComplete, llmNodeLabel, normalizedRouteBuckets, activeNodeIds, doneNodeIds],
  );

  const edges = useMemo(() => buildEdges(isLoading, isDark), [isLoading, isDark]);

  return (
    <div className="h-[360px] w-full overflow-hidden rounded-2xl border border-zinc-200 bg-gradient-to-b from-white to-zinc-50 dark:border-zinc-800 dark:bg-zinc-950/50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypesFull}
        fitView
        fitViewOptions={{ padding: 0.14, maxZoom: 1.05, minZoom: 0.35 }}
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
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} className="opacity-35" />
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
        <div className="flex flex-wrap gap-2 text-[11px] text-zinc-600 dark:text-zinc-400">
          <span className="rounded-full border border-violet-200 bg-violet-50 px-2 py-1 dark:border-violet-900 dark:bg-violet-950/30">
            Guardrail
          </span>
          <span className="rounded-full border border-sky-200 bg-sky-50 px-2 py-1 dark:border-sky-900 dark:bg-sky-950/30">
            Policy
          </span>
          <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 dark:border-emerald-900 dark:bg-emerald-950/30">
            IT
          </span>
          <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-1 dark:border-amber-900 dark:bg-amber-950/30">
            Out of Scope
          </span>
        </div>
        <RAGPipelineFlow {...props} />
      </div>
    </ReactFlowProvider>
  );
}
