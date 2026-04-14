import type { Citation } from "@employee-handbook/shared";
import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

export type EmbeddingPoint = { name: string; x: number; y: number; z: number; kind: "query" | "chunk" };

/** Mock 2D projection: query at origin; top citations placed by angle + distance from similarity. */
function buildEmbeddingPoints(
  queryLabel: string,
  citations: Citation[],
  topK: number,
): { query: EmbeddingPoint[]; chunks: EmbeddingPoint[] } {
  const top = [...citations]
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  const qShort =
    queryLabel.trim().length > 0
      ? queryLabel.trim().slice(0, 48) + (queryLabel.trim().length > 48 ? "…" : "")
      : "User query";

  const query: EmbeddingPoint[] = [
    {
      name: qShort,
      x: 0,
      y: 0,
      z: 120,
      kind: "query",
    },
  ];

  const chunks: EmbeddingPoint[] = top.map((c, i) => {
    const s = Number.isFinite(c.score) ? Math.min(Math.max(c.score, 0), 1) : 0.5;
    const angle = Math.PI / 2 + (i * (2 * Math.PI)) / Math.max(top.length, 1);
    const dist = 0.12 + (1 - s) * 0.88;
    return {
      name: `Chunk ${i + 1}`,
      x: Math.cos(angle) * dist,
      y: Math.sin(angle) * dist,
      z: 60 + s * 40,
      kind: "chunk",
    };
  });

  return { query, chunks };
}

type Props = {
  queryLabel: string;
  citations: Citation[];
  isDark: boolean;
  className?: string;
};

export function EmbeddingSpaceScatter({ queryLabel, citations, isDark, className }: Props) {
  const { query, chunks } = useMemo(
    () => buildEmbeddingPoints(queryLabel, citations, 3),
    [queryLabel, citations],
  );

  const grid = isDark ? "#3f3f46" : "#d4d4d8";
  const tick = isDark ? "#a1a1aa" : "#52525b";
  const axis = isDark ? "#52525b" : "#a1a1aa";
  const tooltip = isDark
    ? {
        backgroundColor: "#18181b",
        border: "1px solid #3f3f46",
        borderRadius: "8px",
        color: "#f4f4f5",
      }
    : {
        backgroundColor: "#ffffff",
        border: "1px solid #e4e4e7",
        borderRadius: "8px",
        color: "#18181b",
      };

  const hasChunks = chunks.length > 0;

  return (
    <div className={className}>
      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-500">
        Embedding space (2D projection)
      </p>
      <p className="mb-2 text-[11px] leading-snug text-zinc-500 dark:text-zinc-500">
        Mock layout from retrieval scores — higher similarity sits closer to the query.
      </p>
      <div className="h-52 w-full rounded-lg border border-zinc-200 bg-zinc-50/80 dark:border-zinc-800 dark:bg-zinc-950/40">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={grid} />
            <XAxis
              type="number"
              dataKey="x"
              domain={[-1.15, 1.15]}
              tick={{ fill: tick, fontSize: 10 }}
              stroke={axis}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[-1.15, 1.15]}
              tick={{ fill: tick, fontSize: 10 }}
              stroke={axis}
            />
            <ZAxis type="number" dataKey="z" range={[70, 140]} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={tooltip}
              formatter={(value, name, item) => {
                const payload = item?.payload as EmbeddingPoint | undefined;
                if (payload?.kind === "query") return [String(value), "Query axis"];
                return [String(value), name];
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: "11px", paddingTop: "4px" }}
              formatter={(value) => <span className="text-zinc-600 dark:text-zinc-400">{value}</span>}
            />
            <Scatter
              name="User query"
              data={query}
              fill="#ef4444"
              fillOpacity={0.92}
              shape="circle"
            />
            {hasChunks ? (
              <Scatter
                name="Top citations"
                data={chunks}
                fill="#22c55e"
                fillOpacity={0.9}
                shape="circle"
              />
            ) : null}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
