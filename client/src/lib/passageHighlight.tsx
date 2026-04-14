import type { ReactNode } from "react";

/** Lowercase tokens from the user query used for lexical overlap highlighting. */
export function extractQueryTerms(query: string): Set<string> {
  const raw = query.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];
  const terms = new Set<string>();
  for (const w of raw) {
    const t = w.replace(/^'+|'+$/g, "");
    if (t.length >= 2) terms.add(t);
  }
  return terms;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Highlights chunk tokens that appear in `terms` (case-insensitive).
 * Uses word-like runs; not true semantic match, but reads as “overlap” in the explainer UI.
 */
export function highlightPassageText(text: string, terms: Set<string>): ReactNode {
  if (terms.size === 0) return text;

  const sorted = [...terms].filter((t) => t.length >= 2).sort((a, b) => b.length - a.length);
  if (sorted.length === 0) return text;

  const re = new RegExp(`(${sorted.map(escapeRegExp).join("|")})`, "giu");
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  for (;;) {
    m = re.exec(text);
    if (m === null) break;
    if (m.index > last) out.push(text.slice(last, m.index));
    const piece = m[0];
    const lower = piece.toLowerCase();
    if (terms.has(lower)) {
      out.push(
        <mark
          key={`h-${key++}`}
          className="rounded-sm bg-amber-200/90 px-0.5 text-inherit dark:bg-amber-500/35"
        >
          {piece}
        </mark>,
      );
    } else {
      out.push(piece);
    }
    last = m.index + piece.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out.length === 0 ? text : out;
}
