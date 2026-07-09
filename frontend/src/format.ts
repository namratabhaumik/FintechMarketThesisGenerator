// formatting helpers for dates and the source-articles label

import type { SourceResponse } from "./types";

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** "Jul 03, 2026" from an ISO date/datetime string. */
export function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const day = String(d.getDate()).padStart(2, "0");
  return `${MONTHS[d.getMonth()]} ${day}, ${d.getFullYear()}`;
}

/** "Source Articles (Jun 01, 2026 - Jul 03, 2026)", collapsing a single date. */
export function sourcesLabel(sources: SourceResponse[]): string {
  const times = sources
    .map((s) => s.published_at)
    .filter((p): p is string => Boolean(p))
    .map((p) => new Date(p).getTime())
    .filter((t) => !Number.isNaN(t));
  if (times.length === 0) return "Source Articles";
  const lo = fmtDate(new Date(Math.min(...times)).toISOString());
  const hi = fmtDate(new Date(Math.max(...times)).toISOString());
  const span = lo === hi ? lo : `${lo} - ${hi}`;
  return `Source Articles (${span})`;
}