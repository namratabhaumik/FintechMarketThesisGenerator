// formatting helpers for dates and the source-articles label

import type { ExecutionEvent, SourceResponse, ThesisResponse } from "./types";

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/**
 * "Jul 03, 2026" from an ISO date/datetime string, shown in the viewer's
 * local timezone.
 */
export function fmtDate(iso: string): string {
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  const d = dateOnly
    ? new Date(Number(dateOnly[1]), Number(dateOnly[2]) - 1, Number(dateOnly[3]))
    : new Date(iso);
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
  const toDateOnly = (t: number) => new Date(t).toISOString().slice(0, 10);
  const lo = fmtDate(toDateOnly(Math.min(...times)));
  const hi = fmtDate(toDateOnly(Math.max(...times)));
  const span = lo === hi ? lo : `${lo} - ${hi}`;
  return `Source Articles (${span})`;
}

/**
 * True when an executed refinement round left the thesis untouched. Newer
 * rounds carry an explicit "No changes made" line from the backend
 * (_diff_thesis); rounds stored before that line existed are recognized by
 * their change list. Both strings must stay in sync with _diff_thesis.
 */
export function isNoOpRound(event: ExecutionEvent | undefined): boolean {
  if (!event || event.status !== "executed") return false;
  const changes = event.changes ?? [];
  return changes.every(
    (c) =>
      c === "Score, confidence, recommendation unchanged" ||
      c.startsWith("No changes made"),
  );
}

/**
 * The friendly stand-in for `raw_output` when the summarizer refused to
 * write a narrative, naming what's still grounded so the reader has
 * somewhere to look instead of an opaque "REFUSED:" sentinel. Wording
 * differs by refusal_reason.
 */
export function refusalSummaryMessage(thesis: ThesisResponse): string {
  const dims = [
    [thesis.key_themes.length, "theme"],
    [thesis.risks.length, "risk"],
    [thesis.investment_signals.length, "signal"],
  ] as const;
  const parts = dims.map(([n, label]) => `${n} ${label}${n === 1 ? "" : "s"}`);
  const dimsList = parts.length > 1
    ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`
    : parts[0];
  const reason = thesis.refusal_reason === "llm_judgment"
    ? "The sources touch on related fintech topics but don't specifically address this query"
    : "The sources didn't give us enough to write a reliable narrative for this query";
  return `${reason} - but the ${dimsList} below are grounded in the same sources and worth reviewing directly.`;
}