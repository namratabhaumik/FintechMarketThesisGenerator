// Export/share: plain-text + Markdown document builders, plus the browser-API
// triggers (clipboard, file download, print) that consume them.

import { fmtDate, refusalSummaryMessage } from "./format";
import type { JobResponse } from "./types";

function header(job: JobResponse): string[] {
  const lines = [`FinThesis: ${job.query}`];
  const meta = [job.created_at ? fmtDate(job.created_at) : null, `job ${job.job_id}`]
    .filter(Boolean)
    .join(" · ");
  lines.push(meta);
  return lines;
}

/** Plain-text rendering for clipboard/paste into email, Slack, etc. */
export function jobToText(job: JobResponse): string {
  const t = job.thesis;
  if (!t) return header(job).join("\n");
  const lines = [
    ...header(job),
    "",
    `Investment Score: ${t.opportunity_score}/5`,
    `Confidence: ${Math.round(t.confidence_level * 100)}%`,
    `Recommendation: ${t.recommendation}`,
    "",
    "Key Themes:",
    ...t.key_themes.map((x) => `- ${x}`),
    "",
    "Risks:",
    ...t.risks.map((x) => `- ${x}`),
    "",
    "Investment Signals:",
    ...t.investment_signals.map((x) => `- ${x}`),
  ];
  if (job.sources.length > 0) {
    lines.push("", "Sources:");
    lines.push(...job.sources.map((s) => `- ${s.title}${s.url ? ` (${s.url})` : ""}`));
  }
  if (t.raw_output) {
    lines.push("", "Summary:", t.summary_status === "refused" ? refusalSummaryMessage(t) : t.raw_output);
  }
  return lines.join("\n");
}

/** Markdown rendering for the downloadable .md file. */
export function jobToMarkdown(job: JobResponse): string {
  const t = job.thesis;
  const [title, meta] = header(job);
  if (!t) return `# ${title}\n\n${meta}\n`;
  const lines = [
    `# ${title}`,
    "",
    meta,
    "",
    `**Investment Score:** ${t.opportunity_score}/5  `,
    `**Confidence:** ${Math.round(t.confidence_level * 100)}%  `,
    `**Recommendation:** ${t.recommendation}`,
    "",
    "## Key Themes",
    ...t.key_themes.map((x) => `- ${x}`),
    "",
    "## Risks",
    ...t.risks.map((x) => `- ${x}`),
    "",
    "## Investment Signals",
    ...t.investment_signals.map((x) => `- ${x}`),
  ];
  if (job.sources.length > 0) {
    lines.push("", "## Sources");
    lines.push(...job.sources.map((s) => (s.url ? `- [${s.title}](${s.url})` : `- ${s.title}`)));
  }
  if (t.raw_output) {
    lines.push("", "## Summary", "", t.summary_status === "refused" ? refusalSummaryMessage(t) : t.raw_output);
  }
  return lines.join("\n");
}

/** Best-effort clipboard write; returns whether it succeeded. */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/** Trigger a browser download of `content` as a file named `filename`. */
export function downloadFile(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** The current run's shareable URL (requires the viewer to be signed in). */
export function shareableUrl(job: JobResponse): string {
  return `${location.origin}${location.pathname}?job_id=${encodeURIComponent(job.job_id)}`;
}