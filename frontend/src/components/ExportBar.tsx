import { useRef, useState } from "react";
import { copyToClipboard, downloadFile, jobToMarkdown, jobToText, shareableUrl } from "../export";
import type { JobResponse } from "../types";

// A ghost button whose label transiently changes (e.g. "Copied" for ~1.5s)
// after its action runs. The action receives a `flash` callback to trigger it.
function ExportButton({
  label,
  onClick,
}: {
  label: string;
  onClick: (flash: (msg: string) => void) => void;
}) {
  const [text, setText] = useState(label);
  const timer = useRef<number | undefined>(undefined);
  const flash = (msg: string) => {
    setText(msg);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setText(label), 1500);
  };
  return (
    <button type="button" className="btn btn-ghost btn-xs font-mono" onClick={() => onClick(flash)}>
      {text}
    </button>
  );
}

// Export / share actions: copy plain text, download Markdown, print-to-PDF, and
// copy a shareable link. Hidden when printing (print:hidden).
export function ExportBar({ job }: { job: JobResponse }) {
  return (
    <div className="print:hidden flex flex-wrap gap-2">
      <ExportButton
        label="Copy as text"
        onClick={(flash) => {
          void copyToClipboard(jobToText(job)).then((ok) => flash(ok ? "Copied" : "Copy failed"));
        }}
      />
      <ExportButton
        label="Download Markdown"
        onClick={(flash) => {
          downloadFile(`finthesis-${job.job_id}.md`, jobToMarkdown(job), "text/markdown");
          flash("Downloaded");
        }}
      />
      <ExportButton label="Export PDF" onClick={() => window.print()} />
      <ExportButton
        label="Copy link"
        onClick={(flash) => {
          void copyToClipboard(shareableUrl(job)).then((ok) =>
            flash(ok ? "Link copied" : "Copy failed"),
          );
        }}
      />
    </div>
  );
}
