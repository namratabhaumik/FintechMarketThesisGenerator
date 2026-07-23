import type { HallucinationAnalysis, JobResponse } from "../types";
import { Collapsible } from "./Collapsible";

// Hallucination analysis, shown only when invalid (non-existent) tools were
// found in the agent's tool calls.
export function Hallucination({ raw }: { raw: JobResponse["hallucination"] }) {
  const h = raw as HallucinationAnalysis | null | undefined;
  if (!h || !h.invalid_tools || h.invalid_tools.length === 0) return null;
  return (
    <section className="bg-error/10 border border-error/30 rounded-box px-6 py-4">
      <p className="text-sm font-semibold text-error mb-2">Hallucinations Detected</p>
      <Collapsible summary="Tool Call Analysis" defaultOpen>
        <div className="space-y-2 text-xs">
          {h.summary && <pre className="whitespace-pre-wrap">{h.summary}</pre>}
          <p>{`Invalid tools (do not exist): ${h.invalid_tools.join(", ")}`}</p>
        </div>
      </Collapsible>
    </section>
  );
}
