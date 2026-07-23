import { Fragment } from "react";
import { isNoOpRound } from "../format";
import type { ExecutionEvent } from "../types";
import { Collapsible } from "./Collapsible";

// Execution trace: the tools that actually ran during refinement, with reasons
// and per-round change notes.
export function ExecutionTrace({ log }: { log: unknown[] }) {
  if (log.length === 0) return null;
  return (
    <Collapsible summary="Execution Trace">
      <div className="space-y-2">
        <p className="text-xs text-base-content/60">
          Tools that actually executed during refinement:
        </p>
        <ol className="space-y-1.5">
          {log.map((raw, idx) => {
            const event = raw as ExecutionEvent;
            // Rounds logged before the backend emitted an explicit "No changes
            // made" line get the same label retroactively, from the stored diff.
            const showNoOp =
              isNoOpRound(event) &&
              !(event.changes ?? []).some((c) => c.startsWith("No changes made"));
            return (
              <Fragment key={idx}>
                <li className="text-xs font-mono text-base-content/60">
                  <span className="text-primary">{`${idx + 1}. `}</span>
                  {`${event.tool_name ?? "unknown"} — ${event.status ?? "unknown"}`}
                </li>
                {event.refinement_number && (
                  <li className="text-[10px] font-mono text-base-content/60">
                    {`Refinement #${event.refinement_number}`}
                  </li>
                )}
                {event.reason && (
                  <li className="text-[10px] font-mono text-base-content/60">
                    {`Reason: ${event.reason}`}
                  </li>
                )}
                {showNoOp && (
                  <li className="text-[10px] font-mono text-base-content/60">
                    No changes made this round
                  </li>
                )}
                {(event.changes ?? []).map((change, ci) => (
                  <li key={ci} className="text-[10px] font-mono text-base-content/60">
                    {change}
                  </li>
                ))}
              </Fragment>
            );
          })}
        </ol>
      </div>
    </Collapsible>
  );
}
