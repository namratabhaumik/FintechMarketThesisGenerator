import { sourcesLabel } from "../format";
import type { SourceResponse } from "../types";
import { Collapsible } from "./Collapsible";

// Collapsible list of source articles. Each row links out (when a URL is
// present) and shows the query-to-article retrieval similarity when stored.
export function SourcesList({ sources }: { sources: SourceResponse[] }) {
  if (sources.length === 0) return null;
  return (
    <Collapsible summary={sourcesLabel(sources)} defaultOpen>
      <ul className="space-y-1.5">
        {sources.map((s, i) => (
          <li key={i}>
            {s.url ? (
              <a
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-primary hover:text-primary/80 underline underline-offset-2 decoration-primary/30 leading-relaxed transition-colors"
              >
                {s.title}
              </a>
            ) : (
              s.title
            )}
            {s.similarity != null && (
              <span className="text-xs text-base-content/50">
                {` · ${Math.round(s.similarity * 100)}% relevant to your query`}
              </span>
            )}
          </li>
        ))}
      </ul>
    </Collapsible>
  );
}
