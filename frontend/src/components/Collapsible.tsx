import type { ReactNode } from "react";

// Collapsible <details>/<summary> styled as a daisyUI collapse. The body is
// wrapped in .collapse-content, matching daisyUI's expected structure.
// `boxed` controls whether it gets its own card background/border (top-level
// sections like Sources/Key Themes) or sits flush against a parent card,
// separated by a top border only (e.g. "Previous versions" nested in the
// action bar). `defaultOpen` sets the initial state only; the native <details>
// element owns toggling from there (uncontrolled).
interface CollapsibleProps {
  summary: string;
  defaultOpen?: boolean;
  boxed?: boolean;
  children: ReactNode;
}

export function Collapsible({
  summary,
  defaultOpen = false,
  boxed = true,
  children,
}: CollapsibleProps) {
  return (
    <details
      className={
        boxed
          ? "collapse collapse-arrow bg-base-200 border border-base-300 rounded-box"
          : "collapse collapse-arrow border-t border-base-300 rounded-none"
      }
      open={defaultOpen}
    >
      <summary
        className={
          boxed
            ? "collapse-title text-xs font-semibold uppercase tracking-widest text-base-content/60 hover:text-base-content transition-colors"
            : "collapse-title text-xs text-base-content/60 min-h-0 py-0"
        }
      >
        {summary}
      </summary>
      <div className="collapse-content text-sm">{children}</div>
    </details>
  );
}
