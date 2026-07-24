import type { ReactNode } from "react";

// Flat editorial section: an uppercase mono label over content, separated from
// what's above by a 1px rule. `collapsible` renders it as a native <details>
// with a chevron (secondary content like Source Articles); otherwise the
// content is always shown. `action` is a slot on the label row for later
// affordances (e.g. annotation controls). No hover-only behavior - the chevron
// and label are always visible, so it works on touch.
interface SectionProps {
  label: string;
  collapsible?: boolean;
  defaultOpen?: boolean;
  action?: ReactNode;
  children: ReactNode;
}

const labelClass = "text-xs font-mono text-base-content/60 uppercase tracking-widest";

function Chevron() {
  return (
    <svg
      className="w-3 h-3 text-base-content/40 transition-transform group-open:rotate-90"
      viewBox="0 0 12 12"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M4.5 2.5L8 6l-3.5 3.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Section({
  label,
  collapsible = false,
  defaultOpen = true,
  action,
  children,
}: SectionProps) {
  if (collapsible) {
    return (
      <details className="group border-t border-base-300 py-6" open={defaultOpen}>
        <summary className="flex items-center justify-between gap-3 cursor-pointer list-none [&::-webkit-details-marker]:hidden">
          <span className="flex items-center gap-2">
            <Chevron />
            <span className={labelClass}>{label}</span>
          </span>
          {action}
        </summary>
        <div className="text-sm mt-4">{children}</div>
      </details>
    );
  }
  return (
    <section className="border-t border-base-300 py-6">
      <div className="flex items-center justify-between gap-3 mb-4">
        <h3 className={labelClass}>{label}</h3>
        {action}
      </div>
      <div className="text-sm">{children}</div>
    </section>
  );
}
