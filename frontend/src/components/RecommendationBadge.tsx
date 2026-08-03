// Pill badge for a recommendation verdict, color-coded by value. The class
// helper is exported separately for the few callers (list rows, compare modal)
// that need the classes without the wrapping span.
export function recommendationBadgeClass(rec: string): string {
  const base = "inline-flex items-center px-2.5 py-1 rounded-md text-xs font-semibold border";
  if (rec === "Pursue") return `${base} bg-primary/15 text-primary border-primary/30`;
  if (rec === "Skip") return `${base} bg-error/15 text-error border-error/30`;
  return `${base} bg-accent/15 text-accent border-accent/30`; // Investigate and any other value
}

export function RecommendationBadge({ recommendation }: { recommendation: string }) {
  return <span className={recommendationBadgeClass(recommendation)}>{recommendation}</span>;
}
