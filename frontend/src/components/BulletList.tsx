// Bulleted list with a colored leading dot per item, or a muted empty message
// when there are none. Ported from dom.ts's bulletList helper.
interface BulletListProps {
  items: string[];
  empty: string;
  dotClass?: string;
}

export function BulletList({ items, empty, dotClass = "bg-primary" }: BulletListProps) {
  if (items.length === 0) {
    return <p className="text-sm text-base-content/60">{empty}</p>;
  }
  return (
    <ul className="space-y-2">
      {items.map((item, i) => (
        <li key={i} className="flex items-center gap-2.5 text-sm">
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dotClass}`} />
          {item}
        </li>
      ))}
    </ul>
  );
}
