// Mapping between a section's plain text and its rendered DOM, so a stored
// annotation range can be found again and painted.
//
// Offsets index the section's PLAIN TEXT - the concatenation of its text nodes
// in document order - not the DOM, so they stay valid regardless of how the
// text happens to be split across elements. Annotations are version-pinned and
// a version's text is frozen once superseded, so an offset can never drift
// under a note.
//
// Highlighting wraps ranges in <mark> rather than using the CSS Custom
// Highlight API: that API is too new to rely on across browsers, and <mark>
// works everywhere and stays visible in print and copied text.

/** Root under which offsets are measured. Marked with data-annotation-section. */
export const SECTION_ATTR = "data-annotation-section";

/** Applied to every <mark> we create, so cleanup only removes our own. */
const MARK_ATTR = "data-annotation-mark";

/** Walk a section's text nodes in document order. */
function textNodes(root: HTMLElement): Text[] {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const out: Text[] = [];
  let node = walker.nextNode();
  while (node) {
    out.push(node as Text);
    node = walker.nextNode();
  }
  return out;
}

/** The plain text an annotation's offsets are measured against. */
export function sectionText(root: HTMLElement): string {
  return textNodes(root)
    .map((n) => n.data)
    .join("");
}

/** Convert a plain-text offset into the (node, offset) pair a Range needs. */
function locate(nodes: Text[], offset: number): { node: Text; offset: number } | null {
  let seen = 0;
  for (const node of nodes) {
    const end = seen + node.data.length;
    // <= so an offset landing exactly on a boundary resolves to the end of this
    // node rather than falling through and being reported as out of range.
    if (offset <= end) return { node, offset: offset - seen };
    seen = end;
  }
  return null;
}

/** Build a DOM Range for a plain-text offset pair, or null if out of range. */
export function rangeFromOffsets(
  root: HTMLElement,
  start: number,
  end: number,
): Range | null {
  if (start >= end) return null;
  const nodes = textNodes(root);
  const from = locate(nodes, start);
  const to = locate(nodes, end);
  if (!from || !to) return null;
  const range = document.createRange();
  try {
    range.setStart(from.node, from.offset);
    range.setEnd(to.node, to.offset);
  } catch {
    // Offsets can outrun the node if the text changed unexpectedly; a missing
    // highlight is recoverable, a thrown render is not.
    return null;
  }
  return range;
}

/** Where the current selection sits in a section's plain text, if it is inside
 * one. Returns null for a collapsed selection or one spanning outside. */
export function offsetsFromSelection(
  root: HTMLElement,
): { start: number; end: number; quote: string } | null {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  if (!root.contains(range.commonAncestorContainer)) return null;

  // Measure by walking to the boundary nodes rather than trusting DOM indices,
  // which are relative to a single node.
  const nodes = textNodes(root);
  let start = -1;
  let end = -1;
  let seen = 0;
  for (const node of nodes) {
    if (node === range.startContainer) start = seen + range.startOffset;
    if (node === range.endContainer) end = seen + range.endOffset;
    seen += node.data.length;
  }
  if (start < 0 || end < 0 || start >= end) return null;

  const quote = range.toString();
  if (!quote.trim()) return null;
  return { start, end, quote };
}

/** Remove every highlight this module added, restoring the original text nodes. */
export function clearHighlights(root: HTMLElement): void {
  const marks = root.querySelectorAll(`mark[${MARK_ATTR}]`);
  marks.forEach((mark) => {
    const parent = mark.parentNode;
    if (!parent) return;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    // Re-join the text nodes the unwrap left adjacent, so the next pass measures
    // the same offsets it would have before any highlighting.
    parent.normalize();
  });
}

/**
 * Wrap a range in <mark>. Splits the range per text node rather than using
 * surroundContents, which throws whenever a selection crosses an element
 * boundary (a highlight spanning two sentences in different spans, say).
 */
function wrapRange(range: Range, onClick: () => void, active: boolean): void {
  const root = range.commonAncestorContainer;
  const container =
    root.nodeType === Node.ELEMENT_NODE ? (root as Element) : root.parentElement;
  if (!container) return;

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const targets: Text[] = [];
  let node = walker.nextNode();
  while (node) {
    if (range.intersectsNode(node)) targets.push(node as Text);
    node = walker.nextNode();
  }

  for (const text of targets) {
    const from = text === range.startContainer ? range.startOffset : 0;
    const to = text === range.endContainer ? range.endOffset : text.data.length;
    if (to <= from) continue;

    // Isolate the covered slice so the <mark> wraps exactly it.
    const middle = text.splitText(from);
    if (to - from < middle.data.length) middle.splitText(to - from);

    const mark = document.createElement("mark");
    mark.setAttribute(MARK_ATTR, "");
    mark.className = active
      ? "bg-primary/30 text-base-content rounded-sm cursor-pointer"
      : "bg-primary/15 text-base-content rounded-sm cursor-pointer hover:bg-primary/25";
    middle.parentNode?.replaceChild(mark, middle);
    mark.appendChild(middle);
    mark.addEventListener("click", (e) => {
      e.stopPropagation();
      onClick();
    });
  }
}

export interface HighlightSpec {
  id: string;
  start: number;
  end: number;
  active: boolean;
}

/**
 * Paint a section's highlights. Clears previous ones first, so this is
 * idempotent and safe to re-run whenever the annotation set changes.
 *
 * Applied longest-first: a shorter range nested inside a longer one would
 * otherwise split the text nodes the longer one still needs.
 */
export function applyHighlights(
  root: HTMLElement,
  specs: HighlightSpec[],
  onSelect: (id: string) => void,
): void {
  clearHighlights(root);
  const ordered = [...specs].sort((a, b) => b.end - b.start - (a.end - a.start));
  for (const spec of ordered) {
    const range = rangeFromOffsets(root, spec.start, spec.end);
    if (!range) continue;
    wrapRange(range, () => onSelect(spec.id), spec.active);
  }
}
