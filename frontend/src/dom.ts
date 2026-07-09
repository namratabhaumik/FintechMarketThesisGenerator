// DOM construction helpers - 
// textContent (LLM/user strings are never interpreted as markup)

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text?: string,
  className?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function bulletList(items: string[], empty: string, dotClass = "bg-primary"): HTMLElement {
  if (items.length === 0) return el("p", empty, "text-sm text-base-content/60");
  const ul = el("ul", undefined, "space-y-2");
  for (const item of items) {
    const li = el("li", undefined, "flex items-center gap-2.5 text-sm");
    li.append(el("span", undefined, `w-1.5 h-1.5 rounded-full flex-shrink-0 ${dotClass}`));
    li.append(document.createTextNode(item));
    ul.append(li);
  }
  return ul;
}