// DOM construction helpers - 
// textContent (LLM/user strings are never interpreted as markup)

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

export function bulletList(items: string[], empty: string): HTMLElement {
  if (items.length === 0) return el("p", empty);
  const ul = el("ul");
  for (const item of items) ul.append(el("li", item));
  return ul;
}