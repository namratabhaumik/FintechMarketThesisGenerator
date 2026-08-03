import { useEffect, useRef, useState } from "react";
import { offsetsFromSelection } from "../anchoring";

export interface Draft {
  start: number;
  end: number;
  quote: string;
  /** Viewport coordinates of the selection, for positioning. */
  x: number;
  y: number;
}

/**
 * Watches for a text selection inside `rootRef` and reports it, so the caller
 * can offer to turn it into a note.
 *
 * Listens to selectionchange rather than mouseup: touch selection on iOS and
 * Android is driven by drag handles that never fire a mouseup on the text, so
 * a mouse-only listener would leave the feature dead on phones. The debounce
 * lets a drag settle before the button appears under the user's finger.
 */
export function useSelectionDraft(
  rootRef: React.RefObject<HTMLElement | null>,
  enabled: boolean,
) {
  const [draft, setDraft] = useState<Draft | null>(null);
  // Once the composer opens it autofocuses a textarea, which moves the caret
  // out of the document and fires selectionchange with nothing selected. Without
  // this lock that event clears the draft and unmounts the composer the instant
  // it appears - the act of opening it would destroy what it is anchored to.
  const locked = useRef(false);

  useEffect(() => {
    if (!enabled) return;
    let timer: number | undefined;

    const onSelectionChange = () => {
      if (locked.current) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        const root = rootRef.current;
        if (!root) return setDraft(null);
        const found = offsetsFromSelection(root);
        if (!found) return setDraft(null);

        const selection = window.getSelection();
        const rect = selection?.getRangeAt(0).getBoundingClientRect();
        if (!rect || (rect.width === 0 && rect.height === 0)) return setDraft(null);
        setDraft({
          start: found.start,
          end: found.end,
          quote: found.quote,
          x: rect.left + rect.width / 2,
          y: rect.top,
        });
      }, 150);
    };

    document.addEventListener("selectionchange", onSelectionChange);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("selectionchange", onSelectionChange);
    };
  }, [rootRef, enabled]);

  return {
    draft,
    /** Freeze the current draft while the composer has focus. */
    lockDraft: () => {
      locked.current = true;
    },
    clearDraft: () => {
      locked.current = false;
      setDraft(null);
    },
  };
}

/**
 * The floating control itself: an "Add note" button over the selection, which
 * expands into a composer. Fixed-positioned against viewport coordinates, so it
 * tracks the selection without needing the document's scroll offset.
 */
export function SelectionPopover({
  draft,
  onCompose,
  onSubmit,
  onDismiss,
}: {
  draft: Draft;
  /** Called when the composer opens, so the draft stops tracking the selection. */
  onCompose: () => void;
  onSubmit: (body: string) => void;
  onDismiss: () => void;
}) {
  const [composing, setComposing] = useState(false);
  const [body, setBody] = useState("");

  // A fresh selection always starts collapsed, never mid-compose.
  useEffect(() => {
    setComposing(false);
    setBody("");
  }, [draft.start, draft.end]);

  // Clamp into the viewport so a selection near an edge cannot push the control
  // off-screen (most likely on a phone).
  const width = composing ? 260 : 104;
  const left = Math.min(Math.max(draft.x - width / 2, 8), window.innerWidth - width - 8);
  // Above the selection when there is room, otherwise below it.
  const above = draft.y > 120;
  const top = above ? draft.y - (composing ? 150 : 44) : draft.y + 28;

  return (
    <div
      className="fixed z-[80] print:hidden"
      style={{ left, top, width }}
      // Keep the selection alive: losing it would drop the anchor offsets.
      onMouseDown={(e) => e.preventDefault()}
    >
      {composing ? (
        <div className="bg-base-200 border border-base-300 rounded-box shadow-xl p-2 space-y-2">
          <p className="text-[10px] italic text-base-content/50 line-clamp-2 border-l-2 border-primary/40 pl-2">
            {draft.quote}
          </p>
          <textarea
            autoFocus
            rows={3}
            className="w-full bg-base-100 border border-base-300 rounded-field px-2 py-1.5 text-xs focus:outline-none"
            placeholder="Add a note..."
            value={body}
            onChange={(e) => setBody(e.target.value)}
            onKeyDown={(e) => {
              // Enter submits; Shift+Enter is a newline.
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (body.trim()) onSubmit(body.trim());
              }
              if (e.key === "Escape") onDismiss();
            }}
          />
          <div className="flex gap-2">
            <button
              type="button"
              className="btn btn-primary btn-xs"
              disabled={!body.trim()}
              onClick={() => onSubmit(body.trim())}
            >
              Save
            </button>
            <button type="button" className="btn btn-ghost btn-xs" onClick={onDismiss}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          className="btn btn-primary btn-xs shadow-lg gap-1.5 w-full"
          onClick={() => {
            onCompose();
            setComposing(true);
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Add note
        </button>
      )}
    </div>
  );
}
