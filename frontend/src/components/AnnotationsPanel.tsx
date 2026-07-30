import { useState } from "react";
import { fmtDate } from "../format";
import type { AnnotationResponse } from "../types";
import type { AnnotationThread } from "../useAnnotations";

// Author label. Falls back to a short id when a profile row has not caught up
// with a fresh signup - a nameless author must not blank out the comment.
function authorName(a: AnnotationResponse): string {
  return a.author.display_name || a.author.user_id.slice(0, 8);
}

function Avatar({ a }: { a: AnnotationResponse }) {
  const url = a.author.avatar_url;
  const name = authorName(a);
  if (url) {
    return <img src={url} alt="" className="w-5 h-5 rounded-full flex-shrink-0" />;
  }
  return (
    <span className="w-5 h-5 rounded-full bg-base-300 text-[9px] flex items-center justify-center flex-shrink-0 uppercase">
      {name.slice(0, 2)}
    </span>
  );
}

function Comment({
  a,
  canEdit,
  onEdit,
  onDelete,
}: {
  a: AnnotationResponse;
  canEdit: boolean;
  onEdit: (body: string) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(a.body);

  return (
    <div className="text-xs">
      <div className="flex items-center gap-2 mb-1">
        <Avatar a={a} />
        <span className="font-medium truncate">{authorName(a)}</span>
        <span className="text-base-content/40 font-mono text-[10px] ml-auto flex-shrink-0">
          {a.created_at ? fmtDate(a.created_at) : ""}
        </span>
      </div>

      {editing ? (
        <div className="space-y-2">
          <textarea
            className="w-full bg-base-100 border border-base-300 rounded-field px-2 py-1.5 text-xs"
            rows={3}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <div className="flex gap-2">
            <button
              type="button"
              className="btn btn-primary btn-xs"
              disabled={!draft.trim()}
              onClick={() => {
                onEdit(draft.trim());
                setEditing(false);
              }}
            >
              Save
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={() => {
                setDraft(a.body);
                setEditing(false);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <>
          <p className="text-base-content/80 whitespace-pre-wrap leading-relaxed">{a.body}</p>
          {canEdit && (
            <div className="flex gap-2 mt-1">
              <button
                type="button"
                className="text-[10px] text-base-content/50 hover:text-base-content"
                onClick={() => setEditing(true)}
              >
                Edit
              </button>
              <button
                type="button"
                className="text-[10px] text-base-content/50 hover:text-error"
                onClick={onDelete}
              >
                Delete
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Thread({
  thread,
  userId,
  isActive,
  onSelect,
  onReply,
  onEdit,
  onDelete,
  onResolve,
}: {
  thread: AnnotationThread;
  userId?: string | null;
  isActive: boolean;
  onSelect: () => void;
  onReply: (body: string) => void;
  onEdit: (id: string, body: string) => void;
  onDelete: (id: string) => void;
  onResolve: (resolution: "accepted" | "rejected" | null, reason?: string) => void;
}) {
  const { root, replies } = thread;
  const [replying, setReplying] = useState(false);
  const [draft, setDraft] = useState("");
  // Crossing asks for a reason, which is posted as a reply so it stays the
  // rejecter's own words rather than an edit to someone else's note.
  const [rejecting, setRejecting] = useState(false);
  const resolved = root.resolution != null;

  const submitReply = () => {
    if (!draft.trim()) return;
    onReply(draft.trim());
    setDraft("");
    setReplying(false);
  };

  return (
    <div
      className={`border rounded-box p-3 space-y-2 cursor-pointer transition-colors ${
        isActive ? "border-primary/50 bg-primary/5" : "border-base-300 hover:border-base-content/20"
      }`}
      onClick={onSelect}
    >
      {root.quote && (
        <p className="text-[11px] italic text-base-content/50 border-l-2 border-primary/40 pl-2 line-clamp-3">
          {root.quote}
        </p>
      )}

      <Comment
        a={root}
        canEdit={root.author.user_id === userId}
        onEdit={(body) => onEdit(root.id, body)}
        onDelete={() => onDelete(root.id)}
      />

      {replies.length > 0 && (
        <div className="pl-3 border-l border-base-300 space-y-2">
          {replies.map((r) => (
            <Comment
              key={r.id}
              a={r}
              canEdit={r.author.user_id === userId}
              onEdit={(body) => onEdit(r.id, body)}
              onDelete={() => onDelete(r.id)}
            />
          ))}
        </div>
      )}

      {rejecting && (
        <div className="space-y-2" onClick={(e) => e.stopPropagation()}>
          <textarea
            className="w-full bg-base-100 border border-base-300 rounded-field px-2 py-1.5 text-xs"
            rows={2}
            placeholder="Why are you dismissing this?"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <div className="flex gap-2">
            <button
              type="button"
              className="btn btn-primary btn-xs"
              disabled={!draft.trim()}
              onClick={() => {
                onResolve("rejected", draft.trim());
                setDraft("");
                setRejecting(false);
              }}
            >
              Dismiss
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={() => {
                setDraft("");
                setRejecting(false);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {replying && (
        <div className="space-y-2" onClick={(e) => e.stopPropagation()}>
          <textarea
            className="w-full bg-base-100 border border-base-300 rounded-field px-2 py-1.5 text-xs"
            rows={2}
            placeholder="Reply..."
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <div className="flex gap-2">
            <button
              type="button"
              className="btn btn-primary btn-xs"
              disabled={!draft.trim()}
              onClick={submitReply}
            >
              Reply
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={() => {
                setDraft("");
                setReplying(false);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {!replying && !rejecting && (
        <div
          className="flex items-center gap-3 pt-1 border-t border-base-300"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            className="text-[10px] text-base-content/50 hover:text-base-content"
            onClick={() => setReplying(true)}
          >
            Reply
          </button>
          {resolved ? (
            <>
              <span
                className={`text-[10px] font-mono ${
                  root.resolution === "accepted" ? "text-primary" : "text-base-content/50"
                }`}
              >
                {root.resolution === "accepted" ? "Accepted" : "Dismissed"}
              </span>
              <button
                type="button"
                className="text-[10px] text-base-content/50 hover:text-base-content ml-auto"
                onClick={() => onResolve(null)}
              >
                Reopen
              </button>
            </>
          ) : (
            <div className="flex items-center gap-2 ml-auto">
              {/* Tick = all good. Cross = dismissed, with a reason. */}
              <button
                type="button"
                aria-label="Accept this note"
                title="Accept"
                className="btn btn-ghost btn-xs text-primary"
                onClick={() => onResolve("accepted")}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path
                    d="M4 12.5l5 5L20 6.5"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              <button
                type="button"
                aria-label="Dismiss this note with a reason"
                title="Dismiss"
                className="btn btn-ghost btn-xs text-error/70 hover:text-error"
                onClick={() => setRejecting(true)}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path
                    d="M6 6l12 12M18 6L6 18"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function AnnotationsPanel({
  threads,
  loading,
  error,
  userId,
  activeId,
  onSelect,
  onReply,
  onEdit,
  onDelete,
  onResolve,
  onClose,
}: {
  threads: AnnotationThread[];
  loading: boolean;
  error: string;
  userId?: string | null;
  activeId: string | null;
  onSelect: (id: string | null) => void;
  onReply: (parentId: string, body: string) => void;
  onEdit: (id: string, body: string) => void;
  onDelete: (id: string) => void;
  onResolve: (id: string, resolution: "accepted" | "rejected" | null, reason?: string) => void;
  onClose?: () => void;
}) {
  const [tab, setTab] = useState<"open" | "resolved">("open");
  const open = threads.filter((t) => t.root.resolution == null);
  const resolved = threads.filter((t) => t.root.resolution != null);
  const shown = tab === "open" ? open : resolved;

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-base-300">
        <h2 className="text-sm font-semibold">
          Annotations{" "}
          <span className="text-xs font-normal text-base-content/50">{open.length}</span>
        </h2>
        {onClose && (
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            onClick={onClose}
            aria-label="Close annotations"
          >
            Close
          </button>
        )}
      </div>

      <div className="flex gap-1 p-2 border-b border-base-300">
        {(["open", "resolved"] as const).map((t) => (
          <button
            key={t}
            type="button"
            className={`flex-1 px-3 py-1.5 text-xs rounded-field capitalize transition-colors ${
              tab === t ? "bg-base-300 text-base-content" : "text-base-content/60 hover:bg-base-300/50"
            }`}
            onClick={() => setTab(t)}
          >
            {t} ({t === "open" ? open.length : resolved.length})
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {error && (
          <p className="text-xs text-error border border-error/30 bg-error/10 rounded-field px-3 py-2">
            {error}
          </p>
        )}
        {!error && loading && (
          <p className="text-xs text-base-content/50 text-center py-6">Loading annotations...</p>
        )}
        {!error && !loading && shown.length === 0 && (
          <p className="text-xs text-base-content/50 text-center py-6">
            {tab === "open"
              ? "No notes yet. Select text in the summary to add one."
              : "Nothing resolved yet."}
          </p>
        )}
        {shown.map((thread) => (
          <Thread
            key={thread.root.id}
            thread={thread}
            userId={userId}
            isActive={activeId === thread.root.id}
            onSelect={() => onSelect(activeId === thread.root.id ? null : thread.root.id)}
            onReply={(body) => onReply(thread.root.id, body)}
            onEdit={onEdit}
            onDelete={onDelete}
            onResolve={(resolution, reason) => onResolve(thread.root.id, resolution, reason)}
          />
        ))}
      </div>
    </div>
  );
}
