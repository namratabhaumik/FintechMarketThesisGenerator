// Annotation state for one thesis version: loading, threading, and the
// mutations the panel exposes.
//
// Annotations are version-pinned, so this always fetches a single version - the
// one on screen. Switching versions is a different set, not a migration.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  createAnnotation,
  deleteAnnotation,
  listAnnotations,
  setAnnotationResolution,
  updateAnnotation,
} from "./api";
import type {
  AnnotationCreateRequest,
  AnnotationResolution,
  AnnotationResponse,
  AnnotationSection,
} from "./types";

/** A root annotation with its replies, which is how the panel renders. */
export interface AnnotationThread {
  root: AnnotationResponse;
  replies: AnnotationResponse[];
}

export interface NewAnnotation {
  section: AnnotationSection;
  start: number;
  end: number;
  quote: string;
  body: string;
}

export function useAnnotations(jobId: string | undefined, version: number) {
  const [items, setItems] = useState<AnnotationResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const describe = (err: unknown, fallback: string) =>
    err instanceof ApiError ? err.message : fallback;

  const load = useCallback(async () => {
    if (!jobId) {
      setItems([]);
      return;
    }
    setLoading(true);
    try {
      setItems(await listAnnotations(jobId, version));
      setError("");
    } catch (err) {
      console.error("Failed to load annotations", err);
      // Kept distinct from "no annotations": an empty panel must not imply the
      // thesis has no notes when the request simply failed.
      setError(describe(err, "Could not load annotations."));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [jobId, version]);

  useEffect(() => {
    void load();
  }, [load]);

  // Roots in document order (by anchor position), each with its replies in the
  // order they were written. Sorting by offset makes the panel track the
  // document rather than creation time.
  const threads = useMemo<AnnotationThread[]>(() => {
    const roots = items
      .filter((a) => !a.parent_id)
      .sort((a, b) => (a.start_offset ?? 0) - (b.start_offset ?? 0));
    return roots.map((root) => ({
      root,
      replies: items.filter((a) => a.parent_id === root.id),
    }));
  }, [items]);

  const add = async (draft: NewAnnotation): Promise<boolean> => {
    if (!jobId) return false;
    const payload: AnnotationCreateRequest = {
      body: draft.body,
      version,
      section: draft.section,
      start_offset: draft.start,
      end_offset: draft.end,
      quote: draft.quote,
    };
    try {
      const created = await createAnnotation(jobId, payload);
      setItems((prev) => [...prev, created]);
      setError("");
      return true;
    } catch (err) {
      console.error("Failed to add annotation", err);
      setError(describe(err, "Could not save the note."));
      return false;
    }
  };

  const reply = async (parentId: string, body: string): Promise<boolean> => {
    if (!jobId) return false;
    try {
      const created = await createAnnotation(jobId, { body, version, parent_id: parentId });
      setItems((prev) => [...prev, created]);
      setError("");
      return true;
    } catch (err) {
      console.error("Failed to reply", err);
      setError(describe(err, "Could not post the reply."));
      return false;
    }
  };

  const edit = async (id: string, body: string): Promise<boolean> => {
    try {
      const updated = await updateAnnotation(id, body);
      setItems((prev) => prev.map((a) => (a.id === id ? updated : a)));
      setError("");
      return true;
    } catch (err) {
      console.error("Failed to edit annotation", err);
      setError(describe(err, "Could not save the edit."));
      return false;
    }
  };

  /** Tick, cross or reopen. A rejection reason is posted separately as a reply,
   * so the resolver's words stay attributed to them. */
  const resolve = async (
    id: string,
    resolution: AnnotationResolution | null,
    reason?: string,
  ): Promise<boolean> => {
    try {
      const updated = await setAnnotationResolution(id, resolution);
      setItems((prev) => prev.map((a) => (a.id === id ? updated : a)));
      if (reason?.trim()) await reply(id, reason.trim());
      setError("");
      return true;
    } catch (err) {
      console.error("Failed to resolve annotation", err);
      setError(describe(err, "Could not update the note."));
      return false;
    }
  };

  const remove = async (id: string): Promise<boolean> => {
    try {
      await deleteAnnotation(id);
      // Replies cascade server-side; mirror that locally so the thread does not
      // linger with orphaned children until the next load.
      setItems((prev) => prev.filter((a) => a.id !== id && a.parent_id !== id));
      setError("");
      return true;
    } catch (err) {
      console.error("Failed to delete annotation", err);
      setError(describe(err, "Could not delete the note."));
      return false;
    }
  };

  return { threads, items, loading, error, reload: load, add, reply, edit, resolve, remove };
}
