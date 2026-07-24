// Server-side pagination over the theses list endpoint. Both the caller's own
// library and the admin cross-user list are the same page/offset/has-more
// machinery differing only by the `all` flag, so they share this hook.

import { useState } from "react";
import { listTheses } from "./api";
import type { ThesisSummaryResponse } from "./types";

export const PAGE_SIZE = 10;

interface Options {
  /** Request the admin cross-user list (the backend 403s for non-admins). */
  allUsers?: boolean;
  /** When false, `load` is a no-op - used to skip the admin list entirely. */
  enabled?: boolean;
  /** Console label for a failed fetch. */
  errorLabel: string;
}

export interface PaginatedTheses {
  page: ThesisSummaryResponse[];
  offset: number;
  hasMore: boolean;
  /** Fetch a page; defaults to the current offset. Never rejects. */
  load: (offset?: number) => Promise<void>;
  /** Step one page forward (1) or back (-1), clamped at zero. */
  goToPage: (direction: number) => void;
}

export function usePaginatedTheses({
  allUsers = false,
  enabled = true,
  errorLabel,
}: Options): PaginatedTheses {
  const [page, setPage] = useState<ThesisSummaryResponse[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);

  const load = async (next = offset) => {
    if (!enabled) return;
    try {
      // Over-fetch by one to detect a next page without a total count.
      const fetched = await listTheses(PAGE_SIZE + 1, next, undefined, allUsers);
      setHasMore(fetched.length > PAGE_SIZE);
      setPage(fetched.slice(0, PAGE_SIZE));
    } catch (err) {
      console.error(errorLabel, err);
      setPage([]);
    }
  };

  // Set the new offset and fetch it explicitly rather than waiting for the
  // state update, so the fetch never reads a stale offset.
  const goToPage = (direction: number) => {
    const next = Math.max(0, offset + direction * PAGE_SIZE);
    setOffset(next);
    void load(next);
  };

  return { page, offset, hasMore, load, goToPage };
}
