"""Domain exceptions for the Silver build."""


class ClassifierOutageError(Exception):
    """Every pending article failed classification - the classifier appears down.

    Raised by the Silver build to abort and fail loud (non-zero exit) for
    investigatigation.
    """


class TransientScrapeError(Exception):
    """A scrape failed for a transient reason (timeout, connection, 5xx, 429).

    Signals that the URL should be left pending and retried on a later run,
    rather than quarantined like a terminal failure (404/403, unparseable page).
    """
