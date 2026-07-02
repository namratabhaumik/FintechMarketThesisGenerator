"""Domain exceptions for article ingestion."""


class ArticleFetchError(Exception):
    """Base class for article-fetch failures."""


class NoArticlesFetchedError(ArticleFetchError):
    """No entries came back from the feed (empty feed or unreachable/network error)."""


class NoRelevantArticlesError(ArticleFetchError):
    """Entries were fetched, but none passed the relevance (fintech) classifier."""


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
