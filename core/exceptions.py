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
