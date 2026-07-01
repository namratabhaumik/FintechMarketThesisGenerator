"""Domain exceptions for article ingestion."""


class ArticleFetchError(Exception):
    """Base class for article-fetch failures."""


class NoArticlesFetchedError(ArticleFetchError):
    """No entries came back from the feed (empty feed or unreachable/network error)."""


class NoRelevantArticlesError(ArticleFetchError):
    """Entries were fetched, but none passed the relevance (fintech) classifier."""
