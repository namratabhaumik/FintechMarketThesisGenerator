"""Reads raw feed entries from the Bronze store, keeps the fintech-relevant ones
(classification), scrapes their full text, and embeds them into the persistent
vector store with their publish date in metadata. Each Bronze article is decided
on exactly once (a verdict is recorded), so later runs never re-classify it.
"""

import logging
from typing import Dict, List

from core.interfaces.article_repository import IArticleRepository
from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.interfaces.scoring_strategy import IScoringStrategy
from core.interfaces.scraper import IWebScraper
from core.interfaces.silver_repository import ISilverRepository
from core.models.article import Article
from core.models.silver_record import SilverVerdict
from core.services.ingestion_service import article_to_document
from core.utils.text_utils import clean_article_text

logger = logging.getLogger(__name__)


class SilverService:
    """Builds the embedded corpus (Silver) from the raw store (Bronze).

    Bronze articles are processed exactly once: each gets a verdict recorded in
    the Silver store (fintech-relevant or not), so later runs skip them and
    never re-run classification. Only fintech-relevant articles are scraped and
    embedded; their themes are assigned here on the full scraped text and stored
    on the verdict, so the Gold layer aggregates them without re-deriving from
    the thinner Bronze text. The vectorstore must be the Supabase pgvector store.
    """

    def __init__(
        self,
        repository: IArticleRepository,
        silver_repository: ISilverRepository,
        classifier: IRelevanceClassifier,
        scraper: IWebScraper,
        scoring_strategy: IScoringStrategy,
        theme_categories: Dict[str, List[str]],
        vectorstore,
    ):
        self._repository = repository
        self._silver_repository = silver_repository
        self._classifier = classifier
        self._scraper = scraper
        self._scoring_strategy = scoring_strategy
        self._theme_categories = theme_categories
        self._vectorstore = vectorstore

    def build(self) -> int:
        """Process all Bronze articles not yet decided on.

        Returns:
            The number of articles newly embedded this run.
        """
        raw_articles = self._repository.fetch_all()
        processed = self._silver_repository.processed_urls()
        pending = [r for r in raw_articles if r.url not in processed]
        logger.info(
            f"Silver: {len(pending)} new of {len(raw_articles)} Bronze articles "
            f"({len(processed)} already processed)"
        )

        documents = []
        verdicts = []
        for raw in pending:
            relevant = self._classifier.is_relevant(raw.title, raw.summary)
            if not relevant:
                verdicts.append(SilverVerdict(url=raw.url, fintech_relevant=False))
                continue
            text = self._scraper.scrape(raw.url) or raw.summary or "No content available"
            try:
                article = Article(
                    title=raw.title,
                    text=clean_article_text(text)[:4000],
                    source=raw.source,
                    url=raw.url,
                    published_at=raw.published_at,
                )
            except ValueError as e:
                # No verdict recorded, so a later run retries this article.
                logger.warning(f"Skipping invalid article {raw.url}: {e}")
                continue
            documents.append(article_to_document(article))
            verdicts.append(
                SilverVerdict(
                    url=raw.url,
                    fintech_relevant=True,
                    themes=self._themes_for(article),
                )
            )

        # Embed first; record verdicts only after embedding succeeds, so a failed
        # embed run retries rather than marking articles as done. The vector
        # store dedups by URL internally and logs how many it actually adds.
        if documents:
            logger.info(
                f"Sending {len(documents)} fintech articles to the vector store "
                "(already-embedded ones are skipped there)"
            )
            self._vectorstore.build(documents)
        else:
            logger.info("No new fintech articles to send")

        self._silver_repository.record(verdicts)

        fintech = len(documents)
        logger.info(
            f"Silver: {fintech} fintech of {len(pending)} new articles processed"
        )
        return fintech

    def _themes_for(self, article: Article) -> List[str]:
        """Themes whose keywords appear in the article's title + full text."""
        text = f"{article.title} {article.text}".lower()
        scores = self._scoring_strategy.score(text, self._theme_categories)
        return [theme for theme, count in scores.items() if count > 0]
