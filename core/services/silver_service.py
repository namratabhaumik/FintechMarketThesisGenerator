"""Reads raw feed entries from the Bronze store, keeps the fintech-relevant ones
(classification), scrapes their full text, and embeds them into the persistent
vector store with their publish date in metadata. Each Bronze article is decided
on exactly once (a verdict is recorded), so later runs never re-classify it.
"""

import logging
from typing import Dict, List, NamedTuple

from core.interfaces.article_content_repository import IArticleContentRepository
from core.interfaces.article_repository import IArticleRepository
from core.interfaces.quarantine_repository import IQuarantineRepository
from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.interfaces.scoring_strategy import IScoringStrategy
from core.interfaces.scraper import IWebScraper
from core.interfaces.silver_repository import ISilverRepository
from core.interfaces.vectorstore import IVectorStore
from core.models.article import Article
from core.models.quarantine_record import (
    INVALID_ARTICLE,
    SCRAPE_FAILED,
    QuarantineRecord,
)
from core.models.silver_record import SilverVerdict
from core.services.ingestion_service import article_to_document
from core.utils.data_quality import check_silver
from core.utils.text_utils import clean_article_text

logger = logging.getLogger(__name__)

# Upper bound on stored article text (~2000 words).
MAX_ARTICLE_CHARS = 12_000


class _Batch(NamedTuple):
    """Work collected from the articles.

    The fields are accumulator lists appended to during collection.
    """

    documents: list
    verdicts: list
    quarantined: list
    new_content: list
    errored: list


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
        content_repository: IArticleContentRepository,
        quarantine_repository: IQuarantineRepository,
        classifier: IRelevanceClassifier,
        scraper: IWebScraper,
        scoring_strategy: IScoringStrategy,
        theme_categories: Dict[str, List[str]],
        vectorstore: IVectorStore,
    ):
        self._repository = repository
        self._silver_repository = silver_repository
        self._content_repository = content_repository
        self._quarantine_repository = quarantine_repository
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
        # Skip articles already decided on (a verdict) or parked in quarantine,
        # so neither is reprocessed.
        skip = self._silver_repository.processed_urls() | self._quarantine_repository.quarantined_urls()
        pending = [r for r in raw_articles if r.url not in skip]
        # Validated text already persisted (e.g. a prior run scraped it but
        # failed to embed). Reuse it so the scrape happens once per URL.
        persisted = {a.url: a for a in self._content_repository.fetch_all()}
        logger.info(
            f"Silver: {len(pending)} new of {len(raw_articles)} Bronze articles "
            f"({len(skip)} already processed or quarantined)"
        )

        batch = self._collect(pending, persisted)
        return self._commit(batch, total_pending=len(pending))

    def _collect(self, pending, persisted) -> _Batch:
        """Classify, enrich and theme each pending article into a _Batch.

        Three classifier cases: errored (no answer -> skip, stays pending for a
        later run), said NO (frozen rejected verdict), or said YES (enrich +
        embed). Scrape/validation failures are quarantined inside _enrich.
        """
        batch = _Batch(
            documents=[], verdicts=[], quarantined=[], new_content=[], errored=[]
        )
        for raw in pending:
            try:
                relevant = self._classifier.is_relevant(raw.title, raw.summary)
            except Exception as e:
                # Could not classify (e.g. model/token/network error). Record
                # nothing so the row stays pending; a later run picks it up once
                # the classifier is healthy.
                logger.warning(
                    f"Classification failed for {raw.url}, left pending for a "
                    f"later run: {e}"
                )
                batch.errored.append(raw.url)
                continue

            if not relevant:  # real NO -> frozen rejected verdict
                batch.verdicts.append(SilverVerdict(url=raw.url, fintech_relevant=False))
                continue

            # real YES -> enrich + embed
            article = persisted.get(raw.url)
            if article is None:
                article = self._enrich(raw, batch.quarantined)
                if article is None:
                    continue  # scrape failed or invalid -> quarantined
                batch.new_content.append(article)

            batch.documents.append(article_to_document(article))
            batch.verdicts.append(
                SilverVerdict(
                    url=raw.url,
                    fintech_relevant=True,
                    themes=self._themes_for(article),
                )
            )
        return batch

    def _commit(self, batch: _Batch, total_pending: int) -> int:
        """Persist a collected batch; return the count of fintech articles.

        Order matters: persist the validated text BEFORE embedding, so if
        embedding fails the next run can re-embed from it without re-scraping.
        Record verdicts only AFTER embedding succeeds (failed embed -> no verdict
        -> a later run re-attempts it, cheaply, from persisted text). The vector
        store dedups by URL.
        """
        self._content_repository.save(batch.new_content)
        if batch.documents:
            logger.info(
                f"Sending {len(batch.documents)} fintech articles to the vector "
                "store (already-embedded ones are skipped there)"
            )
            self._vectorstore.build(batch.documents)
        else:
            logger.info("No new fintech articles to send")

        self._silver_repository.record(batch.verdicts)
        self._quarantine_repository.add(batch.quarantined)

        check_silver(
            pending=total_pending,
            recorded=len(batch.verdicts),
            quarantined=len(batch.quarantined),
            errored=len(batch.errored),
        )

        fintech = len(batch.documents)
        logger.info(
            f"Silver: {fintech} fintech of {total_pending} new articles processed; "
            f"{len(batch.new_content)} newly scraped, {len(batch.quarantined)} quarantined"
        )
        return fintech

    def _enrich(self, raw, quarantined: list):
        """Scrape and validate a raw article into an Article.

        Appends a QuarantineRecord and returns None on scrape failure or
        validation failure.
        """
        scraped = self._scraper.scrape(raw.url)
        if not scraped:
            # Full text unavailable. Quarantine for replay rather than embedding
            # the thin RSS summary or retrying every run.
            quarantined.append(
                QuarantineRecord(
                    url=raw.url,
                    reason=SCRAPE_FAILED,
                    detail="scraper returned no text",
                    title=raw.title,
                )
            )
            return None
        try:
            return Article(
                title=raw.title,
                text=clean_article_text(scraped)[:MAX_ARTICLE_CHARS],
                source=raw.source,
                url=raw.url,
                published_at=raw.published_at,
            )
        except ValueError as e:
            quarantined.append(
                QuarantineRecord(
                    url=raw.url,
                    reason=INVALID_ARTICLE,
                    detail=str(e),
                    title=raw.title,
                )
            )
            return None

    def _themes_for(self, article: Article) -> List[str]:
        """Themes whose keywords appear in the article's title + full text."""
        text = f"{article.title} {article.text}".lower()
        scores = self._scoring_strategy.score(text, self._theme_categories)
        return [theme for theme, count in scores.items() if count > 0]
