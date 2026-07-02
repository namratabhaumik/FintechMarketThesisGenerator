"""Reads raw feed entries from the Bronze store, keeps the fintech-relevant ones
(classification), scrapes their full text, and embeds them into the persistent
vector store with their publish date in metadata. Each Bronze article is decided
on exactly once (a verdict is recorded), so later runs never re-classify it.
"""

import logging
from typing import Dict, List, NamedTuple

from core.exceptions import ClassifierOutageError
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

# Circuit breaker: this many classifier failures in a row (a single success
# resets the count) means the classifier is down, not a flaky article - so the
# Silver build aborts rather than hammering a dead classifier through the batch.
OUTAGE_ABORT_AFTER = 5


class _Batch(NamedTuple):
    """Work collected from the articles.

    The fields are accumulator lists appended to during collection: _collect
    fills them as it walks the pending articles, and _commit then persists each
    list to its store. Keeping them separate lets _commit write in the right
    order (content before vectors before verdicts) and lets the data-quality
    gate reconcile the counts.
    """

    documents: list      # LangChain Documents for the accepted articles, to embed.
    verdicts: list       # One SilverVerdict per decided article (accepted or rejected).
    quarantined: list    # QuarantineRecords for scrape/validation failures.
    new_content: list    # Newly scraped Articles to persist to article_content.
    errored: list        # URLs whose classification threw; left pending for a retry.


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
        risk_categories: Dict[str, List[str]],
        signal_categories: Dict[str, List[str]],
        vectorstore: IVectorStore,
    ):
        self._repository = repository
        self._silver_repository = silver_repository
        self._content_repository = content_repository
        self._quarantine_repository = quarantine_repository
        self._classifier = classifier
        self._scraper = scraper
        self._scoring_strategy = scoring_strategy
        # Three keyword-category maps - one per tag dimension - all scored the
        # same way on the article's full text.
        self._theme_categories = theme_categories
        self._risk_categories = risk_categories
        self._signal_categories = signal_categories
        self._vectorstore = vectorstore

    def build(self) -> int:
        """Process all Bronze articles not yet decided on.

        Returns:
            The number of articles newly embedded this run.
        """
        # raw_articles: the whole Bronze corpus (every URL ever landed).
        raw_articles = self._repository.fetch_all()
        # skip: the set of URLs already handled - either decided on (a verdict)
        # or parked in quarantine - unioned so neither group is reprocessed.
        skip = self._silver_repository.processed_urls() | self._quarantine_repository.quarantined_urls()
        # pending: only the Bronze articles this run still needs to decide on.
        pending = [r for r in raw_articles if r.url not in skip]
        # persisted: {url -> Article} of full text already scraped on a prior run
        # that then failed to embed. Reusing it means a URL is scraped just once.
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
        consecutive_errors = 0  # reset by any success; trips the outage breaker
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
                # Outage circuit breaker: N failures in a row (no success between)
                # means the classifier is down, not a flaky article. Abort now to
                # bound wasted cloud calls and fail loud; untouched rows stay
                # pending for the next run.
                consecutive_errors += 1
                if consecutive_errors >= OUTAGE_ABORT_AFTER:
                    raise ClassifierOutageError(
                        f"Classifier failed on {consecutive_errors} articles in a "
                        f"row; aborting run. Last error: {e}"
                    )
                continue

            consecutive_errors = 0  # a success breaks the failure streak
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

            # Tag the full text on all three dimensions once, then reuse the
            # same tags for both the embedded document's metadata (so retrieval
            # and the thesis can read them) and the stored verdict (so Gold can
            # accumulate them).
            themes, risks, signals = self._tags_for(article)
            batch.documents.append(
                article_to_document(article, themes=themes, risks=risks, signals=signals)
            )
            batch.verdicts.append(
                SilverVerdict(
                    url=raw.url,
                    fintech_relevant=True,
                    themes=themes,
                    risks=risks,
                    signals=signals,
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
        classified = len(batch.verdicts)  # got a fintech / not-fintech verdict
        logger.info(
            f"Silver summary: {total_pending} new articles -> {classified} classified "
            f"({fintech} fintech, {classified - fintech} rejected), "
            f"{len(batch.new_content)} scraped, "
            f"{len(batch.quarantined)} quarantined (scrape failed), "
            f"{len(batch.errored)} errored (classify failed)"
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

    def _tags_for(self, article: Article) -> tuple:
        """Tag the article on all three dimensions from its title + full text.

        Returns (themes, risks, signals), each a list of the categories whose
        keywords appear in the text. All three use the same keyword scoring, just
        with a different category map.
        """
        text = f"{article.title} {article.text}".lower()
        return (
            self._match(text, self._theme_categories),
            self._match(text, self._risk_categories),
            self._match(text, self._signal_categories),
        )

    def _match(self, text: str, categories: Dict[str, List[str]]) -> List[str]:
        """Return the categories whose keywords appear in `text` (score > 0)."""
        scores = self._scoring_strategy.score(text, categories)
        return [category for category, count in scores.items() if count > 0]
