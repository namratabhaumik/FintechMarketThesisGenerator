"""Per-run, cross-record data-quality gates at the medallion layer transitions.

These complement the per-record validation in the model dataclasses
(`RawArticle` / `Article` `__post_init__`): instead of checking one row in
isolation, each gate checks a property that is only visible across the whole set
processed in one run - that the counts reconcile (nothing silently leaked) or
that an aggregate is sane. None of these can be expressed as a single-record
check.

Deliberately threshold-free, catches alerts like "alert if rate > X%". The trigger 
escalates straight to ERROR, because these point at vendor-side drift. 
The gates observe and log.
"""

import logging
from typing import Sequence

from core.models.trend_metric import TrendMetric

logger = logging.getLogger(__name__)


def check_ingestion(seen: int, landed: int) -> None:
    """Bronze gate: flag a feed that returned entries but landed none.

    feedparser tolerates missing fields silently, so a format change at the
    source shows up as entries that parse but yield no usable rows. Per-record
    skips are already logged in the source; the only thing invisible there is
    the aggregate wipeout, which signals schema drift or an unreachable source.
    """
    logger.info(f"Bronze data quality: landed {landed}/{seen} raw articles")
    if seen and not landed:
        logger.error(
            f"Bronze data quality: {seen} feed entries but none landed - "
            "feed schema drift or an unreachable source"
        )


def check_silver(
    pending: int, recorded: int, quarantined: int, errored: int
) -> None:
    """Silver gate: every pending article must end in exactly one bucket.

    Buckets: a recorded verdict (fintech or rejected), quarantined, or errored
    (classification failed, left pending for a later run). The three are counted
    independently, so a mismatch means an article fell through without being
    recorded anywhere - a silent leak this surfaces.
    """
    logger.info(
        f"Silver data quality: {pending} pending -> {recorded} recorded, "
        f"{quarantined} quarantined, {errored} errored"
    )
    if recorded + quarantined + errored != pending:
        logger.error(
            f"Silver data quality: counts do not reconcile - {pending} pending != "
            f"{recorded} recorded + {quarantined} quarantined + {errored} errored"
        )


def check_gold(
    fintech_verdicts: int,
    fintech_with_content: int,
    metrics: Sequence[TrendMetric],
) -> None:
    """Gold gate: every fintech verdict must have a content row, and metric
    counts must be +ve.

    Gold joins 2 Silver stores - the verdicts (which URLs are fintech) and the
    content store (their text + publish date). The invariant is that content is
    persisted before a verdict is recorded, so fewer content rows than verdicts
    means an accepted article lost its content - a cross-store leak. A metric row
    with a <1 count would mean the aggregation itself is broken.
    """
    logger.info(
        f"Gold data quality: {fintech_with_content}/{fintech_verdicts} fintech "
        f"verdicts had content, {len(metrics)} metric rows"
    )
    if fintech_with_content != fintech_verdicts:
        logger.error(
            f"Gold data quality: {fintech_verdicts - fintech_with_content} fintech "
            "verdicts missing a content row - Silver content/verdict stores diverged"
        )
    nonpositive = [m for m in metrics if m.article_count < 1]
    if nonpositive:
        logger.error(
            f"Gold data quality: {len(nonpositive)} metric rows with count < 1"
        )