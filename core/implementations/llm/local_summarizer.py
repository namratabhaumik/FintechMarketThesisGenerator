"""Local extractive summarizer - no LLM, no API calls."""

import logging
import re
from typing import List

from langchain.docstore.document import Document

from config.settings import LLMConfig
from core.interfaces.llm import ILanguageModel

logger = logging.getLogger(__name__)


class LocalSummarizerModel(ILanguageModel):
    """Local extractive summarizer using keyword scoring.

    Uses fintech keyword relevance to extract key sentences.
    """

    FINTECH_KEYWORDS = {
        # General fintech
        "fintech", "financial", "finance", "payment", "lending", "banking",
        # Payments & Transfers
        "payment", "transfer", "transaction", "wallet", "peer-to-peer", "p2p",
        "cash app", "venmo", "stripe", "square", "paypal",
        # Digital Banking
        "neobank", "digital bank", "challenger bank", "online banking",
        # Lending & Credit
        "lending", "loan", "credit", "borrowing", "bnpl", "buy now pay later",
        "credit score", "underwriting",
        # Blockchain & Crypto
        "blockchain", "crypto", "bitcoin", "ethereum", "web3", "defi", "nft",
        "tokenization", "token", "smart contract",
        # B2B Finance
        "b2b", "enterprise", "corporate", "invoicing", "accounts payable", "treasury",
        "expense management", "payroll",
        # WealthTech & Investment
        "wealth", "investment", "portfolio", "robo-advisor", "trading", "stock",
        "asset management", "retail investor",
        # Embedded Finance
        "embedded finance", "baas", "banking as a service", "api-first", "white-label",
        # Compliance & Risk
        "compliance", "regulation", "regulatory", "kyc", "aml", "regtech",
        "fraud", "security", "risk", "governance",
        # Infrastructure
        "api", "integration", "platform", "sdk", "middleware", "infrastructure",
        # AI & Automation
        "ai", "artificial intelligence", "machine learning", "automation",
        "llm", "generative", "chatbot", "agent",
        # Market & Adoption
        "market", "growth", "emerging", "adoption", "expansion", "opportunity",
        "startup", "innovation", "disruption",
    }

    def __init__(self, config: LLMConfig):
        """Initialize local summarizer.

        Args:
            config: LLM configuration (API key not used).
        """
        self._config = config
        logger.info("Initializing Local Extractive Summarizer (no API calls)")

    def summarize(self, documents: List[Document]) -> str:
        """Extract summary from documents using keyword scoring.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            Summarized text (extractive - combination of top sentences).
        """
        try:
            logger.info(f"Summarizing {len(documents)} documents locally (keyword-based extraction)")

            # Collect all sentences from all documents
            all_sentences = []
            for doc in documents:
                sentences = self._split_sentences(doc.page_content)
                all_sentences.extend(sentences)

            if not all_sentences:
                logger.warning("No sentences found in documents")
                return "No content to summarize."

            # Filter out mid-sentence fragments (sentences starting with lowercase)
            complete_sentences = [
                s for s in all_sentences if s[0].isupper()
            ]
            if not complete_sentences:
                complete_sentences = all_sentences  # Fallback if all filtered

            # Score each sentence by fintech keyword count
            scored_sentences = [
                (sent, self._score_sentence(sent))
                for sent in complete_sentences
            ]

            # Sort by score, take top 5-7 sentences
            top_count = min(7, len(scored_sentences))
            top_sentences = sorted(
                scored_sentences,
                key=lambda x: x[1],
                reverse=True
            )[:top_count]

            # Deduplicate near-identical sentences (>70% word overlap)
            deduplicated = []
            for sent, score in top_sentences:
                if not self._is_duplicate(sent, [s for s, _ in deduplicated]):
                    deduplicated.append((sent, score))

            # Preserve original order from documents
            top_sentences_ordered = [
                sent for sent, _ in sorted(
                    deduplicated,
                    key=lambda x: complete_sentences.index(x[0])
                )
            ]

            summary = " ".join(top_sentences_ordered)
            logger.info(f"Local summarization complete: {len(summary)} chars, {len(top_sentences_ordered)} sentences")
            return summary

        except Exception as e:
            logger.error(f"Local summarization failed: {e}")
            raise

    def get_model_name(self) -> str:
        """Get model identifier."""
        return "local-extractor"

    # Promotional/ad phrases to filter out
    _AD_PATTERNS = re.compile(
        r'register now|early bird|save up to|\$\d+ off|buy tickets|get tickets|'
        r'sign up|subscribe now|learn more|click here|limited time',
        re.IGNORECASE
    )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [
            s.strip()
            for s in sentences
            if len(s.strip()) > 20                                  # At least 20 chars
            and s.strip()[-1] in ".!?"                             # Must end with punctuation
            and not re.search(r'[.,][A-Z]', s.strip()[:-1])       # No punctuation immediately followed by uppercase without space (scraping artifact)
            and not LocalSummarizerModel._AD_PATTERNS.search(s)   # No ad/promo phrases
        ]

    @staticmethod
    def _is_duplicate(sentence: str, selected: List[str], threshold: float = 0.7) -> bool:
        """Check if sentence has >threshold word overlap with any already-selected sentence.

        Args:
            sentence: Candidate sentence.
            selected: Already-selected sentences.
            threshold: Overlap ratio above which a sentence is considered duplicate.

        Returns:
            True if the sentence is too similar to an existing one.
        """
        words = set(sentence.lower().split())
        for existing in selected:
            existing_words = set(existing.lower().split())
            if not words or not existing_words:
                continue
            overlap = len(words & existing_words) / max(len(words), len(existing_words))
            if overlap >= threshold:
                return True
        return False

    def _score_sentence(self, sentence: str) -> int:
        """Score sentence by fintech keyword count.

        Args:
            sentence: Text to score.

        Returns:
            Count of fintech keywords found.
        """
        sentence_lower = sentence.lower()
        score = sum(
            1 for keyword in self.FINTECH_KEYWORDS
            if keyword in sentence_lower
        )
        return score
