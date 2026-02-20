"""Thesis structuring service - maps summary keywords to fintech category names."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ThesisStructuringService:
    """Lightweight service for structuring thesis data from summaries.

    Uses keyword-to-category mappings.
    Each dict key is the category label; value is the list of trigger keywords.
    The categories with the most keyword hits in the text are returned.
    """

    THEME_MAP: Dict[str, List[str]] = {
        "AI-Powered Automation":    ["ai agent", "ai-powered", "automation", "invoicing", "settlement", "workflow"],
        "Digital Payments":         ["payment link", "payment request", "cash app", "transfer", "peer-to-peer", "p2p", "payment"],
        "Blockchain & Web3":        ["blockchain", "crypto", "web3", "defi", "tokenization", "token", "wallet"],
        "Digital Lending":          ["lending", "loan", "borrowing", "credit", "bnpl", "buy now pay later"],
        "Neobanking":               ["neobank", "digital bank", "challenger bank", "online banking"],
        "WealthTech":               ["wealth", "robo-advisor", "portfolio", "asset management", "wealthtech"],
        "B2B Finance":              ["b2b", "enterprise", "corporate finance", "treasury", "accounts payable"],
        "RegTech & Compliance":     ["regtech", "kyc", "aml", "compliance", "regulation", "regulatory"],
        "Embedded Finance":         ["embedded finance", "banking as a service", "baas", "api banking"],
        "Consumer Finance":         ["consumer", "retail finance", "personal finance", "gen z", "millennial"],
        "Fintech Infrastructure":   ["infrastructure", "api", "integration", "platform", "sdk", "middleware"],
        "Insurtech":                ["insurance", "insurtech", "underwriting", "premium", "claims"],
    }

    RISK_MAP: Dict[str, List[str]] = {
        "Regulatory Risk":          ["regulatory", "regulation", "compliance", "sec", "gdpr", "enforcement", "ban"],
        "Cybersecurity Risk":       ["breach", "hack", "fraud", "security", "vulnerability", "phishing", "data leak"],
        "Market Adoption Risk":     ["adoption", "user resistance", "slow uptake", "trust", "awareness"],
        "Competitive Pressure":     ["competition", "competitive", "incumbent", "big tech", "rival", "market share"],
        "Credit & Liquidity Risk":  ["credit risk", "default", "liquidity", "insolvency", "bad debt", "npls"],
        "Macroeconomic Risk":       ["recession", "downturn", "inflation", "interest rate", "macro"],
        "Data Privacy Risk":        ["privacy", "data breach", "pii", "personal data", "gdpr", "data protection"],
        "Scalability Risk":         ["scaling", "infrastructure cost", "technical debt", "outage", "downtime"],
        "Geopolitical Risk":        ["geopolit", "sanction", "cross-border", "tariff", "trade war"],
        "Concentration Risk":       ["concentration", "single vendor", "platform dependency", "lock-in"],
    }

    SIGNAL_MAP: Dict[str, List[str]] = {
        "B2B Fintech Expansion":        ["b2b", "enterprise", "corporate", "invoicing", "accounts payable", "treasury"],
        "AI-Driven Financial Tools":    ["ai", "llm", "generative", "chatbot", "financial advisor", "automation"],
        "Emerging Market Growth":       ["emerging market", "india", "africa", "southeast asia", "latam", "developing"],
        "Payment Infrastructure":       ["payment rail", "payment network", "real-time payment", "instant payment"],
        "Embedded Finance Opportunity": ["embedded", "baas", "api-first", "white-label", "platform"],
        "Consumer Fintech Adoption":    ["gen z", "millennial", "consumer adoption", "retail investor", "mass market"],
        "Alternative Lending Growth":   ["bnpl", "alternative lending", "revenue-based", "micro-lending", "credit access"],
        "Crypto & Web3 Opportunity":    ["crypto", "defi", "nft", "tokenization", "web3", "blockchain"],
        "RegTech Investment Signal":    ["regtech", "compliance automation", "kyc", "aml", "regulatory tech"],
        "WealthTech Disruption":        ["robo-advisor", "wealthtech", "wealth management", "retail investing"],
    }

    def structure_thesis(self, summary: str) -> dict:
        """Map summary to structured category labels.

        Args:
            summary: Summarized text from documents.

        Returns:
            Dictionary with key_themes, risks, investment_signals, sources.
        """
        logger.info("Structuring thesis from summary using category mapping")
        text_lower = summary.lower()

        return {
            "key_themes": self._match_categories(text_lower, self.THEME_MAP),
            "risks": self._match_categories(text_lower, self.RISK_MAP),
            "investment_signals": self._match_categories(text_lower, self.SIGNAL_MAP),
        }

    @staticmethod
    def _match_categories(text_lower: str, category_map: Dict[str, List[str]]) -> List[str]:
        """Score each category by keyword hits and return top 3 labels.

        Args:
            text_lower: Lowercased summary text.
            category_map: Dict of {label: [trigger_keywords]}.

        Returns:
            Up to 3 category labels with at least one keyword match, sorted by hit count.
        """
        scored: Dict[str, int] = {
            label: sum(1 for kw in keywords if kw in text_lower)
            for label, keywords in category_map.items()
        }

        # Filter out zero-hit categories, sort by hits descending, return top 3 labels
        return [
            label for label, hits in sorted(scored.items(), key=lambda x: x[1], reverse=True)
            if hits > 0
        ][:3]

