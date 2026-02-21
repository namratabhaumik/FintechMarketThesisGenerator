"""Category mapping data structures for thesis structuring.

These mappings define the keyword-to-category associations used for
structuring market theses.

"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CategoryMapping:
    """Encapsulates a set of category keyword mappings.

    Attributes:
        name: Human-readable name for this mapping set.
        categories: Dict mapping category labels to keyword lists.
    """
    name: str
    categories: Dict[str, List[str]]


class ThemeMappings:
    """Maps keywords to fintech market themes."""

    THEMES: Dict[str, List[str]] = {
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

    @classmethod
    def get_mapping(cls) -> CategoryMapping:
        """Get theme mappings.

        Returns:
            CategoryMapping with theme data.
        """
        return CategoryMapping(name="Themes", categories=cls.THEMES)


class RiskMappings:
    """Maps keywords to fintech risk categories."""

    RISKS: Dict[str, List[str]] = {
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

    @classmethod
    def get_mapping(cls) -> CategoryMapping:
        """Get risk mappings.

        Returns:
            CategoryMapping with risk data.
        """
        return CategoryMapping(name="Risks", categories=cls.RISKS)


class SignalMappings:
    """Maps keywords to investment signal categories."""

    SIGNALS: Dict[str, List[str]] = {
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

    @classmethod
    def get_mapping(cls) -> CategoryMapping:
        """Get signal mappings.

        Returns:
            CategoryMapping with signal data.
        """
        return CategoryMapping(name="Investment Signals", categories=cls.SIGNALS)
