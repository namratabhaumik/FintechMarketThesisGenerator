# Fintech Market Thesis Generator

An AI-powered application that generates investor-style market theses for fintech topics using **LangChain, FAISS, HuggingFace embeddings, and your choice of summarizer**. Supports two modes: a Gemini-powered LLM flow and a fully local, no-API extractive flow.

Try it out here live: [Streamlit Cloud](https://namratabhaumik-fintechmarketthesisGenerator-app-qqdzns.streamlit.app/)

## Demo Video

[![FinThesis Demo](https://img.youtube.com/vi/6TXiDPOqnj0/hqdefault.jpg)](https://youtu.be/6TXiDPOqnj0)


## Features

- **Live News Ingestion** – Fetches real-time fintech articles from TechCrunch RSS feeds
- **Vector Database (FAISS)** – Semantic search over fintech articles
- **LangChain Orchestration** – Integrates retrieval and summarization
- **Two Summarization Modes**:
  - **Gemini (LLM)** – API-powered, analyst-style prose summary via Google Gemini
  - **Local (no-API)** – Keyword-scored extractive summarizer, zero cost, works offline
- **Pattern-Based Structuring** – Maps keywords to fintech taxonomy (same for both modes):
  - Key themes (12 categories: AI, Digital Payments, Blockchain, etc.)
  - Risks (10 categories: Regulatory, Cybersecurity, etc.)
  - Investment signals (10 categories: Market growth, disruption, etc.)
- **Rule-Based Opportunity Scoring** – Generates AI-native investment recommendations:
  - **Opportunity Score** (0–5 scale): Weighted by signals, themes, and risks
  - **Confidence Level** (0–100%): Varies based on source count, signal consistency, and risk factors
  - **Recommendation**: "Pursue" (≥3.75), "Investigate" (2.5–3.75), or "Skip" (<2.5)
  - **Deterministic & Auditable**: Human-interpretable scoring logic, no black-box LLM decisions
- **Streamlit UI** – Interactive web interface with investment scores and recommendations

---

## Tech Stack

- **LangChain** – Chains, retrievers, and integrations
- **FAISS** – Vector database for semantic retrieval
- **HuggingFace Embeddings** – `all-MiniLM-L6-v2`
- **Gemini (Google AI)** – Optional LLM for cloud-powered summarization
- **Streamlit** – Interactive web UI
- **BeautifulSoup** – Article scraping
- **Feedparser** – RSS feed parsing

---

## Project Structure

```
FintechMarketThesisGenerator/
├── app.py                          # Streamlit frontend (main entry point)
├── config/
│   └── settings.py                # Configuration loaded from environment variables
├── core/
│   ├── interfaces/                # Abstract interfaces (DIP)
│   │   ├── article_source.py
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── scraper.py
│   │   ├── scoring_strategy.py
│   │   ├── thesis_structurer.py
│   │   └── vectorstore.py
│   ├── implementations/           # Concrete implementations
│   │   ├── article_sources/
│   │   ├── embeddings/
│   │   ├── llm/
│   │   │   ├── gemini_llm.py      # Gemini (API-based)
│   │   │   └── local_summarizer.py  # Local extractive (no API)
│   │   ├── scrapers/
│   │   ├── vectorstores/
│   │   └── keyword_scoring_strategy.py
│   ├── services/
│   │   ├── category_mappings.py           # Fintech keyword-to-category data
│   │   ├── ingestion_service.py           # RSS feed fetching + article scraping
│   │   ├── retrieval_service.py           # FAISS vectorstore + semantic search
│   │   ├── thesis_generator_service.py    # Main orchestration
│   │   ├── thesis_structuring_service.py  # Pattern-based thesis structuring
│   │   └── opportunity_scoring_service.py # Rule-based investment opportunity scoring
│   ├── models/
│   │   ├── article.py
│   │   └── thesis.py
│   └── utils/
│       ├── text_utils.py          # Article text cleaning
│       └── logging.py
├── dependency_injection/
│   └── container.py               # Service container with provider registries
├── tests/
│   ├── conftest.py                # Shared pytest fixtures
│   └── unit/                      # Unit tests (142 tests)
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables
├── run_tests.py                   # Simple test runner
└── README.md                      # This file
```

### Core Services

| Service | Purpose | Key Methods |
|---------|---------|------------|
| **ThesisGeneratorService** | Main orchestration | `generate_thesis(topic, documents)` |
| **ArticleIngestionService** | Data Collection | `fetch_articles(query)`, `convert_to_documents()` |
| **DocumentRetrievalService** | Vector Search | `build_vectorstore()`, `retrieve()` |
| **ThesisStructuringService** | Thesis Structuring | `structure_thesis(summary)` |
| **OpportunityScoringService** | Investment Scoring | `score_opportunity(themes, risks, signals, sources)` |

---

## Quick Start

### Installation

```bash
git clone https://github.com/namratabhaumik/FintechMarketThesisGenerator.git
cd FintechMarketThesisGenerator
pip install -r requirements.txt
```

### Run with Local Mode (no API key needed)

```bash
# .env
LLM_PROVIDER=local
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

```bash
streamlit run app.py
```

### Run with Gemini (API key required)

```bash
# .env
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your_api_key_here
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

```bash
streamlit run app.py
```

---

## How It Works

Steps 1–3 and 6–7 are identical in both modes. Only step 4 differs.

1. **Fetch Articles** – Scrapes latest fintech news from TechCrunch RSS feeds
2. **Vectorize** – Converts articles to embeddings (HuggingFace) and indexes in FAISS
3. **Retrieve** – Finds top-5 articles most relevant to your query via semantic search
4. **Summarize** *(mode-dependent)*:
   - **Gemini flow** – Sends retrieved articles to Gemini via LangChain map-reduce chain; returns an analyst-style prose summary
   - **Local flow** – Scores each sentence by fintech keyword count, extracts the top 7, deduplicates by word-overlap; no API call, no cost
5. **Structure** – `ThesisStructuringService` maps the summary (from either mode) to fintech taxonomy via keyword scoring:
   - **Themes**: AI-Powered Automation, Digital Payments, Blockchain & Web3, Digital Lending, Neobanking, WealthTech, B2B Finance, RegTech, Embedded Finance, Consumer Finance, Infrastructure, Insurtech
   - **Risks**: Regulatory, Cybersecurity, Market Adoption, Competitive Pressure, Credit & Liquidity, Macroeconomic, Data Privacy, Scalability, Geopolitical, Concentration
   - **Signals**: B2B Expansion, AI-Driven Tools, Emerging Markets, Payment Infrastructure, Embedded Finance, Consumer Adoption, Alternative Lending, Crypto & Web3, RegTech, WealthTech
6. **Score Opportunity** – `OpportunityScoringService` generates an AI-native investment recommendation:
   - **Scores** based on detected signals (0.75 weight), themes (0.25 weight), and risk penalty (−0.25 per risk)
   - **Calculates confidence** (0–100%) based on source coverage, signal strength, and risk balance
   - **Generates recommendation**: "Pursue" (score ≥3.75), "Investigate" (2.5–3.75), or "Skip" (<2.5)
   - **Rule-based & deterministic**: Human reviews the recommendation; final decision remains with human
7. **Display** – Renders results in interactive Streamlit UI with investment scores and recommendations

**For detailed architecture diagrams and data flow**, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Configuration

### Environment Variables

All configuration is loaded from `.env`.

#### Local mode (no API key needed)

```bash
LLM_PROVIDER=local
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional
VECTORSTORE_PROVIDER=faiss
```

#### Gemini mode

```bash
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your_api_key_here
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional
VECTORSTORE_PROVIDER=faiss
```

### Adding New LLM Providers

1. Create implementation in `core/implementations/llm/your_provider.py`:
```python
from core.interfaces.llm import ILanguageModel

class YourLLM(ILanguageModel):
    def summarize(self, documents: List[Document]) -> str:
        # Implementation
        pass

    def get_model_name(self) -> str:
        return "your-model-name"
```

2. Register in `config/settings.py`:
```python
PROVIDER_API_KEY_ENV["your_provider"] = "YOUR_API_KEY_ENV_VAR"
PROVIDER_MODEL_ENV["your_provider"] = "YOUR_MODEL_ENV_VAR"
```

3. Register in `dependency_injection/container.py`:
```python
LLM_PROVIDER_REGISTRY["your_provider"] = YourLLM
```

### RSS Feed Sources

Edit `config/settings.py` to add/remove RSS feeds:

```python
rss_feeds: List[RSSFeedConfig] = field(default_factory=lambda: [
    RSSFeedConfig(
        name="TechCrunch Fintech",
        url="https://techcrunch.com/category/fintech/feed/",
        enabled=True
    ),
    # Add more feeds here
])
```

---

## Example Output

### Input Query
```
"Credit scoring in emerging markets"
```

### Generated Thesis
```json
{
  "key_themes": [
    "Digital Lending",
    "AI-Powered Automation",
    "Fintech Infrastructure"
  ],
  "risks": [
    "Regulatory Risk",
    "Credit & Liquidity Risk",
    "Data Privacy Risk"
  ],
  "investment_signals": [
    "Emerging Market Growth",
    "AI-Driven Financial Tools",
    "Alternative Lending Growth"
  ],
  "sources": [
    "https://techcrunch.com/...",
    "https://techcrunch.com/..."
  ],
  "opportunity_score": 3.8,
  "confidence_level": 0.77,
  "recommendation": "Pursue",
  "key_risk_factors": [
    "Regulatory Risk",
    "Credit & Liquidity Risk",
    "Data Privacy Risk"
  ]
}
```

**Interpretation**:
- **Score 3.8/5** indicates a strong opportunity above the Pursue threshold (≥3.75)
- **Confidence 77%** reflects 5 sources, 2 strong signals, and moderate risk exposure
- **Pursue recommendation** suggests AI system scores this favorably; human makes final decision
- **Key risks** are explicitly listed so investors can conduct due diligence

---

## Testing

```bash
python run_tests.py
```

174 unit tests covering all pure-Python components:
- **test_opportunity_scoring.py** (26 tests) – Scoring formula, confidence calculation, recommendations
- **test_thesis_structuring.py** (31 tests) – Category matching, fallback mechanism, edge cases
- **test_services.py** (8 tests) – Service integration, thesis generation with scoring
- **test_text_utils.py** (19 tests) – Text cleaning, boilerplate removal
- **test_local_summarizer.py** (23 tests) – Sentence extraction, keyword scoring, deduplication
- **test_keyword_scoring.py** (12 tests) – Keyword matching logic
- **test_models.py** (13 tests) – Model validation
- **test_config.py** (17 tests) – Configuration loading
- **test_container.py** (25 tests) – Dependency injection, service wiring

---

## License

MIT License - see LICENSE file for details

---

**Built for fintech market research**
