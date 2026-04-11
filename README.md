# Fintech Market Thesis Generator

An AI-powered application that generates investor-style market theses for fintech topics using **LangChain, FAISS, HuggingFace embeddings, and your choice of summarizer**. Supports two modes: a Gemini-powered LLM flow and a fully local, no-API extractive flow.

Try it out here live: [Streamlit Cloud](https://namratabhaumik-fintechmarketthesisGenerator-app-qqdzns.streamlit.app/)

## Demo Video

[![FinThesis Demo](https://img.youtube.com/vi/73SnVdzeVrg/hqdefault.jpg)](https://youtu.be/73SnVdzeVrg)


## Features

- **Live News Ingestion** – Fetches real-time fintech articles from TechCrunch RSS feeds
- **Vector Database (FAISS)** – Semantic search over fintech articles
- **LangChain Orchestration** – Integrates retrieval and summarization
- **AI Gateway (Cost Optimization)** – Intelligent LLM routing with caching:
  - **Response Caching** – Caches LLM outputs to avoid redundant API calls (15-30% hit rate)
  - **Smart Provider Routing** – Routes to Gemini or Local based on cost/quality tradeoff
  - **Cost Tracking** – Monitors daily/monthly API spend with configurable limits
  - **Three Routing Strategies**: Cost-Optimized, Quality-First, or Hybrid (recommended)
  - **75-95% cost reduction** when combined with intelligent caching
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
- **AI Gateway** – Custom cost optimization layer with caching and routing
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
│   │   │   ├── local_summarizer.py  # Local extractive (no API)
│   │   │   ├── llm_wrapper.py     # Retry + fallback wrapper
│   │   │   ├── ai_gateway.py      # AI Gateway orchestrator
│   │   │   ├── cache_manager.py   # Response caching
│   │   │   ├── cost_tracker.py    # Cost tracking & monitoring
│   │   │   └── routing_strategy.py # Provider routing logic
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
│   │   ├── thesis.py
│   │   ├── cost_metric.py         # Cost tracking model (AI Gateway)
│   │   └── cache_entry.py         # Cache entry model (AI Gateway)
│   └── utils/
│       ├── text_utils.py          # Article text cleaning
│       └── logging.py
├── dependency_injection/
│   └── container.py               # Service container with provider registries + AI Gateway
├── tests/
│   ├── conftest.py                # Shared pytest fixtures
│   └── unit/                      # Unit tests (219 tests)
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

1. **Fetch Articles** – Scrapes latest fintech news from TechCrunch RSS feeds
2. **Vectorize** – Converts articles to embeddings (HuggingFace) and indexes in FAISS
3. **Retrieve** – Finds top-5 articles most relevant to your query via semantic search
4. **AI Gateway** *(cost optimization layer)*:
   - **Check Cache** – If articles were summarized before, return cached result (instant, zero cost)
   - **Route Request** – Intelligent provider selection based on strategy:
     - **Cost-Optimized**: Routes large docs to Local (free), small docs to Gemini
     - **Quality-First**: Always routes to Gemini for best quality
     - **Hybrid** (default): Smart mix based on document size and cost limits
   - **Enforce Cost Limits** – If daily budget exceeded, falls back to Local automatically
5. **Summarize** *(mode-dependent)*:
   - **Gemini flow** – Sends retrieved articles to Gemini via LangChain map-reduce chain; returns an analyst-style prose summary
   - **Local flow** – Scores each sentence by fintech keyword count, extracts the top 7, deduplicates by word-overlap; no API call, no cost
   - **Cache Result** – Stores LLM output for future identical queries
   - **Track Cost** – Records tokens, cost, and latency metrics
6. **Structure** – `ThesisStructuringService` maps the summary to fintech taxonomy via keyword scoring:
   - **Themes**: AI-Powered Automation, Digital Payments, Blockchain & Web3, Digital Lending, Neobanking, WealthTech, B2B Finance, RegTech, Embedded Finance, Consumer Finance, Infrastructure, Insurtech
   - **Risks**: Regulatory, Cybersecurity, Market Adoption, Competitive Pressure, Credit & Liquidity, Macroeconomic, Data Privacy, Scalability, Geopolitical, Concentration
   - **Signals**: B2B Expansion, AI-Driven Tools, Emerging Markets, Payment Infrastructure, Embedded Finance, Consumer Adoption, Alternative Lending, Crypto & Web3, RegTech, WealthTech
7. **Score Opportunity** – `OpportunityScoringService` generates an AI-native investment recommendation:
   - **Scores** based on detected signals (0.75 weight), themes (0.25 weight), and risk penalty (−0.25 per risk)
   - **Calculates confidence** (0–100%) based on source coverage, signal strength, and risk balance
   - **Generates recommendation**: "Pursue" (score ≥3.75), "Investigate" (2.5–3.75), or "Skip" (<2.5)
   - **Rule-based & deterministic**: Human reviews the recommendation; final decision remains with human
8. **Display** – Renders results in interactive Streamlit UI with investment scores and recommendations

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

#### AI Gateway Configuration (Cost Optimization)

```bash
# Enable/disable AI Gateway (default: true)
AI_GATEWAY_ENABLED=true

# Routing strategy: cost_optimized | quality_first | hybrid (default: hybrid)
AI_GATEWAY_STRATEGY=hybrid

# Caching configuration
AI_GATEWAY_CACHE_ENABLED=true
AI_GATEWAY_CACHE_TTL_SECONDS=604800  # 7 days

# Cost limits (in USD)
AI_GATEWAY_COST_LIMIT_DAILY=5.0
AI_GATEWAY_COST_LIMIT_MONTHLY=100.0

# Metrics tracking
AI_GATEWAY_TRACK_METRICS=true
```

**AI Gateway Strategies:**
- **`hybrid`** (recommended): Balances cost and quality. Routes small docs to Gemini, large docs to Local. Respects cost limits.
- **`cost_optimized`**: Maximizes savings. Routes large docs and approaching budget thresholds to Local.
- **`quality_first`**: Prioritizes quality. Always uses Gemini, fallback to Local on API failure.

---

## AI Gateway: Cost Optimization

The AI Gateway is a built-in cost optimization layer that intelligently manages LLM calls through caching and smart provider routing.

### How It Works

1. **Response Caching** – Caches LLM summaries by document content hash
   - Identical queries return cached results instantly (0ms, 0 cost)
   - 7-day TTL by default (configurable)
   - In-memory cache (SQLite persistence planned)

2. **Smart Provider Routing** – Automatically selects Gemini or Local based on:
   - Document size (large docs → Local free summarizer)
   - Daily cost budget (near limit → fallback to Local)
   - Routing strategy (hybrid, cost_optimized, or quality_first)

3. **Cost Tracking** – Real-time monitoring:
   - Per-call cost calculation
   - Daily/monthly spend totals
   - Cache hit rate metrics
   - Provider breakdown

### Expected Savings

**Hybrid Strategy (Recommended):**
- 20% cache hit rate
- 50% of calls to Local (free)
- 30% of calls to Gemini
- **Result**: ~70% cost reduction vs. all-Gemini

**Cost-Optimized Strategy:**
- 25% cache hit rate
- 50% of calls to Local (free)
- 25% of calls to Gemini
- **Result**: ~82% cost reduction vs. all-Gemini

### Example: Cost Comparison

| Scenario | Monthly Theses | Gemini Cost | With AI Gateway | Savings |
|----------|---|---|---|---|
| No optimization | 100 | $20.00 | — | — |
| Hybrid strategy | 100 | — | $5.50 | 72% |
| Cost-optimized | 100 | — | $3.75 | 82% |

### Testing AI Gateway

```bash
# 1. Check gateway initialization
python -c "
import os
os.environ['LLM_PROVIDER']='local'
os.environ['EMBEDDING_PROVIDER']='huggingface'
os.environ['EMBEDDING_MODEL']='all-MiniLM-L6-v2'
os.environ['AI_GATEWAY_ENABLED']='true'

from dependency_injection.container import ServiceContainer
container = ServiceContainer()
llm = container.get_llm()
print(f'Gateway enabled: {\"AIGateway\" in llm.get_model_name()}')
"

# 2. Run AI Gateway tests (27 tests)
pytest tests/unit/test_ai_gateway.py -v

# 3. Run full test suite
python run_tests.py
```

### Disable AI Gateway (if needed)

```bash
AI_GATEWAY_ENABLED=false
```

When disabled, the app behaves exactly as before with no caching or cost optimization.

### Adding New Routing Strategies

Create a custom routing strategy in `core/implementations/llm/routing_strategy.py`:

```python
from core.implementations.llm.routing_strategy import RoutingStrategy

class MyStrategy(RoutingStrategy):
    def select_provider(self, documents, topic, daily_spend, daily_limit):
        # Your logic here
        return ("gemini", "gemini-2.0-flash")  # or ("local", "local-extractor")

# Register in STRATEGY_REGISTRY
STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

Then use: `AI_GATEWAY_STRATEGY=my_strategy` in `.env`

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

**219 unit tests** covering all pure-Python components:

**Core Functionality Tests (174 tests):**
- **test_opportunity_scoring.py** (26 tests) – Scoring formula, confidence calculation, recommendations
- **test_thesis_structuring.py** (31 tests) – Category matching, fallback mechanism, edge cases
- **test_services.py** (8 tests) – Service integration, thesis generation with scoring
- **test_text_utils.py** (19 tests) – Text cleaning, boilerplate removal
- **test_local_summarizer.py** (23 tests) – Sentence extraction, keyword scoring, deduplication
- **test_keyword_scoring.py** (12 tests) – Keyword matching logic
- **test_models.py** (13 tests) – Model validation
- **test_config.py** (17 tests) – Configuration loading
- **test_container.py** (25 tests) – Dependency injection, service wiring

**AI Gateway Tests (27 tests - NEW):**
- **test_ai_gateway.py** (27 tests):
  - **CacheManager** (6 tests) – Key generation, caching, expiration, metrics
  - **CostTracker** (6 tests) – Cost calculation, tracking, spend aggregation
  - **Routing Strategies** (10 tests) – Cost-optimized, quality-first, hybrid strategies
  - **AIGateway Integration** (5 tests) – Cache hits, fallback, cost tracking, metrics

**Run Tests:**
```bash
# Full suite
python run_tests.py

# AI Gateway tests only
pytest tests/unit/test_ai_gateway.py -v

# Specific test
pytest tests/unit/test_ai_gateway.py::TestCacheManager::test_cache_key_generation -v
```

---

## License

MIT License - see LICENSE file for details

---

**Built for fintech market research**
