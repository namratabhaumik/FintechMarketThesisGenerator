# Fintech Market Thesis Generator

An AI-powered application that generates investor-style market theses for fintech topics using **LangChain, FAISS, HuggingFace embeddings, and Gemini**. Features live RSS feed ingestion from TechCrunch for real-time fintech news analysis.

Try it out here live: [Streamlit Cloud](https://namratabhaumik-fintechmarketthesisGenerator-app-qqdzns.streamlit.app/)

## Demo Video

[![FinThesis Demo](https://img.youtube.com/vi/CpHMJ3T2lGY/maxresdefault.jpg)](https://youtu.be/CpHMJ3T2lGY)


## Features

- **Live News Ingestion** – Fetches real-time fintech articles from TechCrunch RSS feeds
- **Vector Database (FAISS)** – Semantic search over fintech articles
- **LangChain Orchestration** – Integrates retrieval and summarization
- **Gemini AI** – Summarizes articles with advanced context understanding
- **Pattern-Based Structuring** – Maps keywords to fintech taxonomy:
  - Key themes (12 categories: AI, Digital Payments, Blockchain, etc.)
  - Risks (10 categories: Regulatory, Cybersecurity, etc.)
  - Investment signals (10 categories: Market growth, disruption, etc.)
- **Streamlit UI** – Interactive web interface

---

## Tech Stack

- **LangChain** – Chains, retrievers, and integrations
- **FAISS** – Vector database for semantic retrieval
- **HuggingFace Embeddings** – `all-MiniLM-L6-v2`
- **Gemini (Google AI)** – LLM for structured outputs
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
│   │   └── vectorstore.py
│   ├── implementations/            # Concrete implementations (Strategy pattern)
│   │   ├── article_sources/
│   │   ├── embeddings/
│   │   ├── llm/
│   │   ├── scrapers/
│   │   └── vectorstores/
│   ├── services/
│   │   ├── ingestion_service.py    # RSS feed fetching + article scraping
│   │   ├── retrieval_service.py    # FAISS vectorstore + semantic search
│   │   ├── thesis_generator_service.py    # Main orchestration
│   │   └── thesis_structuring_service.py  # Pattern-based thesis structuring
│   └── models/
├── dependency_injection/
│   └── container.py               # Service container with provider registries
├── tests/
│   ├── conftest.py                # Shared pytest fixtures
│   └── unit/                      # Unit tests (72 tests)
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
| **ThesisStructuringService** | Thesis Structuring | `structure_thesis(summary)` - maps keywords to fintech taxonomy |

---

## Quick Start

### Installation

```bash
git clone https://github.com/namratabhaumik/FintechMarketThesisGenerator.git
cd FintechMarketThesisGenerator
pip install -r requirements.txt
```

### Set API Key

```bash
# Linux/Mac
export GOOGLE_API_KEY="your_key_here"

# Windows PowerShell
$env:GOOGLE_API_KEY="your_key_here"
```

### Run the App

```bash
streamlit run app.py
```

---

## How It Works

1. **Fetch Articles** – Scrapes latest fintech news from TechCrunch RSS feeds
2. **Vectorize** – Converts articles to embeddings using HuggingFace and indexes them in FAISS
3. **Retrieve** – Finds top-5 articles most relevant to your query using semantic search
4. **Summarize** – Uses Gemini to create an analyst-style summary of retrieved articles (LLM-only)
5. **Structure** – Maps summary keywords to fintech taxonomy (pattern-based, no LLM):
   - **Themes**: AI-Powered Automation, Digital Payments, Blockchain & Web3, Digital Lending, Neobanking, WealthTech, B2B Finance, RegTech, Embedded Finance, Consumer Finance, Infrastructure, Insurtech
   - **Risks**: Regulatory, Cybersecurity, Market Adoption, Competitive Pressure, Credit & Liquidity, Macroeconomic, Data Privacy, Scalability, Geopolitical, Concentration
   - **Signals**: B2B Expansion, AI-Driven Tools, Emerging Markets, Payment Infrastructure, Embedded Finance, Consumer Adoption, Alternative Lending, Crypto & Web3, RegTech, WealthTech
6. **Display** – Renders results in interactive Streamlit UI

**For detailed architecture diagrams and data flow**, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Configuration

### Environment Variables

All configuration is loaded from `.env`. Required variables:

```bash
# LLM Configuration
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your_api_key_here

# Embedding Configuration
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store Configuration (optional, defaults to faiss)
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
"Future of embedded finance in Asia"
```

### Generated Thesis
```json
{
  "key_themes": [
    "Rapid adoption of embedded finance in Southeast Asian markets",
    "Banking-as-a-Service platforms enabling non-financial companies",
    "Regulatory evolution supporting open banking initiatives"
  ],
  "risks": [
    "Regulatory uncertainty across different jurisdictions",
    "Data privacy concerns with embedded financial services",
    "Competition from established financial institutions"
  ],
  "investment_signals": [
    "Growing number of Series A/B rounds in embedded finance",
    "Partnerships between fintechs and e-commerce platforms",
    "Increasing API adoption rates"
  ],
  "sources": [
    "TechCrunch: Embedded finance startup raises $50M",
    "TechCrunch: Asian banks partner with fintech platforms"
  ]
}
```

---

## License

MIT License - see LICENSE file for details

---


**Built for fintech market research**
