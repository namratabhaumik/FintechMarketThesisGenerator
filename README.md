# Fintech Market Thesis Generator

An AI-powered application that generates investor-style market theses for fintech topics using **LangChain, FAISS, HuggingFace embeddings, and Gemini**. Features live RSS feed ingestion from TechCrunch for real-time fintech news analysis.

Try it out here live: [Streamlit Cloud](https://namratabhaumik-fintechmarketthesisGenerator-app-qqdzns.streamlit.app/)

## Demo Video

[![FinThesis Demo](https://img.youtube.com/vi/CpHMJ3T2lGY/maxresdefault.jpg)](https://youtu.be/CpHMJ3T2lGY)


## Features

- **Live News Ingestion** – Fetches real-time fintech articles from TechCrunch RSS feeds
- **Vector Database (FAISS)** – Semantic search over fintech articles
- **LangChain Orchestration** – Integrates retrieval, summarization, and prompting
- **Gemini AI** – Generates structured market theses with:
  - Key themes
  - Risks
  - Investment signals
  - Sources
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
├── core/
│   ├── ingestion.py               # RSS feed fetching + article scraping
│   ├── retrieval.py               # FAISS vectorstore + semantic search
│   ├── gemini_client.py           # Gemini API summarization + structuring
│   └── utils.py                   # Article normalization + logging
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (GOOGLE_API_KEY)
└── README.md                      # This file
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|----------------|
| **app.py** | Main Streamlit UI | Orchestrates entire workflow, handles user input/output |
| **ingestion.py** | Data Collection | `fetch_live_articles()` fetches from RSS, `scrape_article_text()` extracts content |
| **retrieval.py** | Vector Search | `build_vectorstore()` creates FAISS index, enables semantic search |
| **gemini_client.py** | AI Generation | `generate_summary()` and `generate_structured_thesis()` call Gemini API |
| **utils.py** | Utilities | `normalize_articles()` standardizes format, `setup_logging()` configures logs |

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
4. **Summarize** – Uses Gemini to create an analyst-style summary of retrieved articles
5. **Structure** – Formats summary into JSON with key themes, risks, investment signals, and sources
6. **Display** – Renders results in interactive Streamlit UI

**For detailed architecture diagrams and data flow**, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Configuration

### RSS Feed Sources

Edit `core/ingestion.py` to add/remove RSS feeds:

```python
DEFAULT_RSS_FEEDS = [
    {
        "name": "TechCrunch Fintech",
        "url": "https://techcrunch.com/category/fintech/feed/",
        "enabled": True
    },
    {
        "name": "Your Custom Feed",
        "url": "https://example.com/feed",
        "enabled": True
    },
]
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

## Contributing

Contributions are welcome! Areas for improvement:
- Additional RSS feed sources
- UI improvements
- Additional LLM integrations
- Enhanced article filtering

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
- Embeddings from [HuggingFace](https://huggingface.co/)
- UI with [Streamlit](https://streamlit.io/)

---

**Built for fintech market research**
