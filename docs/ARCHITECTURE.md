# FinThesis Architecture & Data Flow

This document provides detailed diagrams and explanations of how the FinThesis application works.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│              (Market topic/question in Streamlit)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    [1] FETCH ARTICLES                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Fetch from TechCrunch RSS feeds                         │ │
│  │ • Scrape article text using BeautifulSoup                │ │
│  │ • Normalize to consistent format (title, url, text)      │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                [2] VECTORIZATION & STORAGE                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Convert articles to text chunks                         │ │
│  │ • Generate embeddings (HuggingFace: all-MiniLM-L6-v2)    │ │
│  │ • Index in FAISS for semantic search                     │ │
│  │ • Cache vectorstore in session (reuse across queries)    │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  [3] SEMANTIC RETRIEVAL                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Convert user query to embedding                         │ │
│  │ • Search FAISS index for top-5 relevant articles          │ │
│  │ • Rank by semantic similarity                            │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              [4] LLM SUMMARIZATION (Gemini)                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Feed retrieved articles to Gemini                       │ │
│  │ • Generate condensed analyst summary                      │ │
│  │ • Uses map-reduce chain for multi-document processing    │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│     [5] STRUCTURE INTO THESIS (Pattern Matching)               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Map summary keywords to fintech taxonomy               │ │
│  │ • Extract: key_themes, risks, signals                    │ │
│  │ • Extract sources from article metadata                  │ │
│  │ • No second LLM call needed                              │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    [6] DISPLAY RESULTS                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ •Show fetched article links                             │ │
│  │ •Display raw Gemini output                              │ │
│  │ •Show condensed analyst summary                         │ │
│  │ •Render structured thesis (themes, risks, etc)          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)                         │
│  - Input form for market queries                                │
│  - Display results                                              │
│  - Session management (caching vectorstore)                     │
└─────────────────┬──────────────────────────────────────────────┘
                  │
        ┌─────────▼──────────────────────────┐
        │                                    │
        ▼                                    ▼
┌────────────────────────────┐   ┌──────────────────────────────┐
│  SERVICE CONTAINER         │   │  CONFIGURATION (config/)     │
│  (dependency_injection/)   │   │                              │
│                            │   │ ┌─────────────────────────┐  │
│ ┌──────────────────────┐   │   │ │ Load from .env vars:   │  │
│ │ LLM_PROVIDER_REGISTRY│   │   │ │ • LLM_PROVIDER         │  │
│ │ • gemini             │   │   │ │ • EMBEDDING_PROVIDER   │  │
│ └──────────────────────┘   │   │ │ • Validate & Parse     │  │
│                            │   │ └─────────────────────────┘  │
│ ┌──────────────────────┐   │   │                              │
│ │EMBEDDING_PROVIDER... │   │   └──────────────────────────────┘
│ │ • huggingface        │   │
│ └──────────────────────┘   │
└────────────────────────────┘
        │
        │ Creates & Injects
        │
        ▼
┌────────────────────────────────────────────────────────┐
│         THESIS GENERATOR SERVICE                       │
│  (ThesisGeneratorService)                              │
│                                                        │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 1. ArticleIngestionService                     │ │
│  │    ├── IArticleSource (RSSArticleSource)       │ │
│  │    └── IWebScraper (BeautifulSoupScraper)      │ │
│  └─────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 2. DocumentRetrievalService                    │ │
│  │    ├── IEmbeddingModel (HuggingFace)           │ │
│  │    └── IVectorStore (FAISS)                    │ │
│  └─────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 3. ILanguageModel (Gemini)                     │ │
│  │    └── Summarize documents (map-reduce)       │ │
│  └─────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 4. ThesisStructuringService                    │ │
│  │    └── Pattern-based keyword mapping to       │ │
│  │        fintech taxonomy (12+10+10 categories)  │ │
│  └─────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Processing

1. **Fetch Articles** – `ArticleIngestionService` fetches live articles from TechCrunch RSS
2. **Vectorize** – `DocumentRetrievalService` creates FAISS embeddings (one-time per session)
3. **Retrieve** – FAISS finds top-5 articles semantically similar to your query
4. **Summarize** – `GeminiLanguageModel` uses Gemini to create analyst summary (LLM-only step)
5. **Structure** – `ThesisStructuringService` maps keywords to fintech taxonomy:
   - Scores categories by keyword hits in summary
   - Returns top-3 categories per type (themes, risks, signals)
   - Extracts sources from article metadata (no LLM needed)
6. **Display** – `app.py` renders results in Streamlit UI

---

## Performance Considerations

- **Vectorstore Caching**: Built once per Streamlit session and reused across queries (significant speedup)
- **Top-K Retrieval**: Returns top-5 most relevant articles based on semantic similarity
- **Map-Reduce Summarization**: LangChain's chain handles large document sets efficiently
- **Pattern-Based Structuring**: Lightweight keyword matching (no LLM call) → instant structuring vs. previous LLM-based approach
- **Streaming**: Could be optimized with Streamlit streaming for real-time token output

## Design Patterns Used

- **Dependency Injection (DIP)**: All components depend on interfaces, not concrete implementations
- **Strategy Pattern**: Pluggable LLM/embedding/vectorstore providers via registries
- **Factory Pattern**: ServiceContainer creates and manages singleton instances
- **Interface Segregation**: Minimal interface methods (only what's needed)
- **Single Responsibility**: Each service has one clear responsibility

## Testing

- **72 unit tests** covering:
  - ThesisStructuringService: 24 tests (keyword mapping, category ranking, edge cases)
  - AppConfig: 17 tests (env var loading, validation, provider resolution)
  - ServiceContainer: 25 tests (registries, provider lookup, error handling)
  - Services: 6 tests (ingestion, retrieval, thesis generation)

Run tests with:
```bash
python run_tests.py
```
