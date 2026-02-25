# FinThesis Architecture & Data Flow

This document describes how FinThesis works, covering both the **Gemini (LLM)** and **Local (no-API)** summarization flows.

---

## Two Summarization Flows

The application supports two modes, selected via `LLM_PROVIDER` in `.env`. Steps 1–3 and 5–6 are identical; only step 4 differs.

| | Gemini flow | Local flow |
|---|---|---|
| **LLM_PROVIDER** | `gemini` | `local` |
| **API key required** | Yes (`GOOGLE_API_KEY`) | No |
| **Summarization** | Gemini via LangChain map-reduce | Keyword-scored sentence extraction |
| **Summary quality** | Analyst-style prose | Extractive (top sentences by fintech keyword score) |
| **Latency** | API round-trip (~2–5 s) | Local computation (~0.1 s) |
| **Cost** | Per-token billing | Free |

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
│                    [1] FETCH ARTICLES                           │
│  • Fetch from TechCrunch RSS feeds (feedparser)                 │
│  • Scrape article text using BeautifulSoup                      │
│  • Clean text (ads, boilerplate, whitespace)                    │
│  • Normalize to Article(title, url, text, source)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                [2] VECTORIZATION & STORAGE                      │
│  • Convert articles to LangChain Documents                      │
│  • Chunk text (800 chars, 100 overlap)                          │
│  • Generate embeddings (HuggingFace: all-MiniLM-L6-v2)         │
│  • Index in FAISS for semantic search                           │
│  • Cache vectorstore in session (reuse across queries)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  [3] SEMANTIC RETRIEVAL                         │
│  • Convert user query to embedding                              │
│  • Search FAISS index for top-5 relevant chunks                 │
│  • Rank by semantic similarity                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────┴──────────────┐
              │                             │
    LLM_PROVIDER=gemini           LLM_PROVIDER=local
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│  [4a] GEMINI SUMMARY    │   │  [4b] LOCAL EXTRACTIVE SUMMARY  │
│                         │   │                                 │
│ • Send docs to Gemini   │   │ • Split all doc text into       │
│   via LangChain         │   │   sentences                     │
│   map-reduce chain      │   │ • Score each sentence by        │
│ • Returns analyst-style │   │   fintech keyword count         │
│   prose summary         │   │ • Extract top 7 sentences       │
│ • Requires GOOGLE_API   │   │ • Deduplicate (>70% word        │
│   _KEY                  │   │   overlap threshold)            │
│                         │   │ • Preserve document order       │
│                         │   │ • No API call, no cost          │
└────────────┬────────────┘   └────────────────┬────────────────┘
             │                                 │
             └──────────────┬──────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│     [5] STRUCTURE INTO THESIS (Pattern Matching)                │
│  • Lowercase summary                                            │
│  • Score 32 fintech categories against summary keywords:        │
│    - 12 themes  (AI, Payments, Blockchain, Lending...)          │
│    - 10 risks   (Regulatory, Cybersecurity, Credit...)          │
│    - 10 signals (B2B Expansion, AI Tools, Emerging Markets...)  │
│  • Return top-3 per category type by keyword hit count          │
│  • Extract source URLs from document metadata                   │
│  • No LLM call - same logic for both flows                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│   [6] SCORE OPPORTUNITY (Rule-Based Scoring)                    │
│  • Calculate opportunity score (0-5 scale):                     │
│    - Base: 2.5                                                  │
│    - Signal boost: +0.75 per signal (max 3)                     │
│    - Theme boost: +0.25 per theme (max 3)                       │
│    - Risk penalty: -0.25 per risk (max 3)                       │
│  • Calculate confidence (0-100%):                               │
│    - Source factor: min(sources/5.0, 1.0)                       │
│    - Signal factor: min(signals/3.0, 1.0)                       │
│    - Risk factor: max(0.5, 1.0 - risks/4.0)                     │
│    - Weighted: 40% sources + 40% signals + 20% risks            │
│  • Generate recommendation:                                     │
│    - "Pursue" if score >= 3.75                                  │
│    - "Investigate" if 2.5 <= score < 3.75                       │
│    - "Skip" if score < 2.5                                      │
│  • Deterministic & auditable; human makes final decision        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    [7] DISPLAY RESULTS                          │
│  • Show fetched article source links                            │
│  • Display raw summary output                                   │
│  • Render structured thesis with scoring:                       │
│    - Themes, risks, signals (from step 5)                       │
│    - Opportunity score, confidence, recommendation (from step 6)│
│    - Key risk factors highlighted                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)                         │
│  - Input form for market queries                                 │
│  - Display results                                               │
│  - Session management (caching vectorstore)                      │
└─────────────────┬────────────────────────────────────────────────┘
                  │
        ┌─────────▼──────────────────────────┐
        │                                    │
        ▼                                    ▼
┌────────────────────────────┐   ┌──────────────────────────────┐
│  SERVICE CONTAINER         │   │  CONFIGURATION (config/)     │
│  (dependency_injection/)   │   │                              │
│                            │   │  Load from .env:             │
│  LLM_PROVIDER_REGISTRY:    │   │  • LLM_PROVIDER              │
│  • "gemini" -> GeminiLLM   │   │  • EMBEDDING_PROVIDER        │
│  • "local"  -> LocalSumm.  │   │  • GOOGLE_API_KEY (gemini)   │
│                            │   │  • GEMINI_MODEL   (gemini)   │
│  EMBEDDING_PROVIDER_REG:   │   │  • EMBEDDING_MODEL           │
│  • "huggingface" -> HFEmb. │   │                              │
└────────────────────────────┘   └──────────────────────────────┘
        │
        │ Creates & Injects
        │
        ▼
┌────────────────────────────────────────────────────────┐
│         THESIS GENERATOR SERVICE                       │
│  (ThesisGeneratorService)                              │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 1. ArticleIngestionService                      │  │
│  │    ├── IArticleSource  (RSSArticleSource)       │  │
│  │    └── IWebScraper     (BeautifulSoupScraper)   │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 2. DocumentRetrievalService                     │  │
│  │    ├── IEmbeddingModel (HuggingFaceEmbeddings)  │  │
│  │    └── IVectorStore    (FAISSVectorStore)        │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 3. ILanguageModel  <- selected by LLM_PROVIDER  │  │
│  │    ├── GeminiLanguageModel  (LLM_PROVIDER=gemini)│  │
│  │    │     map-reduce chain via LangChain          │  │
│  │    └── LocalSummarizerModel (LLM_PROVIDER=local) │  │
│  │          keyword-scored extractive summarization │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 4. ThesisStructuringService  (both flows)       │  │
│  │    ├── IScoringStrategy (KeywordCountScoring)   │  │
│  │    └── CategoryMappings (12+10+10 categories)   │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 5. OpportunityScoringService (rule-based)       │  │
│  │    • Score: 0-5 scale (themes, signals, risks)  │  │
│  │    • Confidence: 0-100% (sources, signals, risk)│  │
│  │    • Recommendation: Pursue/Investigate/Skip    │  │
│  │    • Deterministic & auditable logic            │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Processing

1. **Fetch Articles** – `ArticleIngestionService` fetches live articles from TechCrunch RSS, scrapes full text, cleans noise via `clean_article_text()`
2. **Vectorize** – `DocumentRetrievalService` chunks documents and builds a FAISS index with HuggingFace embeddings (cached per session)
3. **Retrieve** – FAISS finds top-5 chunks semantically similar to the user's query
4. **Summarize** – determined by `LLM_PROVIDER`:
   - **`gemini`**: `GeminiLanguageModel` sends docs to Gemini via a LangChain map-reduce chain → analyst prose summary
   - **`local`**: `LocalSummarizerModel` scores every sentence by fintech keyword count, extracts the top 7, deduplicates by word overlap → extractive summary, no API call
5. **Structure** – `ThesisStructuringService` receives the summary (identical interface for both flows) and keyword-scores it against 32 fintech categories; returns top-3 per type
6. **Score** – `OpportunityScoringService` generates an AI-native investment recommendation:
   - **Opportunity Score** (0–5): Base 2.5 + signal boost (0.75 per signal) + theme boost (0.25 per theme) − risk penalty (0.25 per risk)
   - **Confidence** (0–100%): Weighted average of source coverage (40%), signal strength (40%), and risk balance (20%)
   - **Recommendation**: "Pursue" (≥3.75), "Investigate" (2.5–3.75), or "Skip" (<2.5)
   - **Rule-based & deterministic**: Auditable scoring logic; human makes final yes/no decision
7. **Display** – `app.py` renders themes, risks, signals, scores, confidence, recommendation, and source URLs in Streamlit

---

## Performance Considerations

- **Vectorstore Caching** – Built once per Streamlit session; reused across queries
- **Gemini Latency** – ~2–5 s API round-trip; suited for quality-first use cases
- **Local Latency** – ~0.1 s; suited for offline use, CI, or cost-sensitive deployments
- **Pattern-Based Structuring** – Lightweight keyword matching, instant for both flows
- **Top-K Retrieval** – Returns top-5 most relevant chunks; configurable in `retrieval_service.py`

---

## Design Patterns Used

- **Dependency Injection (DIP)** – All components depend on interfaces, not concrete implementations
- **Strategy Pattern** – `ILanguageModel` is swapped at runtime via `LLM_PROVIDER_REGISTRY`; scoring is injectable via `IScoringStrategy`
- **Factory Pattern** – `ServiceContainer` creates and manages lazy-loaded singleton instances
- **Interface Segregation** – Minimal interface methods (only what's needed)
- **Single Responsibility** – Each service has one clear responsibility

---

## Testing

Run all tests:
```bash
python run_tests.py
```

174 unit tests, all pure Python (no network calls):

| File | Tests | Covers |
|---|---|---|
| `test_opportunity_scoring.py` | 26 | Scoring formula, confidence calculation, recommendations |
| `test_thesis_structuring.py` | 31 | Category matching, ranking, fallback mechanism, edge cases |
| `test_services.py` | 12 | Ingestion, document conversion, thesis generation with scoring |
| `test_text_utils.py` | 19 | Ad/boilerplate removal, whitespace normalisation |
| `test_local_summarizer.py` | 23 | Sentence splitting, keyword scoring, deduplication, summarize |
| `test_keyword_scoring.py` | 12 | `KeywordCountScoringStrategy` counting and edge cases |
| `test_models.py` | 13 | `Article` validation, `StructuredThesis` defaults |
| `test_config.py` | 17 | Env var loading, provider resolution, missing var errors |
| `test_container.py` | 25 | Provider registries, singleton caching, error messages |
