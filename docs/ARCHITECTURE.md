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
│          [5] STRUCTURE INTO JSON (Gemini)                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Ask Gemini to structure summary into JSON              │ │
│  │ • Extract: key_themes, risks, signals, sources           │ │
│  │ • Parse JSON output                                       │ │
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
        ┌─────────┴──────────┐
        │                    │
        ▼                    ▼
┌─────────────────┐  ┌──────────────────┐
│  INGESTION      │  │  RETRIEVAL       │
│  (ingestion.py) │  │  (retrieval.py)  │
│                 │  │                  │
│ ┌─────────────┐ │  │ ┌──────────────┐ │
│ │ Feedparser  │ │  │ │ FAISS Index  │ │
│ │ RSS Feeds   │ │  │ │              │ │
│ └────────┬────┘ │  │ │ Semantic     │ │
│          │      │  │ │ Search       │ │
│ ┌────────▼────┐ │  │ └──────────────┘ │
│ │BeautifulSoup│ │  │                  │
│ │Scrape Text  │ │  │ ┌──────────────┐ │
│ └─────────────┘ │  │ │HuggingFace   │ │
│                 │  │ │Embeddings    │ │
└────────┬────────┘  │ │(all-MiniLM)  │ │
         │           │ └──────────────┘ │
         │           └──────────────────┘
         │                    │
         └─────────┬──────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │   UTILS (utils.py)  │
         │                     │
         │ • Normalize articles│
         │ • Setup logging     │
         └─────────────────────┘
                   │
                   ▼
         ┌──────────────────────────┐
         │  GEMINI CLIENT           │
         │  (gemini_client.py)      │
         │                          │
         │  ┌──────────────────────┐│
         │  │ Summarization Chain  ││
         │  │ (Gemini 2.5 Flash)   ││
         │  └──────────────────────┘│
         │                          │
         │  ┌──────────────────────┐│
         │  │ Structured Output    ││
         │  │ (JSON Parsing)       ││
         │  └──────────────────────┘│
         └──────────────────────────┘
```

---

## Step-by-Step Processing

1. **Fetch Articles** – `core/ingestion.py` fetches live articles from TechCrunch RSS
2. **Normalize** – `core/utils.py` standardizes article format
3. **Vectorize** – `core/retrieval.py` creates FAISS embeddings (one-time per session)
4. **Retrieve** – FAISS finds top-5 articles semantically similar to your query
5. **Summarize** – `core/gemini_client.py` uses Gemini to create analyst summary
6. **Structure** – Gemini formats summary into JSON (themes, risks, signals, sources)
7. **Display** – `app.py` renders results in Streamlit UI

---

## Performance Considerations

- **Vectorstore Caching**: Built once per Streamlit session and reused across queries (significant speedup)
- **Top-K Retrieval**: Returns top-5 most relevant articles based on semantic similarity
- **Map-Reduce Summarization**: LangChain's chain handles large document sets efficiently
- **Streaming**: Could be optimized with Streamlit streaming for real-time token output
