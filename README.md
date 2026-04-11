# FinThesis

> An agentic pipeline that reads fintech news so you don't have to.

[![FinThesis Demo](https://img.youtube.com/vi/73SnVdzeVrg/hqdefault.jpg)](https://youtu.be/73SnVdzeVrg)

Try it live: [Streamlit Cloud](https://namratabhaumik-fintechmarketthesisgenerator-app-qqdzns.streamlit.app/)

---

## What it does

Give FinThesis a fintech market query - "crypto in Asia", "BNPL in Europe", "embedded insurance" - and it produces a structured investment thesis built from real-time fintech news.

Each thesis includes:

- A pattern-structured breakdown: key themes, risks, and investment signals mapped to a fintech taxonomy
- A rule-based opportunity score (0–5), confidence level, and Pursue / Investigate / Skip recommendation
- The source articles used to reach the conclusion
- An optional refinement loop - if the output isn't right, pick from a fixed set of feedback reasons and a LangGraph agent rewrites the thesis via tool calls

## How it works

- **Agentic refinement with LangGraph.** The refinement layer is an agent, not a pipeline re-run. It has access to `refine_thesis`, `structure_thesis`, and `score_opportunity` as tools and decides which to call. Tool calling uses `InjectedState` so the graph state is passed to tools cleanly.
- **Hallucination detection on structured tool calls.** Every agent turn is validated against the actual tool calls in the message history, not the free-text output. If the agent invents a tool or a result, it is caught before the user sees it.
- **Rule-based opportunity scoring, not LLM-generated.** The score, confidence, and recommendation come from a deterministic formula weighted by detected signals, themes, and risks. An investment recommendation black-boxed inside an LLM is not useful to an analyst; this one is auditable and human-interpretable.
- **Dual-mode summarization.** Gemini (LLM) or a local keyword-scored extractive summarizer. The local mode was originally a workaround for the Gemini free tier; it now serves as a legitimate no-API fallback and a safety net when cost limits kick in.
- **Langfuse observability.** Every agent run, tool call, and LLM call is traced so behavior is inspectable end-to-end, not just guessable from final output.

For detailed architecture and data flow, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Try it locally

```bash
git clone https://github.com/namratabhaumik/FintechMarketThesisGenerator.git
cd FintechMarketThesisGenerator
pip install -r requirements.txt
```

Create a `.env` file:

```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

For no-API mode, set `LLM_PROVIDER=local` and omit the Gemini keys.

Run it:

```bash
streamlit run app.py
```

## Stack

Python, LangChain, LangGraph, FAISS, FastEmbed (ONNX), Gemini, Langfuse, Streamlit.

## License

MIT License - see LICENSE for details.
