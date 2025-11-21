# app.py
"""
Streamlit frontend for FinThesis
Uses core modules:
 - core.utils: setup_logging, normalize_articles
 - core.retrieval: build_vectorstore
 - core.gemini_client: generate_summary, generate_structured_thesis
"""

import logging
import os
import json
import streamlit as st

from core.utils import setup_logging, normalize_articles
from core.retrieval import build_vectorstore
from core.gemini_client import generate_summary, generate_structured_thesis
from core.ingestion import fetch_live_articles


# ---- Setup ----
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FinThesis - AI Market Research Assistant", layout="wide")

st.title("FinThesis: AI-Powered Fintech Market Research Assistant")
st.markdown(
    "Generate structured **market theses** from financial context using semantic retrieval "
    "and Gemini-powered reasoning."
)

# quick note about API key
if not os.getenv("GOOGLE_API_KEY"):
    st.warning(
        "No GOOGLE_API_KEY detected in environment. "
        "Gemini calls will fail without a valid API key. (Set GOOGLE_API_KEY in your environment.)"
    )

# Input
query = st.text_input("Enter a market topic or question:",
                      placeholder="e.g., Future of Digital Lending in Asia")

# Helper: safe display of parsed JSON fields


def display_parsed_thesis(parsed: dict):
    st.subheader("Key Themes")
    themes = parsed.get("key_themes", [])
    st.write("\n".join(f"- {t}" for t in themes) or "No themes found.")

    st.subheader("Risks")
    risks = parsed.get("risks", [])
    st.write("\n".join(f"- {r}" for r in risks) or "No risks found.")

    st.subheader("Investment Signals")
    signals = parsed.get("investment_signals", [])
    st.write("\n".join(f"- {s}" for s in signals) or "No signals found.")

    st.subheader("Sources")
    sources = parsed.get("sources", [])
    st.write("\n".join(f"- {s}" for s in sources) or "No sources found.")


# Main action
if st.button("Generate Thesis"):
    if not query or not query.strip():
        st.warning("Please enter a non-empty query.")
    else:
        try:
            # Fetch live articles from RSS feeds
            with st.spinner("Fetching latest fintech news from RSS feeds..."):
                live_articles = fetch_live_articles(limit=5)
                articles = normalize_articles(live_articles)
                
                if not articles:
                    st.warning("No articles found. Please try again later.")

            # Show clickable article links
            if articles:
                st.subheader("Latest Fintech Articles")
                for a in articles[:5]:  # show all fetched articles
                    if a.get("url"):
                        st.markdown(f"• [{a['title']}]({a['url']})")
                    else:
                        st.markdown(f"• {a['title']}")

            # Build or reuse vectorstore (cache in session_state)
            if "vectorstore" not in st.session_state:
                with st.spinner("Building FAISS vectorstore (one-time)..."):
                    vs = build_vectorstore(articles)
                    st.session_state["vectorstore"] = vs
                    logger.info(
                        "Vectorstore built and cached in session_state.")
            else:
                vs = st.session_state["vectorstore"]

            # Retrieve relevant docs
            with st.spinner("Retrieving relevant context from vectorstore..."):
                retriever = vs.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(query)
                if not docs:
                    st.warning("No relevant documents found for this query.")
                    docs = []

            # Summarize retrieved docs
            with st.spinner("Summarizing retrieved context with Gemini..."):
                thesis_text = generate_summary(docs)
                if not thesis_text:
                    st.error(
                        "Failed to generate a summary from retrieved documents.")
                    raise RuntimeError("Empty summary returned by Gemini.")

            # Ask Gemini to structure the thesis into JSON
            with st.spinner("Structuring thesis into JSON..."):
                structured = generate_structured_thesis(query, thesis_text)
                raw_output = structured.get("raw", "")
                parsed = structured.get("json", None)

            # Show outputs
            st.subheader("Raw LLM Output")
            st.code(raw_output or "No raw output available.", language="json")

            st.subheader("Analyst Summary (condensed)")
            st.write(thesis_text or "No summary available.")

            if parsed:
                st.success("Parsed structured thesis")
                display_parsed_thesis(parsed)
            else:
                st.warning(
                    "Could not parse structured JSON from the model. See raw output above.")

        except Exception as exc:
            logger.exception("Error while generating thesis")
            st.error(f"An unexpected error occurred: {str(exc)}")

else:
    st.info("Enter a query and click 'Generate Thesis' to start.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangChain, FAISS, and Gemini API")
