# app.py
"""
Streamlit frontend for FinThesis (Refactored with SOLID principles)
Uses dependency injection to manage all dependencies.
"""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from config.settings import AppConfig
from core.models.thesis import StructuredThesis
from core.utils.logging import setup_logging
from dependency_injection.container import ServiceContainer

# Load environment variables from .env
load_dotenv()

# ---- Setup ----
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FinThesis - AI Market Research Assistant", layout="wide")

st.title("FinThesis: Fintech Market Research Assistant")
st.markdown(
    "Generate structured **market theses** from live fintech news using semantic retrieval "
    "and keyword-driven analysis — powered by a local summarizer or an LLM of your choice."
)

# Cache control for testing
if st.button("🔄 Clear Cache & Reset"):
    st.cache_resource.clear()
    if "vectorstore_built" in st.session_state:
        del st.session_state["vectorstore_built"]
    st.success("Cache cleared! Refresh the page.")
    st.stop()

# Warning about API key
if not os.getenv("GOOGLE_API_KEY"):
    st.warning(
        "No GOOGLE_API_KEY detected in environment. "
        "Gemini calls will fail without a valid API key. (Set GOOGLE_API_KEY in your environment.)"
    )


# === Initialize DI Container (cached for performance) ===
@st.cache_resource
def get_container() -> ServiceContainer:
    """Initialize and cache service container."""
    config = AppConfig.from_env()
    container = ServiceContainer(config)
    logger.info("Service container initialized and cached")
    return container


container = get_container()


# === UI Helper Functions ===

def display_structured_thesis(thesis: StructuredThesis):
    """Display structured thesis in Streamlit UI.

    Args:
        thesis: StructuredThesis object to display.
    """
    # Display opportunity score and recommendation prominently
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Investment Score", f"{thesis.opportunity_score}/5")

    with col2:
        st.metric("Confidence Level", f"{int(thesis.confidence_level * 100)}%")

    with col3:
        # Color code the recommendation
        if thesis.recommendation == "Pursue":
            st.success(f"✅ {thesis.recommendation}")
        elif thesis.recommendation == "Investigate":
            st.info(f"🔍 {thesis.recommendation}")
        else:
            st.warning(f"⏭️ {thesis.recommendation}")

    # Approval toggle — unique key per thesis ensures it resets for each new generation
    col_toggle = st.columns(3)[0]
    with col_toggle:
        toggle_key = f"approval_toggle_{st.session_state.get('thesis_count', 0)}"
        st.toggle("Approved", value=False, key=toggle_key)

    st.divider()

    st.subheader("Key Themes")
    themes = thesis.key_themes
    st.write("\n".join(f"- {t}" for t in themes) or "No themes found.")

    st.subheader("Risks")
    risks = thesis.risks
    st.write("\n".join(f"- {r}" for r in risks) or "No risks found.")

    if thesis.key_risk_factors:
        st.caption(f"⚠️ Key Risk Factors: {', '.join(thesis.key_risk_factors)}")

    st.subheader("Investment Signals")
    signals = thesis.investment_signals
    st.write("\n".join(f"- {s}" for s in signals) or "No signals found.")


# === Main Application ===

query = st.text_input(
    "Enter a market topic or question:",
    placeholder="e.g., Future of Digital Lending in Asia"
)

if st.button("Generate Thesis"):
    if not query or not query.strip():
        st.warning("Please enter a non-empty query.")
    else:
        try:
            # Get services from DI container
            ingestion_service = container.get_ingestion_service()
            retrieval_service = container.get_retrieval_service()
            thesis_service = container.get_thesis_service()

            # Step 1: Fetch articles
            with st.spinner("Fetching latest fintech news from RSS feeds..."):
                articles = ingestion_service.fetch_articles(query="fintech", limit=5)

                if not articles:
                    st.warning("No articles found. Please try again later.")
                    st.stop()

                # Store articles in session state for display outside button block
                st.session_state["articles"] = articles

            # Step 2: Build vectorstore (cache in session)
            if "vectorstore_built" not in st.session_state:
                with st.spinner("Building FAISS vectorstore (one-time)..."):
                    documents = ingestion_service.convert_to_documents(articles)
                    retrieval_service.build_vectorstore(documents)
                    st.session_state["vectorstore_built"] = True
                    logger.info("Vectorstore built and cached in session")

            # Step 3: Retrieve relevant documents
            with st.spinner("Retrieving relevant context from vectorstore..."):
                docs = retrieval_service.retrieve(query, k=5)

                if not docs:
                    st.warning("No relevant documents found for this query.")
                    st.stop()

            # Step 4: Generate thesis
            with st.spinner("Generating market thesis with Gemini..."):
                thesis = thesis_service.generate_thesis(query, docs)
                st.session_state["generated_thesis"] = thesis
                # Increment counter so the toggle gets a new unique key (resets automatically)
                st.session_state["thesis_count"] = st.session_state.get("thesis_count", 0) + 1

        except Exception as exc:
            logger.exception("Error while generating thesis")
            st.error(f"An unexpected error occurred: {str(exc)}")

# Step 5: Display articles (outside button block so they persist across reruns)
if "articles" in st.session_state:
    st.subheader("Latest Fintech Articles")
    articles = st.session_state["articles"]
    for article in articles:
        if article.url:
            st.markdown(f"• [{article.title}]({article.url})")
        else:
            st.markdown(f"• {article.title}")

# Step 6: Display results (outside button block so toggle doesn't disappear)
if "generated_thesis" in st.session_state:
    thesis = st.session_state["generated_thesis"]

    if thesis.raw_output:
        st.subheader("Raw Summary")
        st.code(thesis.raw_output, language="json")

    if thesis.key_themes:
        st.success("Structured thesis generated successfully")
        display_structured_thesis(thesis)
    else:
        st.warning("Could not parse structured output. See raw output above.")
else:
    st.info("Enter a query and click 'Generate Thesis' to start.")

st.markdown("---")
