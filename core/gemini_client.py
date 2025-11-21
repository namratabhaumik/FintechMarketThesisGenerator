# core/gemini_client.py
import re
import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain

logger = logging.getLogger(__name__)


def generate_summary(docs, model_name="gemini-2.5-flash"):
    """Summarize retrieved docs using Gemini."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    try:
        result = chain.invoke(docs)
        if isinstance(result, dict) and "output_text" in result:
            return result["output_text"]
        return str(result)
    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        raise


def generate_structured_thesis(topic: str, thesis_text: str, model_name="gemini-2.5-flash"):
    """Ask Gemini to output a structured JSON thesis."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    prompt = f"""
You are an expert VC analyst. Based on this summarized evidence about "{topic}":

{thesis_text}

Return a JSON object with keys:
- key_themes: list of 3 concise themes
- risks: list of 3 concise risks
- investment_signals: list of 3 startup focus areas
- sources: list of source titles or URLs
Only output valid JSON.
"""
    try:
        result = llm.invoke(prompt)
        raw = getattr(result, "content", str(result))
        cleaned = re.sub(r"^```json|```$", "", raw.strip(),
                         flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        return {"raw": raw, "json": parsed}
    except Exception as e:
        logger.warning(f"Failed to parse Gemini output as JSON: {e}")
        return {"raw": raw, "json": None}
