"""
app.py
Streamlit frontend for the Fintech Market Thesis Generator.

- Lets user input a fintech topic
- Uses FAISS + Gemini to build a mini "market thesis"
- Displays structured JSON (themes, risks, signals, sources) + analyst summary
"""

import streamlit as st
import json
from finthesis_gemini_faiss import build_vectorstore, SAMPLE_ARTICLES, generate_thesis

st.set_page_config(page_title="Fintech Market Thesis Generator", layout="wide")

st.title("📊 Fintech Market Thesis Generator")
st.write("Generate an investor-style thesis for a fintech sector using Gemini + FAISS.")

# Input topic
topic = st.text_input("Enter a fintech topic:", "B2B Payments")

if st.button("Generate Thesis"):
    try:
        with st.spinner("Building thesis..."):
            vs = build_vectorstore(SAMPLE_ARTICLES)
            result = generate_thesis(topic, vs)

        # Always show raw JSON first
        st.subheader("📄 Thesis (Raw JSON)")
        st.code(result["raw"], language="json")

        # If structured JSON parsed correctly
        if result["json"]:
            parsed = result["json"]

            st.subheader("🔑 Key Themes")
            st.write("\n".join(
                [f"- {t}" for t in parsed.get("key_themes", [])]) or "No themes found.")

            st.subheader("⚠️ Risks")
            st.write(
                "\n".join([f"- {r}" for r in parsed.get("risks", [])]) or "No risks found.")

            st.subheader("🚀 Investment Signals")
            st.write("\n".join(
                [f"- {s}" for s in parsed.get("investment_signals", [])]) or "No signals found.")

            st.subheader("🔗 Sources")
            st.write("\n".join(
                [f"- {src}" for src in parsed.get("sources", [])]) or "No sources found.")
        else:
            st.warning(
                "⚠️ Could not parse structured JSON. Showing raw output above.")

        # Analyst summary is always useful
        st.subheader("📝 Analyst Summary")
        st.write(result.get("summary", "No summary available."))

    except Exception as e:
        st.error(f"An error occurred while generating the thesis: {str(e)}")
