# 📊 Fintech Market Thesis Generator

A demo app that uses **LangChain, FAISS, HuggingFace embeddings, and Gemini** to generate an investor-style market thesis for fintech topics.  
Built as a proof-of-concept for showing how LLMs + vector databases can be combined for applied research and decision support.

👉 **Live Demo**: [Streamlit Cloud App](https://namratabhaumik-fintechmarketthesisgenerator-app-qqdzns.streamlit.app/)

---

## Features

- **LangChain Orchestration** – integrates retrieval, summarization, and prompting.
- **Vector Database (FAISS)** – stores fintech articles and retrieves the most relevant evidence for a topic.
- **Local Embeddings** – uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (no paid API calls).
- **Gemini (Google Generative AI)** – structures the thesis into a clean JSON with:
  - Key themes
  - Risks
  - Investment signals
  - Sources
- **Streamlit UI** – simple frontend to enter a topic and view structured outputs.

---

## Tech Stack

- **LangChain** (chains, retrievers, and integrations)
- **FAISS** (vector database for semantic retrieval)
- **HuggingFace embeddings** (open-source transformer embeddings)
- **Gemini via LangChain** (LLM for structured outputs)
- **Streamlit** (interactive web UI, deployed free on Streamlit Cloud)

---

## Project Structure

### Streamlit frontend: app.py

### Core pipeline: FAISS + Gemini summarization: finthesis_gemini_faiss.py

### Python dependencies: requirements.txt

---

## Run Locally

```bash
git clone https://github.com/namratabhaumik/FintechMarketThesisGenerator.git
cd FintechMarketThesisGenerator
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key_here"
streamlit run app.py

```
