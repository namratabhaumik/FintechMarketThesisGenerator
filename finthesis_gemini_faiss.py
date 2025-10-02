"""
finthesis_gemini_faiss.py
Core logic for the Fintech Market Thesis Generator.

- Build FAISS vector store from sample articles
- Retrieve relevant chunks for a topic
- Use Gemini (via LangChain) to summarize + structure into JSON
"""

import os
import json
import re
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain

# -----------------------
# 1) Sample corpus
# -----------------------
SAMPLE_ARTICLES = [
    {
        "title": "Real-time payments adoption",
        "source": "example.com/article1",
        "text": "Real-time payments networks are being adopted by banks and fintechs..."
    },
    {
        "title": "New regulatory updates in payments",
        "source": "example.com/article2",
        "text": "New EU regulations around PSD3 are creating compliance work..."
    },
    {
        "title": "B2B payments startups raising capital",
        "source": "example.com/article3",
        "text": "Several startups focused on reconciliation and payouts for marketplaces..."
    },
    {
        "title": "Embedded finance trend",
        "source": "example.com/article4",
        "text": "Embedded finance continues to expand: platforms embed payment/credit rails..."
    },
]

# -----------------------
# 2) Build FAISS vector store
# -----------------------


def build_vectorstore(articles):
    """Convert article dicts into FAISS vector store."""
    docs = [
        Document(
            page_content=f"{a['title']}\n\n{a['text']}",
            metadata={"source": a["source"], "title": a["title"]}
        )
        for a in articles
    ]

    if not docs:
        raise ValueError("No documents provided for vectorstore.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Use HuggingFace embeddings locally (no API cost)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# -----------------------
# 3) Generate structured thesis
# -----------------------


def generate_thesis(topic, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(topic)

    if not docs:
        return {"error": f"No relevant documents found for topic: {topic}"}

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    chain_result = chain.invoke(docs)

    # Make sure we have plain text
    if isinstance(chain_result, dict) and "output_text" in chain_result:
        thesis_text = chain_result["output_text"]
    else:
        thesis_text = str(chain_result)

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
    result = llm.invoke(prompt)
    raw_output = getattr(result, "content", str(result))
    cleaned = re.sub(r"^```json|```$", "", raw_output.strip(),
                     flags=re.MULTILINE).strip()

    parsed_json = None
    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return {
        "raw": raw_output,
        "json": parsed_json,
        "summary": thesis_text
    }


# -----------------------
# 4) Run standalone
# -----------------------
if __name__ == "__main__":
    assert os.getenv(
        "GOOGLE_API_KEY"), "Set env var GOOGLE_API_KEY before running."
    print("Building vectorstore from sample articles...")
    vs = build_vectorstore(SAMPLE_ARTICLES)
    topic = "B2B Payments"
    print(f"\nGenerating thesis for: {topic}\n")
    thesis = generate_thesis(topic, vs)
    print(json.dumps(thesis, indent=2))
