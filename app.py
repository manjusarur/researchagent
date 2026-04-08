import streamlit as st
from groq import Groq
import json
from duckduckgo_search import DDGS
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime

# ====================== TEXT SANITIZER ======================
def clean_text(text: str) -> str:
    """Remove non-ASCII characters (fix UnicodeEncodeError)"""
    if not text:
        return ""
    return text.encode("ascii", "ignore").decode()

# ====================== PAGE ======================
st.set_page_config(page_title="Research Intelligence Agent", layout="wide")
st.title("🔍 Research Intelligence Agent")
st.markdown("**Gathers latest web data + RAG on your PDFs • Delivers professional insights report**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Groq API Key", type="password", value=st.session_state.get("groq_key", ""))

    if api_key:
        st.session_state.groq_key = api_key
        st.success("Groq connected")

    st.divider()
    st.header("Upload Documents (RAG)")
    uploaded_files = st.file_uploader(
        "Upload PDFs for your research",
        accept_multiple_files=True,
        type=["pdf"]
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} PDF(s) uploaded")

# ====================== SESSION ======================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====================== EMBEDDINGS ======================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ====================== PDF PROCESSING ======================
def process_pdfs(files):
    chunks = []

    for file in files:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        text = clean_text(text)

        chunk_size = 800
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

    if chunks:
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))

        st.session_state.vectorstore = index
        st.session_state.chunks = chunks

        st.success(f"Processed {len(chunks)} chunks")

    return chunks

# Process PDFs
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Processing PDFs..."):
        process_pdfs(uploaded_files)

# ====================== TOOLS ======================
def search_web(query: str) -> str:
    try:
        query = clean_text(query)
        results = list(DDGS().text(query, max_results=4))

        output = "\n\n".join(
            [f"Source: {r['title']}\n{r['body']}" for r in results]
        )

        return clean_text(output)

    except:
        return "No web results found."


def retrieve_from_documents(query: str) -> str:
    if st.session_state.vectorstore is None:
        return "No documents uploaded."

    query = clean_text(query)

    query_embedding = embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state.vectorstore.search(query_embedding, k=3)

    relevant_chunks = [st.session_state.chunks[i] for i in indices[0]]

    return clean_text("\n\n---\n\n".join(relevant_chunks))

# ====================== AGENT ======================
def run_research_agent(topic: str):
    if not st.session_state.get("groq_key"):
        return "❌ Please enter your Groq API key."

    try:
        client = Groq(api_key=st.session_state.groq_key)

        topic = clean_text(topic)
        
        # Step 1: Gather web search results
        st.info("🔍 Searching the web...")
        web_results = search_web(topic)
        
        # Step 2: Retrieve from documents (RAG)
        st.info("📚 Retrieving from documents...")
        doc_results = retrieve_from_documents(topic)
        
        # Step 3: Send to Groq for analysis
        st.info("🧠 Analyzing with AI...")
        
        analysis_prompt = f"""You are a professional Research Intelligence Agent.
Analyze the following research data on '{topic}' and provide:

1. **Key Findings** - Main discoveries and facts
2. **Trends** - Emerging patterns and developments  
3. **Insights** - What this means for the future
4. **Recommendations** - Actionable next steps

RESEARCH DATA:

Web Search Results:
{web_results}

Document Analysis:
{doc_results}

Provide a comprehensive, professional insights report with proper formatting and citations."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": clean_text(analysis_prompt)}],
            temperature=0.5,
            max_tokens=2000
        )

        report = response.choices[0].message.content
        st.success("✓ Analysis complete")
        
        # Format the report
        formatted_report = f"""
╔════════════════════════════════════════════════════════════════╗
║         RESEARCH INTELLIGENCE AGENT - INSIGHTS REPORT         ║
╚════════════════════════════════════════════════════════════════╝

📊 RESEARCH TOPIC: {topic}
📅 GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

────────────────────────────────────────────────────────────────

{clean_text(report)}

────────────────────────────────────────────────────────────────
Model: llama-3.3-70b-versatile | Sources: Web + Documents
"""
        
        return formatted_report

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.error(error_msg)
        return error_msg

# ====================== UI ======================
st.subheader("What would you like to research?")

research_topic = st.text_input(
    "Research Topic",
    placeholder="Latest advancements in Agentic AI in 2026",
    value="Latest advancements in Agentic AI in 2026"
)

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("Run Research Agent", type="primary", use_container_width=True):
        if not st.session_state.get("groq_key"):
            st.error("Enter Groq API key")
        else:
            with st.spinner("Researching..."):
                final_report = run_research_agent(research_topic)

            st.session_state.chat_history.append({
                "timestamp": datetime.now().strftime("%H:%M"),
                "topic": research_topic,
                "report": final_report
            })

            st.subheader("Professional Insights Report")
            st.markdown(final_report)
            
            # Add download button
            st.download_button(
                label="📥 Download Report (TXT)",
                data=final_report,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

with col2:
    if st.button("Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ====================== HISTORY ======================
if st.session_state.chat_history:
    st.divider()
    st.subheader("Previous Research")

    for entry in reversed(st.session_state.chat_history):
        with st.expander(f"[{entry['timestamp']}] {entry['topic'][:60]}"):
            st.write(entry["report"])
            st.download_button(
                label="📥 Download",
                data=entry["report"],
                file_name=f"research_{entry['timestamp'].replace(':', '')}_{entry['topic'][:20]}.txt",
                mime="text/plain",
                key=f"download_{entry['timestamp']}"
            )

st.caption("Groq + FAISS RAG • Streamlit App")