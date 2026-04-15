"""Streamlit Chat with PDF app using LangChain + Chroma + Groq."""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import List, Tuple

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------------
# Streamlit page configuration
# ------------------------------
st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")


# ------------------------------
# Constants
# ------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"


# ------------------------------
# Utility functions
# ------------------------------
def pdf_file_hash(file_bytes: bytes) -> str:
    """Create a stable hash for uploaded PDF bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def write_pdf_to_tempfile(file_bytes: bytes) -> str:
    """Write uploaded bytes to a temp PDF file path for PyPDFLoader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        return tmp_file.name


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """Load PDF pages and split text into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


@st.cache_resource(show_spinner=False)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Create and cache the embedding model once per app session."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=True)
def build_vectorstore_cached(file_hash: str, file_bytes: bytes) -> Chroma:
    """Build and cache Chroma DB based on PDF hash to avoid recomputation."""
    # NOTE: file_hash is intentionally part of the cache signature.
    _ = file_hash

    temp_pdf_path = write_pdf_to_tempfile(file_bytes)
    try:
        chunks = load_and_split_pdf(temp_pdf_path)
        if not chunks:
            raise ValueError("The uploaded PDF appears empty or unreadable.")

        embeddings = get_embeddings_model()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"pdf_{file_hash[:12]}",
        )
        return vectorstore
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def get_llm() -> ChatGroq:
    """Initialize Groq chat model."""
    return ChatGroq(model=GROQ_MODEL_NAME, temperature=0)


def get_prompt() -> ChatPromptTemplate:
    """Prompt enforcing grounded answers from retrieved context only."""
    return ChatPromptTemplate.from_template(
        """
You are a helpful assistant for question-answering over a PDF.

Use the context below to answer the user's question.
Answer ONLY from the provided context. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""".strip()
    )


def format_docs(documents: List[Document]) -> str:
    """Convert retrieved docs into a single context string."""
    return "\n\n".join(doc.page_content for doc in documents)


def answer_question(question: str, vectorstore: Chroma) -> Tuple[str, List[Document]]:
    """Run retrieval + generation and return answer plus retrieved docs."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)

    prompt = get_prompt()
    llm = get_llm()
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content, retrieved_docs


def inject_supercomputer_theme() -> None:
    """Inject custom CSS for an interactive 'AI brain' visual style."""
    st.markdown(
        """
<style>
/* Main canvas */
.stApp {
    background: radial-gradient(circle at 20% 20%, #0d1b4d 0%, #060b21 38%, #03050f 100%);
    color: #e8f2ff;
}

/* Animated neural glow layer */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
      radial-gradient(circle at 15% 35%, rgba(47, 128, 237, 0.25), transparent 32%),
      radial-gradient(circle at 80% 20%, rgba(123, 97, 255, 0.22), transparent 30%),
      radial-gradient(circle at 70% 75%, rgba(0, 215, 201, 0.15), transparent 38%);
    animation: brainPulse 10s ease-in-out infinite;
    z-index: 0;
}
@keyframes brainPulse {
    0%, 100% { opacity: 0.55; transform: scale(1); }
    50% { opacity: 0.85; transform: scale(1.03); }
}

/* Ensure content sits above glow layer */
div[data-testid="stAppViewContainer"] > .main,
div[data-testid="stSidebar"] { position: relative; z-index: 1; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8, 16, 45, 0.98), rgba(8, 12, 30, 0.98));
    border-right: 1px solid rgba(76, 122, 255, 0.35);
}

/* Hero card */
.hero-card {
    border: 1px solid rgba(100, 165, 255, 0.45);
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(10, 24, 60, 0.88), rgba(9, 14, 36, 0.9));
    box-shadow: 0 0 22px rgba(56, 126, 255, 0.18), inset 0 0 18px rgba(33, 79, 180, 0.2);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #dff0ff;
    margin-bottom: 0.25rem;
}

.hero-subtitle {
    color: #96c2ff;
    margin-bottom: 0.75rem;
}

.status-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.status-pill {
    padding: 0.3rem 0.6rem;
    border-radius: 999px;
    border: 1px solid rgba(110, 173, 255, 0.45);
    font-size: 0.78rem;
    background: rgba(10, 26, 65, 0.7);
}

/* Chat bubble upgrade */
div[data-testid="stChatMessage"] {
    border: 1px solid rgba(97, 151, 255, 0.18);
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(8, 19, 47, 0.82), rgba(8, 13, 30, 0.7));
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
}

div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #eef5ff;
}

/* Buttons and inputs */
.stButton > button,
.stDownloadButton > button {
    border: 1px solid rgba(103, 177, 255, 0.55);
    color: #e8f3ff;
    background: linear-gradient(135deg, rgba(16, 44, 102, 0.8), rgba(12, 26, 63, 0.9));
}
.stButton > button:hover {
    border-color: rgba(124, 195, 255, 0.9);
    box-shadow: 0 0 16px rgba(74, 146, 255, 0.35);
}

/* Hide default top menu/deploy for cleaner sci-fi frame */
#MainMenu, header[data-testid="stHeader"] { visibility: hidden; }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_panel() -> None:
    """Render AI themed header panel."""
    vector_ready = "READY" if st.session_state.vectorstore is not None else "WAITING FOR PDF"
    turns = len(st.session_state.messages)
    st.markdown(
        f"""
<div class="hero-card">
    <div class="hero-title">🧠 Neural PDF Intelligence Console</div>
    <div class="hero-subtitle">Query your document like an AI supercomputer searching memory shards.</div>
    <div class="status-row">
        <span class="status-pill">Vector Brain: {vector_ready}</span>
        <span class="status-pill">Conversation Turns: {turns}</span>
        <span class="status-pill">Retriever Top-K: {TOP_K}</span>
        <span class="status-pill">Model: {GROQ_MODEL_NAME}</span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------
# Session state initialization
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "active_pdf_hash" not in st.session_state:
    st.session_state.active_pdf_hash = None


# ------------------------------
# UI
# ------------------------------
inject_supercomputer_theme()
render_hero_panel()
st.markdown("### ⚡ Mission Control")
st.markdown("Upload a PDF, then ask questions while the AI retrieves memory chunks from its vector brain.")

with st.sidebar:
    st.header("🛰️ Document Dock")
    st.caption("Feed one PDF into the intelligence core.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("♻️ Clear chat history"):
        st.session_state.messages = []
        st.success("Chat history cleared.")


# ------------------------------
# PDF processing (cached)
# ------------------------------
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        st.error("Uploaded file is empty. Please upload a valid PDF.")
    else:
        file_hash = pdf_file_hash(file_bytes)

        if st.session_state.active_pdf_hash != file_hash:
            try:
                with st.spinner("Processing PDF and building embeddings..."):
                    st.session_state.vectorstore = build_vectorstore_cached(file_hash, file_bytes)
                    st.session_state.active_pdf_hash = file_hash
                    st.session_state.messages = []  # reset chat when new PDF uploaded
                st.success("PDF processed successfully. You can start asking questions.")
            except Exception as exc:
                st.session_state.vectorstore = None
                st.session_state.active_pdf_hash = None
                st.error(f"Failed to process PDF: {exc}")


# ------------------------------
# Chat history display
# ------------------------------
st.markdown("### 💬 Ask Questions")
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# ------------------------------
# Chat input + RAG response
# ------------------------------
user_query = st.chat_input("Ask the neural console about your PDF...")

if user_query is not None:
    if not user_query.strip():
        st.warning("Please enter a non-empty question.")
    elif st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF before asking questions.")
    else:
        # Add user message to chat state.
        st.session_state.messages.append(HumanMessage(content=user_query))

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    answer, retrieved_docs = answer_question(
                        question=user_query,
                        vectorstore=st.session_state.vectorstore,
                    )

                    # Display model answer.
                    st.markdown(answer)

                    # Optional expandable section to inspect retrieved chunks.
                    with st.expander("Retrieved context chunks (top 3)"):
                        if retrieved_docs:
                            for idx, doc in enumerate(retrieved_docs, start=1):
                                source = doc.metadata.get("source", "Unknown source")
                                page = doc.metadata.get("page", "N/A")
                                st.markdown(f"**Chunk {idx}** — Source: `{source}`, Page: `{page}`")
                                st.write(doc.page_content)
                                st.divider()
                        else:
                            st.write("No context chunks were retrieved.")

                    # Save assistant response in chat state.
                    st.session_state.messages.append(AIMessage(content=answer))

                except Exception as exc:
                    error_message = f"Error while generating answer: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(AIMessage(content=error_message))


# ------------------------------
# Footer instructions in UI
# ------------------------------
st.markdown("---")
st.caption(
    "Setup: install dependencies, set GROQ_API_KEY, then run `streamlit run app.py`."
)
