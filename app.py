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
st.title("📄 Chat with PDF")
st.markdown("Upload a PDF and ask questions about its content using RAG.")

with st.sidebar:
    st.header("1) Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("Clear chat history"):
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
st.header("2) Ask Questions")
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
user_query = st.chat_input("Ask something about your PDF...")

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
