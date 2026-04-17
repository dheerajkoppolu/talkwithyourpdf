"""Streamlit Chat with PDF/Image app using LangChain + Chroma + Groq."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
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


st.set_page_config(page_title="Chat with PDF/Image", page_icon="📄", layout="wide")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
VISION_MODEL_NAME = "llama-3.2-90b-vision-preview"
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"]
GROQ_API_KEY_ENV = "GROQ_API_KEY"


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def write_pdf_to_tempfile(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        return tmp_file.name


def load_pdf(pdf_path: str) -> List[Document]:
    return PyPDFLoader(pdf_path).load()


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_source_type(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    if uploaded_file.type == "application/pdf":
        return "pdf"
    guessed_mime, _ = mimetypes.guess_type(uploaded_file.name)
    if guessed_mime == "application/pdf":
        return "pdf"
    return "image"


def extract_text_from_image_with_llm(image_bytes: bytes, mime_type: str) -> str:
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    llm = ChatGroq(model=VISION_MODEL_NAME, temperature=0, api_key=get_groq_api_key())
    response = llm.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all readable text and important visual details from this image. "
                            "Return plain text only for downstream retrieval."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                    },
                ]
            )
        ]
    )
    return str(response.content).strip()


@st.cache_resource(show_spinner=False)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=True)
def build_vectorstore_cached(
    source_hash: str,
    file_bytes: bytes,
    source_type: str,
    source_name: str,
    mime_type: str,
) -> Chroma:
    _ = source_hash

    if source_type == "pdf":
        temp_pdf_path = write_pdf_to_tempfile(file_bytes)
        try:
            loaded_docs = load_pdf(temp_pdf_path)
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    else:
        extracted_text = extract_text_from_image_with_llm(file_bytes, mime_type)
        loaded_docs = [
            Document(
                page_content=extracted_text,
                metadata={"source": source_name, "page": 1, "source_type": "image"},
            )
        ]

    chunks = split_documents(loaded_docs)
    if not chunks:
        raise ValueError("The uploaded file appears empty or unreadable.")

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings_model(),
        collection_name=f"src_{source_hash[:12]}",
    )


def get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL_NAME, temperature=0, api_key=get_groq_api_key())


def get_groq_api_key() -> str:
    """Read Groq API key from environment variable."""
    api_key = os.getenv(GROQ_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(
            f"Missing API key. Set {GROQ_API_KEY_ENV} before running the app."
        )
    return api_key


def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """
You are a helpful assistant for question-answering over a document/image.

Use only the context below to answer the question.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""".strip()
    )


def format_docs(documents: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def answer_question(question: str, vectorstore: Chroma) -> Tuple[str, List[Document]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)
    chain = get_prompt() | get_llm()
    response = chain.invoke({"context": format_docs(retrieved_docs), "question": question})
    return response.content, retrieved_docs


def process_uploaded_source(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
) -> None:
    """Process PDF/image upload and initialize vector store when source changes."""
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        st.error("Uploaded file is empty. Please upload a valid PDF or image.")
        return

    source_hash = file_hash(file_bytes)
    source_type = get_source_type(uploaded_file)
    mime_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or "image/jpeg"

    if st.session_state.active_source_hash == source_hash:
        return

    try:
        spinner_text = (
            "Processing PDF and building embeddings..."
            if source_type == "pdf"
            else "Reading image content and building embeddings..."
        )
        with st.spinner(spinner_text):
            st.session_state.vectorstore = build_vectorstore_cached(
                source_hash=source_hash,
                file_bytes=file_bytes,
                source_type=source_type,
                source_name=uploaded_file.name,
                mime_type=mime_type,
            )
            st.session_state.active_source_hash = source_hash
            st.session_state.messages = []
        st.success("Source processed successfully. You can start asking questions.")
    except Exception as exc:
        st.session_state.vectorstore = None
        st.session_state.active_source_hash = None
        st.error(f"Failed to process source: {exc}")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "active_source_hash" not in st.session_state:
    st.session_state.active_source_hash = None


st.title("📄🖼️ Chat with PDF or Image")
st.markdown("Upload a PDF/image and ask questions about its content using RAG.")

with st.sidebar:
    st.header("1) Upload Source")
    uploaded_file = st.file_uploader("Choose a PDF or image", type=SUPPORTED_UPLOAD_TYPES)

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.success("Chat history cleared.")


if uploaded_file is not None:
    process_uploaded_source(uploaded_file)


st.header("2) Ask Questions")
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


user_query = st.chat_input("Ask something about your uploaded PDF/image...")
if user_query is not None:
    if not user_query.strip():
        st.warning("Please enter a non-empty question.")
    elif st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF/image before asking questions.")
    else:
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
                    st.markdown(answer)

                    with st.expander(f"Retrieved context chunks (top {TOP_K})"):
                        if retrieved_docs:
                            for idx, doc in enumerate(retrieved_docs, start=1):
                                source = doc.metadata.get("source", "Unknown source")
                                page = doc.metadata.get("page", "N/A")
                                st.markdown(f"**Chunk {idx}** — Source: `{source}`, Page: `{page}`")
                                st.write(doc.page_content)
                                st.divider()
                        else:
                            st.write("No context chunks were retrieved.")

                    st.session_state.messages.append(AIMessage(content=answer))
                except Exception as exc:
                    error_message = f"Error while generating answer: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(AIMessage(content=error_message))


st.markdown("---")
st.caption("Setup: install dependencies, set GROQ_API_KEY, then run `streamlit run app.py`.")
