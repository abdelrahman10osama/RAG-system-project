import streamlit as st
import os
import tempfile

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================
# 🔧 Streamlit Setup
# =========================
st.set_page_config(page_title="GIS RAG", page_icon="🗺️", layout="wide")
st.title("🗺️ Chat with Your GIS Documents (RAG System)")


# =========================
# 🔑 Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key (for Chat LLM)", type="password")
    uploaded_files = st.file_uploader(
        "📄 Upload GIS PDF(s)",
        type="pdf",
        accept_multiple_files=True
    )

if not api_key:
    st.warning("⚠️ Please enter your Gemini API key")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# =========================
# ✂️ Text Splitter
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# =========================
# 🔢 Embeddings (FIXED - NO GEMINI)
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# 📚 Process PDFs
# =========================
if uploaded_files and "vectorstore" not in st.session_state:

    with st.spinner("📚 Processing PDFs..."):

        all_chunks = []

        for uploaded_file in uploaded_files:

            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Chunking (IMPORTANT)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

            os.unlink(tmp_path)

        # =========================
        # 💾 Vector DB
        # =========================
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name="gis_rag_lab"
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.chunks_count = len(all_chunks)
        st.session_state.files_count = len(uploaded_files)

        # Stats (LAB requirement)
        st.session_state.stats = {
            "total_questions": 0,
            "retrieved_chunks": 0
        }

    st.success(
        f"✅ Processed {len(uploaded_files)} files | {len(all_chunks)} chunks"
    )


# =========================
# 🤖 LLM (Gemini for answers only)
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# =========================
# 💬 Chat UI
# =========================
if "vectorstore" in st.session_state:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # input
    if question := st.chat_input("Ask about your GIS documents..."):

        st.session_state.stats["total_questions"] += 1

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching GIS knowledge base..."):

                # =========================
                # 🔍 Retrieval (CORE RAG)
                # =========================
                docs = st.session_state.vectorstore.similarity_search(
                    question,
                    k=3
                )

                st.session_state.stats["retrieved_chunks"] += len(docs)

                context = "\n\n".join(
                    [d.page_content for d in docs]
                )

                # =========================
                # 🧠 Prompt
                # =========================
                prompt = f"""
You are a GIS expert assistant.

Use ONLY the context below.
If answer is not found, say:
"I don't have enough information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""

                # =========================
                # 🤖 LLM call
                # =========================
                response = llm.invoke(prompt)
                answer = response.content

                st.write(answer)

                # =========================
                # 📚 Sources
                # =========================
                with st.expander("📚 Retrieved Sources"):
                    for i, doc in enumerate(docs, 1):
                        page = doc.metadata.get("page", "?")
                        st.write(f"**Source {i} (Page {page})**")
                        st.write(doc.page_content[:300] + "...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# =========================
# 📊 Stats Sidebar
# =========================
with st.sidebar:
    if "stats" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Stats")

        st.write("Questions:", st.session_state.stats["total_questions"])
        st.write("Chunks retrieved:", st.session_state.stats["retrieved_chunks"])
        st.write("Files:", st.session_state.get("files_count", 0))
        st.write("Chunks:", st.session_state.get("chunks_count", 0))