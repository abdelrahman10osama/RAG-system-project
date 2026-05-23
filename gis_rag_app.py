import streamlit as st
import os
import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# =========================
# 🔧 Streamlit Setup
# =========================
st.set_page_config(page_title="GIS RAG", page_icon="🗺️", layout="wide")
st.title("🗺️ Chat with Your GIS Documents (RAG System)")


# =========================
# 🔑 Sidebar (API + Upload)
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")
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
# ✂️ Text Splitter (IMPORTANT)
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# =========================
# 📚 Process PDFs → RAG Pipeline
# =========================
if uploaded_files and "vectorstore" not in st.session_state:

    with st.spinner("📚 Processing PDFs..."):

        all_chunks = []

        for uploaded_file in uploaded_files:

            # Save temp PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Split into CHUNKS (IMPORTANT FIX)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

            # Cleanup
            os.unlink(tmp_path)

        # =========================
        # 🔢 Embeddings
        # =========================
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )

        # =========================
        # 💾 Vector DB (FIXED)
        # =========================
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name="gis_rag_lab"
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.chunks_count = len(all_chunks)
        st.session_state.files_count = len(uploaded_files)

        # =========================
        # 📊 Stats (LAB REQUIREMENT)
        # =========================
        st.session_state.stats = {
            "total_questions": 0,
            "retrieved_chunks": 0
        }

    st.success(
        f"✅ Processed {len(uploaded_files)} files | "
        f"{len(all_chunks)} chunks created!"
    )


# =========================
# 🤖 LLM Setup
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

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if question := st.chat_input("Ask about your GIS documents..."):

        # Update stats
        st.session_state.stats["total_questions"] += 1

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching GIS knowledge base..."):

                # =========================
                # 🔍 RETRIEVAL STEP (RAG CORE)
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
                # 🧠 Prompt Engineering (RAG)
                # =========================
                prompt = f"""
You are a GIS expert assistant.

Use ONLY the context below to answer the question.
If answer is not in context, say:
"I don't have enough information in the documents."

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

                # =========================
                # 🤖 LLM Call
                # =========================
                response = llm.invoke(prompt)
                answer = response.content

                st.write(answer)

                # =========================
                # 📚 Sources (LAB REQUIREMENT)
                # =========================
                with st.expander("📚 Retrieved Sources"):
                    for i, doc in enumerate(docs, 1):
                        page = doc.metadata.get("page", "?")
                        st.write(f"**Source {i} - Page {page}**")
                        st.write(doc.page_content[:300] + "...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# =========================
# 📊 Sidebar Stats Display
# =========================
with st.sidebar:
    if "stats" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Session Stats")
        st.write("Questions:", st.session_state.stats["total_questions"])
        st.write("Chunks Retrieved:", st.session_state.stats["retrieved_chunks"])
        st.write("Files:", st.session_state.get("files_count", 0))
        st.write("Chunks:", st.session_state.get("chunks_count", 0))