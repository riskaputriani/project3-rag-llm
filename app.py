import streamlit as st
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

# --- Konfigurasi dan Inisialisasi Model ---

# Path untuk menyimpan cache model
CACHE_DIR = Path("./models_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Nama model embedding dan model LLM dari Hugging Face
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_REPO = "TheBloke/Llama-2-7B-Chat-GGUF"
LLM_MODEL_FILE = "llama-2-7b-chat.Q4_K_M.gguf"


@st.cache_resource
def load_embedding_model():
    """Mengunduh dan me-load model embedding dari Hugging Face."""
    st.write(f"Memuat model embedding: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=str(CACHE_DIR),
        model_kwargs={"device": "cpu"},  # Gunakan CPU
    )
    st.write("Model embedding berhasil dimuat.")
    return embeddings


@st.cache_resource
def load_llm():
    """Mengunduh (jika perlu) dan me-load model LLM GGUF."""
    st.write("Memuat model LLM lokal...")
    llm = CTransformers(
        model=LLM_MODEL_REPO,
        model_file=LLM_MODEL_FILE,
        model_type="llama",
        config={
            "max_new_tokens": 256,
            "temperature": 0.7,
            "context_length": 2048,
        },
    )
    st.write("Model LLM berhasil dimuat.")
    return llm


# --- Fungsi Inti RAG ---

def get_text_chunks(text_data: str):
    """Memecah teks menjadi potongan-potongan (chunks)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text_data)
    return chunks


def get_text_chunks_from_docs(documents):
    """Memecah dokumen (dari loader) menjadi potongan-potongan (chunks)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store_from_text(text_chunks, embeddings):
    """Membuat vector store dari potongan teks."""
    if not text_chunks:
        st.warning("Tidak ada teks untuk diproses. Silakan masukkan teks atau unggah file.")
        return None
    st.write("Membuat vector store...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    st.write("Vector store berhasil dibuat.")
    return vectorstore


def create_vector_store_from_docs(doc_chunks, embeddings):
    """Membuat vector store dari potongan dokumen."""
    if not doc_chunks:
        st.warning("Gagal memproses dokumen. Pastikan format file didukung (PDF, TXT).")
        return None
    st.write("Membuat vector store dari dokumen...")
    vectorstore = FAISS.from_documents(documents=doc_chunks, embedding=embeddings)
    st.write("Vector store berhasil dibuat.")
    return vectorstore


def rag_answer(question: str, llm, vector_store, chat_history):
    """Lakukan retrieval + generate jawaban dengan manual chain (tanpa langchain.chains)."""
    # Ambil dokumen paling relevan
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    # Ambil sedikit riwayat percakapan sebelumnya
    history_text = ""
    for q, a in chat_history[-5:]:
        history_text += f"User: {q}\nAI: {a}\n"

    prompt = f"""
Kamu adalah asisten AI yang menjawab berdasarkan dokumen yang diberikan.

Riwayat percakapan sebelumnya (jika ada):
{history_text}

Konteks dokumen yang relevan:
{context}

Pertanyaan user: {question}

Jawab dalam bahasa Indonesia secara jelas dan ringkas.
Jika jawaban tidak ada di dalam konteks dokumen, jujur katakan bahwa kamu tidak yakin berdasarkan dokumen yang tersedia.
"""

    # CTransformers di LangChain baru biasanya pakai .invoke()
    try:
        response = llm.invoke(prompt)
    except AttributeError:
        # fallback kalau versi yang terinstall pakai __call__
        response = llm(prompt)

    # Kalau response berupa dict (beberapa versi), ambil "content" / "text"
    if isinstance(response, dict):
        response_text = response.get("content") or response.get("text") or str(response)
    else:
        response_text = str(response)

    return response_text


# --- Antarmuka Streamlit ---

st.set_page_config(page_title="Chat dengan Dokumen Lokal (RAG)", layout="wide")
st.title("Chatbot RAG dengan LLM Lokal")
st.markdown("Unggah dokumen atau masukkan teks, lalu ajukan pertanyaan tentang isinya.")

# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# Sidebar untuk input data
with st.sidebar:
    st.header("Sumber Data")

    input_method = st.radio("Pilih metode input:", ("Unggah File", "Ketik Teks"))

    if input_method == "Ketik Teks":
        user_text = st.text_area("Ketik atau tempel teks di sini:")
        if st.button("Proses Teks"):
            if user_text:
                with st.spinner("Memproses teks..."):
                    embeddings = load_embedding_model()
                    text_chunks = get_text_chunks(user_text)
                    vector_store = create_vector_store_from_text(text_chunks, embeddings)
                    if vector_store:
                        llm = load_llm()
                        st.session_state.vector_store = vector_store
                        st.session_state.llm = llm
                        st.success("Teks berhasil diproses! Anda bisa mulai chat.")
            else:
                st.warning("Teks tidak boleh kosong.")

    elif input_method == "Unggah File":
        uploaded_files = st.file_uploader(
            "Unggah file (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Proses File"):
            if uploaded_files:
                with st.spinner("Memproses file..."):
                    embeddings = load_embedding_model()

                    temp_files = []
                    temp_dir = Path("temp_docs")
                    temp_dir.mkdir(exist_ok=True)
                    for uploaded_file in uploaded_files:
                        temp_path = temp_dir / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_files.append(str(temp_path))

                    docs = []
                    for path in temp_files:
                        if path.endswith(".pdf"):
                            loader = PyPDFLoader(path)
                        else:
                            loader = TextLoader(path, encoding="utf-8")
                        docs.extend(loader.load())

                    doc_chunks = get_text_chunks_from_docs(docs)
                    vector_store = create_vector_store_from_docs(doc_chunks, embeddings)

                    if vector_store:
                        llm = load_llm()
                        st.session_state.vector_store = vector_store
                        st.session_state.llm = llm
                        st.success("File berhasil diproses! Anda bisa mulai chat.")

                    # Hapus file sementara
                    for path in temp_files:
                        Path(path).unlink()
            else:
                st.warning("Silakan unggah setidaknya satu file.")

    st.divider()
    if st.button("Bersihkan Riwayat Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# --- Tampilan Chat ---

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user
if user_prompt := st.chat_input("Tanyakan sesuatu mengenai dokumen Anda..."):
    if st.session_state.vector_store is None or st.session_state.llm is None:
        st.warning("Harap proses dokumen atau teks terlebih dahulu.")
    else:
        # Simpan pesan user
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Jawab dengan RAG manual
        with st.chat_message("assistant"):
            with st.spinner("Memikirkan jawaban..."):
                response = rag_answer(
                    question=user_prompt,
                    llm=st.session_state.llm,
                    vector_store=st.session_state.vector_store,
                    chat_history=st.session_state.chat_history,
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Update chat history untuk konteks berikutnya
        st.session_state.chat_history.append((user_prompt, response))
