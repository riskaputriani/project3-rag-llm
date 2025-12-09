
import streamlit as st
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import create_conversational_retrieval_chain
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

# Fungsi untuk mengunduh dan me-load model embedding
# @st.cache_resource memastikan model hanya di-load sekali
@st.cache_resource
def load_embedding_model():
    """Mengunduh dan me-load model embedding dari Hugging Face."""
    st.write(f"Memuat model embedding: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=str(CACHE_DIR),
        model_kwargs={'device': 'cpu'} # Gunakan CPU
    )
    st.write("Model embedding berhasil dimuat.")
    return embeddings

# Fungsi untuk mengunduh dan me-load model LLM
# @st.cache_resource memastikan model hanya di-load sekali
@st.cache_resource
def load_llm():
    """Mengunduh (jika perlu) dan me-load model LLM GGUF."""
    st.write("Memuat model LLM lokal...")
    # CTransformers adalah library untuk menjalankan model GGUF secara efisien di CPU
    llm = CTransformers(
        model=LLM_MODEL_REPO,
        model_file=LLM_MODEL_FILE,
        model_type="llama",
        config={
            'max_new_tokens': 256,
            'temperature': 0.7,
            'context_length': 2048
        }
    )
    st.write("Model LLM berhasil dimuat.")
    return llm

# --- Fungsi Inti RAG ---

def get_text_chunks(text_data):
    """Memecah teks menjadi potongan-potongan (chunks)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text_data)
    return chunks

def get_text_chunks_from_docs(documents):
    """Memecah dokumen (dari loader) menjadi potongan-potongan (chunks)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(text_chunks, embeddings):
    """Membuat vector store dari potongan teks."""
    if not text_chunks:
        st.warning("Tidak ada teks untuk diproses. Silakan masukkan teks atau unggah file.")
        return None
    st.write("Membuat vector store...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    st.write("Vector store berhasil dibuat.")
    return vectorstore

def create_doc_vector_store(doc_chunks, embeddings):
    """Membuat vector store dari potongan dokumen."""
    if not doc_chunks:
        st.warning("Gagal memproses dokumen. Pastikan format file didukung (PDF, TXT).")
        return None
    st.write("Membuat vector store dari dokumen...")
    vectorstore = FAISS.from_documents(documents=doc_chunks, embedding=embeddings)
    st.write("Vector store berhasil dibuat.")
    return vectorstore


def create_conversational_chain(vector_store, llm):
    """Membuat chain untuk percakapan RAG."""
    retriever = vector_store.as_retriever()
    return create_conversational_retrieval_chain(llm, retriever)

# --- Antarmuka Streamlit ---

st.set_page_config(page_title="Chat dengan Dokumen Lokal (RAG)", layout="wide")
st.title("Chatbot RAG dengan LLM Lokal")
st.markdown("Unggah dokumen atau masukkan teks, lalu ajukan pertanyaan tentang isinya.")

# Inisialisasi session state
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
                    vector_store = create_vector_store(text_chunks, embeddings)
                    if vector_store:
                        llm = load_llm()
                        st.session_state.conversation_chain = create_conversational_chain(vector_store, llm)
                        st.success("Teks berhasil diproses! Anda bisa mulai chat.")
            else:
                st.warning("Teks tidak boleh kosong.")

    elif input_method == "Unggah File":
        uploaded_files = st.file_uploader(
            "Unggah file (PDF, TXT)", 
            type=["pdf", "txt"], 
            accept_multiple_files=True
        )
        if st.button("Proses File"):
            if uploaded_files:
                with st.spinner("Memproses file..."):
                    embeddings = load_embedding_model()
                    
                    # Simpan file yang diunggah sementara agar PyPDFLoader bisa membacanya
                    temp_files = []
                    temp_dir = Path("temp_docs")
                    temp_dir.mkdir(exist_ok=True)
                    for uploaded_file in uploaded_files:
                        temp_path = temp_dir / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_files.append(str(temp_path))

                    # Gunakan PyPDFLoader dengan path file sementara
                    docs = []
                    for path in temp_files:
                        if path.endswith(".pdf"):
                            loader = PyPDFLoader(path)
                        else:
                            loader = TextLoader(path, encoding="utf-8")
                        docs.extend(loader.load())

                    doc_chunks = get_text_chunks_from_docs(docs)
                    vector_store = create_doc_vector_store(doc_chunks, embeddings)
                    
                    if vector_store:
                        llm = load_llm()
                        st.session_state.conversation_chain = create_conversational_chain(vector_store, llm)
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

# Terima input dari user
if user_prompt := st.chat_input("Tanyakan sesuatu mengenai dokumen Anda..."):
    if st.session_state.conversation_chain is None:
        st.warning("Harap proses dokumen atau teks terlebih dahulu.")
    else:
        # Tambahkan pertanyaan user ke riwayat
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Dapatkan jawaban dari RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Memikirkan jawaban..."):
                result = st.session_state.conversation_chain({
                    "question": user_prompt,
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]
                st.markdown(response)

                # Tambahkan jawaban AI ke riwayat
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Perbarui history untuk chain
        st.session_state.chat_history.append((user_prompt, response))
        
