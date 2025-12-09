import streamlit as st
import faiss
import numpy as np
import torch

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------
# Konstanta model (akan di-download otomatis dari Hugging Face)
# -----------------------------------------------------------
EMBED_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# -----------------------------------------------------------
# Cache: load embedding model sekali per server
# -----------------------------------------------------------
@st.cache_resource(show_spinner="Mengunduh & memuat embedding Qwen...")
def load_embed_model():
    model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)
    return model

# -----------------------------------------------------------
# Cache: load LLM Qwen sekali per server
# -----------------------------------------------------------
@st.cache_resource(show_spinner="Mengunduh & memuat LLM Qwen... (bisa agak lama)")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model

# -----------------------------------------------------------
# Utility: chunking teks
# -----------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------------------------------------
# Build FAISS index dari list teks
# -----------------------------------------------------------
def build_faiss_index(texts: List[str]):
    embed_model = load_embed_model()
    embeddings = embed_model.encode(texts, batch_size=16, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# -----------------------------------------------------------
# Retrieve top-k context dari query
# -----------------------------------------------------------
def retrieve(query: str, top_k: int = 4):
    if "faiss_index" not in st.session_state:
        st.warning("Index belum dibuat. Silakan upload teks dan klik 'Bangun RAG Index' dulu.")
        return []

    embed_model = load_embed_model()
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")

    index = st.session_state.faiss_index
    distances, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(st.session_state.chunks):
            continue
        results.append(st.session_state.chunks[idx])
    return results

# -----------------------------------------------------------
# Build prompt untuk Qwen (dengan history + context)
# -----------------------------------------------------------
def build_prompt_with_history(question: str, contexts: List[str]) -> List[Dict[str, str]]:
    # Riwayat chat terakhir (misal 3 pasangan QA)
    history = st.session_state.chat_history
    max_turns = 3
    trimmed_history = history[-max_turns * 2:]  # user+assistant

    # Bentuk string history sederhana
    history_lines = []
    for msg in trimmed_history:
        role_label = "User" if msg["role"] == "user" else "Asisten"
        history_lines.append(f"{role_label}: {msg['content']}")
    history_str = "\n".join(history_lines) if history_lines else "(belum ada)"

    # Context dari dokumen
    context_str = "\n\n---\n\n".join(contexts) if contexts else "(tidak ada konteks)"

    system_prompt = (
        "Kamu adalah asisten yang menjawab berdasarkan konteks dokumen (novel/teks) yang diberikan.\n"
        "Jika jawaban tidak ada di konteks, katakan bahwa kamu tidak yakin dan jangan mengarang."
    )

    user_content = (
        f"Riwayat percakapan sejauh ini:\n{history_str}\n\n"
        f"Konteks dokumen yang relevan:\n{context_str}\n\n"
        f"Pertanyaan terbaru user: {question}\n\n"
        "Jawab dengan bahasa Indonesia yang jelas dan ringkas."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages

# -----------------------------------------------------------
# Panggil Qwen LLM
# -----------------------------------------------------------
def call_qwen(messages: List[Dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    tokenizer, model = load_llm()

    # Qwen pakai chat template dari tokenizer
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer

# -----------------------------------------------------------
# Streamlit main app
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title="Qwen RAG di Streamlit", page_icon="📚")

    st.title("📚 Qwen RAG LLM di Streamlit")
    st.caption("Upload teks → Bangun RAG → Chat dengan Qwen yang ingat history (per sesi).")

    # Inisialisasi session_state
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {role, content}

    st.sidebar.header("Pengaturan Chat")
    top_k = st.sidebar.slider("Top K dokumen", 1, 10, 4)
    max_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 512, step=64)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.7, step=0.1)

    # -------------------------
    # Bagian upload teks / input teks
    # -------------------------
    st.subheader("1. Upload / Input Teks untuk RAG")

    uploaded_files = st.file_uploader(
        "Upload file teks (.txt). Bisa lebih dari satu.",
        type=["txt"],
        accept_multiple_files=True,
    )

    manual_text = st.text_area(
        "Atau tempel teks langsung di sini",
        height=150,
        placeholder="Tempel novel / dokumen di sini jika tidak memakai file .txt...",
    )

    if st.button("Bangun RAG Index"):
        all_texts = []

        # Dari file upload
        if uploaded_files:
            for f in uploaded_files:
                content = f.read().decode("utf-8", errors="ignore")
                if content.strip():
                    all_texts.append(content)

        # Dari text area
        if manual_text.strip():
            all_texts.append(manual_text)

        if not all_texts:
            st.error("Tidak ada teks yang bisa dipakai. Upload file atau isi text area dulu.")
        else:
            with st.spinner("Melakukan chunking & membangun index RAG..."):
                chunks = []
                for t in all_texts:
                    chunks.extend(chunk_text(t, chunk_size=500, overlap=100))

                st.session_state.chunks = chunks
                st.session_state.faiss_index = build_faiss_index(chunks)

            st.success(f"Index berhasil dibuat dengan {len(st.session_state.chunks)} chunk.")

    st.markdown("---")

    # -------------------------
    # Bagian Chat
    # -------------------------
    st.subheader("2. Chat dengan Qwen (RAG)")

    col1, col2 = st.columns(2)
    with col1:
        clear = st.button("🧹 Clear chat")
    with col2:
        st.write("")  # spacer

    if clear:
        st.session_state.chat_history = []
        st.success("Chat dibersihkan (dokumen & index tetap ada).")

    # Tampilkan riwayat chat
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    # Input chat
    user_input = st.chat_input("Tulis pertanyaan kamu di sini...")

    if user_input:
        if st.session_state.faiss_index is None:
            st.warning("Index belum ada. Bangun dulu dari teks di atas.")
        else:
            # Simpan pertanyaan ke history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Mencari konteks & memanggil Qwen..."):
                    contexts = retrieve(user_input, top_k=top_k)
                    messages = build_prompt_with_history(user_input, contexts)
                    answer = call_qwen(
                        messages,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                    )

                st.markdown(answer)

            # Simpan jawaban ke history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Opsional: tampilkan konteks
            if contexts:
                with st.expander("Lihat konteks dokumen yang dipakai"):
                    for i, c in enumerate(contexts):
                        st.markdown(f"**Context {i+1}:**")
                        st.write(c)
                        st.markdown("---")

if __name__ == "__main__":
    main()
