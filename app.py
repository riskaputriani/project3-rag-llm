import streamlit as st
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# --- Using langchain-classic to avoid import errors ---
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

def get_vectorstore_from_text(text):
    """Splits text, creates embeddings, and returns a vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    text_chunks = text_splitter.split_text(text)
    
    if not text_chunks:
        st.error("Could not create text chunks. Is the text long enough?")
        return None

    embeddings = OllamaEmbeddings(model="mistral", show_progress=True)
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        collection_name="local-rag"
    )
    return vector_store

def create_rag_chain(retriever):
    """Creates a RetrievalQA chain using the classic approach."""
    try:
        llm = Ollama(model="mistral", temperature=0)

        # This prompt template is for the RetrievalQA chain
        prompt_template = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise."
            "\n\n"
            "Context: {context}"
            "\n\n"
            "Question: {question}"
            "\n\n"
            "Helpful Answer:"
        )
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Create the RetrievalQA chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False, # Set to True if you want to see the source docs
        )
        return rag_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None

# --- Streamlit App ---

st.set_page_config(page_title="Classic RAG with Ollama", page_icon="🤖")
st.title("Classic RAG with Ollama")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("1. Add Your Novel")
    novel_text = st.text_area("Paste chapters of your novel here:", height=300)
    if st.button("Process Novel"):
        if novel_text:
            with st.spinner("Processing novel... This may take a moment."):
                try:
                    ollama.pull("mistral")
                    vector_store = get_vectorstore_from_text(novel_text)
                    if vector_store:
                        st.session_state.retriever = vector_store.as_retriever()
                        st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                        st.session_state.chat_history = [AIMessage(content="I've processed the novel. How can I help you?")]
                        st.success("Novel processed! You can now ask questions.")
                    else:
                        st.error("Failed to create vector store.")
                except Exception as e:
                    st.error(f"Could not connect to Ollama or process the text. Please ensure Ollama is running. Error: {e}")
        else:
            st.warning("Please paste some text to process.")

st.header("2. Chat with Your Novel")

if "rag_chain" not in st.session_state:
    st.info("Begin by processing your novel in the sidebar.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.write(msg.content)

    # User input
    user_query = st.chat_input("Ask a question...")
    if user_query:
        st.chat_message("user").write(user_query)
        
        # --- Manual History Management ---
        # Format previous messages into a string for context
        history_str = ""
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                history_str += f"Human: {msg.content}\n"
            else:
                history_str += f"Assistant: {msg.content}\n"
        
        # Combine history with the new question
        full_query = f"Using the conversation history below, answer the new question.\n\nHistory:\n{history_str}\nNew Question: {user_query}"

        with st.spinner("Thinking..."):
            try:
                # The classic chain expects a 'query'
                response = st.session_state.rag_chain.invoke({"query": full_query})
                answer = response.get("result", "Sorry, I couldn't find an answer.")

                # Update history
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=answer))
                
                # Display final answer
                st.chat_message("assistant").write(answer)

            except Exception as e:
                st.error(f"An error occurred while invoking the chain: {e}")
