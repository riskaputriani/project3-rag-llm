import streamlit as st
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

def get_vectorstore_from_text(text):
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    text_chunks = text_splitter.split_text(text)
    
    if not text_chunks:
        st.error("Could not create text chunks. Is the text long enough?")
        return None

    # Create a vector store from the chunks
    embeddings = OllamaEmbeddings(model="mistral", show_progress=True)
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        collection_name="local-rag" # Added collection name
    )
    return vector_store

def create_rag_chain(retriever):
    # This function creates the RAG chain
    try:
        # Define the prompt template for the conversational RAG chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Initialize the Ollama LLM
        llm = Ollama(model="mistral", temperature=0)

        # Create the document chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the retrieval chain
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None

# --- Streamlit App ---

st.set_page_config(page_title="RAG with Ollama", page_icon="🤖")
st.title("RAG with Ollama and Novel Chapters")

# Sidebar for text input
with st.sidebar:
    st.header("1. Add Your Novel")
    novel_text = st.text_area("Paste one or more chapters of your novel here:", height=300)
    process_button = st.button("Process Novel")
    
    if process_button and novel_text:
        with st.spinner("Chunking text and creating vector store... This may take a moment."):
            try:
                # Ensure the Ollama service is running and the model is available
                ollama.pull("mistral") 
            except Exception as e:
                st.error("Could not connect to Ollama. Please make sure Ollama is running and accessible.")
                st.stop()

            vector_store = get_vectorstore_from_text(novel_text)
            if vector_store:
                st.session_state.retriever = vector_store.as_retriever()
                st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                st.session_state.chat_history = []
                st.success("Novel processed successfully! You can now ask questions.")
            else:
                st.error("Failed to process the novel.")

# Main chat interface
st.header("2. Chat with Your Novel")

if "rag_chain" not in st.session_state:
    st.info("Please add your novel chapters and click 'Process Novel' to begin.")
else:
    # Display previous messages
    if "chat_history" in st.session_state:
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            else: # AIMessage
                with st.chat_message("assistant"):
                    st.write(msg.content)

    # Chat input
    user_query = st.chat_input("Ask a question about your novel...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                # Update chat history
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(response["answer"])

                with st.chat_message("assistant"):
                    st.write(response["answer"])
            
            except Exception as e:
                st.error(f"An error occurred: {e}")