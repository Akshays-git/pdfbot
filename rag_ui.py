import os
import streamlit as st
import chromadb
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

# Page configuration
st.set_page_config(page_title="PDF Knowledge Base", page_icon="📚", layout="wide")
st.title("📚 PDF Knowledge Base")

# Sidebar for uploading PDFs and configurations
with st.sidebar:
    st.header("Configuration")
    
    # PDF upload section
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    # Configure model parameters
    st.subheader("Model Settings")
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
    
    # Embedding model selection
    embedding_options = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
    }
    selected_embedding = st.selectbox(
        "Embedding Model",
        list(embedding_options.keys()),
        index=0
    )
    
    # LLM model selection
    llm_options = ["llama3.2", "llama3.1", "mistral"]
    selected_llm = st.selectbox("LLM Model", llm_options, index=0)
    
    process_button = st.button("Process PDFs")

# Main content area
main_container = st.container()

# Initialize session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Function to save uploaded PDFs
def save_uploaded_files(uploaded_files):
    pdf_folder = "./my_pdfs"
    os.makedirs(pdf_folder, exist_ok=True)
    
    # Clear existing PDFs
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            os.remove(os.path.join(pdf_folder, file))
    
    # Save new PDFs
    for uploaded_file in uploaded_files:
        file_path = os.path.join(pdf_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return pdf_folder

# Process PDFs and create vector database
if uploaded_files and process_button:
    with st.spinner("Processing PDFs and creating vector database..."):
        # Save uploaded PDFs
        pdf_folder = save_uploaded_files(uploaded_files)
        
        # Load PDFs
        documents = []
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                with st.status(f"Loading {file}..."):
                    loader = PyPDFLoader(os.path.join(pdf_folder, file))
                    documents.extend(loader.load())
        
        # Split text into chunks
        with st.status("Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            doc_splits = text_splitter.split_documents(documents)
        
        # Convert text chunks into vector embeddings
        with st.status("Creating embeddings..."):
            embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_options[selected_embedding]
            )
            chroma_db = Chroma.from_documents(
                doc_splits, 
                embedding_model, 
                persist_directory="chroma_db"
            )
        
        # Save the vector database
        chroma_db.persist()
        
        # Load the vector database
        chroma_db = Chroma(
            persist_directory="chroma_db", 
            embedding_function=embedding_model
        )
        st.session_state.retriever = chroma_db.as_retriever()
        
        # Initialize Locally Installed LLM with Ollama
        llm = Ollama(model=selected_llm)
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=st.session_state.retriever
        )
        
        st.success("PDF processing complete! You can now ask questions about your documents.")

# Check if existing vector store is available
if st.session_state.retriever is None and os.path.exists("chroma_db"):
    try:
        with st.spinner("Loading existing vector database..."):
            embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_options[selected_embedding]
            )
            chroma_db = Chroma(
                persist_directory="chroma_db", 
                embedding_function=embedding_model
            )
            st.session_state.retriever = chroma_db.as_retriever()
            
            # Initialize Locally Installed LLM with Ollama
            llm = Ollama(model=selected_llm)
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                retriever=st.session_state.retriever
            )
            st.success("Existing vector database loaded successfully!")
    except Exception as e:
        st.error(f"Error loading existing database: {e}")

# Create the chat interface
with main_container:
    # Display chat history
    for i, (query, response) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.write(response)
    
    # User input
    user_query = st.chat_input("Ask a question about your PDFs")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        if st.session_state.qa_chain is not None:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Create a format for chat history that the chain expects
                    formatted_chat_history = [(q, a) for q, a in st.session_state.chat_history]
                    
                    # Get response from QA chain
                    response = st.session_state.qa_chain.invoke({
                        "question": user_query, 
                        "chat_history": formatted_chat_history
                    })
                    
                    # Display the response
                    st.write(response["answer"])
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_query, response["answer"]))
        else:
            st.error("Please upload and process PDFs first.")

# Add information section
with st.expander("About this app"):
    st.markdown("""
    This PDF Knowledge Base application allows you to:
    1. Upload PDF documents
    2. Process them into a searchable vector database
    3. Ask questions about the content of your PDFs
    4. Maintain a conversational history for context
    
    The app uses:
    - LangChain for document processing and retrieval
    - Hugging Face embeddings for vectorizing text
    - Ollama for running LLM inference locally
    - ChromaDB for vector storage
    """)