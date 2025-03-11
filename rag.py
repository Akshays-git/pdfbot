import os
import chromadb
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama


# Step 3: Load PDFs and Extract Text
def load_pdfs(pdf_folder):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
    return documents

pdf_folder = "./my_pdfs"  # Ensure PDFs are inside this folder
os.makedirs(pdf_folder, exist_ok=True)
documents = load_pdfs(pdf_folder)

# Step 4: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_splits = text_splitter.split_documents(documents)

# Step 5: Convert Text Chunks into Vector Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma.from_documents(doc_splits, embedding_model, persist_directory="chroma_db")

# Step 6: Save the Vector Database
chroma_db.persist()

# Step 7: Load the Vector Database
chroma_db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
retriever = chroma_db.as_retriever()

# Step 8: Initialize Locally Installed LLaMA 3.2 with Ollama
llm = Ollama(model="llama3.2")
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

chat_history = []  # Stores past queries for conversational memory
query = """How much did Apple spend on Research and Development in fiscal year 2018, and how did it change compared to 2017?"""
response = qa_chain.invoke({"question": query, "chat_history": chat_history})

print("Answer:", response["answer"])
