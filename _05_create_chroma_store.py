import os
from langchain_chroma import Chroma
from _01_documents import load_pdfs
from _03_chunking import chunk_documents
from _04_vector_representation import get_embeddings_model

def create_vectorstore(
    documents_dir: str = "data",
    persist_dir: str = "chroma_db",
    chunk_size: int = 700,
    chunk_overlap: int = 100
):
    print("Loading PDFs ...")
    documents = load_pdfs(documents_dir)
    print(f"Loaded {len(documents)} pages.")
    print("Chunking documents ...")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(chunks)} chunks.")
    print("Loading embedding model ...")
    embeddings = get_embeddings_model()
    print("Creating Chroma vector store ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Vector store created and persisted at '{persist_dir}'.")
    return vectorstore

if __name__ == "__main__":
    create_vectorstore()
