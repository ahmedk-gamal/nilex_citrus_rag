import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdfs(directory: str = "data") -> List[Document]:
    documents = []
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found.")
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            try:
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                documents.extend(docs)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
    return documents

if __name__ == "__main__":
    docs = load_pdfs("data")
    print(f"Loaded {len(docs)} pages from PDFs.")
