from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from _02_preprocessing import clean_arabic_text

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = clean_arabic_text(chunk.page_content)
    chunks = [c for c in chunks if c.page_content.strip()]
    return chunks
