from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma

def retrieve_context(
    query: str,
    vectorstore: Chroma,
    k: int = 3
) -> Tuple[List[Document], str]:
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content
        context_parts.append(
            f"[Source {i}] File: {source}, Page: {page}\n{content}\n"
        )
    context_str = "\n---\n".join(context_parts)
    return retrieved_docs, context_str
