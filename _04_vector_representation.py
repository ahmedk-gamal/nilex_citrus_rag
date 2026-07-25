from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings_model(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings

if __name__ == "__main__":
    emb = get_embeddings_model()
    print("Embedding model loaded successfully.")
    print("Vector dimension:", len(emb.embed_query("test")))
