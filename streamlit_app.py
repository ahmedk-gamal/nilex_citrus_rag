import os
import streamlit as st
from openai import OpenAI
from _04_vector_representation import get_embeddings_model
from _06_retrieve_context import retrieve_context
from _07_prompting import build_prompt
from langchain_chroma import Chroma

st.set_page_config(page_title="NILEX.AI Citrus Advisor", page_icon="🍊", layout="centered")
st.title("🍊 NILEX.AI Citrus & Orange Disease Advisor")
st.markdown("**Beheira Governorate, Egypt – Pilot Phase**")
st.caption("Ask about citrus diseases, symptoms, treatments, and authorized pesticides.")

try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
    model_name = st.secrets["OPENROUTER_MODEL"]
    hf_token = st.secrets.get("HF_TOKEN")
except KeyError:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
    hf_token = os.getenv("HF_TOKEN")

if not api_key:
    st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in Streamlit secrets or as an environment variable.")
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_cached_embeddings():
    return get_embeddings_model(token=hf_token)

def load_vectorstore(persist_dir: str = None):
    if persist_dir is None:
        persist_dir = CHROMA_DIR
    
    embeddings = get_cached_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = 0
    
    if count == 0:
        with st.spinner("جاري تعبئة قاعدة البيانات لأول مرة... قد يستغرق هذا بضع دقائق."):
            from _01_documents import load_pdfs
            from _03_chunking import chunk_documents
            
            if os.path.isdir(DATA_DIR):
                documents = load_pdfs(DATA_DIR)
                chunks = chunk_documents(documents)
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_dir
                )
                st.success(f"تم تعبئة قاعدة البيانات تلقائياً بـ {len(chunks)} مقطع.")
            else:
                st.warning("لم يتم العثور على مجلد البيانات. تأكد من وجود مجلد 'data' يحتوي على ملفات PDF.")
    else:
        try:
            test_results = vectorstore.similarity_search("اختبار", k=1)
            if not test_results:
                st.warning("تم العثور على مستندات لكن الاسترجاع لا يعمل. جاري إعادة تعبئة قاعدة البيانات...")
                raise RuntimeError("Empty search results")
        except Exception:
            with st.spinner("جاري إعادة تعبئة قاعدة البيانات... قد يستغرق هذا بضع دقائق."):
                from _01_documents import load_pdfs
                from _03_chunking import chunk_documents
                
                if os.path.isdir(DATA_DIR):
                    documents = load_pdfs(DATA_DIR)
                    chunks = chunk_documents(documents)
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=persist_dir
                    )
                    st.success(f"تم إعادة تعبئة قاعدة البيانات بـ {len(chunks)} مقطع.")
    
    return vectorstore

vectorstore = load_vectorstore()

# Debug: show count to verify
try:
    db_count = vectorstore._collection.count()
    st.caption(f"📚 قاعدة البيانات: {db_count} مقطع")
except Exception:
    pass

query = st.text_input("Enter your question about citrus diseases:", placeholder="e.g., ما هي أعراض مرض التصمغ في الموالح؟")

if query:
    with st.spinner("Searching knowledge base and generating answer..."):
        try:
            retrieved_docs, context_str = retrieve_context(query, vectorstore, k=3)
            st.caption(f"🔍 Retrieved {len(retrieved_docs)} documents")
            if not retrieved_docs:
                st.warning("No relevant documents found in the knowledge base.")
            else:
                messages = build_prompt(query, context_str)
                try:
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_key,
                    )
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.2,
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 🩺 Expert Answer")
                    st.markdown(answer)
                    with st.expander("📚 View Retrieved Sources"):
                        for i, doc in enumerate(retrieved_docs, start=1):
                            src = doc.metadata.get("source", "unknown")
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(f"**[{i}] File:** {src} | **Page:** {page}")
                            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                            st.divider()
                except Exception as e:
                    st.error(f"Error communicating with OpenRouter: {e}")
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
