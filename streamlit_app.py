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
except KeyError:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

if not api_key:
    st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in Streamlit secrets or as an environment variable.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_vectorstore(persist_dir: str = "chroma_db"):
    try:
        embeddings = get_embeddings_model()
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vector database. Error: {e}")
        st.stop()

vectorstore = load_vectorstore()

query = st.text_input("Enter your question about citrus diseases:", placeholder="e.g., ما هي أعراض مرض التصمغ في الموالح؟")

if query:
    with st.spinner("Searching knowledge base and generating answer..."):
        retrieved_docs, context_str = retrieve_context(query, vectorstore, k=3)
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
