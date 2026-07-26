import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. إعداد صفحة الواجهة
st.set_page_config(page_title="NILEX.AI - Citrus RAG", page_icon="🍊")
st.title("🍊 نظام NILEX.AI - المساعد الذكي لأمراض الموالح")

# 2. شريط جانبي لإدخال المفتاح (تم وضع مفتاحك كقيمة افتراضية للسرعة)
st.sidebar.header("الإعدادات")
api_key = st.sidebar.text_input(
    "مفتاح OpenRouter:",
    value=st.secrets.get("OPENROUTER_API_KEY", ""),
    type="password"
)

try:
    hf_token = st.secrets.get("HF_TOKEN")
except Exception:
    hf_token = os.getenv("HF_TOKEN")



# 3. دالة لتحميل قاعدة البيانات
def load_database():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu", **({"token": hf_token} if hf_token else {})},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = 0
    
    if count == 0:
        with st.spinner("جاري تعبئة قاعدة البيانات لأول مرة... قد يستغرق هذا بضع دقائق."):
            from langchain_community.document_loaders import PyPDFDirectoryLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            if os.path.isdir(DATA_DIR):
                loader = PyPDFDirectoryLoader(DATA_DIR)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=CHROMA_DIR
                )
                st.success(f"تم تعبئة قاعدة البيانات تلقائياً بـ {len(chunks)} مقطع.")
            else:
                st.warning("لم يتم العثور على مجلد البيانات. تأكد من وجود مجلد 'data' يحتوي على ملفات PDF.")
    
    return vectorstore

vectorstore = load_database()

# دالة لتنسيق النصوص المسترجعة
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if api_key:
    # 4. تجهيز قاعدة البيانات
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 5. تجهيز موديل الذكاء الاصطناعي وربطه بـ OpenRouter
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free", # يمكنك تغييره لاحقاً لأي موديل متاح على OpenRouter
        temperature=0.3
    )
    
    # 6. توجيهات النظام (System Prompt)
    system_prompt = (
        "أنت خبير زراعي متخصص في أمراض الموالح والبرتقال في مصر وتعمل ضمن نظام NILEX.AI. "
        "أجب على سؤال المزارع بناءً على السياق التالي فقط. "
        "إذا لم تكن الإجابة موجودة في السياق، قل بوضوح 'لا أملك معلومات دقيقة حول هذا الموضوع بناءً على النشرات المتاحة'. "
        "استخدم لغة بسيطة وواضحة تناسب المزارعين.\n\n"
        "السياق: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 7. ربط سلسلة العمليات (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 8. إدارة سجل المحادثة في الواجهة
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input("اكتب سؤالك هنا (مثال: ما هو علاج العفن الهبابي في البرتقال؟)")
    
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("جاري البحث في النشرات الزراعية الخاصة بـ NILEX.AI..."):
                try:
                    # إرسال السؤال للموديل واستقبال الإجابة
                    response = rag_chain.invoke(user_query)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"حدث خطأ في الاتصال بالخادم: {e}")
else:
    st.info("👈 رجاءً إدخال مفتاح OpenRouter في القائمة الجانبية للبدء.")