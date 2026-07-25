import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("جاري قراءة ملفات الـ PDF...")
# 1. قراءة كل ملفات الـ PDF الموجودة في فولدر data
loader = PyPDFDirectoryLoader("data")
documents = loader.load()

print(f"تم قراءة {len(documents)} صفحة. جاري تقسيم النصوص...")
# 2. تقسيم النصوص لقطع صغيرة عشان الموديل يقدر يستوعبها (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200 # تداخل بين القطع عشان المعلومات متتقطعش في النص
)
chunks = text_splitter.split_documents(documents)

print("جاري تحميل نموذج اللغة (Embeddings)...")
# 3. استخدام نموذج مجاني بيدعم اللغة العربية لتحويل النصوص لبيانات متجهة
# أول مرة هترن الكود هياخد شوية وقت عشان بيحمل الموديل من النت (حوالي 400 ميجا)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("جاري حفظ البيانات في قاعدة البيانات (ChromaDB)...")
# 4. حفظ البيانات في فولدر جديد اسمه chroma_db
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"🎉 تم بنجاح! تم حفظ {len(chunks)} مقطع في قاعدة البيانات وبقت جاهزة للاستخدام.")