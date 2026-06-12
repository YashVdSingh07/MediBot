from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DATA_PATH="data/"
def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents

print("Loading PDF files...")
documents=load_pdf_files(data=DATA_PATH)
print(f"✓ Loaded {len(documents)} pages from PDF files")

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

print("Creating text chunks...")
text_chunks=create_chunks(extracted_data=documents)
print(f"✓ Created {len(text_chunks)} text chunks")

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

print("Generating embeddings...")
embedding_model=get_embedding_model()
print("✓ Embedding model loaded")

DB_FAISS_PATH="vectorstore/db_faiss"
print("Building FAISS database...")
db=FAISS.from_documents(text_chunks,embedding_model)
print("✓ FAISS database created")

print(f"Saving database to {DB_FAISS_PATH}...")
db.save_local(DB_FAISS_PATH)
print("✓ Database saved successfully!")