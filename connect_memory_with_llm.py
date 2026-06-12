import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def format_docs(docs):
    """Format documents into context string"""
    return "\n\n".join(doc.page_content for doc in docs)

def check_relevance(query, context, threshold=0.2):
    """
    Check if the retrieved context is relevant to the query.
    Returns True if relevant, False otherwise.
    Uses a simple heuristic: checks if important keywords from query appear in context
    """
    import re
    
    # Convert to lowercase for comparison
    query_lower = query.lower()
    context_lower = context.lower()
    
    # Extract key terms from query (remove common words and question words)
    stop_words = {'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 
                  'who', 'what', 'when', 'where', 'why', 'how', 'best', 'world', 'you', 'your', 'can', 'will', 'be', 'have', 'has', 'do', 'does'}
    
    # Remove punctuation and split into words
    query_text = re.sub(r'[^\w\s]', '', query_lower)
    query_words = [w for w in query_text.split() if len(w) > 2 and w not in stop_words]
    
    if not query_words:
        return True
    
    # Count how many query words appear in context
    matches = sum(1 for word in query_words if word in context_lower)
    relevance_score = matches / len(query_words) if query_words else 0
    
    return relevance_score >= threshold

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Loading FAISS database...")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
print("✓ Database loaded!")

# Create retriever and QA components
retriever = db.as_retriever(search_kwargs={'k': 3})
prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

# Now invoke with a single query
user_query = input("Write Query Here: ")

print("\nRetrieving documents and generating answer...")

# Retrieve documents
docs = retriever.invoke(user_query)
context = format_docs(docs)

# Check if the retrieved context is relevant to the query
if check_relevance(user_query, context):
    result = context
else:
    # If not relevant, return a message saying we don't know
    result = "I don't know the answer to this question. The database contains medical information and may not have information about '" + user_query + "'."

# Format output to match the expected structure
print("RESULT: ", result)
print("SOURCE DOCUMENTS: ", docs)