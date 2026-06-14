import os
import base64
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def format_llm_response(text):
    """Format LLM response with proper paragraph breaks and capitalization."""
    if not text:
        return text
    
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            if line and line[0].islower():
                line = line[0].upper() + line[1:]
            formatted_lines.append(line)
    
    return '\n\n'.join(formatted_lines)


LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MediBot logo.png")


@st.cache_data
def get_logo_base64():
    """Load the MediBot logo and return as base64 string."""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_greeting():
    """Return a greeting based on current time of day."""
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning", "🌅", "Hope you're having a fresh start!"
    elif 12 <= hour < 17:
        return "Good Afternoon", "☀️", "How can I assist you today?"
    elif 17 <= hour < 21:
        return "Good Evening", "🌆", "Winding down? I'm here to help!"
    else:
        return "Good Night", "🌙", "Burning the midnight oil? Let's chat!"


def inject_custom_css():
    """Inject dark teal/emerald themed CSS."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* ===== Global Dark Teal Theme ===== */
        .stApp {
            background: linear-gradient(145deg, #080f0d 0%, #0a1a16 25%, #0d2420 50%, #0a1a16 75%, #080f0d 100%) !important;
            font-family: 'Inter', sans-serif;
        }

        /* ===== Hide default Streamlit elements ===== */
        #MainMenu, footer { visibility: hidden; }
        header { background: transparent !important; }
        .stAppDeployButton { display: none !important; }

        /* ===== Animations ===== */
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slide-up {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(45, 212, 168, 0.12); }
            50% { box-shadow: 0 0 40px rgba(45, 212, 168, 0.25), 0 0 60px rgba(5, 150, 105, 0.1); }
        }

        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes heartbeat {
            0%, 100% { transform: scale(1); }
            15% { transform: scale(1.12); }
            30% { transform: scale(1); }
            45% { transform: scale(1.08); }
        }

        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }

        /* ===== Hero Container ===== */
        .hero-container {
            text-align: center;
            padding: 2rem 1rem 1rem 1rem;
            animation: slide-up 0.8s ease-out;
        }

        .medibot-logo {
            font-size: 3.5rem;
            margin-bottom: 0.3rem;
            animation: heartbeat 2s ease-in-out infinite;
            display: inline-block;
            filter: drop-shadow(0 0 20px rgba(45, 212, 168, 0.35));
        }

        .medibot-logo img {
            border-radius: 20px;
        }

        .medibot-title {
            font-family: 'Outfit', sans-serif;
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #2dd4a8 0%, #059669 30%, #34d399 60%, #10b981 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradient-shift 4s ease infinite, slide-up 0.8s ease-out;
            margin: 0;
            padding: 0;
            letter-spacing: -1px;
            line-height: 1.2;
        }

        .medibot-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.45);
            font-weight: 300;
            margin-top: 0.2rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            animation: fade-in 1.2s ease-out 0.3s both;
        }

        /* ===== Greeting Card ===== */
        .greeting-card {
            background: linear-gradient(135deg, rgba(45, 212, 168, 0.07) 0%, rgba(5, 150, 105, 0.07) 50%, rgba(16, 185, 129, 0.07) 100%);
            border: 1px solid rgba(45, 212, 168, 0.12);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            margin: 1.5rem auto;
            max-width: 600px;
            backdrop-filter: blur(20px);
            animation: slide-up 0.8s ease-out 0.4s both, pulse-glow 3s ease-in-out infinite;
            position: relative;
            overflow: hidden;
        }

        .greeting-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -200%;
            width: 200%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(45, 212, 168, 0.03), transparent);
            animation: shimmer 6s ease-in-out infinite;
        }

        .greeting-emoji {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            display: inline-block;
            animation: heartbeat 3s ease-in-out infinite;
        }

        .greeting-text {
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            font-weight: 600;
            color: #e8e8e8;
            margin: 0.2rem 0;
        }

        .greeting-sub {
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.5);
            font-weight: 300;
            margin-top: 0.3rem;
        }

        /* ===== Feature Pills ===== */
        .feature-container {
            display: flex;
            justify-content: center;
            gap: 0.8rem;
            flex-wrap: wrap;
            margin: 1.5rem auto;
            max-width: 650px;
            animation: slide-up 0.8s ease-out 0.7s both;
        }

        .feature-pill {
            background: rgba(45, 212, 168, 0.04);
            border: 1px solid rgba(45, 212, 168, 0.1);
            border-radius: 50px;
            padding: 0.5rem 1.2rem;
            font-size: 0.82rem;
            color: rgba(255, 255, 255, 0.55);
            font-weight: 400;
            transition: all 0.3s ease;
            cursor: default;
        }

        .feature-pill:hover {
            background: rgba(45, 212, 168, 0.12);
            border-color: rgba(45, 212, 168, 0.35);
            color: #2dd4a8;
            transform: translateY(-2px);
        }

        .feature-pill .pill-icon {
            margin-right: 0.4rem;
        }

        /* ===== Divider ===== */
        .styled-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(45, 212, 168, 0.2), rgba(16, 185, 129, 0.2), transparent);
            border: none;
            margin: 1.5rem auto;
            max-width: 500px;
            animation: fade-in 1s ease-out 0.9s both;
        }

        /* ===== Chat Messages ===== */
        .stChatMessage {
            border-radius: 16px !important;
            border: 1px solid rgba(45, 212, 168, 0.06) !important;
            backdrop-filter: blur(10px) !important;
            animation: slide-up 0.4s ease-out;
        }

        /* ===== Chat Input ===== */
        .stChatInput {
            background: transparent !important;
        }

        .stChatInput > div {
            border-radius: 16px !important;
            border: 1px solid rgba(45, 212, 168, 0.2) !important;
            background: rgba(8, 15, 13, 0.85) !important;
            backdrop-filter: blur(10px) !important;
            transition: all 0.3s ease;
            padding: 2px !important;
        }

        .stChatInput > div:focus-within {
            border-color: rgba(45, 212, 168, 0.45) !important;
            box-shadow: 0 0 25px rgba(45, 212, 168, 0.1) !important;
        }

        .stChatInput textarea,
        .stChatInput input {
            background: transparent !important;
            border: none !important;
            color: #e0e0e0 !important;
            caret-color: #2dd4a8 !important;
            font-family: 'Inter', sans-serif !important;
        }

        .stChatInput textarea::placeholder,
        .stChatInput input::placeholder {
            color: rgba(255, 255, 255, 0.3) !important;
        }

        /* ===== Send Button ===== */
        .stChatInput button {
            background: linear-gradient(135deg, #2dd4a8, #059669) !important;
            border: none !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }

        .stChatInput button:hover {
            box-shadow: 0 0 15px rgba(45, 212, 168, 0.3) !important;
            transform: scale(1.05) !important;
        }

        /* ===== Bottom Strip ===== */
        .stBottom, .stBottomBlockContainer,
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"] {
            background: transparent !important;
            background-color: transparent !important;
            border-top: none !important;
        }

        .block-container {
            padding-bottom: 1rem !important;
        }

        [data-testid="stAppViewBlockContainer"] {
            padding-bottom: 80px !important;
        }

        .main .block-container {
            max-width: 800px;
            padding-top: 2rem;
        }

        section[data-testid="stBottomBlockContainer"] {
            background: linear-gradient(135deg, #080f0d 0%, #0a1a16 50%, #080f0d 100%) !important;
            border-top: 1px solid rgba(45, 212, 168, 0.06) !important;
            padding: 0.8rem 1rem !important;
        }

        /* ===== Sidebar ===== */
        [data-testid="stSidebar"] {
            background-color: #081210 !important;
            border-right: 1px solid rgba(45, 212, 168, 0.08) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: rgba(255, 255, 255, 0.7);
        }

        /* ===== Sidebar Buttons ===== */
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(45, 212, 168, 0.06) !important;
            border: 1px solid rgba(45, 212, 168, 0.1) !important;
            border-radius: 10px !important;
            color: rgba(255, 255, 255, 0.75) !important;
            font-size: 0.82rem !important;
            text-align: left !important;
            padding: 0.5rem 0.8rem !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(45, 212, 168, 0.12) !important;
            border-color: rgba(45, 212, 168, 0.25) !important;
            color: #e0e0e0 !important;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(45, 212, 168, 0.1) !important;
            border-color: rgba(45, 212, 168, 0.2) !important;
            color: #2dd4a8 !important;
        }

        /* ===== Expander ===== */
        .streamlit-expanderHeader {
            background: rgba(45, 212, 168, 0.04) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(45, 212, 168, 0.08) !important;
        }

        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(45, 212, 168, 0.15);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(45, 212, 168, 0.3); }
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    """Render the animated hero section with greeting."""
    greeting, emoji, sub_text = get_greeting()
    logo_b64 = get_logo_base64()

    # Title section with logo
    st.markdown(f"""
    <div class="hero-container">
        <div class="medibot-logo"><img src="data:image/png;base64,{logo_b64}" alt="MediBot" style="width:150px; height:150px; object-fit:contain;"></div>
        <h1 class="medibot-title">MediBot</h1>
        <p class="medibot-subtitle">Your AI-Powered Medical Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Greeting card
    st.markdown(f"""
    <div class="greeting-card">
        <div class="greeting-emoji">{emoji}</div>
        <div class="greeting-text">{greeting}!</div>
        <div class="greeting-sub">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature pills
    st.markdown("""
    <div class="feature-container">
        <span class="feature-pill"><span class="pill-icon">💊</span>Drug Info</span>
        <span class="feature-pill"><span class="pill-icon">🩺</span>Symptoms</span>
        <span class="feature-pill"><span class="pill-icon">📋</span>Conditions</span>
        <span class="feature-pill"><span class="pill-icon">🧬</span>Treatments</span>
    </div>
    """, unsafe_allow_html=True)

    # Divider
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


FAVICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medibot_favicon.png")


def main():
    st.set_page_config(
        page_title="MediBot",
        page_icon=FAVICON_PATH,
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    inject_custom_css()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Show hero only when chat is empty
    if not st.session_state.messages:
        render_hero()
    else:
        # Compact header when chatting
        logo_b64 = get_logo_base64()
        st.markdown(f"""
        <div style="text-align:center; padding: 0.8rem 0; animation: fade-in 0.5s ease-out; display:flex; align-items:center; justify-content:center; gap:0.5rem;">
            <img src="data:image/png;base64,{logo_b64}" alt="MediBot" style="width:32px; height:32px; object-fit:contain;">
            <span style="font-family:'Outfit',sans-serif; font-size:1.5rem; font-weight:700;
                  background:linear-gradient(135deg,#2dd4a8,#059669);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;">MediBot</span>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'sources' in message and message['sources']:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(message['sources'], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f"- **Page:** {doc.metadata.get('page_label', 'N/A')}")
                        source_file = doc.metadata.get('source', 'N/A').split('\\')[-1] if '\\' in doc.metadata.get('source', '') else doc.metadata.get('source', 'N/A')
                        st.markdown(f"- **File:** {source_file}")

    prompt=st.chat_input("Ask MediBot anything about health...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Format your response clearly with proper paragraphs and grammar. Use capital letters at the start of each paragraph.

Context: {context}
Question: {question}

Answer:"""
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"].strip()
            result_formatted = format_llm_response(result)
            source_documents=response["source_documents"]
            
            with st.chat_message('assistant'):
                st.markdown(result_formatted)
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(source_documents, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f"- **Page:** {doc.metadata.get('page_label', 'N/A')}")
                        source_file = doc.metadata.get('source', 'N/A').split('\\')[-1] if '\\' in doc.metadata.get('source', '') else doc.metadata.get('source', 'N/A')
                        st.markdown(f"- **File:** {source_file}")

            st.session_state.messages.append({
                'role':'assistant', 
                'content': result_formatted,
                'sources': source_documents
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
