import os
import base64
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv
import db

load_dotenv(find_dotenv())

# Initialize the Database
db.init_db()

DB_FAISS_PATH = "vectorstore/db_faiss"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medibot_logo_cropped.png")
FAVICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medibot_favicon.png")


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db_vector = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db_vector


@st.cache_data
def get_logo_base64():
    """Load the MediBot logo and return as base64 string."""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
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
            if line[0].islower():
                line = line[0].upper() + line[1:]
            formatted_lines.append(line)
    return '\n\n'.join(formatted_lines)


def get_greeting():
    """Return a greeting based on current time of day."""
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning", "Rise and shine! How can I help you today?"
    elif 12 <= hour < 17:
        return "Good Afternoon", "How can I assist you today?"
    elif 17 <= hour < 21:
        return "Good Evening", "What would you like to know?"
    else:
        return "Good Night", "I'm here whenever you need me."


def inject_custom_css():
    """Inject clean, professional CSS."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            background-color: #212121 !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer { visibility: hidden; }
        header { background: transparent !important; }
        .stAppDeployButton { display: none !important; }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ===== Hero ===== */
        .hero {
            text-align: center;
            padding: 5rem 1.5rem 1.5rem 1.5rem;
            animation: fadeUp 0.5s ease-out;
        }

        .hero-logo {
            margin-bottom: -1.0rem;
        }

        .hero-logo img {
            border-radius: 18px;
        }

        .hero-greeting {
            font-family: 'Inter', sans-serif;
            font-size: 1.75rem;
            font-weight: 600;
            color: #f5f5f5;
            margin: 0;
            letter-spacing: -0.3px;
        }

        .hero-sub {
            font-size: 0.9rem;
            color: #8e8e93;
            font-weight: 400;
            margin-top: 0.4rem;
        }

        /* ===== Auth Screen ===== */
        div[data-testid="stTabs"] {
            background: #2f2f2f;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #3a3a3a;
            animation: fadeUp 0.3s ease-out;
        }

        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }

        /* ===== Sidebar ===== */
        [data-testid="stSidebar"] {
            background-color: #1a1a1a !important;
            border-right: 1px solid #2d2d2d !important;
        }

        [data-testid="stSidebar"] button {
            border: 1px solid #333 !important;
            background-color: #242424 !important;
            color: #d1d1d1 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
        }

        [data-testid="stSidebar"] button p {
            text-align: left !important;
            width: 100% !important;
        }
        
        [data-testid="stSidebar"] div[data-testid="column"]:nth-of-type(2) button p {
            text-align: center !important;
        }

        [data-testid="stSidebar"] .element-container:last-child button {
            background-color: #3b1c1c !important;
            border-color: #592424 !important;
            color: #ffb3b3 !important;
        }

        [data-testid="stSidebar"] .element-container:last-child button:hover {
            background-color: #4a2121 !important;
            border-color: #6e2a2a !important;
        }


        [data-testid="stSidebar"] button:hover {
            background-color: #2d2d2d !important;
            border-color: #444 !important;
            color: #fff !important;
        }

        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #2d2d2d !important;
            border-color: #555 !important;
            color: #fff !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] .stMarkdown h3 {
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            color: #888 !important;
            margin-bottom: 0.5rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: #333 !important;
            margin: 1.5rem 0 !important;
        }

        /* ===== Chat ===== */
        .stChatMessage {
            border-radius: 12px !important;
            border: none !important;
            animation: fadeUp 0.25s ease-out;
        }

        .stChatInput {
            background: transparent !important;
        }

        .stChatInput > div {
            border-radius: 26px !important;
            border: 1px solid #3a3a3a !important;
            background: transparent !important;
            transition: border-color 0.2s ease;
            padding: 2px !important;
        }

        .stChatInput > div:focus-within {
            border-color: #525252 !important;
        }

        .stChatInput textarea,
        .stChatInput input {
            background: transparent !important;
            border: none !important;
            color: #f5f5f5 !important;
            caret-color: #f5f5f5 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.9rem !important;
        }

        .stChatInput textarea::placeholder,
        .stChatInput input::placeholder {
            color: #8e8e93 !important;
        }

        .stChatInput button {
            background: #f5f5f5 !important;
            border: none !important;
            border-radius: 50% !important;
            color: #212121 !important;
            transition: all 0.15s ease !important;
        }

        .stChatInput button:hover {
            background: #ffffff !important;
        }

        .stChatInput button svg {
            fill: #212121 !important;
        }

        /* ===== Bottom Bar Fix ===== */
        .stBottom, .stBottomBlockContainer,
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"] {
            background: transparent !important;
            border-top: none !important;
        }

        section[data-testid="stBottomBlockContainer"] {
            background: transparent !important;
            border-top: none !important;
            padding: 0.6rem 1rem !important;
        }

        .block-container { padding-bottom: 1rem !important; }

        [data-testid="stAppViewBlockContainer"] {
            padding-bottom: 80px !important;
        }

        .main .block-container {
            max-width: 800px;
            padding-top: 1rem;
        }

        /* ===== Expander ===== */
        .streamlit-expanderHeader {
            background: #2f2f2f !important;
            border-radius: 8px !important;
            border: 1px solid #3a3a3a !important;
        }

        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: #3a3a3a;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: #525252; }
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    """Render a clean, ChatGPT-style hero section."""
    greeting, sub_text = get_greeting()
    
    st.markdown(f"""
    <div class="hero" style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 40vh;">
        <div class="hero-greeting">{greeting}</div>
        <div class="hero-sub">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_login_signup():
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem; margin-top: 2rem;">
        <h2 style='color: #f5f5f5; font-family: Inter, sans-serif; font-weight: 600; margin-top: 1rem;'>Welcome to MediBot</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    user_id = db.verify_user(username, password)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose a Username")
                new_password = st.text_input("Choose a Password", type="password")
                submit_signup = st.form_submit_button("Sign Up", use_container_width=True)
                
                if submit_signup:
                    if len(new_username) < 3 or len(new_password) < 6:
                        st.error("Username must be at least 3 chars and password 6 chars.")
                    else:
                        user_id = db.create_user(new_username, new_password)
                        if user_id:
                            st.success("Account created! You can now log in.")
                        else:
                            st.error("Username already exists.")


def render_sidebar():
    with st.sidebar:
        col1, col2 = st.columns([4, 1.2])
        with col1:
            st.markdown(f"<h3 style='margin-top: 0.2rem; font-size: 0.95rem; font-weight: 600; color: #f5f5f5;'>{st.session_state.username}</h3>", unsafe_allow_html=True)
        with col2:
            if st.button("＋", key="new_chat", help="New Chat", use_container_width=True):
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.rerun()
            
        st.markdown("---")
        st.markdown("### Chat History")
        
        sessions = db.get_user_sessions(st.session_state.user_id)
        if not sessions:
            st.caption("No past chats found.")
            
        for session in sessions:
            title = session['title']
            btn_type = "primary" if session['id'] == st.session_state.current_session_id else "secondary"
            if st.button(f"{title}", key=f"session_{session['id']}", use_container_width=True, type=btn_type):
                st.session_state.current_session_id = session['id']
                st.session_state.messages = db.get_session_messages(session['id'])
                st.rerun()
                
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for key in ['user_id', 'username', 'current_session_id', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def main():
    st.set_page_config(
        page_title="MediBot - AI Medical Assistant",
        page_icon=FAVICON_PATH,
        layout="centered",
        initial_sidebar_state="expanded" if 'user_id' in st.session_state else "collapsed"
    )

    inject_custom_css()

    # --- Authentication Guard ---
    if 'user_id' not in st.session_state:
        render_login_signup()
        return

    # --- Authenticated State ---
    render_sidebar()

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        if 'current_session_id' in st.session_state and st.session_state.current_session_id:
            st.session_state.messages = db.get_session_messages(st.session_state.current_session_id)

    hero_placeholder = st.empty()
    
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'sources' in message and message['sources']:
                with st.expander("📚 Source Documents"):
                    for i, source_dict in enumerate(message['sources'], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f"- **Page:** {source_dict.get('page_label', 'N/A')}")
                        st.markdown(f"- **File:** {source_dict.get('source', 'N/A')}")

    prompt = st.chat_input("Ask MediBot anything about health...")

    if not st.session_state.messages and not prompt:
        with hero_placeholder.container():
            render_hero()

    if prompt:
        # Create a new session if this is the first message
        if st.session_state.get('current_session_id') is None:
            title = prompt[:25] + "..." if len(prompt) > 25 else prompt
            st.session_state.current_session_id = db.create_chat_session(st.session_state.user_id, title)
            
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        db.save_message(st.session_state.current_session_id, 'user', prompt, None)

        CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Format your response clearly with proper paragraphs and grammar. Use capital letters at the start of each paragraph.

Context: {context}
Question: {question}

Answer:"""

        try:
            vectorstore = get_vectorstore()
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
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"].strip()
            result_formatted = format_llm_response(result)
            source_documents = response["source_documents"]
            
            # Format sources into dictionaries for storage and display
            source_data = []
            for doc in source_documents:
                source_data.append({
                    "page_label": doc.metadata.get('page_label', 'N/A'),
                    "source": doc.metadata.get('source', 'N/A').split('\\')[-1] if '\\' in doc.metadata.get('source', '') else doc.metadata.get('source', 'N/A')
                })

            with st.chat_message('assistant'):
                st.markdown(result_formatted)
                with st.expander("📚 Source Documents"):
                    for i, s_dict in enumerate(source_data, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f"- **Page:** {s_dict.get('page_label', 'N/A')}")
                        st.markdown(f"- **File:** {s_dict.get('source', 'N/A')}")

            st.session_state.messages.append({
                'role': 'assistant',
                'content': result_formatted,
                'sources': source_data
            })
            db.save_message(st.session_state.current_session_id, 'assistant', result_formatted, source_data)
            
            st.rerun() # Ensure sidebar updates with new session title if it was just created

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
