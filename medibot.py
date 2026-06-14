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
from streamlit_cookies_controller import CookieController

load_dotenv(find_dotenv())

# Initialize the Database
db.init_db()


DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db_faiss = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db_faiss


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
        return "Good Morning", "Hope you're having a fresh start!"
    elif 12 <= hour < 17:
        return "Good Afternoon", "How can I assist you today?"
    elif 17 <= hour < 21:
        return "Good Evening", "Winding down? I'm here to help!"
    else:
        return "Good Night", "I'm here whenever you need me."


def inject_custom_css():
    """Inject dark teal/emerald themed CSS."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ===== Global Dark Teal Theme ===== */
        .stApp, .stAppViewContainer, [data-testid="stAppViewContainer"] {
            background: linear-gradient(145deg, #080f0d 0%, #0a1a16 25%, #0d2420 50%, #0a1a16 75%, #080f0d 100%) !important;
            background-attachment: fixed !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ===== Hide default Streamlit elements ===== */
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
            color: rgba(45, 212, 168, 0.5);
            font-weight: 400;
            margin-top: 0.4rem;
        }

        /* ===== Auth Screen ===== */
        div[data-testid="stTabs"] {
            background: rgba(10, 26, 22, 0.8);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(45, 212, 168, 0.12);
            animation: fadeUp 0.3s ease-out;
        }

        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }

        /* Hide 'Press Enter to submit form' text */
        div[data-testid="stForm"] .stFormSubmitContent {
            display: none !important;
        }

        .stForm [data-testid="InputInstructions"] {
            display: none !important;
        }

        /* ===== Tab styling: green underline ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            color: rgba(255, 255, 255, 0.5) !important;
            font-weight: 500 !important;
        }

        .stTabs [aria-selected="true"] {
            color: #2dd4a8 !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #2dd4a8 !important;
        }

        .stTabs [data-baseweb="tab-border"] {
            background-color: rgba(45, 212, 168, 0.1) !important;
        }

        /* ===== Input field text alignment ===== */
        .stTextInput input {
            padding-left: 0.75rem !important;
            color: #f5f5f5 !important;
        }

        .stTextInput label {
            color: rgba(255, 255, 255, 0.7) !important;
        }

        /* ===== Sidebar ===== */
        [data-testid="stSidebar"] {
            background-color: #081210 !important;
            border-right: 1px solid rgba(45, 212, 168, 0.08) !important;
        }

        [data-testid="stSidebar"] button {
            border: 1px solid rgba(45, 212, 168, 0.12) !important;
            background-color: rgba(45, 212, 168, 0.04) !important;
            color: #c8c8c8 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
        }

        [data-testid="stSidebar"] button p {
            text-align: left !important;
            width: 100% !important;
        }

        /* Logout button: reddish tint */
        [data-testid="stSidebar"] .element-container:last-child button {
            background-color: rgba(180, 40, 40, 0.15) !important;
            border-color: rgba(180, 40, 40, 0.25) !important;
            color: #ffb3b3 !important;
        }

        [data-testid="stSidebar"] .element-container:last-child button:hover {
            background-color: rgba(180, 40, 40, 0.25) !important;
            border-color: rgba(180, 40, 40, 0.4) !important;
        }

        [data-testid="stSidebar"] button:hover {
            background-color: rgba(45, 212, 168, 0.1) !important;
            border-color: rgba(45, 212, 168, 0.25) !important;
            color: #fff !important;
        }

        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: rgba(45, 212, 168, 0.1) !important;
            border-color: rgba(45, 212, 168, 0.25) !important;
            color: #2dd4a8 !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] .stMarkdown h3 {
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            color: rgba(45, 212, 168, 0.4) !important;
            margin-bottom: 0.5rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(45, 212, 168, 0.08) !important;
            margin: 1.5rem 0 !important;
        }

        /* ===== Chat Messages ===== */
        .stChatMessage {
            border-radius: 12px !important;
            border: none !important;
            animation: fadeUp 0.25s ease-out;
        }

        /* ===== Modern Chat Input Bar ===== */
        [data-testid="stChatInput"] {
            background-color: transparent !important;
            padding-bottom: 2rem !important; /* Lift it up from the absolute bottom */
        }
        
        [data-testid="stChatInput"] > div {
            border-radius: 30px !important;
            border: 1px solid rgba(45, 212, 168, 0.3) !important;
            background: rgba(8, 20, 15, 0.5) !important; /* Frosted glass teal */
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
            padding: 4px 6px 4px 16px !important; /* extra left padding for text */
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        [data-testid="stChatInput"] > div:focus-within {
            border-color: rgba(45, 212, 168, 0.7) !important;
            box-shadow: 0 0 20px rgba(45, 212, 168, 0.15) !important;
        }

        /* Clear all internal backgrounds from Streamlit's BaseWeb components */
        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="base-input"] {
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
        }

        /* Text Area Styling */
        [data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            background: transparent !important;
            color: #f5f5f5 !important;
            caret-color: #2dd4a8 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: rgba(255, 255, 255, 0.4) !important;
        }

        /* Send Button */
        [data-testid="stChatInput"] button {
            background: linear-gradient(135deg, #2dd4a8, #059669) !important;
            border: none !important;
            border-radius: 50% !important;
            color: #fff !important;
            transition: all 0.2s ease !important;
            height: 2.6rem !important;
            width: 2.6rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        [data-testid="stChatInput"] button:hover {
            box-shadow: 0 0 15px rgba(45, 212, 168, 0.4) !important;
            transform: scale(1.05) !important;
        }

        [data-testid="stChatInput"] button svg {
            fill: #fff !important;
            width: 1.2rem !important;
            height: 1.2rem !important;
        }

        /* ===== Bottom Bar Fix ===== */
        .stBottom, 
        .stBottom > div,
        .stBottomBlockContainer,
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"],
        section[data-testid="stBottomBlockContainer"] {
            background: transparent !important;
            background-color: transparent !important;
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
            background: rgba(45, 212, 168, 0.05) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(45, 212, 168, 0.1) !important;
        }

        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(45, 212, 168, 0.15);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(45, 212, 168, 0.3); }
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    """Render a clean hero section."""
    greeting, sub_text = get_greeting()
    
    st.markdown(f"""
    <div class="hero" style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 40vh;">
        <div class="hero-greeting">{greeting}</div>
        <div class="hero-sub">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_login_signup(controller):
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
                username = st.text_input("Name")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    user_id = db.verify_user(username, password)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                        controller.set('user_id', user_id)
                        controller.set('username', username)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose a Name")
                new_password = st.text_input("Choose a Password", type="password")
                submit_signup = st.form_submit_button("Sign Up", use_container_width=True)
                
                if submit_signup:
                    if len(new_username) < 3 or len(new_password) < 6:
                        st.error("Name must be at least 3 chars and password 6 chars.")
                    else:
                        user_id = db.create_user(new_username, new_password)
                        if user_id:
                            st.success("Account created! You can now log in.")
                        else:
                            st.error("Name already exists.")


def render_sidebar(controller):
    with st.sidebar:
        # Username display
        st.markdown(f"<div style='font-size: 1.05rem; font-weight: 600; color: #f5f5f5; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2rem; color: #2dd4a8;'>👤</span> {st.session_state.username}</div>", unsafe_allow_html=True)
        
        # New Chat Button
        if st.button("➕ New Chat", key="new_chat", use_container_width=True):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("---")
        st.markdown("### Recent Chats")
        
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
            controller.remove('user_id')
            controller.remove('username')
            st.rerun()


FAVICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medibot_favicon.png")
USER_AVATAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_avatar.svg")
BOT_AVATAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_avatar.svg")

def main():
    st.set_page_config(
        page_title="MediBot - AI Medical Assistant",
        page_icon=FAVICON_PATH,
        layout="centered",
        initial_sidebar_state="expanded" if 'user_id' in st.session_state else "collapsed"
    )

    inject_custom_css()

    # --- Restore Authentication from Cookies ---
    controller = CookieController()
    
    if 'user_id' not in st.session_state:
        cookie_user_id = controller.get('user_id')
        cookie_username = controller.get('username')
        if cookie_user_id and cookie_username:
            st.session_state.user_id = cookie_user_id
            st.session_state.username = cookie_username

    # --- Authentication Guard ---
    if 'user_id' not in st.session_state:
        render_login_signup(controller)
        return

    # --- Authenticated State ---
    render_sidebar(controller)

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        if 'current_session_id' in st.session_state and st.session_state.current_session_id:
            st.session_state.messages = db.get_session_messages(st.session_state.current_session_id)

    hero_placeholder = st.empty()
    
    for message in st.session_state.messages:
        avatar_path = USER_AVATAR_PATH if message['role'] == 'user' else BOT_AVATAR_PATH
        with st.chat_message(message['role'], avatar=avatar_path):
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
            
        st.chat_message('user', avatar=USER_AVATAR_PATH).markdown(prompt)
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

            with st.chat_message('assistant', avatar=BOT_AVATAR_PATH):
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
