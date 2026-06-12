import sqlite3
import bcrypt
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "medibot.db")

def init_db():
    """Initialize the SQLite database with required tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Chat Sessions Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Chat Messages Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    ''')

    conn.commit()
    conn.close()

def create_user(username, password):
    """Create a new user. Returns user_id on success, None if username exists."""
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials. Returns user_id on success, None on failure."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row:
        user_id, password_hash = row
        if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
            return user_id
    return None

def create_chat_session(user_id, title="New Chat"):
    """Create a new chat session for a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)", (user_id, title))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def update_session_title(session_id, title):
    """Update the title of a chat session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE chat_sessions SET title = ? WHERE id = ?", (title, session_id))
    conn.commit()
    conn.close()

def get_user_sessions(user_id):
    """Get all chat sessions for a user, ordered by newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    sessions = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "title": row[1], "created_at": row[2]} for row in sessions]

def save_message(session_id, role, content, sources=None):
    """Save a single message to a session."""
    import json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sources_json = json.dumps(sources) if sources else None
    cursor.execute("INSERT INTO chat_messages (session_id, role, content, sources) VALUES (?, ?, ?, ?)", 
                   (session_id, role, content, sources_json))
    conn.commit()
    conn.close()

def get_session_messages(session_id):
    """Retrieve all messages for a given session."""
    import json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, sources FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
    messages = cursor.fetchall()
    conn.close()
    
    result = []
    for row in messages:
        role, content, sources_json = row
        sources = json.loads(sources_json) if sources_json else None
        result.append({"role": role, "content": content, "sources": sources})
    return result
