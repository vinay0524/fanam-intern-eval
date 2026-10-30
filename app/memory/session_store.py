"""Session and conversation memory store."""


import json
import os
import threading
from pathlib import Path
from datetime import datetime, timezone

SESSION_DIR = Path("data/sessions")
SESSION_FILE = SESSION_DIR / "conversations.json"

_file_lock = threading.Lock()


def _read_sessions():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    if not SESSION_FILE.exists():
        return {}

    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        raise RuntimeError(
            "Conversation history JSON is corrupted. "
            "Restore a valid backup before continuing."
        )


def load_history(session_id: str) -> list:
    with _file_lock:
        sessions = _read_sessions()
        return sessions.get(session_id, [])


def save_history(session_id: str, query: str, answer: str):
    with _file_lock:
        sessions = _read_sessions()

        if session_id not in sessions:
            sessions[session_id] = []

        sessions[session_id].append({
            "query": query,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        SESSION_DIR.mkdir(parents=True, exist_ok=True)

        temp_file = SESSION_FILE.with_suffix(".tmp")

        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(sessions, file, indent=4, ensure_ascii=False)

        os.replace(temp_file, SESSION_FILE)


def delete_history(session_id: str) -> bool:
    with _file_lock:
        sessions = _read_sessions()

        if session_id not in sessions:
            return False

        del sessions[session_id]

        temp_file = SESSION_FILE.with_suffix(".tmp")

        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(sessions, file, indent=4, ensure_ascii=False)

        os.replace(temp_file, SESSION_FILE)

        return True
