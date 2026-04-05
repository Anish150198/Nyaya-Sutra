"""
Manages multi-turn conversation history and Streamlit session state.
"""

import streamlit as st
from typing import Optional
from core.data_models import UserProfile


def init_session_state():
    """Initialize all session state keys if not already present."""
    defaults = {
        "chat_history": [],
        "persona": "citizen",
        "language": "en",
        "act_filter": "ALL",
        "compare_models": False,
        "user_profile": None,
        "active_tab": "Chat (Citizen)",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_message(role: str, content: str, metadata: Optional[dict] = None):
    """
    Add a message to chat history.

    Parameters
    ----------
    role : str
        "user" or "assistant".
    content : str
        Message text.
    metadata : dict or None
        Extra info (citations, confidence, model_id, etc.).
    """
    entry = {"role": role, "content": content}
    if metadata:
        entry["metadata"] = metadata
    st.session_state.chat_history.append(entry)


def get_chat_history() -> list[dict]:
    """Return the current chat history."""
    return st.session_state.get("chat_history", [])


def clear_chat():
    """Clear the conversation history."""
    st.session_state.chat_history = []


def set_user_profile(profile: UserProfile):
    """Store the user profile from the Scheme Wizard."""
    st.session_state.user_profile = profile


def get_user_profile() -> Optional[UserProfile]:
    """Retrieve the stored user profile."""
    return st.session_state.get("user_profile")


def get_persona() -> str:
    return st.session_state.get("persona", "citizen")


def get_language() -> str:
    return st.session_state.get("language", "en")


def get_act_filter() -> str:
    return st.session_state.get("act_filter", "ALL")


def get_compare_models() -> bool:
    return st.session_state.get("compare_models", False)
