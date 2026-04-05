"""
Conversational UI component for the Legal Navigator (Nyaya).
Renders chat bubbles with citations and confidence indicators.
"""

import streamlit as st
from core.data_models import UserQuery, Persona, ActFilter, OrchestratorResponse
from app.state_manager import (
    add_message, get_chat_history, clear_chat,
    get_persona, get_language, get_act_filter, get_compare_models,
)


def render_chat_view(handle_fn, tab_suffix=""):
    """
    Render the chat interface.

    Parameters
    ----------
    handle_fn : Callable[[UserQuery], OrchestratorResponse]
        The orchestrator.handle function.
    tab_suffix : str
        Suffix to make widget keys unique across tabs.
    """
    # Chat history
    for msg in get_chat_history():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("metadata", {})
            if meta.get("citations"):
                with st.expander("📚 Citations"):
                    for c in meta["citations"]:
                        st.markdown(f"- **{c.get('code', '')} §{c.get('section_no', '')}** "
                                    f"{c.get('title', '')}")
            if meta.get("confidence"):
                _render_confidence(meta["confidence"])

    # Input
    if prompt := st.chat_input("Ask about Indian law or government schemes...", key=f"chat_input_{tab_suffix}"):
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query = UserQuery(
                    text=prompt,
                    user_lang=st.session_state.language,
                    persona=Persona(st.session_state.persona),
                    act_filter=ActFilter(st.session_state.act_filter),
                    compare_models=st.session_state.compare_models,
                )
                try:
                    response: OrchestratorResponse = handle_fn(query)
                    st.markdown(response.answer_text)

                    if response.disclaimer:
                        st.caption(response.disclaimer)

                    metadata = {
                        "citations": [c.model_dump() for c in response.citations] if response.citations else [],
                        "confidence": response.confidence,
                        "model_ids": response.model_ids_used,
                        "intent": response.intent.value,
                    }

                    if response.citations:
                        with st.expander("📚 Citations"):
                            for c in response.citations:
                                st.markdown(
                                    f"- **{c.code or ''} §{c.section_no or ''}** "
                                    f"{c.title or ''}"
                                )

                    if response.schemes:
                        with st.expander("🏛️ Relevant Schemes"):
                            for s in response.schemes:
                                st.markdown(f"**{s.get('name', 'Unknown')}**")
                                st.markdown(f"  {s.get('description', '')[:200]}")

                    if response.comparison:
                        with st.expander("⚖️ Model Comparison"):
                            st.json(response.comparison)

                    _render_confidence(response.confidence)

                    add_message("assistant", response.answer_text, metadata)

                except Exception as exc:
                    err_msg = f"An error occurred: {exc}"
                    st.error(err_msg)
                    add_message("assistant", err_msg)


def _render_confidence(confidence: str):
    """Render a confidence badge."""
    colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    icon = colors.get(confidence, "⚪")
    st.caption(f"{icon} Confidence: **{confidence.title()}**")
