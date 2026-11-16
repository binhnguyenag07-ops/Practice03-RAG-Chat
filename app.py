"""Practice 03 - Firebase-protected Streamlit travel guide RAG chat."""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from firebase_client import FirebaseAuthError, sign_in, sign_up
from rag import OllamaError, answer_question

# Reload local settings when Streamlit reruns after .env is edited.
load_dotenv(Path(__file__).with_name(".env"), override=True)
st.set_page_config(page_title="Lumen Travel - RAG Assistant", page_icon="🧭", layout="centered")

FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "").strip().rstrip("/")
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))

if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧭 Lumen Travel")
st.caption("A sign-in-only travel assistant grounded in the Ho Chi Minh City guide")

if st.session_state.user is None:
    st.info("Sign in or create an account to access the chat.")
    sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Sign up"])

    with sign_in_tab:
        with st.form("signin_form"):
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            try:
                st.session_state.user = sign_in(FIREBASE_API_KEY, email, password)
                st.session_state.messages = []
                st.rerun()
            except FirebaseAuthError as exc:
                st.error(str(exc))

    with sign_up_tab:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password (at least 6 characters)", type="password", key="signup_password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            try:
                st.session_state.user = sign_up(FIREBASE_API_KEY, email, password)
                st.session_state.messages = []
                st.rerun()
            except FirebaseAuthError as exc:
                st.error(str(exc))
    st.stop()  # The protected chat UI is not rendered without sign-in.

with st.sidebar:
    st.success(f"Signed in as {st.session_state.user['email']}")
    st.write("**City:** Ho Chi Minh City")
    st.write(f"**Embedding:** embeddinggemma\n\n**Chat:** qwen3:4b\n\n**Top-k:** {TOP_K}")
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()  # Clear current user's chat and auth state.
        st.rerun()

st.subheader("Ask about the travel guide")
if not OLLAMA_URL:
    st.error("OLLAMA_URL is missing. Configure .env before chatting.")
    st.stop()

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander("Retrieved guide entries and cosine similarity scores"):
                for entry in item["sources"]:
                    st.markdown(f"**[{entry['id']}] {entry['title']} — score: {entry['score']:.3f}**")
                    st.caption(f"Source: {entry['source']} | Category: {entry.get('category', 'N/A')}")
                    st.write(entry["text"])

question = st.chat_input("Example: What can I see at the Central Post Office?")
if question is not None:
    if not question.strip():
        st.warning("Please enter a non-empty question.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.status("Retrieving the guide...", expanded=True) as status:
                response, sources = answer_question(
                    question, OLLAMA_URL, TOP_K, MIN_SIMILARITY,
                    on_status=lambda message: status.update(label=message),
                )
                status.update(label="Answer ready", state="complete", expanded=False)
            st.markdown(response)
            with st.expander("Retrieved guide entries and cosine similarity scores"):
                for entry in sources:
                    st.markdown(f"**[{entry['id']}] {entry['title']} — score: {entry['score']:.3f}**")
                    st.caption(f"Source: {entry['source']} | Category: {entry.get('category', 'N/A')}")
                    st.write(entry["text"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources}
            )
        except (OllamaError, ValueError, OSError) as exc:
            st.error(str(exc))
            # The failure does not fabricate an assistant answer.
