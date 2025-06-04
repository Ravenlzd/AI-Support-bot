import streamlit as st
from sentence_transformers import SentenceTransformer, util
import importlib

st.set_page_config(page_title="SmartHelp Chatbot", layout="centered")



@st.cache_resource
def load_nlp():
    torch_mod = importlib.import_module("torch")
    st_mod = importlib.import_module("sentence_transformers")
    model = st_mod.SentenceTransformer("all-MiniLM-L6-v2")
    return model, st_mod.util

model, util = load_nlp()



import os

# Disable symlink warnings and watcher on PyTorch internals
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import random
import json
import re
from googletrans import Translator
translator = Translator()


language_map = {
    "English": "en",
    "Turkish": "tr",
    "Russian": "ru",
    "Azerbaijani": "az"
}


# ─── 1) PAGE CONFIG & CSS ─────────────────────────────────────────────────────
# NOTE: No separate CSS file needed—inline styles handle everything.
st.markdown("""
            
            <style>
/* Chat container */
.chat-window {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 10px;
    max-height: 500px;
    overflow-y: auto;
    margin-top: 20px;
    margin-bottom: 20px;
}

/* Common bubble style */
.message {
    display: inline-block;
    padding: 10px 14px;
    border-radius:200px;
    margin-bottom: 8px;
    max-width: 75%;
    word-wrap: break-word;
    font-size: 16px;
}

/* Bot message (left aligned, gray) */
.bot {
    background-color: #313542;
    color: #ffffff;
    text-align: left;
    margin-right: auto;
}

/* User message (right aligned, green) */
.user {
    background-color: #4CAF50;
    color: #ffffff;
    text-align: left;
    margin-left: auto;
}
</style>

<style>
/* Unified custom chat button styling */
div.stButton > button {
    position: fixed;
    bottom: 30px;
    right: 30px;
    padding: 20px 30px;
    font-size: 60px;
    border-radius: 50px;
    background-color: #222;
    color: white;
    font-weight: bold;
    border: none;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    cursor: pointer;
    z-index: 999;
    transition: all 0.3s ease-in-out;
    background-image: linear-gradient(#ffffff 0 0);  /* white underline */
    background-size: 0% 500px;
    background-position: 0 20%;
    background-repeat: no-repeat;
}

/* Hover underline animation on button */
div.stButton > button:hover {
    background-size: 100% 80px;
    background-color: #080707#;
    color: black;
}
</style>
""", unsafe_allow_html=True)






# ─── 2) INTENT LOADER & NLP ───────────────────────────────────────────────────

def load_intents():
    # Ensure JSON is read as UTF-8 (so emojis render correctly)
    with open("intents.json", "r", encoding="utf-8") as f:
        return json.load(f)["intents"]

def preprocess(text: str) -> str:
    """Lowercase + remove punctuation for matching."""
    return re.sub(r"[^\w\s]", "", text.strip().lower())

def find_response_nlp(user_input, intents):
    pattern_to_intent = []
    pattern_texts = []

    for intent in intents:
        for pattern in intent["patterns"]:
            pattern_to_intent.append(intent)
            pattern_texts.append(pattern)

    # Embed user input and patterns
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    pattern_embeddings = model.encode(pattern_texts, convert_to_tensor=True)

    # Find best match
    similarity_scores = util.pytorch_cos_sim(user_embedding, pattern_embeddings)[0]
    best_match_idx = similarity_scores.argmax().item()
    best_intent = pattern_to_intent[best_match_idx]

    if similarity_scores[best_match_idx] < 0.45:
        return "Sorry, I’m not sure how to help with that."

    return random.choice(best_intent["responses"])


# ─── 3) STREAMLIT STATE SETUP ────────────────────────────────────────────────

if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

if "chat_log" not in st.session_state:
    # Each entry will be ("You"/"Bot", "message text")
    st.session_state.chat_log = []
    


# ─── 4) FLOATING BUTTON TO TOGGLE CHAT ───────────────────────────────────────

# Note: we use st.button directly, not st.form, for toggling visibility.
if st.button("💬 Chat with us", key="toggle_chat"):
    st.session_state.show_chat = not st.session_state.get("show_chat", False)

# ─── 5) CHAT WINDOW ──────────────────────────────────────────────────────────

if st.session_state.show_chat:
    
    st.markdown("### SmartHelp Support Chat")

    # ─ Show existing messages in order ───────────────────────────────────────
    for sender, msg in st.session_state.chat_log:
        if sender == "You":
         st.markdown(
               f'<div class="message user"><b>You:</b> {msg}</div>',
              unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="message bot"><b>Bot:</b> {msg}</div>',
            unsafe_allow_html=True,
        )


    # ─ Input Form at the bottom ──────────────────────────────────────────────
    with st.form(key="chat_input_form", clear_on_submit=True):
        user_input = st.text_input("Your message:", key="chat_input_field")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        translator = Translator()
        detected_lang = translator.detect(user_input).lang

    # Translate user input to English for intent detection
        translated_input = translator.translate(user_input, src=detected_lang, dest="en").text

    # Load intents
        intents = load_intents()

    # Get response from NLP engine
        response_en = find_response_nlp(translated_input, intents)

    # Translate response back to user's original language
        response = translator.translate(response_en, src="en", dest=detected_lang).text

    # Save messages in chat history
        st.session_state.chat_log.append(("You", user_input))
        st.session_state.chat_log.append(("Bot", response))

    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


    st.markdown("</div>", unsafe_allow_html=True)

# ─── 6) OPTIONAL: Placeholder Below Chat to Push It Up ───────────────────────
    # (Ensures that the chat window isn't flush against the bottom of the page)
st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
