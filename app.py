import streamlit as st # type: ignore
import pickle
import os
import spacy # type: ignore
import time


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)


# ── Load CSS from templates folder ────────────────────────────────────────────
def load_css(filepath: str):
    with open(filepath, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ── Load HTML template from templates folder ──────────────────────────────────
def load_html(filepath: str, **kwargs) -> str:
    with open(filepath, "r") as f:
        html = f.read()
    # Replace all {placeholders} with actual values passed as kwargs
    for key, value in kwargs.items():
        html = html.replace("{" + key + "}", str(value))
    return html


# ── Apply styles ──────────────────────────────────────────────────────────────
load_css("templates/style.css")


# ── Load model and vectorizer ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/spam_classifier.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_vectorizer():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


model      = load_model()
vectorizer = load_vectorizer()
nlp        = load_nlp()


# ── spaCy preprocessing ───────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    doc    = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if not token.text.isalnum():
            continue
        tokens.append(token.lemma_.lower())
    return " ".join(tokens)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(message: str):
    cleaned    = preprocess(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    return prediction, probability, cleaned


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='header'><h1>📱 SMS Spam Classifier</h1><p>Paste any SMS message to detect if it is spam or legitimate.</p></div>", unsafe_allow_html=True)
st.divider()


# ── Example buttons ───────────────────────────────────────────────────────────
st.markdown("#### Try an example")
col1, col2 = st.columns(2)

with col1:
    if st.button("🚨 Spam Example", use_container_width=True):
        st.session_state.input_text = (
            "Congratulations! You have won a FREE iPhone. "
            "Click here to claim your prize: bit.ly/win123"
        )

with col2:
    if st.button("✅ Ham Example", use_container_width=True):
        st.session_state.input_text = (
            "Hey, are you coming to the meeting tomorrow at 10am? Let me know."
        )


# ── Text input ────────────────────────────────────────────────────────────────
st.markdown("#### Enter your message")
message = st.text_area(
    label="SMS Message",
    value=st.session_state.get("input_text", ""),
    height=150,
    placeholder="Type or paste your SMS message here...",
    label_visibility="collapsed"
)

if message.strip():
    word_count = len(message.strip().split())
    char_count = len(message.strip())
    st.caption(f"{word_count} words · {char_count} characters")

st.markdown("")


# ── Classify button ───────────────────────────────────────────────────────────
classify_btn = st.button("🔍 Classify Message", type="primary", use_container_width=True)

if classify_btn:
    if not message.strip():
        st.warning("Please enter a message before classifying.")
    else:
        with st.spinner("Analysing message..."):
            time.sleep(0.4)
            prediction, probability, cleaned = predict(message)

        spam_prob = round(probability[1] * 100, 2)
        ham_prob  = round(probability[0] * 100, 2)

        st.divider()
        st.markdown("#### Result")

        # ── Load and render result card HTML ──────────────────────────────────
        result_html = load_html(
            "templates/result_card.html",
            result_class      = "spam" if prediction == 1 else "ham",
            result_icon       = "🚨" if prediction == 1 else "✅",
            result_label      = "SPAM" if prediction == 1 else "HAM (Legitimate)",
            result_message    = (
                "This message is likely spam. Do not click any links."
                if prediction == 1 else
                "This message appears to be a normal, legitimate SMS."
            ),
            ham_prob          = ham_prob,
            spam_prob         = spam_prob,
            original_message  = message,
            preprocessed_message = cleaned if cleaned else "(empty after preprocessing)"
        )

        st.markdown(result_html, unsafe_allow_html=True)

        # ── Spam confidence progress bar ──────────────────────────────────────
        st.markdown("")
        st.markdown("**Spam confidence**")
        st.progress(spam_prob / 100)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Trained on UCI SMS Spam Dataset · Dockerized")