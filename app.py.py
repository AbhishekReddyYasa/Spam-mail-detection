"""
app.py
------
Self-learning Spam Mail Detection — Streamlit application.

Self-learning loop
==================
1. User enters a message → model predicts spam / not-spam.
2. A confirmation widget appears asking whether the prediction was correct.
3. If the user marks it as WRONG the app asks for the true label.
4. The confirmed (text, label) pair is appended to feedback_data.csv.
5. model.partial_fit() updates the model weights immediately (no full retraining).
6. The updated model is saved to disk so it persists across restarts.

Run
===
    streamlit run app.py
"""

import os
import pickle
import time

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH      = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
FEEDBACK_PATH   = "feedback_data.csv"
CLASSES         = [0, 1]   # required for partial_fit

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Helper: load / save model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model and vectorizer from disk (cached across re-runs)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error(
            "Model files not found. Please run `python create_model.py` first."
        )
        st.stop()
    model      = pickle.load(open(MODEL_PATH,      "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    return model, vectorizer


def save_model(model: MultinomialNB) -> None:
    """Persist updated model to disk."""
    pickle.dump(model, open(MODEL_PATH, "wb"))


def append_feedback(text: str, label: int) -> None:
    """Append a confirmed sample to the feedback CSV."""
    row = pd.DataFrame({"text": [text], "label": [label]})
    if os.path.exists(FEEDBACK_PATH):
        row.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(FEEDBACK_PATH, index=False)


def incremental_update(model: MultinomialNB, vectorizer: TfidfVectorizer,
                        text: str, label: int) -> MultinomialNB:
    """Update the model with a single new labelled sample via partial_fit."""
    X_new = vectorizer.transform([text])
    model.partial_fit(X_new, [label], classes=CLASSES)
    return model


def feedback_count() -> int:
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    try:
        return len(pd.read_csv(FEEDBACK_PATH))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
model, vectorizer = load_artifacts()

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
for key, default in {
    "prediction":        None,
    "confidence":        None,
    "message":           "",
    "awaiting_feedback": False,
    "last_feedback":     None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# UI — header
# ---------------------------------------------------------------------------
st.title("📧 Spam Mail Detector")
st.caption("Powered by Naïve Bayes · Self-learning via user feedback")

col_stats1, col_stats2 = st.columns(2)
with col_stats1:
    st.metric("📚 Feedback samples collected", feedback_count())
with col_stats2:
    model_size = os.path.getsize(MODEL_PATH) // 1024 if os.path.exists(MODEL_PATH) else 0
    st.metric("💾 Model size", f"{model_size} KB")

st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
message = st.text_area(
    "✉️ Enter the email message below:",
    height=180,
    placeholder="Paste or type your email content here…",
    key="input_message",
)

check_clicked = st.button("🔍 Check Message", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if check_clicked:
    if not message.strip():
        st.warning("⚠️ Please enter a message before checking.")
    else:
        st.session_state.message           = message.strip()
        st.session_state.awaiting_feedback = True
        st.session_state.last_feedback     = None

        X = vectorizer.transform([st.session_state.message])
        proba = model.predict_proba(X)[0]
        pred  = int(model.predict(X)[0])

        st.session_state.prediction  = pred
        st.session_state.confidence  = round(float(max(proba)) * 100, 1)

# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------
if st.session_state.prediction is not None:
    pred = st.session_state.prediction
    conf = st.session_state.confidence

    st.divider()
    if pred == 1:
        st.error(f"🚨 **SPAM** detected  —  confidence {conf}%")
    else:
        st.success(f"✅ **Not Spam**  —  confidence {conf}%")

    # Confidence bar
    st.progress(conf / 100, text=f"Model confidence: {conf}%")

# ---------------------------------------------------------------------------
# Feedback section
# ---------------------------------------------------------------------------
if st.session_state.awaiting_feedback and st.session_state.last_feedback is None:
    st.divider()
    st.subheader("💬 Was this prediction correct?")
    st.caption(
        "Your feedback trains the model in real time — "
        "even a single correction helps."
    )

    fb_col1, fb_col2 = st.columns(2)

    with fb_col1:
        if st.button("👍 Yes, correct", use_container_width=True):
            # Confirmed: use the predicted label as ground truth
            confirmed_label = st.session_state.prediction
            model = incremental_update(model, vectorizer,
                                       st.session_state.message, confirmed_label)
            save_model(model)
            append_feedback(st.session_state.message, confirmed_label)
            st.session_state.awaiting_feedback = False
            st.session_state.last_feedback     = "correct"
            st.cache_resource.clear()
            st.rerun()

    with fb_col2:
        if st.button("👎 No, it was wrong", use_container_width=True):
            st.session_state.awaiting_feedback = False
            st.session_state.last_feedback     = "wrong"
            st.rerun()

# ---------------------------------------------------------------------------
# If wrong: ask for true label and update
# ---------------------------------------------------------------------------
if st.session_state.last_feedback == "wrong":
    st.divider()
    st.subheader("🔧 What is the correct label?")
    true_label_choice = st.radio(
        "Select the correct classification:",
        options=["📧 Not Spam (Ham)", "🚨 Spam"],
        horizontal=True,
    )
    if st.button("✅ Submit correction", type="primary"):
        true_label = 1 if "Spam" in true_label_choice and "Not" not in true_label_choice else 0
        model = incremental_update(model, vectorizer,
                                   st.session_state.message, true_label)
        save_model(model)
        append_feedback(st.session_state.message, true_label)
        st.session_state.last_feedback = "corrected"
        st.cache_resource.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Post-feedback acknowledgement
# ---------------------------------------------------------------------------
if st.session_state.last_feedback == "correct":
    st.success("✅ Prediction confirmed and model updated. Thank you!")

if st.session_state.last_feedback == "corrected":
    st.success("🔧 Correction saved! The model has been updated. Thank you!")

# ---------------------------------------------------------------------------
# Sidebar — feedback history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Feedback History")

    if os.path.exists(FEEDBACK_PATH):
        try:
            df_fb = pd.read_csv(FEEDBACK_PATH)
            if df_fb.empty:
                st.info("No feedback collected yet.")
            else:
                spam_count = int((df_fb["label"] == 1).sum())
                ham_count  = int((df_fb["label"] == 0).sum())

                st.metric("Total samples", len(df_fb))
                st.metric("Spam confirmed", spam_count)
                st.metric("Ham confirmed", ham_count)

                st.divider()
                st.subheader("Recent feedback")
                recent = df_fb.tail(5).copy()
                recent["label"] = recent["label"].map({1: "🚨 Spam", 0: "✅ Ham"})
                recent["text"]  = recent["text"].str[:40] + "…"
                st.dataframe(recent[::-1], use_container_width=True, hide_index=True)

                st.divider()
                if st.button("🗑️ Clear all feedback", type="secondary"):
                    pd.DataFrame(columns=["text", "label"]).to_csv(
                        FEEDBACK_PATH, index=False
                    )
                    st.success("Feedback cleared.")
                    time.sleep(1)
                    st.rerun()
        except Exception as e:
            st.error(f"Could not load feedback: {e}")
    else:
        st.info("No feedback file found yet.")

    st.divider()
    st.caption(
        "Model updates happen in real time via `partial_fit`. "
        "Each correction shifts decision boundaries without full retraining."
    )
