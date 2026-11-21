from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import clean_text

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "spam_pipeline.pkl"
RESULTS_PATH = ROOT / "reports" / "final_test_results.csv"

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

EXAMPLES = {
    "Prize scam": "Congratulations! You have won N5,000,000 in the MTN promo. Send your account details to claim now",
    "Fake account warning": "URGENT! Your MTN line will be blocked. Click http://bit.ly/mtn-verify to update your details",
    "Free offer": "Free entry! Text WIN to 80086 to get a chance to win a free iPhone",
    "Casual message (pidgin)": "Abeg call me when you reach house, I dey wait",
    "Work message": "Meeting has been moved to 3pm tomorrow. Please confirm.",
    "Bank credit alert": "Dear customer, your account has been credited with NGN 25,000. Bal: NGN 40,200",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def spam_scores(model, cleaned_texts):
    """Spam probability for each cleaned message."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(cleaned_texts)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(cleaned_texts)))


def influential_words(model, cleaned, n=5):
    """Words pushing this message toward spam / ham (linear models only)."""
    try:
        vec = model.named_steps["tfidf"]
        coefs = model.named_steps["clf"].coef_.ravel()
        contrib = vec.transform([cleaned]).multiply(coefs).toarray().ravel()
        feats = np.array(vec.get_feature_names_out())
        order = contrib.argsort()
        spam_idx = [i for i in order[::-1][:n] if contrib[i] > 0]
        ham_idx = [i for i in order[:n] if contrib[i] < 0]
        return feats[spam_idx].tolist(), feats[ham_idx].tolist()
    except Exception:
        return None


model = load_model()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About")
    st.write(
        "Classifies SMS messages as **spam** or **ham** (legitimate) using a "
        "TF-IDF + Logistic Regression pipeline trained on the UCI SMS Spam Collection."
    )
    threshold = st.slider("Spam threshold", 0.30, 0.90, 0.50, 0.05,
                          help="A message is flagged as spam when its score is at or above this value. "
                               "Raise it to reduce false alarms, lower it to catch more spam.")
    if RESULTS_PATH.exists():
        st.subheader("Test-set performance")
        st.dataframe(pd.read_csv(RESULTS_PATH).set_index("Model"), use_container_width=True)
    st.subheader("Limitations")
    st.caption(
        "Trained mainly on UK and Singapore English SMS. Nigerian bank alerts, local "
        "short codes and pidgin may be misjudged. Use as a decision aid, not a final verdict."
    )
    st.caption("Data: Almeida & Gómez Hidalgo, UCI SMS Spam Collection.")

# ---------------- Main ----------------
st.title("📩 SMS Spam Detector")
st.write("Paste a message to check whether it looks like spam.")

tab_single, tab_batch = st.tabs(["Single message", "Batch (CSV)"])

# ----- Single message -----
with tab_single:
    def _load_example():
        choice = st.session_state.example
        if choice != "—":
            st.session_state.msg = EXAMPLES[choice]

    st.selectbox("Try an example", ["—"] + list(EXAMPLES), key="example", on_change=_load_example)
    message = st.text_area("Message", key="msg", height=130,
                           placeholder="Type or paste an SMS here...")

    if st.button("Classify", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            cleaned = clean_text(message)
            if not cleaned.strip():
                st.info("This message has no informative words after cleaning "
                        "(only stopwords or symbols), so the model cannot judge it.")
            else:
                score = float(spam_scores(model, [cleaned])[0])
                if score >= threshold:
                    st.error(f"🚨 Likely SPAM (score {score:.0%})")
                else:
                    st.success(f"✅ Likely legitimate / ham (spam score {score:.0%})")
                st.progress(min(max(score, 0.0), 1.0))
                st.caption(f"Threshold: {threshold:.2f}. Scores near the threshold are uncertain.")

                with st.expander("How the model saw this message"):
                    st.write("**Cleaned text:**", cleaned)
                    words = influential_words(model, cleaned)
                    if words:
                        spam_w, ham_w = words
                        st.write("**Words pushing toward spam:**", ", ".join(spam_w) or "none")
                        st.write("**Words pushing toward ham:**", ", ".join(ham_w) or "none")

# ----- Batch -----
with tab_batch:
    st.write("Upload a CSV with a column of messages.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        default = list(data.columns).index("message") if "message" in data.columns else 0
        col = st.selectbox("Column containing the messages", data.columns, index=default)
        if len(data) > 5000:
            st.warning("Only the first 5,000 rows will be processed.")
            data = data.head(5000)
        if st.button("Classify all messages"):
            cleaned = data[col].astype(str).apply(clean_text)
            scores = spam_scores(model, cleaned.tolist())
            out = data.copy()
            out["spam_score"] = scores.round(3)
            out["prediction"] = np.where(scores >= threshold, "spam", "ham")
            c1, c2 = st.columns(2)
            c1.metric("Messages", len(out))
            c2.metric("Flagged as spam", int((out["prediction"] == "spam").sum()))
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="spam_predictions.csv", mime="text/csv")