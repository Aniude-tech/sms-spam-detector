"""SMS Spam Detector: Streamlit app (TF-IDF + Logistic Regression)."""
import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import clean_text

# ------------------------------------------------------------------ settings
REPO_URL = "https://github.com/Aniude-tech/sms-spam-detector"   

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "spam_pipeline.pkl"
RESULTS_PATH = ROOT / "reports" / "final_test_results.csv"
FIG_DIR = ROOT / "reports" / "figures"

EXAMPLES = {
    "🎁 Prize scam": "Congratulations! You have won N5,000,000 in the MTN promo. Send your account details to claim now",
    "⚠️ Fake alert": "URGENT! Your MTN line will be blocked. Click http://bit.ly/mtn-verify to update your details",
    "📱 Free offer": "Free entry! Text WIN to 80086 to get a chance to win a free iPhone",
    "💬 Pidgin": "Abeg call me when you reach house, I dey wait",
    "🗓️ Work": "Meeting has been moved to 3pm tomorrow. Please confirm.",
    "🏦 Bank alert": "Dear customer, your account has been credited with NGN 25,000. Bal: NGN 40,200",
}

FIGURES = [
    ("confusion_matrices_tuned.png", "Confusion matrices",
     "Predictions versus true labels on the held-out test set, for each tuned model."),
    ("model_comparison_tuned.png", "Model comparison",
     "Accuracy, precision, recall and F1 (spam class) for the three tuned models."),
    ("roc_pr_curves.png", "ROC and precision-recall curves",
     "The precision-recall curve is the more informative one when spam is only ~12% of the data."),
    ("top_features_lr.png", "Most influential words",
     "Words (stemmed) with the largest Logistic Regression weights toward spam and toward ham."),
    ("custom_test_by_category.png", "Nigerian-context test",
     "Accuracy on a small hand-written set of Nigerian-style messages, by category."),
    ("custom_test_scores.png", "Spam score per custom message",
     "Messages on the wrong side of the dashed 0.5 line were misclassified."),
]

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩",
                   layout="wide", initial_sidebar_state="auto")


# ------------------------------------------------------------------ helpers
def html_block(markup: str) -> None:
    """Render raw HTML; newlines and indentation are collapsed so Markdown
    never mistakes indented lines for a code block."""
    st.markdown(re.sub(r"\n\s*", " ", markup).strip(), unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH).set_index("Model")
    return None


def spam_scores(model, cleaned_texts):
    """Spam probability for each cleaned message."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(cleaned_texts)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(cleaned_texts)))


def influential_words(model, cleaned, n=6):
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
        return [], []


def message_facts(text):
    has_link = bool(re.search(r"http|www\.|\b[a-z0-9-]+\.(com|net|org|ng|co)\b", text, re.I))
    return [
        ("Characters", len(text)),
        ("Words", len(text.split())),
        ("Digits", sum(c.isdigit() for c in text)),
        ("Link", "Yes" if has_link else "No"),
    ]


def confidence_label(score, threshold):
    gap = abs(score - threshold)
    if gap < 0.10:
        return "Uncertain: close to the threshold"
    if gap < 0.30:
        return "Moderate confidence"
    return "High confidence"


def chips(words, kind):
    if not words:
        return '<span class="muted">none</span>'
    return "".join(f'<span class="chip {kind}">{html.escape(w)}</span>' for w in words)


def render_result(text, model, threshold):
    cleaned = clean_text(text)
    if not cleaned.strip():
        html_block("""
            <div class="result neutral">
              <p class="verdict">🤔 Nothing to judge</p>
              <p class="sub">After cleaning, no informative words are left (only stopwords,
              symbols or very short words), so the model has nothing to work with.</p>
            </div>""")
        return

    score = float(spam_scores(model, [cleaned])[0])
    is_spam = score >= threshold
    kind = "spam" if is_spam else "ham"
    verdict = "🚨 Likely spam" if is_spam else "✅ Looks legitimate"
    spam_w, ham_w = influential_words(model, cleaned)
    facts = "".join(f'<div class="fact"><b>{v}</b><span>{k}</span></div>' for k, v in message_facts(text))

    html_block(f"""
        <div class="result {kind}">
          <div class="top">
            <p class="verdict">{verdict}</p>
            <div class="score"><b>{score:.0%}</b><span>spam score</span></div>
          </div>
          <div class="meter">
            <div class="tick" style="left:{threshold * 100:.1f}%"></div>
            <div class="marker" style="left:{score * 100:.1f}%"></div>
          </div>
          <div class="meter-labels"><span>Legitimate</span><span>Threshold {threshold:.2f}</span><span>Spam</span></div>
          <p class="sub">{confidence_label(score, threshold)}</p>
        </div>
        <div class="facts">{facts}</div>
        <div class="why">
          <p class="why-title">What pushed the score up</p>{chips(spam_w, "spam")}
          <p class="why-title">What pushed it down</p>{chips(ham_w, "ham")}
        </div>""")
    with st.expander("See the cleaned text the model received"):
        st.code(cleaned, language=None)
        st.caption("Words are lower-cased and stemmed, numbers become tokens, and stopwords are removed.")


# ------------------------------------------------------------------ styling
html_block("""
<style>
:root { --ink:#1F2937; --muted:#6B7280; --line:#E5E7EB; --brand:#4F46E5;
        --spam:#B91C1C; --spam-bg:#FEF2F2; --spam-line:#FCA5A5;
        --ham:#047857;  --ham-bg:#ECFDF5;  --ham-line:#6EE7B7; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1120px; }

.hero { background: linear-gradient(120deg,#4338CA 0%,#6D28D9 55%,#A855F7 100%);
        border-radius: 22px; padding: 2rem 2.2rem; margin-bottom: 1.4rem; }
.hero h1 { color:#fff; font-size:2.2rem; font-weight:800; margin:0 0 .45rem 0; padding:0; line-height:1.15; }
.hero p  { color:rgba(255,255,255,.9); margin:0; font-size:1.04rem; max-width:44rem; }
.pill { display:inline-block; background:rgba(255,255,255,.18); color:#fff; border-radius:999px;
        padding:.2rem .8rem; font-size:.8rem; margin:1rem .45rem 0 0; }

.stTabs [data-baseweb="tab-list"] { gap:.4rem; }
.stTabs [data-baseweb="tab"] { height:2.6rem; padding:0 1rem; border-radius:10px 10px 0 0; }
.stTextArea textarea { border-radius:14px; font-size:1.02rem; }
.stButton, div[data-testid="stButton"] { width:100%; }
.stButton > button { border-radius:12px; font-weight:600; width:100%; }

.result { border-radius:18px; padding:1.2rem 1.4rem 1rem; border:1.5px solid var(--line); background:#fff; }
.result.spam { background:var(--spam-bg); border-color:var(--spam-line); }
.result.ham  { background:var(--ham-bg);  border-color:var(--ham-line); }
.result.neutral { background:#F9FAFB; }
.result .top { display:flex; justify-content:space-between; align-items:center; gap:1rem; }
.result .verdict { font-size:1.55rem; font-weight:800; margin:0; }
.result.spam .verdict { color:var(--spam); }
.result.ham  .verdict { color:var(--ham); }
.result .score { text-align:right; line-height:1.1; }
.result .score b { display:block; font-size:2rem; color:var(--ink); }
.result .score span { font-size:.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.result .sub { color:var(--muted); margin:.7rem 0 0 0; font-size:.92rem; }

.meter { position:relative; height:12px; border-radius:999px; margin:1.1rem 0 .35rem;
         background:linear-gradient(90deg,#34D399 0%,#FBBF24 55%,#F87171 100%); }
.meter .marker { position:absolute; top:-6px; width:6px; height:24px; background:#111827;
                 border:2px solid #fff; border-radius:4px; transform:translateX(-50%); }
.meter .tick { position:absolute; top:-3px; width:2px; height:18px; background:rgba(17,24,39,.35); }
.meter-labels { display:flex; justify-content:space-between; font-size:.74rem; color:var(--muted); }

.facts { display:flex; gap:.6rem; margin:.9rem 0; flex-wrap:wrap; }
.fact { flex:1 1 70px; background:#fff; border:1px solid var(--line); border-radius:12px;
        padding:.55rem .7rem; text-align:center; }
.fact b { display:block; font-size:1.15rem; color:var(--ink); }
.fact span { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }

.why { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem .9rem; margin-bottom:.6rem; }
.why-title { font-size:.78rem; font-weight:700; color:var(--muted); text-transform:uppercase;
             letter-spacing:.05em; margin:.4rem 0 .35rem 0; }
.chip { display:inline-block; border-radius:999px; padding:.15rem .7rem; margin:0 .35rem .35rem 0;
        font-size:.85rem; font-weight:600; }
.chip.spam { background:var(--spam-bg); color:var(--spam); border:1px solid var(--spam-line); }
.chip.ham  { background:var(--ham-bg);  color:var(--ham);  border:1px solid var(--ham-line); }
.muted { color:var(--muted); font-size:.88rem; }

.placeholder { border:2px dashed #D1D5DB; border-radius:18px; padding:2.4rem 1.4rem; text-align:center;
               color:var(--muted); background:rgba(255,255,255,.6); }
.placeholder b { display:block; font-size:1.1rem; color:var(--ink); margin-bottom:.3rem; }
.footer { text-align:center; color:var(--muted); font-size:.82rem; margin-top:2.2rem; }
</style>
""")

# ------------------------------------------------------------------ data
model = load_model()
results = load_results()

best_f1_text = None
if results is not None and "F1 (spam)" in results.columns:
    _final = next((r for r in results.index if "Logistic" in r), results["F1 (spam)"].idxmax())
    best_f1_text = f"Test F1 {results.loc[_final, 'F1 (spam)']:.2f}"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Spam threshold", 0.30, 0.90, 0.50, 0.05,
        help="A message is flagged as spam when its score is at or above this value. "
             "Raise it for fewer false alarms; lower it to catch more spam.",
    )
    st.caption("Default 0.50. Scores close to the threshold are uncertain.")
    st.divider()
    st.markdown("### ⚠️ Know the limits")
    st.caption(
        "Trained mainly on UK and Singapore English SMS. Nigerian bank alerts, local short "
        "codes and pidgin may be misjudged. Use this as a decision aid, not a final verdict. "
        "Messages you enter are not stored."
    )
    if "YOUR-USERNAME" not in REPO_URL:
        st.divider()
        st.markdown(f"[View the code on GitHub ↗]({REPO_URL})")

# ------------------------------------------------------------------ hero
pills = ["TF-IDF + Logistic Regression", "UCI SMS Spam Collection", "Stratified 5-fold CV"]
if best_f1_text:
    pills.append(best_f1_text)
html_block(f"""
    <div class="hero">
      <h1>📩 SMS Spam Detector</h1>
      <p>Paste a text message and see whether it looks like spam, how confident the model is,
      and which words drove the decision.</p>
      {"".join(f'<span class="pill">{p}</span>' for p in pills)}
    </div>""")

tab_check, tab_batch, tab_insights, tab_about = st.tabs(
    ["🔍 Check a message", "📄 Batch check", "📊 Model insights", "ℹ️ About"]
)

# ------------------------------------------------------------------ tab 1
with tab_check:
    def _on_example():
        label = st.session_state.get("example_pick")
        if label:
            st.session_state["msg"] = EXAMPLES[label]
            st.session_state["checked"] = EXAMPLES[label]

    def _on_classify():
        text = st.session_state.get("msg", "")
        st.session_state["example_pick"] = None
        st.session_state["empty_warning"] = not text.strip()
        st.session_state["checked"] = text if text.strip() else None

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.selectbox("Try an example", list(EXAMPLES), index=None, key="example_pick",
                     placeholder="Pick an example message...", on_change=_on_example)
        st.text_area("Your message", key="msg", height=150,
                     placeholder="Type or paste an SMS here...")
        st.button("Classify message", type="primary", key="classify", on_click=_on_classify)
        if st.session_state.pop("empty_warning", False):
            st.warning("Please enter a message first.")

    with right:
        checked = st.session_state.get("checked")
        if checked:
            render_result(checked, model, threshold)
        else:
            html_block("""
                <div class="placeholder">
                  <b>Your result will appear here</b>
                  Choose an example or type a message, then press <i>Classify message</i>.
                </div>""")

# ------------------------------------------------------------------ tab 2
with tab_batch:
    st.markdown("Upload a CSV with a column of messages and get a spam score for every row.")
    sample = pd.DataFrame({"message": list(EXAMPLES.values())})
    st.download_button("Download a sample CSV", sample.to_csv(index=False).encode("utf-8"),
                       file_name="sample_messages.csv", mime="text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        if data.empty:
            st.warning("That file has no rows.")
        else:
            default = list(data.columns).index("message") if "message" in data.columns else 0
            col = st.selectbox("Column containing the messages", data.columns, index=default)
            if len(data) > 5000:
                st.warning("Only the first 5,000 rows will be processed.")
                data = data.head(5000)
            if st.button("Classify all messages", type="primary", key="classify_all"):
                cleaned = data[col].astype(str).apply(clean_text)
                scores = spam_scores(model, cleaned.tolist())
                out = data.copy()
                out["spam_score"] = scores.round(3)
                out["prediction"] = np.where(scores >= threshold, "spam", "ham")
                out = out.sort_values("spam_score", ascending=False)
                n_spam = int((out["prediction"] == "spam").sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Messages checked", len(out))
                m2.metric("Flagged as spam", n_spam)
                m3.metric("Share flagged", f"{n_spam / len(out):.0%}")
                st.dataframe(
                    out,
                    column_config={"spam_score": st.column_config.ProgressColumn(
                        "Spam score", min_value=0.0, max_value=1.0, format="%.2f")},
                    hide_index=True,
                )
                st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                                   file_name="spam_predictions.csv", mime="text/csv")

# ------------------------------------------------------------------ tab 3
with tab_insights:
    st.markdown("### How well does the model work?")
    if results is None:
        st.info("Test results are not available yet (reports/final_test_results.csv is missing).")
    else:
        final_row = next((r for r in results.index if "Logistic" in r), results["F1 (spam)"].idxmax())
        st.caption(f"Final model: **{final_row}**, evaluated once on a held-out test set (20% of the data).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{results.loc[final_row, 'Accuracy']:.1%}")
        c2.metric("Spam precision", f"{results.loc[final_row, 'Precision (spam)']:.1%}",
                  help="Of the messages flagged as spam, how many really were.")
        c3.metric("Spam recall", f"{results.loc[final_row, 'Recall (spam)']:.1%}",
                  help="Of the real spam, how much was caught.")
        c4.metric("Spam F1", f"{results.loc[final_row, 'F1 (spam)']:.3f}")
        st.markdown("**All tuned models**")
        st.dataframe(results.round(4))

    st.markdown("### Figures")
    shown = [f for f in FIGURES if (FIG_DIR / f[0]).exists()]
    if not shown:
        st.info("No figures found in reports/figures/ yet.")
    for i in range(0, len(shown), 2):
        cols = st.columns(2, gap="large")
        for col, (fname, title, caption) in zip(cols, shown[i:i + 2]):
            with col:
                st.markdown(f"**{title}**")
                st.image(str(FIG_DIR / fname))
                st.caption(caption)

# ------------------------------------------------------------------ tab 4
with tab_about:
    st.markdown("""
### About this project
This app is a capstone project for the **Tech4Youth Advanced ICT Skills Training Programme**
(Machine Learning / AI track). It classifies SMS messages as *spam* or *ham* (legitimate).

**Data.** UCI SMS Spam Collection (Almeida & Gómez Hidalgo): about 5,500 labelled English
messages, roughly 13% spam. Exact duplicates were removed before splitting.

**Method.**
1. Clean the text: decode HTML entities, replace links, phone numbers, amounts and numbers with
   tokens, lower-case, remove stopwords, stem.
2. Convert to TF-IDF features (unigrams and bigrams).
3. Train Naive Bayes, Logistic Regression and Linear SVM, tune with stratified 5-fold
   cross-validation, and choose the final model by cross-validated F1.
4. Evaluate once on a stratified 20% test set, reporting precision, recall and F1 for the spam class
   (accuracy alone is misleading with imbalanced classes).

**Limitations.**
- The training data is mostly UK and Singapore English, so local formats (Naira amounts, bank and
  telecom alerts, pidgin) can be misjudged. Legitimate bank alerts are the most likely false alarms.
- Spam written like a personal message or a service notification is the hardest to catch.
- The model is a decision aid; it should not be the only safeguard.

**Privacy.** Messages entered here are scored in memory and are not stored by the app.
""")
    if "YOUR-USERNAME" not in REPO_URL:
        st.markdown(f"[Source code and full write-up on GitHub ↗]({REPO_URL})")

html_block('<div class="footer">Built with scikit-learn and Streamlit · Tech4Youth capstone project</div>')