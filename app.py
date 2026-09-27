from pathlib import Path

import pickle
import streamlit as st

st.set_page_config(page_title="Disaster Tweet Classifier")

BASE = Path(__file__).parent

EXAMPLES = [
    "Forest fire near La Ronge Sask. Canada",
    "This song is fire, I can't stop listening",
    "Magnitude 7.2 earthquake hits the coast, thousands evacuated",
    "I'm on fire today, nailed every meeting",
]


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "disaster_model.pkl", "rb"))


@st.cache_resource
def load_vectorizer():
    return pickle.load(open(BASE / "disaster_vectorizer.pkl", "rb"))


model = load_model()
vectorizer = load_vectorizer()

st.title("Disaster Tweet Classifier")
st.write(
    "A TF-IDF + Logistic Regression model (trained on the Kaggle 'NLP with Disaster Tweets' dataset) predicts "
    "whether a tweet is about a real disaster or uses disaster words in another sense (e.g. 'this song is fire')."
)

choice = st.selectbox("Try an example (optional)", ["Write my own"] + EXAMPLES)
default_text = "" if choice == "Write my own" else choice
text = st.text_area("Tweet text", default_text, height=100)

if st.button("Classify") and text.strip():
    proba = float(model.predict_proba(vectorizer.transform([text]))[0, 1])
    if proba >= 0.5:
        st.error(f"🚨 Predicted: **real disaster** ({proba:.1%} probability)")
    else:
        st.success(f"✅ Predicted: **not a disaster** ({1 - proba:.1%} probability of not being one)")
    st.progress(proba)
    if 0.35 < proba < 0.65:
        st.info("The model is genuinely unsure here — a sign the text is ambiguous, possibly using disaster words metaphorically.")

st.caption(
    "Model: TF-IDF (5000 words) + Logistic Regression (validation F1 ≈ 0.77). Tweets that use words like "
    "'fire' or 'flood' figuratively are the hardest cases for this kind of model."
)
