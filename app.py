import streamlit as st
from transformers import pipeline

st.title("AI Emoji Predictor")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

text = st.text_input("Enter your message")

if st.button("Predict Emoji"):
    result = classifier(text)

    label = result[0]["label"]

    if label == "POSITIVE":
        st.success("😊 Happy")
    else:
        st.error("😡 Angry")