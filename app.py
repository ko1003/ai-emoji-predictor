import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Emoji Predictor", page_icon="🤖")

st.title("🤖 AI Emotion Emoji Predictor")
st.write("Type a sentence and AI will detect the emotion.")

classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base")

text = st.text_input("Enter your text")

if st.button("Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        result = classifier(text)[0]

        label = result['label']
        score = result['score']

        emoji_dict = {
            "joy": "😊 Happy",
            "sadness": "😢 Sad",
            "anger": "😡 Angry",
            "love": "😍 Love",
            "fear": "😨 Fear",
            "surprise": "😲 Surprise"
        }

        emotion = emoji_dict.get(label.lower(), "😐 Neutral")

        st.success(f"Emotion: {emotion}")
        st.write(f"Confidence: {score:.2f}")
