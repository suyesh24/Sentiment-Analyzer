import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("my_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Same clean_text function used before
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=250)  # Replace 250 with your X.shape[1]
    pred = model.predict(padded)
    return label_map[np.argmax(pred)]

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("💬 Customer Review Sentiment Analyzer")
st.write("Enter your product review below to predict sentiment.")

review = st.text_area("Review:")

if st.button("Predict Sentiment"):
    if review.strip() != "":
        result = predict_sentiment(review)
        st.success(f"**Predicted Sentiment:** {result}")
    else:
        st.warning("Please enter a review!")
