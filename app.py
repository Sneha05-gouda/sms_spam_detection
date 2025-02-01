import streamlit as st
import joblib
from textblob import TextBlob

# Load the trained model and vectorizer
model = joblib.load('model_files/spam_classifier.pkl')
vectorizer = joblib.load('model_files/tfidf_vectorizer.pkl')

# Sentiment Analysis Function
def analyze_sentiment(message):
    sentiment_score = TextBlob(message).sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Streamlit App UI
st.title("SMS Spam Detector & Sentiment Analyzer")
user_message = st.text_area("Enter a message:")

if st.button("Analyze"):
    message_tfidf = vectorizer.transform([user_message])
    prediction = model.predict(message_tfidf)
    spam_or_ham = "Spam" if prediction[0] == 1 else "Ham"
    sentiment = analyze_sentiment(user_message)
    st.write(f"Spam Detection: **{spam_or_ham}**")
    st.write(f"Sentiment: **{sentiment}**")
