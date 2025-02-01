import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from textblob import TextBlob
import joblib

# Step 1: Load the dataset
try:
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df.rename(columns={'v1': 'Label', 'v2': 'Message'})[['Label', 'Message']]
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})  # Encode labels as 0 (ham) and 1 (spam)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: spam.csv file not found. Ensure it is in the correct directory.")
    exit()

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"], test_size=0.2, random_state=42)

# Step 3: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the trained model and vectorizer
model_dir = "model_files"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

joblib.dump(model, os.path.join(model_dir, 'spam_classifier.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
print("Model and vectorizer have been saved successfully in the model_files directory.")

# Step 7: Sentiment Analysis Function
def analyze_sentiment(message):
    sentiment_score = TextBlob(message).sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Step 8: Test with new messages
new_messages = [
    "Congratulations! You have won a lottery.",
    "Can we meet for coffee tomorrow?",
    "I hate the traffic today; it ruined my day!",
    "Hope you are doing well!"
]

new_messages_tfidf = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_tfidf)

print("\nNew Message Predictions:")
for msg, pred in zip(new_messages, predictions):
    sentiment = analyze_sentiment(msg)
    spam_or_ham = "Spam" if pred == 1 else "Ham"
    print(f"Message: {msg} => {spam_or_ham}, Sentiment: {sentiment}")
import os
import joblib

if not os.path.exists('model_files'):
    os.makedirs('model_files')

joblib.dump(model, 'model_files/spam_classifier.pkl')
joblib.dump(vectorizer, 'model_files/tfidf_vectorizer.pkl')

print("Model and vectorizer have been saved successfully.")
import seaborn as sns
import matplotlib.pyplot as plt

# Plot Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix Heatmap")
plt.show()

# Plot Spam vs. Ham Distribution
df['Label'].value_counts().plot(kind='bar', title='Spam vs. Ham Distribution')
plt.xlabel('Message Type')
plt.ylabel('Frequency')
plt.show()
