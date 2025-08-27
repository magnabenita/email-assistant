# app/nlp/train_recommender.py

import os
import pickle
from app.database import connect_to_mongo, get_email_collection
from sklearn.feature_extraction.text import TfidfVectorizer

# 1️⃣ Connect to MongoDB
connect_to_mongo()
emails_col = get_email_collection()

# 2️⃣ Fetch emails
emails = list(emails_col.find({}))
print(f"Total emails fetched: {len(emails)}")

# 3️⃣ Prepare corpus safely
corpus = [email.get("body", "") for email in emails if email.get("body")]
print(f"Corpus prepared with {len(corpus)} emails")

# 4️⃣ Build TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(corpus)

print("✅ TF-IDF vectorizer trained")

# 5️⃣ Save vectorizer and corpus
os.makedirs("models", exist_ok=True)  # Create models/ folder if not exists

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    print("✅ Saved TF-IDF vectorizer to models/tfidf_vectorizer.pkl")

with open("models/email_corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
    print("✅ Saved email corpus to models/email_corpus.pkl")

print("🎉 Training complete")
