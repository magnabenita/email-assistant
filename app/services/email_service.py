# app/services/email_service.py
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from app.database import get_email_collection

# -------------------------
# Load TF-IDF vectorizer and corpus once at startup
# -------------------------
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/email_corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

# Transform the corpus once
tfidf_matrix = vectorizer.transform(corpus)

# -------------------------
# Fetch emails from MongoDB
# -------------------------
def fetch_emails():
    emails_col = get_email_collection()
    emails = list(emails_col.find({}))

    # Ensure each email has an 'id' and 'body'
    for e in emails:
        e["id"] = str(e["_id"])
        if "body" not in e:
            e["body"] = ""
        if "subject" not in e:
            e["subject"] = ""
    return emails

# -------------------------
# Search top K similar emails
# -------------------------
def search_similar_emails(query, top_k=5):
    emails = fetch_emails()
    if not emails:
        return {"error": "No emails found in DB"}

    try:
        # Transform query to TF-IDF
        query_vec = vectorizer.transform([query])

        # Compute cosine similarity
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Get top K indices
        top_indices = sims.argsort()[-top_k:][::-1]

        # Prepare results with similarity scores
        results = []
        for i in top_indices:
            results.append({
                "id": emails[i]["id"],
                "subject": emails[i]["subject"],
                "body": emails[i]["body"],
                "similarity_score": float(sims[i])
            })
        return results

    except Exception as e:
        return {"error": str(e)}
