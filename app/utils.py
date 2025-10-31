#app/utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Initialize globally
vectorizer = TfidfVectorizer(max_features=5000)

def build_tfidf_matrix(emails):
    corpus = [e["body"] for e in emails]
    return vectorizer.fit_transform(corpus)

def get_similar_emails(query, tfidf_matrix, emails, top_k=5):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [emails[i] for i in top_indices]
