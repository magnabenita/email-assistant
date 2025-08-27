import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)  # collapse spaces
    text = re.sub(r"[^a-zA-Z ]", "", text)  # remove special chars
    words = [w.lower() for w in text.split() if w.lower() not in STOPWORDS]
    return " ".join(words)
