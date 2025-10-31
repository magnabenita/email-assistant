# dashboard.py
import streamlit as st
import requests
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "enron_nlp")

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
emails_col = db["emails"]

FASTAPI_URL = "http://127.0.0.1:8000/suggest_email/"  # Adjust if different

# ----------------------------
# STREAMLIT APP
# ----------------------------

st.set_page_config(page_title="NLP Email Assistant Dashboard", layout="wide")

st.title("📧 NLP Email Assistant Dashboard")
st.markdown("Interact with the FastAPI NLP backend and view insights on Enron emails.")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Suggest Email", "Email Dataset Explorer", "Statistics Dashboard"])

# ----------------------------
# PAGE 1: Email Suggestion
# ----------------------------
if page == "Suggest Email":
    st.subheader("💡 Generate Smart Email Suggestions")

    name = st.text_input("Your Name", "Magna")
    email_text = st.text_area("Enter email text", "Need to request a meeting update.")

    if st.button("Generate Suggestions"):
        payload = {"email_text": email_text, "name": name}
        response = requests.post(FASTAPI_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            st.success(f"**Predicted Intent:** {data['intent']}")
            st.write("### ✉️ Suggested Emails:")
            for i, email in enumerate(data["templates"], start=1):
                st.markdown(f"**Suggestion {i}:**\n\n{email}")
        else:
            st.error("❌ Failed to get response from backend")

# ----------------------------
# PAGE 2: Email Dataset Explorer
# ----------------------------
elif page == "Email Dataset Explorer":
    st.subheader("📚 Explore Stored Emails")

    emails = list(emails_col.find({}, {"_id": 0, "id": 1, "subject": 1, "body": 1}).limit(50))
    df = pd.DataFrame(emails)

    if not df.empty:
        st.dataframe(df)
        search_query = st.text_input("🔍 Search Emails by Keyword")

        if search_query:
            filtered_df = df[df["body"].str.contains(search_query, case=False, na=False)]
            st.dataframe(filtered_df)
    else:
        st.warning("No emails found in the database.")

# ----------------------------
# PAGE 3: Statistics Dashboard
# ----------------------------
elif page == "Statistics Dashboard":
    st.subheader("📊 Email Statistics")

    # Basic metrics
    total_emails = emails_col.count_documents({})
    st.metric("Total Emails", total_emails)

    # Intent distribution (if stored)
    if "intent" in emails_col.find_one() or True:
        # Example placeholder: random data
        import random
        intents = ["request", "information", "complaint", "thank_you"]
        counts = [random.randint(10, 100) for _ in intents]

        df_stats = pd.DataFrame({"Intent": intents, "Count": counts})
        st.bar_chart(df_stats.set_index("Intent"))
    else:
        st.info("Intent data not available yet.")
