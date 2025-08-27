# app/database.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "enron_nlp")

client = None
db = None

def connect_to_mongo():
    global client, db
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print(f"✅ Connected to MongoDB: {DB_NAME}")

def close_mongo_connection():
    global client
    if client:
        client.close()
        print("❌ MongoDB connection closed")

def get_email_collection():
    db = client["enron_nlp"]
    if db is not None:   # ✅ correct
        return db["emails"]

    raise Exception("Database not connected. Call connect_to_mongo() first.")
