# check_data.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "enron_nlp")

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def check_data():
    try:
        collection = db["emails"]  # use the same collection name as in load_data.py
        count = collection.count_documents({})
        print(f"✅ Total documents in {DB_NAME}.emails: {count}")

        # Show one example document
        sample = collection.find_one()
        print("\n🔎 Example document:")
        print(sample)

    except Exception as e:
        print("❌ Error checking data:", e)

if __name__ == "__main__":
    check_data()
