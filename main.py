# main.py
from fastapi import FastAPI, Query
from app.database import connect_to_mongo, get_email_collection, close_mongo_connection
from app.services.email_service import search_similar_emails
from app.routes import router as email_router

app = FastAPI(title="Enron NLP Backend")

# Include /suggest_email/ router
app.include_router(email_router)

# Connect to MongoDB when the app starts
@app.on_event("startup")
def startup_db():
    connect_to_mongo()

# Close MongoDB connection on shutdown
@app.on_event("shutdown")
def shutdown_db_event():
    close_mongo_connection()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Enron NLP backend!"}

@app.get("/test-db")
def test_db():
    emails_col = get_email_collection()
    total = emails_col.count_documents({})
    example = emails_col.find_one()
    # Convert ObjectId to string for JSON serialization
    if example and "_id" in example:
        example["_id"] = str(example["_id"])
    return {"total_emails": total, "example_email": example}

@app.get("/search-emails")
def search_emails_endpoint(q: str = Query(..., description="Text query to search similar emails")):
    """
    Search top 5 emails similar to the given query text.
    """
    results = search_similar_emails(q)

    # Handle errors if DB is empty or any other issue
    if "error" in results:
        return {"query": q, "error": results["error"]}

    return {"query": q, "top_emails": results}
