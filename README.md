# 📧 Enron NLP Backend

A **backend service for searching and suggesting emails** using NLP techniques on the Enron email dataset. Built with FastAPI and integrated with MongoDB, this project enables intelligent email search and paraphrasing.

---

## 🚀 Features

- 🔍 Search emails similar to a given query  
- 🗄️ MongoDB integration for storing and retrieving emails  
- ⚡ FastAPI backend for serving API endpoints  
- ✍️ Paraphrasing support using transformer models (T5)  
- 🧩 Easy to extend for additional NLP features  

---

## 📁 Project Structure

```
project/
│
├─ app/
│   ├─ database.py              # MongoDB connection functions
│   ├─ services/
│   │   └─ email_service.py     # Email search and NLP logic
│   └─ routes.py                # FastAPI routers
│
├─ main.py                      # FastAPI entrypoint
├─ requirements.txt             # Python dependencies
└─ README.md
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux / macOS
   venv\Scripts\activate        # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

---

## 📡 API Endpoints

| Endpoint         | Method | Description                                      |
|------------------|--------|--------------------------------------------------|
| `/`              | GET    | Root endpoint, returns welcome message           |
| `/test-db`       | GET    | Checks MongoDB connection and returns a sample email |
| `/search-emails` | GET    | Returns top 5 emails similar to query text       |

---

## 🔮 Future Extensions

- Integrate paraphrasing model to suggest alternate email drafts  
- Add authentication for multiple users  
- Deploy on cloud (AWS / GCP / Azure) for public access  
- Add endpoints for categorizing emails by intent (e.g., information, request, complaint)  

---

## 👩‍💻 Author

**Magna Benita P** – Initial development of the backend

---

## 🛠️ Notes for Contributors

- Ensure MongoDB is running and accessible  
- Maintain the structure inside `app/`  
- Use `app/services/email_service.py` for adding NLP models or extending search functionality  
- Add new endpoints to `app/routes.py` and include them in `main.py`  
- Update `requirements.txt` with any new dependencies  