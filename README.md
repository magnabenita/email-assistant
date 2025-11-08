# Email Assistant – NLP-Powered Smart Email Generator

An AI-powered backend for analyzing, classifying, and generating polite, context-aware email suggestions using NLP techniques.
Built with FastAPI, integrated with MongoDB, and powered by transformer-based models for intent detection and email rephrasing.

---

## 🚀 Features

- 🧠 Email Intent Detection – Automatically identifies the purpose (e.g., request, complaint, apology, --information).
-  Smart Email Suggestion Generator – Suggests polite and professional versions of user-provided text.
-  🧩 Template-Based Customization – Controlled by editable JSON templates (email_templates.json).
-  📊 Interactive Dashboard – View intent stats and model insights via a Streamlit-powered dashboard.
-  ⚡ FastAPI Backend – Lightweight, async API serving NLP-powered endpoints.
-  🗄️ MongoDB Integration – Store, query, and retrieve email data efficiently.

---

## 📁 Project Structure

```
project/
│
├─ app/
│   ├─ routes.py                # FastAPI routes for NLP endpoints
│   ├─ utils.py                 # NLP logic, intent detection, suggestion generation
│   ├─ database.py              # MongoDB connection helper
│
├─ enron_dataset/
│   └─ email_templates.json     # Editable email tone/style templates
│
├─ dashboard.py                 # Streamlit-based dashboard UI
├─ main.py                      # FastAPI app entrypoint
├─ requirements.txt             # Dependencies
└─ README.md                    # Documentation

```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/magnabenita/email-assistant
   cd email-assistant
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

| Endpoint           | Method | Description                                                         |
| ------------------ | ------ | ------------------------------------------------------------------- |
| `/`                | GET    | Root endpoint – health check                                        |
| `/predict-intent`  | POST   | Detects the intent (e.g., *request*, *complaint*) of the input text |
| `/suggest-email`   | POST   | Generates multiple polite email versions for a given user input     |
| `/analyze-dataset` | GET    | Returns basic dataset statistics (intent counts, etc.)              |

---

## 🔮 Future Extensions

- Integrate paraphrasing model to suggest alternate email drafts  
- Add authentication for multiple users  
- Deploy on cloud (AWS / GCP / Azure) for public access  
- Add endpoints for categorizing emails by intent (e.g., information, request, complaint)  

---

## Author

**Magna Benita P** 

🛠️ Developer & Researcher – NLP + Smart Communication Systems

📫 GitHub: @magnabenita

---

## 🛠️ Notes for Contributors

- Ensure MongoDB is running and accessible  
- Maintain the structure inside `app/`  
- Use `app/services/email_service.py` for adding NLP models or extending search functionality  
- Add new endpoints to `app/routes.py` and include them in `main.py`  
- Update `requirements.txt` with any new dependencies  
