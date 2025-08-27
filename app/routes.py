# app/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.nlp.intent_model import predict_intent
from app.nlp.templates import get_multiple_responses, random_greeting, random_sign_off
from app.nlp.paraphraser_infer import paraphrase  # only one import

router = APIRouter()

class EmailRequest(BaseModel):
    email_text: str
    name: str = "Magna"

DEFAULT_PLACEHOLDERS = {
    "request": {"action": "perform task", "topic": "project report"},
    "information": {"info": "details", "topic": "project report"},
    "complaint": {"issue": "problem"},
    "thank_you": {},
}

@router.post("/suggest_email/")
def suggest_email(request: EmailRequest):
    # 1️⃣ Detect intent
    intent_result = predict_intent(request.email_text)
    predicted_intent = intent_result.get("predicted_intent", "other")

    # 2️⃣ Prepare placeholders
    placeholders = DEFAULT_PLACEHOLDERS.get(predicted_intent, {})
    placeholders["name"] = request.name

    # 3️⃣ Get templates
    templates = get_multiple_responses(predicted_intent, n=3, **placeholders)

    # 4️⃣ Paraphrase user's input (generate multiple distinct paraphrases)
    input_paraphrases = [paraphrase(request.email_text) for _ in range(3)]

    # 5️⃣ Paraphrase templates
    template_paraphrases = [paraphrase(t) for t in templates]

    # 6️⃣ Combine all bodies
    all_bodies = input_paraphrases + template_paraphrases

    # 7️⃣ Format with greeting & sign-off
    full_emails = [
        f"{random_greeting(request.name)}\n\n{body}\n\n{random_sign_off()}"
        for body in all_bodies
    ]

    return {
        "intent": predicted_intent,
        "confidence_scores": intent_result.get("scores", {}),
        "templates": full_emails
    }


