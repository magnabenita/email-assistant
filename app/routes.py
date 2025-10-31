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
    "request": {"action": "perform task"},
    "information": {"info": "details"},
    "complaint": {"issue": "problem"},
    "thank_you": {},
}

@router.post("/suggest_email/")
def suggest_email(request: EmailRequest):
    # 1️⃣ Detect intent
    intent_result = predict_intent(request.email_text)
    predicted_intent = intent_result.get("predicted_intent", "other")

    # 2️⃣ Extract a keyword/topic from the user’s input dynamically
    # (simple heuristic: pick the most relevant noun or last key term)
    import re
    words = re.findall(r"\b[a-zA-Z]{4,}\b", request.email_text)
    topic = words[-1] if words else "matter"

    # 3️⃣ Prepare placeholders dynamically
    placeholders = {"name": request.name}
    if predicted_intent == "request":
        placeholders["action"] = f"handle {topic}"
    elif predicted_intent == "information":
        placeholders["info"] = topic
    elif predicted_intent == "complaint":
        placeholders["issue"] = topic
    elif predicted_intent == "thank_you":
        placeholders["context"] = topic

    # 4️⃣ Get templates
    templates = get_multiple_responses(predicted_intent, n=3, **placeholders)

    # 5️⃣ Paraphrase user's input (generate multiple distinct paraphrases)
    input_paraphrases = []
    for _ in range(3):
        paras = paraphrase(request.email_text)
        if isinstance(paras, list):
            input_paraphrases.append(paras[0])
        else:
            input_paraphrases.append(paras)

    # 6️⃣ Paraphrase templates
    template_paraphrases = []
    for t in templates:
        paras = paraphrase(t)
        if isinstance(paras, list):
            template_paraphrases.append(paras[0])
        else:
            template_paraphrases.append(paras)

    # 7️⃣ Combine all bodies
    all_bodies = input_paraphrases + template_paraphrases

    # 8️⃣ Format with greeting & sign-off
    full_emails = [
        f"{random_greeting(request.name)}\n\n{body}\n\n{random_sign_off()}"
        for body in all_bodies
    ]

    return {
        "intent": predicted_intent,
        "topic": topic,
        "confidence_scores": intent_result.get("scores", {}),
        "templates": full_emails
    }
