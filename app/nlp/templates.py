import json
import random
from pathlib import Path

# ---------------------------
# Load templates from JSON
# ---------------------------
JSON_FILE = Path(__file__).parent.parent.parent / "enron_dataset/email_templates.json"
#print(JSON_FILE.exists())  # Should print True

with open(JSON_FILE, "r", encoding="utf-8") as f:
    TEMPLATES = json.load(f)
#print(TEMPLATES.keys())
# ---------------------------
# Utility function
# ---------------------------
def format_template(template: str, **kwargs) -> str:
    """Safely format a template string with keyword arguments."""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"[Template missing key: {e}]"

# ---------------------------
# Random template pickers
# ---------------------------
def random_greeting(name: str = "") -> str:
    return format_template(random.choice(TEMPLATES.get("greeting", [])), name=name)

def random_sign_off() -> str:
    return random.choice(TEMPLATES.get("sign_off", []))

def random_polite_request(action: str = "") -> str:
    return format_template(random.choice(TEMPLATES.get("polite_request", [])), action=action)

def random_polite_suggestion(suggestion: str = "") -> str:
    return format_template(random.choice(TEMPLATES.get("polite_suggestion", [])), suggestion=suggestion)

def random_apology(issue: str = "") -> str:
    return format_template(random.choice(TEMPLATES.get("apology", [])), issue=issue)

def random_thank_you(action: str = "") -> str:
    return format_template(random.choice(TEMPLATES.get("thank_you", [])), action=action)

def random_paraphrase(text: str = "") -> str:
    return format_template(random.choice(TEMPLATES.get("paraphrase_template", [])), text=text)

def random_intent_response(intent: str, **kwargs) -> str:
    """Return a single random template for a given intent."""
    if intent in TEMPLATES.get("intent_responses", {}):
        return format_template(random.choice(TEMPLATES["intent_responses"][intent]), **kwargs)
    return "[No template available for this intent]"

# ---------------------------
# Multiple responses for intent
# ---------------------------
def get_multiple_responses(intent, n=3, **placeholders):
    """
    Fetch 'n' templates for a given intent and fill placeholders.
    """
    # Use the correct key
    templates = TEMPLATES['intent_responses'].get(intent, [])

    # Take first n templates (or fewer if not enough)
    selected = templates[:n]

    # Fill placeholders
    filled = [t.format(**placeholders) for t in selected]
    return filled

def get_response_for_intent(intent, **placeholders):
    """
    Return the first template for an intent
    """
    templates = TEMPLATES['intent_responses'].get(intent, [])
    if templates:
        return templates[0].format(**placeholders)
    return ""