import sys
from pathlib import Path

# Add project root to sys.path so imports work
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.nlp.templates import (
    random_greeting,
    random_sign_off,
    random_polite_request,
    random_polite_suggestion,
    random_apology,
    random_thank_you,
    random_paraphrase,
    get_response_for_intent,
    get_multiple_responses
)

# ---------------------------
# Test examples
# ---------------------------

print(random_greeting(name="Magna"))
print(random_sign_off())
print(random_polite_request(action="send the report"))
print(random_polite_suggestion(suggestion="review the document"))
print(random_apology(issue="the delay"))
print(random_thank_you(action="your help"))
print(random_paraphrase(text="Please send this by tomorrow"))
print(get_response_for_intent("greeting", name="Magna"))
print(get_multiple_responses("meeting_request", n=2, time="3 PM"))
