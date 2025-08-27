# app/nlp/paraphraser_model.py

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load a pre-trained T5 model (small and fast for testing)
MODEL_NAME = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def paraphrase(text: str, num_variations: int = 3) -> list:
    """
    Generate paraphrased versions of input text.
    Args:
        text (str): Original text to paraphrase
        num_variations (int): Number of paraphrases to return
    Returns:
        List[str]: Paraphrased texts
    """
    input_text = f"paraphrase: {text} </s>"
    encoding = tokenizer.encode_plus(
        input_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    outputs = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=256,
        num_beams=10,
        num_return_sequences=num_variations,
        temperature=1.5,
        top_p=0.95,
        early_stopping=True
    )

    paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrases
