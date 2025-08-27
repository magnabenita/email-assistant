# app/nlp/paraphraser_infer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load your big model (change the name/path if needed)
MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"  # or your fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def paraphrase(text: str, num_return_sequences: int = 5, max_length: int = 128):
    """
    Generate multiple paraphrases for a given input text.

    Args:
        text (str): Input sentence to paraphrase
        num_return_sequences (int): Number of paraphrases to generate
        max_length (int): Maximum length of generated sentences

    Returns:
        List[str]: A list of paraphrased sentences
    """
    # Encode input text
    inputs = tokenizer(f"paraphrase: {text}", return_tensors="pt", truncation=True).to(device)

    # Generate multiple sequences
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0
    )

    # Decode outputs and remove special tokens
    paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return paraphrases

# Quick test
if __name__ == "__main__":
    test_text = "Please schedule a meeting by tomorrow to discuss project updates."
    results = paraphrase(test_text)
    for i, sentence in enumerate(results, 1):
        print(f"{i}: {sentence}")
