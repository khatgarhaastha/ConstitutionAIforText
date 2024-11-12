from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the Llama 3.1 model and tokenizer from Hugging Face
# Replace 'Llama-3.1' with the actual model name if needed
model_name = "llama-3b"  # Use the appropriate Llama model version from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate the prompt for Llama
def generate_llama_prompt(text, constitution):
    prompt = f"""
    Analyze the following text and check if it follows the constitution provided below.
    Constitution: "{constitution}"

    Text: "{text}"

    If the text violates the constitution, explain where and how it violates specific sections of the constitution.
    """
    return prompt

# Processed text and constitution as inputs
processed_text = "The government has decided to impose curfews without any valid reason."
constitution = """
Article 1: Every citizen has the right to freedom of speech and movement.
Article 2: No government authority shall impose curfews or restrictions without a valid reason, as defined by law.
"""

# Generate the prompt
prompt = generate_llama_prompt(processed_text, constitution)

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output using Llama model
outputs = model.generate(
    inputs.input_ids,
    max_length=500,  # You can adjust this depending on the output length
    num_beams=4,     # Optional: use beam search to improve the response
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Decode and print the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
