import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# Define the prompt
prompt = (
    "hi what you're name?"
)

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    max_length=50,
)

# Decode the generated tokens
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)

# 현재 실행시 노트북 나감