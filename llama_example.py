from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = "facebook/llama-7b"  # Use "facebook/llama-13b" or another LLaMA variant if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if CUDA (GPU) is available, and if so, move the model to GPU for faster inference
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example prompt for the model
prompt = "Who is Adar in Ring of Powers?"

# Tokenize the input text
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output from the model
output = model.generate(**inputs, max_length=100)

# Decode the output tokens to text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:", generated_text)
