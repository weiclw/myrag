
# Use the following command to install necessary libraries,
# including bert. It has to be used with either tensorflow
# or pytorch:
#
# pip install transformers
#
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode some text
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")  # 'pt' for PyTorch tensors
print(inputs)

# Get the BERT output (hidden states)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)
print(last_hidden_states)
