# This demonstrates how to get sentense level, instead of word level,
# embedding.

import torch
from transformers import BertTokenizer, BertModel

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "Hello, how are you?"

# Tokenize the sentence (output will include the [CLS] and [SEP] tokens)
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

# Get the model output (last_hidden_state contains embeddings for all tokens)
with torch.no_grad():
    outputs = model(**inputs)

# Extract the last hidden state (the embeddings for all tokens)
last_hidden_state = outputs.last_hidden_state

# The embedding for the [CLS] token is the first token in the sequence
cls_embedding = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

# Print the shape of the embedding (it will be of size [1, 768] for BERT-base)
print(cls_embedding.shape)  # Output: torch.Size([1, 768])

# cls_embedding now contains the sentence embedding
print(cls_embedding)
