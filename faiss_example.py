# This is an example of vector database.
# To install FAISS, run following command first:
#
# pip install faiss-cpu
#
import faiss
import numpy as np

# Step 1: Generate random vectors (e.g., embeddings)
# Let's create 100 vectors of dimension 128 (e.g., 100 feature vectors, each of length 128)
d = 128  # Dimension of each vector
n = 100  # Number of vectors

# Generate random vectors (e.g., simulate feature embeddings)
np.random.seed(42)
xb = np.random.random((n, d)).astype('float32')

# Step 2: Create a FAISS index for the vectors
# FAISS supports various types of indices. We'll use the simplest one, the flat (brute-force) index.
index = faiss.IndexFlatL2(d)  # L2 distance metric (Euclidean distance)

# Add the vectors to the index
index.add(xb)

# Step 3: Perform a query (search) for nearest neighbors
# Generate a random query vector
xq = np.random.random((1, d)).astype('float32')  # 1 query vector

# Search for the 5 nearest neighbors of the query vector
k = 5  # Number of neighbors to find
distances, indices = index.search(xq, k)

# Step 4: Print the results
print("Query vector:", xq)
print("\nNearest neighbors:")
print("Indices:", indices)
print("Distances:", distances)
