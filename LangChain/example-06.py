import openai
import faiss
import numpy as np

# GPT-3 API key setup
openai.api_key = 'YOUR_API_KEY'

# Initialize FAISS index
d = 768  # Dimension of the GPT-3 vectors
index = faiss.IndexFlatL2(d)

# Example documents
documents = [
    "Das Wetter ist sonnig und angenehm.",
    "Es regnet heute den ganzen Tag.",
    "Morgen wird es bewölkt sein."
]


# Function to get embeddings using OpenAI API
def get_embeddings(docs):
    embeddings = []
    for doc in docs:
        response = openai.Embedding.create(model="text-embedding-ada-002-v2", input=[doc])
        embedding = response['data'][0]['embedding']
        embeddings.append(embedding)
    return embeddings


# Create embeddings and insert them into the FAISS index
embeddings = get_embeddings(documents)
index.add(np.array(embeddings))

# Check the number of vectors in the index
print(f"Anzahl der Vektoren im Index: {index.ntotal}")
