import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Locate knowledge-base files
# -----------------------------

RAG_FOLDER = os.path.dirname(os.path.abspath(__file__))

documents = []
document_names = []

for filename in os.listdir(RAG_FOLDER):

    if filename.endswith(".md"):

        filepath = os.path.join(RAG_FOLDER, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        documents.append(content)
        document_names.append(filename)


print("Knowledge base loaded!")
print("Documents:", document_names)


# -----------------------------
# 2. Load embedding model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")


# -----------------------------
# 4. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


print("FAISS index created!")
print("Number of documents:", index.ntotal)


# -----------------------------
# 5. Retrieval function
# -----------------------------

def retrieve(query, top_k=1):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        results.append({
            "document": document_names[i],
            "content": documents[i]
        })

    return results


# -----------------------------
# 6. Test the retriever
# -----------------------------

if __name__ == "__main__":

    # Simulated prediction from our computer vision model
    predicted_class = "plastic"

    # User's question
    user_question = "How should I recycle this item?"

    # Combine prediction + question
    query = f"{predicted_class} waste. {user_question}"

    # Retrieve relevant knowledge
    results = retrieve(query)

    print("\nDetected Waste Type:", predicted_class)
    print("User Question:", user_question)

    print("\n--- RAG Knowledge Retrieved ---")

    for result in results:

        print("\nSource:", result["document"])
        print(result["content"])