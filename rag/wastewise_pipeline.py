import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predict import predict_image
from rag.rag_retriever import retrieve



def wastewise_pipeline(image_path, user_question):
    # Step 1: Computer Vision prediction
    predicted_class, confidence = predict_image(image_path)

    # Step 2: Combine prediction with user's question
    query = f"{predicted_class} waste. {user_question}"

    # Step 3: Retrieve relevant information using RAG
    results = retrieve(query)

    return predicted_class, confidence, results


if __name__ == "__main__":

    image_path = "test.jpg"
    user_question = "How should I dispose of this item?"

    predicted_class, confidence, results = wastewise_pipeline(
        image_path,
        user_question
    )

    print("\n========== AI WASTEWISE ==========")

    print("\nDetected Waste Type:", predicted_class)
    print("Confidence:", round(confidence * 100, 2), "%")

    print("\nUser Question:", user_question)

    print("\n--- RAG Knowledge Retrieved ---")

    for result in results:
        print("\nSource:", result["document"])
        print(result["content"])