from flask import Flask, render_template, request, jsonify
import os
import sys

from predict import predict_image


# -----------------------------------
# Format RAG content
# -----------------------------------

def format_rag_content(content):

    sections = {
        "before_recycling": [],
        "reuse_ideas": [],
        "disposal_guidance": "",
        "important_note": ""
    }

    current_section = None

    for line in content.splitlines():

        line = line.strip()

        if line == "## Before Recycling":
            current_section = "before_recycling"

        elif line == "## Reuse Ideas":
            current_section = "reuse_ideas"

        elif line == "## Disposal Guidance":
            current_section = "disposal_guidance"

        elif line == "## Important Note":
            current_section = "important_note"

        elif line.startswith("## "):
            current_section = None

        elif line.startswith("- ") and current_section in [
            "before_recycling",
            "reuse_ideas"
        ]:
            sections[current_section].append(line[2:])

        elif line and current_section in [
            "disposal_guidance",
            "important_note"
        ]:

            if sections[current_section]:
                sections[current_section] += " " + line
            else:
                sections[current_section] = line

    return sections


# -----------------------------------
# Project root
# -----------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from rag.rag_retriever import retrieve


# -----------------------------------
# Flask
# -----------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------------
# Home page
# -----------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------
# Waste prediction
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    image = request.files["image"]

    if image.filename == "":
        return "No image selected"

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(image_path)

    predicted_class, confidence = predict_image(image_path)

    # Get RAG information
    user_question = (
        "How should I dispose of this item "
        "and can I reuse or recycle it?"
    )

    query = f"{predicted_class} waste. {user_question}"

    results = retrieve(query)

    rag_content = results[0]["content"]

    sections = format_rag_content(rag_content)

    return render_template(
        "result.html",
        predicted_class=predicted_class,
        confidence=round(confidence * 100, 2),
        sections=sections
    )


# -----------------------------------
# AI CHAT
# -----------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data:
        return jsonify({
            "answer": "Please enter a question."
        })

    question = data.get("question", "").strip()
    waste_type = data.get("waste_type", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question."
        })

    if not waste_type:
        return jsonify({
            "answer": "Please scan a waste item first."
        })

    # -----------------------------------
    # Ask RAG
    # -----------------------------------

    query = f"{waste_type} waste. {question}"

    results = retrieve(query)

    if not results:
        return jsonify({
            "answer": "I could not find information about this waste type."
        })

    rag_content = results[0]["content"]

    sections = format_rag_content(rag_content)

    # -----------------------------------
    # Decide which information is relevant
    # -----------------------------------

    question_lower = question.lower()

    # Recycling / preparation questions
    if any(word in question_lower for word in [
        "recycle",
        "recycling",
        "clean",
        "wash",
        "prepare"
    ]):

        items = sections["before_recycling"]

        if items:
            answer = "Here is what you should do before recycling:\n\n"

            for item in items:
                answer += "• " + item + "\n"

        else:
            answer = sections["disposal_guidance"]


    # Reuse questions
    elif any(word in question_lower for word in [
        "reuse",
        "re-use",
        "use again",
        "repurpose",
        "use it again"
    ]):

        items = sections["reuse_ideas"]

        if items:
            answer = "Here are some reuse ideas:\n\n"

            for item in items:
                answer += "• " + item + "\n"

        else:
            answer = (
                "I couldn't find specific reuse ideas "
                "for this item in my knowledge base."
            )


    # Disposal questions
    elif any(word in question_lower for word in [
        "dispose",
        "disposal",
        "throw",
        "garbage",
        "bin",
        "waste"
    ]):

        answer = sections["disposal_guidance"]


    # General question
    else:

        answer = (
            f"Here's what I know about {waste_type} waste:\n\n"
        )

        if sections["disposal_guidance"]:
            answer += (
                "🗑️ Disposal:\n"
                + sections["disposal_guidance"]
                + "\n\n"
            )

        if sections["reuse_ideas"]:
            answer += "💡 Reuse ideas:\n"

            for item in sections["reuse_ideas"]:
                answer += "• " + item + "\n"

            answer += "\n"

        if sections["important_note"]:
            answer += (
                "⚠️ Important:\n"
                + sections["important_note"]
            )

    

    return jsonify({
    "answer": answer,
    "waste_type": waste_type,
    "source": results[0]["document"]
})


# -----------------------------------
# Run application
# -----------------------------------

if __name__ == "__main__":

    app.run(debug=True)