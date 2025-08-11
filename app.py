from flask import Flask, request, jsonify
import os
import pymysql
import joblib
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load env
load_dotenv()

# DB config
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": os.getenv("DB_CHARSET")
}

MODEL_FILE = "user_model.pkl"
VEC_FILE = "vectorizer.pkl"

app = Flask(__name__)

def load_training_data():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT input, intent FROM intents")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    if not rows:
        return [], []
    return zip(*rows)

def train_model():
    inputs, intents = load_training_data()
    if not inputs:
        return False
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(inputs)
    model = MultinomialNB()
    model.fit(X_train, intents)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VEC_FILE)
    return True

def predict_intent(text):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VEC_FILE)
    X_test = vectorizer.transform([text])
    proba = model.predict_proba(X_test)[0]
    labels = model.classes_
    best_idx = proba.argmax()
    return labels[best_idx], float(proba[best_idx])

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask AI Server is running!"
@app.route("/train", methods=["POST"])
def api_train():
    if train_model():
        return jsonify({"status": "ok", "message": "Model trained successfully"})
    return jsonify({"status": "error", "message": "No training data"})

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.json
    old_messages = data.get("oldMessages", [])  # mảng 10 tin nhắn mới nhất từ FE

    if not old_messages:
        return jsonify({"error": "Missing oldMessages"}), 400
    
    # Chuyển mỗi tin thành string có format: "role: content"
    formatted_history = []
    for m in old_messages:
        role = m.get("role", "user")  # mặc định user nếu không có
        content = m.get("content", "").strip()
        if content:
            formatted_history.append(f"{role}: {content}")

    # Ghép thành context string
    full_input = "\n".join(formatted_history)

    # Predict intent
    intent, confidence = predict_intent(full_input)

    return jsonify({
        "intent": intent,
        "confidence": confidence,
        "debug": {
            "formattedHistory": formatted_history
        }
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
