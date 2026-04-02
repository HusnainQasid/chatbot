# ==============================
# FAQ CHATBOT (NLP BASED)
# ==============================

import nltk
import string
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# डाउनलोड NLTK data (first time only)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

# ==============================
# FAQ DATASET
# ==============================

faqs = [
    {"question": "What is your name?", "answer": "I am a chatbot."},
    {"question": "How are you?", "answer": "I am fine, thank you!"},
    {"question": "What is AI?", "answer": "AI stands for Artificial Intelligence."},
    {"question": "What is machine learning?", "answer": "Machine learning is a subset of AI."},
    {"question": "How can I contact support?", "answer": "You can contact support via email support@example.com."},
    {"question": "What services do you provide?", "answer": "We provide AI and software solutions."},
    {"question": "Where are you located?", "answer": "We operate online globally."}
]

# ==============================
# TEXT PREPROCESSING
# ==============================

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    tokens = [
        word for word in tokens
        if word not in stopwords.words('english')
        and word not in string.punctuation
    ]
    
    return " ".join(tokens)

# ==============================
# VECTORIZE QUESTIONS
# ==============================

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# ==============================
# CHATBOT RESPONSE FUNCTION
# ==============================

def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    
    similarity = cosine_similarity(user_vector, X)
    
    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]
    
    if best_score < 0.3:
        return "Sorry, I don't understand your question."
    
    return answers[best_match_index]

# ==============================
# FLASK WEB APP
# ==============================

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h2>FAQ Chatbot 🤖</h2>
    <p>Use POST /chat with JSON {"message": "your question"}</p>
    """

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    
    response = get_response(user_input)
    
    return jsonify({"response": response})

# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    print("Chatbot running on http://127.0.0.1:5000")
    app.run(debug=True)