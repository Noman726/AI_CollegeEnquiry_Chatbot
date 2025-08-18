from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import random

app = Flask(__name__)

# Download resources (only the first time)
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents = {
    "greeting": {
        "patterns": ["hello", "hi", "good morning", "good evening", "hey"],
        "responses": ["Hello! How can I assist you about the college today?",
                      "Hi there! What would you like to know about the college?"]
    },
    "courses": {
        "patterns": ["what courses are available", "course list", "degree programs", "courses offered"],
        "responses": ["We offer BSc IT, BSc CS, BCOM, and BAF at undergraduate level."]
    },
    "admissions": {
        "patterns": ["how to apply", "admission process", "eligibility", "admissions"],
        "responses": ["The admission process is online through our college website. Eligibility criteria depends on the chosen program."]
    },
    "fees": {
        "patterns": ["what are the fees", "fee structure", "college fees"],
        "responses": ["The annual fee for BSc IT is around ₹45,000. Other courses vary accordingly."]
    },
    "hostel": {
        "patterns": ["is hostel available", "hostel facilities", "accommodation"],
        "responses": ["Yes, the college provides separate hostel facilities for boys and girls."]
    },
    "exams": {
        "patterns": ["exam schedule", "when are exams conducted", "exam pattern"],
        "responses": ["Exams are conducted twice a year – Semester 1 in Nov-Dec and Semester 2 in April-May."]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "take care"],
        "responses": ["Goodbye! Feel free to chat again if you have more questions.", "Take care!"]
    }
}

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in tokens]

words = []
classes = []
documents = []

for intent in intents:
    for pattern in intents[intent]['patterns']:
        w = preprocess(pattern)
        words.extend(w)
        documents.append((w, intent))
    if intent not in classes:
        classes.append(intent)

words = sorted(set(words))
classes = sorted(set(classes))

def bow(sentence, words):
    tokens = preprocess(sentence)
    bag = [0] * len(words)
    for s in tokens:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bag_vector = bow(sentence, words)
    scores = []
    for d in documents:
        _, intent = d
        patterns = intents[intent]['patterns']
        match = 0
        for p in patterns:
            p_bow = bow(p, words)
            score = np.dot(bag_vector, p_bow)
            match = max(match, score)
        scores.append((intent, match))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    if scores[0][1] == 0:
        return "noanswer"
    return scores[0][0]

def chatbot_response(msg):
    intent = predict_class(msg)
    if intent == "noanswer":
        return "I’m sorry, I don’t understand that. Can you please rephrase your question?"
    return random.choice(intents[intent]['responses'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return chatbot_response(user_text)

if __name__ == "__main__":
    app.run(debug=True)
