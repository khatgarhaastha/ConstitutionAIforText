# backend/app.py

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)  # Use your original code structure to create OpenAI client

app = Flask(__name__)

# Original function with minimal changes to fit within Flask
def classify_description_with_rules(rules, input_text):
    prompt = f"Please label the following description according to these rules. If it violates the rules, label as 1; otherwise, label as 0:\n\nRules: {rules}\n\nDescription: {input_text}\n\nLabel:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to classify text according to provided rules."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1,
            temperature=0
        )
        label = response.choices[0].message.content.strip()
        return "Flagged" if label == "1" else "Not Flagged"
    except Exception as e:
        return f"Error: {e}"

# API endpoint to classify text based on rules
@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    rules = data.get('rules')
    input_text = data.get('input_text')
    result = classify_description_with_rules(rules, input_text)
    return jsonify({"result": result})

# Serve the frontend HTML file
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)