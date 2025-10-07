import os
import pickle
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
sys.path.insert(1,r'C:\Users\kisho\OneDrive\Desktop\webapp\model')
from preprocessor import preprocess_text, create_sequences # Custom preprocessor


app = Flask(__name__)

# --- Load Model and Tokenizer ---
try:
    model_path = os.path.join('model', 'next_word_model.h5')
    tokenizer_path = os.path.join('model', 'tokenizer.pkl')

    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    MAX_SEQUENCE_LEN = model.input_shape[1] # Get sequence length from model input
    print(f"Model and tokenizer loaded successfully. Max sequence length: {MAX_SEQUENCE_LEN}")

except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    model = None
    tokenizer = None
    MAX_SEQUENCE_LEN = 50 # Default or fallback

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model or tokenizer not loaded. Please check server logs.'}), 500

    data = request.json
    seed_text = data.get('text', '')

    if not seed_text:
        return jsonify({'prediction': []})

    # Preprocess the input text
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    
    # Pad the sequence to the fixed length
    # Ensure token_list does not exceed MAX_SEQUENCE_LEN for slicing
    if len(token_list) >= MAX_SEQUENCE_LEN:
        token_list = token_list[-(MAX_SEQUENCE_LEN - 1):] # Take last N-1 words to predict next word
    
    padded_token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN-1, padding='pre')

    # Get predictions
    predicted_probs = model.predict(padded_token_list, verbose=0)[0]
    
    # Get the indices of the top 3 probable words
    top_indices = predicted_probs.argsort()[-3:][::-1] # Top 3, descending probability

    predicted_words = []
    for word, index in tokenizer.word_index.items():
        if index in top_indices:
            predicted_words.append(word)

    # Sort predicted_words based on their probabilities (optional, for consistency)
    # This requires matching indices back to probabilities, which is a bit more complex.
    # For now, we just return the words found.

    # Alternatively, map indices back to words directly and sort by probability:
    sorted_predictions = []
    for i in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == i:
                sorted_predictions.append(word)
                break
    
    return jsonify({'prediction': sorted_predictions})

if __name__ == '__main__':
    
    app.run(debug=True)