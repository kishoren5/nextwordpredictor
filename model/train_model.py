import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from preprocessor import preprocess_text, create_sequences # Import from your utility

# --- 1. Load and Preprocess Data ---
with open('data/corpus.txt', 'r') as f:
    corpus = f.read().splitlines()

# Preprocess each line
processed_corpus = [preprocess_text(line) for line in corpus]

# --- 2. Tokenization ---
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(processed_corpus)
total_words = len(tokenizer.word_index) + 1 # +1 for OOV token / index 0

# --- 3. Create N-gram Sequences ---
# Determine max_sequence_len from your data or set a reasonable value
# For simplicity, let's find the max length after tokenization
max_sequence_len = max([len(tokenizer.texts_to_sequences([line])[0]) for line in processed_corpus])
if max_sequence_len < 10: # Ensure a minimum reasonable length
    max_sequence_len = 10
print(f"Calculated Max Sequence Length: {max_sequence_len}")


input_sequences = []
for line in processed_corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
# Pad to max_sequence_len, which should be the input length of the model
# The label will be the last word, so input_len = max_sequence_len - 1
padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and labels
# All but the last word for predictors, last word for label
xs, labels = padded_sequences[:,:-1], padded_sequences[:,-1]
ys = to_categorical(labels, num_classes=total_words)

# --- 4. Build Model ---
embedding_dim = 100

model = Sequential()
model.add(Embedding(total_words, embedding_dim, input_length=max_sequence_len-1)) # input_length for predictors
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# --- 5. Train Model ---
history = model.fit(xs, ys, epochs=100, verbose=1) # Adjust epochs as needed

# --- 6. Save Model and Tokenizer ---
model.save('model/next_word_model.h5')
with open('model/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer saved!")

# --- Function to test prediction (Optional, for verification) ---
def predict_next_words(seed_text, n_words, model, tokenizer, max_sequence_len):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Ensure token_list does not exceed MAX_SEQUENCE_LEN for slicing
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]

        padded_token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(padded_token_list, verbose=0)[0]
        predicted_word_index = np.argmax(predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Test the trained model
# print(predict_next_words("deep learning is", 5, model, tokenizer, max_sequence_len))