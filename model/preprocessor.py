import string
from tensorflow.keras.preprocessing.text import Tokenizer

def preprocess_text(text):
    """
    Cleans and tokenizes text.
    - Converts to lowercase
    - Removes punctuation
    """
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def create_sequences(tokenizer, corpus, max_sequence_len):
    """
    Creates sequences from a corpus for training.
    """
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            if len(n_gram_sequence) <= max_sequence_len: # Ensure sequences fit model input
                 input_sequences.append(n_gram_sequence)
    return input_sequences