# N-gram Based Next Word Prediction

## Overview
This project implements a statistical N-gram language model for next-word prediction using the NLTK Gutenberg corpus. The system is based on unigram, bigram, and trigram frequency distributions and employs a backoff strategy to handle data sparsity. The implementation demonstrates classical probabilistic language modeling techniques used in natural language processing.

---

## Objectives
- Build a next-word prediction system using statistical language modeling
- Implement unigram, bigram, and trigram frequency-based models
- Address unseen contexts using a hierarchical backoff strategy
- Enable efficient reuse of trained models through serialization

---

## Dataset
- Corpus: NLTK Gutenberg Corpus
- Content: Public-domain literary texts
- Data Source: NLTK library
- Preprocessing: Tokenization and normalization of text data

---

## Methodology
- Text from all Gutenberg corpus documents is aggregated
- Tokenization is performed using a regular expression to extract alphabetic and apostrophe-containing tokens
- All tokens are converted to lowercase for consistency
- Frequency counts are computed for:
  - Unigrams (single words)
  - Bigrams (consecutive word pairs)
  - Trigrams (consecutive word triples)
- Frequency models are stored using Python data structures

---

## Model Design
- Unigram model captures global word frequencies
- Bigram model captures conditional probabilities based on the previous word
- Trigram model captures conditional probabilities based on the previous two words
- A backoff strategy is applied:
  - Trigram prediction is attempted first
  - Bigram prediction is used if trigram context is unavailable
  - Unigram prediction is used as the final fallback

---

## Prediction Process
- Input text is tokenized using the same preprocessing pipeline
- The last two words of the input are used for trigram prediction
- If trigram context is missing, bigram probabilities are evaluated
- If both trigram and bigram contexts are missing, unigram frequencies are used
- The word with the highest probability is selected as the prediction

---

## Implementation
- The entire system is implemented using Python
- Frequency distributions are stored using `Counter`
- The model logic is encapsulated within an object-oriented class structure
- Model persistence is supported using Python’s pickle module

---

## Model Persistence
- Trained N-gram frequency models can be saved to disk
- Serialized models can be loaded for inference without retraining
- This improves efficiency during repeated executions

---

## Applications
- Next-word prediction
- Text auto-completion
- Educational demonstrations of statistical NLP
- Baseline language modeling systems

---

## Limitations
- Suffers from data sparsity, especially at higher N-gram levels
- Limited contextual window
- No semantic understanding of language
- Performance depends heavily on corpus size and coverage

---

## Future Work
- Incorporation of advanced smoothing techniques such as Kneser-Ney or Witten-Bell
- Expansion of training data beyond the Gutenberg corpus
- Quantitative evaluation using perplexity metrics
- Extension toward hybrid statistical and neural language models

---

## Conclusion
This project demonstrates the effectiveness of classical N-gram language models for next-word prediction. By combining trigram context with a backoff strategy, the system balances contextual accuracy and robustness against sparse data. The implementation highlights the interpretability and efficiency of statistical approaches in natural language processing.
