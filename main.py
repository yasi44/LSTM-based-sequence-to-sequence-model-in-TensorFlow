import numpy as np
import tensorflow as tf
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample English-French dataset
english_sentences = ["hello how are you", "what is your name", "where is the bathroom"]
french_sentences = ["bonjour comment ça va", "quel est ton nom", "où sont les toilettes"]

# Tokenization & Padding
eng_tokenizer = Tokenizer()
french_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_sentences)
french_tokenizer.fit_on_texts(french_sentences)

eng_vocab_size = len(eng_tokenizer.word_index) + 1
french_vocab_size = len(french_tokenizer.word_index) + 1

eng_sequences = pad_sequences(eng_tokenizer.texts_to_sequences(english_sentences), padding='post')
french_sequences = pad_sequences(french_tokenizer.texts_to_sequences(french_sentences), padding='post')

# Model Parameters
embedding_dim = 64
units = 128
max_len = eng_sequences.shape[1]

# Encoder
encoder_inputs = Input(shape=(max_len,))
enc_emb = Embedding(eng_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(units, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_len,))
dec_emb_layer = Embedding(french_vocab_size, embedding_dim, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# Attention
attention = Attention()
context_vector = attention([decoder_outputs, encoder_outputs])
decoder_combined_context = tf.concat([decoder_outputs, context_vector], axis=-1)
decoder_dense = Dense(french_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# Model Compilation
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Convert target sequences to one-hot format for training
def preprocess_target_sequences(sequences, vocab_size):
    one_hot_targets = np.zeros((len(sequences), max_len, vocab_size), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for t, word_idx in enumerate(seq):
            if word_idx > 0:
                one_hot_targets[i, t, word_idx] = 1.0
    return one_hot_targets

french_targets = preprocess_target_sequences(french_sequences, french_vocab_size)

# Training
model.fit([eng_sequences, french_sequences], french_targets, epochs=100, batch_size=2)

# Inference Function
def translate_sentence(input_text):
    input_seq = pad_sequences(eng_tokenizer.texts_to_sequences([input_text]), maxlen=max_len, padding='post')
    states_value = model.predict([input_seq, np.zeros((1, max_len))])
    predicted_seq = np.argmax(states_value, axis=-1)
    translated_text = ' '.join(french_tokenizer.index_word[idx] for idx in predicted_seq[0] if idx > 0)
    return translated_text

# Testing Translation
print(translate_sentence("Hi. May name is Yasaman. What is your name?"))
