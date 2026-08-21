import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Build absolute paths relative to this script's own location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'dad_jokes_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')

# Load model + tokenizer once
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    token = pickle.load(f)

st.title("🤣 Dad Jokes Generator")
seed = st.text_input("Start your joke:", "what do")
num_words = st.slider("How many words to generate?", 1, 15, 10)

if st.button("Generate"):
    text = seed
    for i in range(num_words):
        token_text = token.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=51, padding='pre')
        pos = np.argmax(model.predict(padded_token_text, verbose=0))
        for word, index in token.word_index.items():
            if index == pos:
                text = text + " " + word
                break
    st.write(text)
