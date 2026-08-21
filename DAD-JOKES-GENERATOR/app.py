import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model + tokenizer once
model = load_model('dad_jokes_model.h5')
with open('tokenizer.pkl', 'rb') as f:
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
