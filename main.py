import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import  load_model

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}


model=load_model('simple_rnn_imdb.h5')
model.summary()

# helper function
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])
#  function to pre precoess the user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


# prediction function



 ## stream lit
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as postive or negative review.')

user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)

    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else "Negative"

    st.write(f'sentiment : {sentiment}')
    st.write(f'Prediction Score :{prediction[0][0]}')
else:
    st.write("please enter a movie review.")
    st.write("Example review : This movie is fantastic!The acting was great and the plot was thrilling.")


# streamlit run app.py
# > conda activate tf2.17