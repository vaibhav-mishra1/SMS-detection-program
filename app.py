import nltk
import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords

stop_words = stopwords.words("English")
def transform_text(text):
    useful_text = []
    text = text.lower()
    removedSC = list()
    for i in text:
        if i.isalnum() or i.isspace():
            removedSC.append(i)
    text = "".join(removedSC)

    words = nltk.word_tokenize(text)
    for word in words:
        if word not in stop_words:
            useful_text.append(word)
    return " ".join(useful_text)

tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("SMS Spam Detection Model")
st.write("*This is a Machine Learning application to classify SMS as spam or ham*")
st.write("*Made by Vaibhav Mishra*")

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):

    #preprocessing the input sms using tokenization
    transformed_sms = transform_text(input_sms)
    #vectorize the preprocessed sms
    vector_input = tk.transform([transformed_sms])
    #predicting the result
    result = model.predict(vector_input)[0]
    #displaying result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")