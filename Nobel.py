
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


st.write(''' # NOBEL PRIZES CATEGORY PREDICTION ''')
st.image("Nobelphoto.webp", caption="It was created by the Swedish inventor Alfred Nobel through his will in 1895.")

st.header('Text')

def user_input_features():
    # Entrada
    texto = st.text_input("Enter the text to evaluate:")

    user_input_data = {'Text': texto}

    features = pd.DataFrame(user_input_data, index=[0])

    return features

df = user_input_features()

nobel = pd.read_csv('nobel_unido.csv', encoding='latin-1')
X = nobel.Motivation
y = nobel.Category

vect = CountVectorizer()
X_dtm = vect.fit_transform(X)

nb = MultinomialNB()
nb.fit(X_dtm, y)

df_dtm = vect.transform(df['Text'])
prediction = nb.predict(df_dtm)

#{'physics':0, 'medicine':1, 'peace':2, 'literature':3, 'chemistry':4, 'economics':5}
#'Physics', 'Medicine', 'Peace', 'Literature', 'Chemistry', 'Economics'
st.subheader('Prediction')
if prediction == 0:
    st.write('Physics')
elif prediction == 1:
    st.write('Medicine')
elif prediction == 2:
    st.write('Peace')
elif prediction == 3:
    st.write('Literature')
elif prediction == 4:
    st.write('Chemistry')
elif prediction == 5:
    st.write('Economics')
else:
    st.write('No Prediction')
