import numpy as np
import pickle
import dill
import pandas as pd
import streamlit as st
import weakref
import joblib

from PIL import Image

pickle_in = open("model.pkl", "rb")
model1 = joblib.load(pickle_in)

def welcome():
    return "Welcome all"
    
def predict_values(arr):   
    prediction=model1.predict([[arr]])
    print(prediction)
    return prediction
    
def main():
    st.title("LSTM predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit LSTM ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    arr = st.text_input("Array: ","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_values(arr)
    st.success('The output is {}'.format(result))
   
    
if __name__ == '__main__':
    main()