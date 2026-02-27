import streamlit as st
from model import predict

st.title("Cyberbullying Detection App")

st.write("This app detects whether a text contains cyberbullying content.")

user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict(user_input)
        st.success(f"Prediction: {result}")