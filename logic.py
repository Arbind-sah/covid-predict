import streamlit as st


def display_tabs(tabs):
    if tabs == "Home":
        st.title("Online COVID-19 Prediction")
        st.markdown("<hr style='border: 1px solid green;'>", unsafe_allow_html=True)
        st.write(
            "This is a simple web app to predict whether you have COVID-19 based on your symptoms."
        )


def safety_measures():
    st.title("Safety Measures:")
    st.write(
        "Here are some safety measures you should follow to protect yourself from COVID-19:"
    )
