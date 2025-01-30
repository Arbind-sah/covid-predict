import numpy as np
import streamlit as st
from logic import display_tabs, safety_measures
import pickle
import altair as alt
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier


# Page layout
st.set_page_config(
    page_title="Online COVID-19 Prediction", page_icon="🧊", layout="wide"
)

# sidebar
st.sidebar.title("Online COVID-19 Prediction")

# Navigation
tabs = st.sidebar.radio("Choose a page", ["Home", "Safety Measures"])
display_tabs(tabs)

if tabs == "Home":
    # load the model
    with open("finalized_model.sav", "rb") as file:
        model = pickle.load(file)

    # main page
    st.write("(Fill the symptoms to check COVID-19)")
    st.write("")
    # Input features
    features = [
        "Fever",
        "Tiredness",
        "Dry-Cough",
        "Difficulty-in-Breathing",
        "Sore-Throat",
        "Pains",
        "Nasal-Congestion",
        "Runny-Nose",
        "Diarrhea",
        "None_Experiencing",
        "Severity_Mild",
        "Severity_Moderate",
        "Severity_None",
        "Severity_Severe",
        "Contact_Dont-Know",
        "Contact_No",
        "Contact_Yes",
    ]

    # Create input fields
    input_data = []
    form = st.form(key="covid_form")
    with form:
        for feature in features:
            response = st.radio(
                f"Do you have {feature.replace('_', ' ')}?", ("Yes", "No")
            )
            input_data.append(response)
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        if "" in input_data:
            st.warning("Please fill out all the fields before submitting.")
        else:
            st.write("Checking for COVID-19...")
            st.header("Your COVID-19 result is here:")
            input_data = [1 if response == "Yes" else 0 for response in input_data]
            input_data = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_data)
            st.write(
                f"{'You are suffering from COVID-19 😷. Please consult a doctor immediately' if prediction[0] == 1 else 'You are not suffering from COVID-19 😊. Stay safe and healthy'}"
            )

            # Data visualization

            # Bar chart for features
            feature_counts = np.array(input_data).flatten()
            feature_names = [feature.replace("_", " ") for feature in features]
            feature_data = pd.DataFrame(
                {"Feature": feature_names, "Presence": feature_counts}
            )

            feature_chart = (
                alt.Chart(feature_data)
                .mark_bar()
                .encode(x="Feature", y="Presence", color="Presence:N")
                .properties(title="Presence of Symptoms")
            )

            st.altair_chart(feature_chart, use_container_width=True)

elif tabs == "Safety Measures":

    safety_measures()

    # Measures to handle COVID-19
    st.markdown(
        """
        <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
            <h2 style="color: #ff4d4d;">Steps to Handle COVID-19 Infection</h2>
            <ol style="font-size: 18px; line-height: 1.6; color: black;">
                <li>🧼 <strong>Wash your hands frequently:</strong> Use soap and water for at least 20 seconds.</li>
                <li>📏 <strong>Maintain social distancing:</strong> Keep at least 6 feet distance from others.</li>
                <li>🚫 <strong>Avoid touching eyes, nose, and mouth:</strong> Hands touch many surfaces and can pick up viruses.</li>
                <li>🤧 <strong>Practice respiratory hygiene:</strong> Cover your mouth and nose with your bent elbow or tissue when you cough or sneeze.</li>
                <li>🏥 <strong>Seek medical care early:</strong> If you have fever, cough, and difficulty breathing, seek medical attention.</li>
                <li>📢 <strong>Stay informed:</strong> Follow advice given by your healthcare provider.</li>
                <li>😷 <strong>Wear a mask:</strong> Wear a mask when physical distancing is not possible.</li>
                <li>🏠 <strong>Stay home if unwell:</strong> Stay home and self-isolate even with minor symptoms.</li>
                <li>🧻 <strong>Cover your nose and mouth:</strong> Use your bent elbow or a tissue when you cough or sneeze.</li>
                <li>🧴 <strong>Clean and disinfect:</strong> Clean and disinfect frequently touched objects and surfaces daily.</li>
            </ol>
            <p style="color: #ff4d4d; font-size: 18px; text-align: center;">Stay safe and healthy!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")

# prevention guidelines

st.markdown(
    """
    <style>
    .guidelines {
        border: 2px solid black;
        background-color: #333;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="guidelines">
        <h2>Prevention Guidelines</h2>
        <p>1. Wash your hands frequently.</p>
        <p>2. Maintain social distancing.</p>
        <p>3. Avoid touching eyes, nose and mouth.</p>
        <p>4. Practice respiratory hygiene.</p>
        <p>5. If you have fever, cough and difficulty breathing, seek medical care early.</p>
        <p>6. Stay informed and follow advice given by your healthcare provider.</p>
        <p>7. Wear a mask when physical distancing is not possible.</p>
        <p>8. Stay home if you feel unwell.</p>
        <p>9. Cover your nose and mouth with your bent elbow or a tissue when you cough or sneeze.</p>
        <p>10. Clean and disinfect frequently touched objects and surfaces.</p>
        <p>Stay safe and healthy!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")


# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #000015;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="footer">
    Made with ❤️ by [Arbind Kumar Sah](https://github.com/Arbind-sah)
    """,
    unsafe_allow_html=True,
)


# horizontal line under which copy right is written

st.write(
    '<hr style="border: 0.2px solid white; border-radius: 2px; margin: 10px 0;">',
    unsafe_allow_html=True,
)
st.write(
    '<p style="color: white; text-align: center;">&copy; 2020 All rights reserved</p>',
    unsafe_allow_html=True,
)
st.write(
    '<p style="color: white; text-align: center;">Contact us: info@example.com</p>',
    unsafe_allow_html=True,
)
st.write(
    '<p style="color: white; text-align: center;">Follow us on <a href="https://twitter.com/example" style="color: #1DA1F2;">Twitter</a> and <a href="https://facebook.com/example" style="color: #4267B2;">Facebook</a></p>',
    unsafe_allow_html=True,
)
