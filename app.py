import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from nrclex import NRCLex

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div.stButton > button {
        background-color: blue;
        color: white;
    }

    div.stButton > button:hover {
        background-color: #e6ffe6;   /* Darker blue on hover */
        color: black;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("Emotion Detection System")

text = st.text_area("Enter text:")

if st.button("Detect Emotion"):
    if text.strip() != "":
        # Extract features
        features = vectorizer.transform([text])

        # Prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Labels
        emotions = model.classes_

        # Display the predicted emotion
        # st.success(f"**Predicted Emotion:** {emotions[prediction]}")

        # Emotional categories and counts
        categories = {
            0: "Negative/Unpleasant Emotions (Label 0)",
            1: "Happy/Positive/Joyful Emotions (Label 1)",
            2: "Neutral/Other/Ambivalent Emotions (Label 2)",
            3: "Anger/Irritation/Annoyance (Label 3)",
            4: "Fear/Anxiety/Nervousness (Label 4)",
            5: "Surprise/Confusion (Label 5)"
        }

        # Display category message based on predicted emotion
        if prediction in categories:
            st.info(f"**Emotion Details:** {categories[prediction]}")
        
        # DataFrame for bar chart
        df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probabilities
        })

        df["Emotion"] = df["Emotion"].replace({
            0: "Sad",
            1: "Happy",
            2: "Neutral",
            3: "Angry",
            4: "Fear",
            5: "Surprise"
        })

        colors = ["#4CAF50", "#2196F3", "#F44336", "#FF9800", "#9E9E9E", "#9C27B0"]  # 6 colors

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(df["Emotion"], df["Probability"], color=colors[:len(df)])

        # Title
        ax.set_title("Emotion Detection Probabilities", fontsize=14, fontweight="bold")
        ax.set_xlabel("Emotions", fontsize=12 )
        ax.set_ylabel("Probability", fontsize=12, fontweight="bold")

     

        # Add probability values on top of bars
        # for bar, prob in zip(bars, df["Probability"]):
        #     ax.text(bar.get_x() + bar.get_width()/2, prob + 0.01,
        #             f"{prob:.2f}", ha='center', va='bottom', fontsize=10)

        ax.set_ylim(0, 1)  # probability scale
        st.pyplot(fig)

        # Show bar chart
        # st.bar_chart(df.set_index("Emotion"))
    else:
        st.warning("⚠️ Please enter some text to analyze.")



