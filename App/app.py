import streamlit as st
from autocorrect import Speller
import matplotlib.pyplot as plt
from transformers import pipeline

classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

# Page configuration
st.set_page_config(
    page_title="Sentify",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Function to perform Sentiment Analysis
def analyze_sentiment(text):

    result = classifier(text)[0]
    sentiment = result['label']
    polarity = result['score']
    
    
    if sentiment > '3 stars':
        sentiment = "positive"
    elif sentiment < '3 stars':
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return sentiment, polarity

# Autocorrect function
def autocorrect_text(text):
    spell = Speller(lang='en')
    corrected_text = spell(text)
    return corrected_text

# Streamlit User Interface
st.title("Sentify - Sentiment Analysis App")
st.write("Sentify is a Sentiment Analysis model that determines whether a given text has a positive, negative, or neutral sentiment using natural language processing techniques.")

# User Input
user_text = st.text_input("Share how you feel: ")

# Autocorrect option
autocorrect_option = st.checkbox("Enable Autocorrect")

if autocorrect_option and user_text:
    user_text = autocorrect_text(user_text)

# Button to analyze
if st.button("Analyze"):
    if user_text:
        # Performs sentiment analysis on user's input
        sentiment, polarity = analyze_sentiment(user_text)

        # Displays sentiment and polarity score
        st.write("Sentiment:", sentiment)
        st.write("Polarity Score:", polarity)

        if sentiment == "positive":
            st.write("This is a positive statement!")
        elif sentiment == "negative":
            st.write("This is a negative statement.")
        else:
            st.write("The sentiment is neutral.")

        # Visualization: Bar chart of sentiment distribution
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        sentiment_counts[sentiment.capitalize()] = 1


        # Displays the bar chart
        st.header("Sentiment Visualization")
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.keys(), sentiment_counts.values())
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

link = 'Created by [Ashish Dabral](https://www.linkedin.com/in/ashish-dabral-6428ba195/)'
st.markdown(link, unsafe_allow_html=True)