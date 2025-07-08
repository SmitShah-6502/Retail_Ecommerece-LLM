import streamlit as st
from deep_translator import GoogleTranslator
from transformers import pipeline
from gtts import gTTS
import speech_recognition as sr
import io

# Load model
chatbot = pipeline("text-generation", model="distilgpt2")

# Language codes
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu"
}

# Translate to English
def translate_to_english(text, source_lang_code):
    if source_lang_code != "en":
        translated = GoogleTranslator(source=source_lang_code, target="en").translate(text)
        return translated
    return text

# Translate from English to user language
def translate_back(text, target_lang_code):
    if target_lang_code != "en":
        translated = GoogleTranslator(source="en", target=target_lang_code).translate(text)
        return translated
    return text

# Generate detailed response
def handle_query_detailed_en(query):
    query = query.lower()
    if "order" in query or "parcel" in query:
        return (
            "Thank you for your query regarding your order. Your package is currently being processed and will "
            "be dispatched soon. You can track the order status from your order history. Estimated delivery is within 3–5 business days."
        )
    elif "return" in query:
        return (
            "Returns are hassle-free! You can return your order within 7 days of delivery by going to the Returns section. "
            "Make sure the product is unused and in its original packaging."
        )
    elif "availability" in query or "stock" in query:
        return (
            "The product you're looking for is currently available. However, due to high demand, availability may change. "
            "Please place your order soon to ensure you receive the item on time."
        )
    elif "delivery" in query:
        return (
            "We offer standard delivery within 3–5 business days. You will receive a tracking link once your order is shipped. "
            "Thank you for shopping with us!"
        )
    else:
        response = chatbot(query, max_length=100, do_sample=True)[0]['generated_text']
        return response.strip()

# Voice input using microphone
def recognize_speech():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            with st.spinner("Listening..."):
                audio = r.listen(source, phrase_time_limit=5)
            try:
                return r.recognize_google(audio)
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand your voice."
            except sr.RequestError:
                return "Speech recognition service failed. Try again."
    except AttributeError:
        return "PyAudio is not installed or microphone not accessible."

# Text-to-speech output using Streamlit audio widget
def speak_text(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    st.audio(mp3_fp, format="audio/mp3")

# --- Added: Initialize session state variables ---
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

if 'final_response' not in st.session_state:
    st.session_state.final_response = ""

# Streamlit UI
st.set_page_config(page_title="Multilingual Order Assistant")
st.title("🛍️ Multilingual Order Chat Assistant")

language = st.selectbox("🌐 Choose your language:", list(language_codes.keys()))
lang_code = language_codes[language]

col1, col2 = st.columns(2)

# Use session state for input box to keep value
user_input = col1.text_input("💬 Type your query here:", value=st.session_state.user_input)

use_voice = col2.button("🎤 Speak your query")

if use_voice:
    recognized_text = recognize_speech()
    st.session_state.user_input = recognized_text  # Save recognized text to session state
    user_input = recognized_text
    st.write(f"🗣️ You said: {recognized_text}")

if user_input:
    # Save user input to session state
    st.session_state.user_input = user_input

    # Translate to English
    query_en = translate_to_english(user_input, lang_code)

    # Get detailed response
    response_en = handle_query_detailed_en(query_en)

    # Translate back to user language
    final_response = translate_back(response_en, lang_code)

    # Save response to session state
    st.session_state.final_response = final_response

# Show the response if available
if st.session_state.final_response:
    st.markdown("🧠 **Assistant Response:**")
    st.success(st.session_state.final_response)

    # Button to speak response
    if st.button("🔊 Speak Response"):
        speak_text(st.session_state.final_response, lang_code)
