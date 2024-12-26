import os
import io
import nltk
import streamlit as st
from PIL import Image
import PyPDF2
import easyocr
from googletrans import Translator
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
import hashlib

from stt import record_text
from summary import summarize_and_translate

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def download_nltk_data():
    nltk_data_path = 'C:\\Users\\poova\\nltk_data'  # Adjust this path if necessary
    if not os.path.exists(nltk_data_path):
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('wordnet', download_dir=nltk_data_path)
        nltk.download('omw-1.4', download_dir=nltk_data_path)
    nltk.data.path.append(nltk_data_path)

download_nltk_data()

translator = Translator()

st.title('English to Indian Regional Language Translation and Summarization')

uploaded_file = st.sidebar.file_uploader("Choose a file (PDF, JPG, JPEG, PNG)", type=['pdf', 'jpg', 'jpeg', 'png'])

lang_options = {
    'Hindi': 'hi', 'Bengali': 'bn', 'Telugu': 'te', 'Marathi': 'mr',
    'Tamil': 'ta', 'Gujarati': 'gu', 'Kannada': 'kn', 'Malayalam': 'ml',
    'Punjabi': 'pa', 'Urdu': 'ur', 'Odia': 'or', 'Assamese': 'as',
    'Sanskrit': 'sa', 'English': 'en',
}

accuracy_scores = {
    'Hindi': 90, 'Bengali': 87, 'Telugu': 75, 'Marathi': 85,
    'Tamil': 85, 'Gujarati': 80, 'Kannada': 75, 'Malayalam': 75,
    'Punjabi': 80, 'Urdu': 90, 'Odia': 70, 'Assamese': 65,
    'Sanskrit': 60, 'English': 95,
}

target_lang = st.sidebar.selectbox("Select target language for translation:", options=list(lang_options.keys()))

text = ""

if 'history' not in st.session_state:
    st.session_state.history = []

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_image(uploaded_file):
    text = ""
    try:
        image = Image.open(uploaded_file)
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
        
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(img_byte_array)
        text = " ".join([entry[1] for entry in result])
    except Exception as e:
        st.error(f"Error during OCR: {e}")
    return text

def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    else:
        return extract_text_from_image(uploaded_file)

def generate_title(text):
    title = text[:50]
    if len(title) < 50: 
        title = title + "..."
    return title

if uploaded_file is not None:
    text = process_uploaded_file(uploaded_file)

if text:
    try:
        detected_language = detect(text)
        st.subheader("Detected Language:")
        st.write(detected_language)

        st.subheader("Original Text:")
        st.write(text)

        translated_text = translator.translate(text, dest=lang_options[target_lang]).text
        st.subheader(f"Translated Text ({target_lang}):")
        st.write(translated_text)

        accuracy = accuracy_scores.get(target_lang, 70)  # Default accuracy if not found
        st.write(f"Estimated Translation Accuracy {target_lang} is: {accuracy}%")

        translation_title = generate_title(text)
        translation_data = {
            'title': translation_title,
            'original_text': text,
            'translated_text': translated_text,
            'target_lang': target_lang
        }
        st.session_state.history.append(translation_data)

        if st.button("Summarize"):
            try:
                translated_summary = summarize_and_translate(text, target_language=lang_options[target_lang])
                if translated_summary[1]:
                    st.subheader(f"Summary ({target_lang}):")
                    st.write(translated_summary[1])
                else:
                    st.write("Summary translation not available.")
            except Exception as e:
                st.error(f"Error during summarization: {e}")

    except Exception as e:
        st.error(f"Error during translation or summarization: {e}")

else:
    st.write("No text could be extracted.")

if st.session_state.history:
    st.sidebar.subheader("Translation History")

    for idx, entry in enumerate(st.session_state.history):
        # Generate a unique key by combining index and title
        button_key = f"view_{idx}_{hashlib.md5(entry['title'].encode()).hexdigest()[:8]}"  
        if st.sidebar.button(f"View {entry['title']}", key=button_key):
            st.subheader(entry['title'])
            st.write("Original Text:")
            st.write(entry['original_text'])
            st.write(f"Translated Text ({entry['target_lang']}):")
            st.write(entry['translated_text'])

