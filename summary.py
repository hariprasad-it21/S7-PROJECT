from googletrans import Translator
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq

def get_stopwords(language):
    if language == 'hi':
        # Load Hindi stopwords from the custom file
        try:
            with open('hindi_words.txt', 'r', encoding='utf-8') as file:
                stopwords_list = [line.strip() for line in file]
            return set(stopwords_list)
        except Exception as e:
            print(f"Error loading stopwords: {e}")
            return set()
    else:
        return set(stopwords.words('english'))

def summarize_and_translate(text, summary_percentage=0.45, target_language='tamil'):
    # Step 1: Summarize the text in English
    try:
        english_summary = summarize_text(text, summary_percentage)
    except Exception as e:
        print(f"Error during summarization: {e}")
        english_summary = "Summary not available"

    # Step 2: Translate the original text to the target language
    translator = Translator()
    try:
        translated_text = translator.translate(text, dest=target_language).text
    except Exception as e:
        print(f"Error during translation: {e}")
        translated_text = "Translation error"

    # Step 3: Translate the English summary to the target language
    try:
        translated_summary = translator.translate(english_summary, dest=target_language).text
    except Exception as e:
        print(f"Error during summary translation: {e}")
        translated_summary = "Summary translation not available"

    return translated_text, translated_summary

def summarize_text(text, summary_percentage=0.45):
    # Tokenize the text
    stop_words = get_stopwords('en')
    sentences = sent_tokenize(text)
    word_tokens = word_tokenize(text.lower())
    filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]

    # Calculate TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A.flatten()

    # Get the most important sentences
    summary_length = int(len(sentences) * summary_percentage)
    important_sentences_indices = heapq.nlargest(summary_length, range(len(sentence_scores)), sentence_scores.take)
    summary = ' '.join([sentences[i] for i in sorted(important_sentences_indices)])

    return summary
