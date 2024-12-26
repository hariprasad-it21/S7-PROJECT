import cv2
import os
import PyPDF2
from langdetect import detect
import easyocr
import fitz  
import io
import time
from deep_translator import GoogleTranslator

class ImageOCR:
    def __init__(self, file_path):
        self.file_path = file_path

    def convert_to_jpeg(self):
        _, file_extension = os.path.splitext(self.file_path)
        if file_extension.lower() not in ['.jpg', '.jpeg']:
            img = cv2.imread(self.file_path)
            jpeg_image_path = os.path.splitext(self.file_path)[0] + '.jpeg'
            cv2.imwrite(jpeg_image_path, img)
            self.file_path = jpeg_image_path

    def process_pdf_and_summarize(self):
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf()
        print("Recognized Text from PDF:")
        print(pdf_text if pdf_text else "No text found in PDF.")

        # Extract and process images from PDF
        image_text = self.extract_and_process_images_from_pdf()
        print("Recognized Text from Images:")
        print(image_text if image_text else "No text found in images.")

        combined_text = pdf_text + " " + image_text

        if combined_text.strip():
            original_language = self.detect_language(combined_text)
            print("Original Language:", original_language)

            target_language = self.select_target_language()
            translated_text = self.translate_text(combined_text, target_language)
            print("Translated Text from PDF and Images:")
            print(translated_text)
        else:
            print("No text could be extracted from the PDF or images.")

    def extract_text_from_pdf(self):
        pdf_text = ""
        try:
            with open(self.file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text + " "
                    else:
                        print(f"No text found on page {page_num + 1}")
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return pdf_text.strip()

    def translate_text(self, text, target_language='en'):
        retries = 3
        for attempt in range(retries):
            try:
                translator = GoogleTranslator(source='auto', target=target_language)
                translated = translator.translate(text)
                if translated:
                    return translated
            except Exception as e:
                print(f"Translation attempt {attempt + 1} failed: {e}")
                time.sleep(2)  
        return "Translation error after retries"

    def select_target_language(self):
        languages = {
            '1': 'hi', '2': 'bn', '3': 'te', '4': 'mr', '5': 'ta',
            '6': 'gu', '7': 'kn', '8': 'ml', '9': 'pa', '10': 'ur',
            '11': 'as', '12': 'or', '13': 'sa', '14': 'en'
        }
        print("Select a target language for translation:")
        for key, lang in languages.items():
            print(f"{key}. {lang.capitalize()}")
        choice = input("Enter the number of your choice: ")
        return languages.get(choice, 'en')

    def detect_language(self, text):
        try:
            detected_lang = detect(text)
        except Exception as e:
            print(f"Error during language detection: {e}")
            detected_lang = "unknown"
        return detected_lang

    def perform_easyocr(self, image_bytes):
        try:
            reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if GPU is available and compatible
            result = reader.readtext(image_bytes)
            return " ".join([item[1] for item in result])
        except Exception as e:
            print(f"Error during OCR: {e}")
            return ""

    def extract_and_process_images_from_pdf(self):
        text_from_images = ""
        try:
            with fitz.open(self.file_path) as doc:
                for page_num, page in enumerate(doc):
                    image_list = page.get_images(full=True)
                    for image_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_text = self.perform_easyocr(io.BytesIO(image_bytes))
                        text_from_images += image_text + " "
                        print(f"Text from image on page {page_num + 1}, image {image_index + 1}: {image_text}")
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
        return text_from_images.strip()

if __name__ == "__main__":
    file_path = input("Enter the path of the PDF or image: ")
    ocr_processor = ImageOCR(file_path)

    if file_path.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
        ocr_processor.process_pdf_and_summarize()
    else:
        print("Invalid file format. Supported formats: PDF, JPG, JPEG, PNG.")
