import pdf2image
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

import spacy
import pathlib

from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

# pdf_file = ""

# def pdf_to_img(pdf_file):
#     return pdf2image.convert_from_path(pdf_file)


# def ocr_core(file):
#     text = pytesseract.image_to_string(file)
#     return text


# def print_pages(pdf_file):
#     images = pdf_to_img(pdf_file)
#     counter = 1
#     for pg, img in enumerate(images):
#         filename = "Grab Q2" + str(counter)
#         with open(file=filename +'.txt', mode = 'w') as f:
#             f.write(ocr_core(img))
#         counter += 1


# print_pages('Grab Q2 22.pdf')

nlp_spacy = spacy.load("en_core_web_sm")
# doc = nlp_spacy("Grab Q22.txt")
file_name = "Grab Q28.txt"
introduction_doc = nlp_spacy(pathlib.Path(file_name).read_text(encoding="utf-8"))
print ([token.text for token in introduction_doc])

preprocess_doc = []

for token in introduction_doc:
    if token.text == "\n\n":
        preprocess_doc.append(".")
    elif token.text == "\n":
        preprocess_doc.append(" ")
    else: 
        preprocess_doc.append(token.text)

sentences_annual_report = []
sentence = ""
for token in preprocess_doc:
    if token == ".":
        sentence += "."
        sentences_annual_report.append(sentence)
        sentence = ""
    else: 
        sentence += token
        sentence += " "

print (sentences_annual_report)
