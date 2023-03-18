import pytesseract
from pdf2image import convert_from_path
import io
import os
from PyPDF2 import PdfFileReader
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_colwidth=-1
pd.options.display.min_rows=100
import re
from rank_bm25 import BM25Okapi
import string 
from sklearn.feature_extraction import _stop_words
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

import pdfminer
from pdfminer.high_level import extract_text
from termcolor import colored
import spacy
import matplotlib.pyplot as plt
import os
import torch
import pathlib
import nltk




# convert PDF to image
pages = convert_from_path(r'C:/Users/arkgn/Desktop/public/grab/Grab Q2 22.pdf', 300)


pytesseract.pytesseract.tesseract_cmd = (
		r"C:\Program Files\Tesseract-OCR\tesseract.exe"
	)

# extract text from image
text = ''
for page in pages:
    text += pytesseract.image_to_string(page)




# extract metadata from PDF
with open(r'C:/Users/arkgn/Desktop/public/grab/Grab Q2 22.pdf', 'rb') as f:
    pdf = PdfFileReader(f)
    info = pdf.getDocumentInfo()
    author = info.author
    title = info.title
    num_pages = pdf.getNumPages()

print('Author:', author)
print('Title:', title)
print('Number of pages:', num_pages)
print('Text:', text)


nlp_spacy = spacy.load("en_core_web_sm")
doc = nlp_spacy(text)

sentences_annual_report = []
for sent in doc.sents:
    #print (sent)
    if len(sent) > 6:
        new_lane = sent.text.split(".")
        for x in new_lane:
            if len(x.strip()) > 2:
                sentences_annual_report.append(x)
                print(x)
         
sentences_annual_report2 = sentences_annual_report

sentences_annual_report = []

# Download the words corpus
nltk.download('words')

for string in sentences_annual_report2:
    
    # Split the string into words
      words = string.split()

    # Filter out non-dictionary words
      dictionary_words = [word for word in words if word.lower() in nltk.corpus.words.words()]

    # Join the remaining words back into a string
      filtered_string = ' '.join(dictionary_words)
                
    #clean sentences
      if len(filtered_string) > 2:
          sentences_annual_report.append(filtered_string)
            

        
     
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

results = nlp(sentences_annual_report)



sentiment = pd.DataFrame({"docs": sentences_annual_report,
                          "label": [r["label"] for r in results],
                          "score":[r["score"] for r in results],
                          "docs": sentences_annual_report})




Positive = 0
Negative = 0
Neutral = 0

#print(sentiment.label)
    
    #if x.label == "Positive":
    #    Positive += 1
    
#print("Positive: " + Positive)


sentence_list=[]
label_list=[]

for x, y in zip(sentences_annual_report, sentiment.label):
    sentence_list.append(x)
    label_list.append(y)
    if y == "Positive":
        Positive += 1

        
    if y == "Negative":
        Negative += 1
        
        
    if y == "Neutral":
        Neutral += 1
    
sentiment = pd.DataFrame({"sentence": sentence_list, "label":  label_list })



print("Positive :" + str(Positive))
print("Negative :" + str(Negative))
print("Neutral :" + str(Neutral))


sentiment.to_excel("output.xlsx")
    
#print(sentences_annual_report)





