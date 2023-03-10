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

text = extract_text(r'/Users/ronaldsee/Documents/GitHub/NLPeople/Grab Q2 22_OCR.pdf')
#print(text)


nlp_spacy = spacy.load("en_core_web_sm")
doc = nlp_spacy(text)

sentences_annual_report = []
for sent in doc.sents:
    if len(sent.text.split()) > 6:
        print(sent.text)
        sentences_annual_report.append(sent.text)
    
#print(sentences_annual_report)


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

for x,y,z in zip(sentiment.label,sentiment.docs,sentiment.score):
    if x == "Positive":
        Positive += 1
        print(y)
        print()
       
        
    if x == "Negative":
        Negative += 1
        
        
    if x == "Neutral":
        Neutral += 1

print("Positive :" + str(Positive))
print("Negative :" + str(Negative))
print("Neutral :" + str(Neutral))