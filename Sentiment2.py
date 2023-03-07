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

text = extract_text('input.pdf')
print(text)


nlp_spacy = spacy.load("en_core_web_sm")
doc = nlp_spacy(text)

sentences_annual_report = []
for sent in doc.sents:
    if len(sent.text.split()) > 3:
        sentences_annual_report.append(sent.text)
    
print(sentences_annual_report)