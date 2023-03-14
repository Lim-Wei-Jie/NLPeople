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

import dash
from dash import Dash, dash_table, dcc, html, exceptions, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

dash.register_page(__name__)

#Upload component:
pdf_load = dcc.Upload(
    id='pdf-upload',
    children=html.Div(['Drag and Drop or ', html.A('Select PDF files')]),
    style={'width': '90%', 'height': '60px', 'lineHeight': '60px',
    'borderWidth': '1px', 'borderStyle': 'dashed',
    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
)



layout = html.Div([
    html.H4('Upload PDF file to do Sentiment Analysis'),
    pdf_load
])