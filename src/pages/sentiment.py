import base64
import pytesseract
from pdf2image import convert_from_bytes
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

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

dash.register_page(__name__)

@dash.callback(
    Output('sen-viewer', 'data'),
    Input('sen-upload', 'contents'),
    State('sen-upload', 'filename')
)
def sen_output(contents, filename):
    if contents is None:
        raise exceptions.PreventUpdate
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        input_sen = io.BytesIO(decoded)

        with open("input.sen", "wb") as f:
            base64_sen = base64.b64encode(input_sen.read()).decode('utf-8')
            f.write(base64.b64decode(base64_sen))
        f.close()

        sen = convert_from_bytes('input_sen', 300)
        
        # extract text from image
        text = ''
        for page in sen:
            text += pytesseract.image_to_string(page)

        print(text)

#Upload component:
sen_load = dcc.Upload(
    id='sen-upload',
    children=html.Div(['Drag and Drop or ', html.A('Select PDF files')]),
    style={'width': '90%', 'height': '60px', 'lineHeight': '60px',
    'borderWidth': '1px', 'borderStyle': 'dashed',
    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
)

#Table to view output from pdf:
sen_table = dash_table.DataTable(
    editable=True,
    row_deletable=True,
    export_format='xlsx',
    id='sen-viewer',
    row_selectable='multi', 
    selected_columns=[], 
    selected_rows=[],
    style_table={'height':'1000px', 'overflowY': 'auto'},
    style_cell={
        'minWidth': 95, 'maxWidth': 95, 'width': 95
    },
    style_data={
        'whiteSpace': 'normal',
        'height': 'auto'
    }
)

#Place into the app
layout = html.Div([
    html.H4('Upload PDF file to do Sentiment Analysis'),
    sen_load,
    sen_table
])