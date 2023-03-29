import base64
import os
import tempfile
import pandas as pd
pd.options.display.max_colwidth=-1
pd.options.display.min_rows=100

import plotly.express as px

import dash
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pytesseract
from pdf2image import convert_from_path

import spacy
import nltk

from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

#tesseract file directory
pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe") 

dash.register_page(__name__)

@dash.callback(
    Output('output-pdf', 'children'),
    Input('upload-pdf', 'contents'),
    State('upload-pdf', 'filename')
)
def display_pdf(contents, filename):
    if contents is not None:
        # decode the contents of the uploaded PDF file
        decoded_content = base64.b64decode(contents.split(',')[1])

        # create a temporary file to save the decoded PDF data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = os.path.join(temp_dir, filename)
            with open(temp_filename, 'wb') as f:
                f.write(decoded_content)
            
            # use pdf2image 
            pages = convert_from_path(temp_filename, 300)

            # extract text from image
            text = ''
            for page in pages:
                text += pytesseract.image_to_string(page)
                
        #apply  NLP to extract text from pdf      
        nlp_spacy = spacy.load("en_core_web_sm")
        doc = nlp_spacy(text)

        sentences_annual_report = []
        for sent in doc.sents:
            if len(sent) > 6:
                new_lane = sent.text.split(".")
                for x in new_lane:
                    if len(x.strip()) > 2:
                        sentences_annual_report.append(x)
                
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

        #apply model to give label to each sentence
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
        results = nlp(sentences_annual_report)

        #put the result in a dataframe
        sentiment = pd.DataFrame({
            "docs": sentences_annual_report,
            "label": [r["label"] for r in results],
            "score":[r["score"] for r in results],
            "docs": sentences_annual_report
        })

        Positive = 0
        Negative = 0
        Neutral = 0

        sentence_list=[]
        label_list=[]
        score_list=[]

        for x, y,z in zip(sentences_annual_report, sentiment.label ,  sentiment.score):
            sentence_list.append(x)
            label_list.append(y)
            score_list.append(z)
            
            if y == "Positive":
                Positive += 1

            if y == "Negative":
                Negative += 1
                
            if y == "Neutral":
                Neutral += 1
        
        #arrange the order of the dataframe result 
        sentiment = pd.DataFrame({"sentence": sentence_list, "label":  label_list, "score":  score_list })

        #sort the dataframe by score in descending order
        sentiment = sentiment.sort_values(by='score', ascending=False)        

        # Filter for rows with the 'positive' label and get the top 5 by score
        sentiment_positive_top5 = sentiment[sentiment['label'] == 'Positive'].nlargest(5, 'score')

        # Filter for rows with the 'neutral' label and get the top 5 by score
        sentiment_neutral_top5 = sentiment[sentiment['label'] == 'Neutral'].nlargest(5, 'score')

        # Filter for rows with the 'neutral' label and get the top 5 by score
        sentiment_negative_top5 = sentiment[sentiment['label'] == 'Negative'].nlargest(5, 'score')

        table1 = dash_table.DataTable(
            data=sentiment_positive_top5.to_dict('records'),
            columns=[{"name": i, "id": i} for i in sentiment_positive_top5.columns],
            style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px','textAlign': 'left','whiteSpace': 'pre-wrap'},
            style_cell_conditional=[{'if': {'column_id': 'sentence'}, 'width': '70%'}]
        )
        
        table2 = dash_table.DataTable(
            data=sentiment_negative_top5.to_dict('records'),
            columns=[{"name": i, "id": i} for i in sentiment_negative_top5.columns],
            style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px','textAlign': 'left','whiteSpace': 'pre-wrap'},
            style_cell_conditional=[{'if': {'column_id': 'sentence'}, 'width': '70%'}]
        )
        
        table3 = dash_table.DataTable(
            data=sentiment_neutral_top5.to_dict('records'),
            columns=[{"name": i, "id": i} for i in sentiment_neutral_top5.columns],
            style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px', 'maxHeight': 'none','textAlign': 'left','whiteSpace': 'pre-wrap'},
            style_cell_conditional=[{'if': {'column_id': 'sentence'}, 'width': '70%'}]
        )

        # sentiment.to_excel("output.xlsx")

        # bar chart
        dff = sentiment
        fig = px.bar(
            dff,
            x='label',
            y='score',
            title="Scores of total Postive, Negative, Neutral sentences"
        )

        # pie chart
        piefig = px.pie(
            dff,
            values='score',
            names='label',
            title="Scores of total Postive, Negative, Neutral sentences (in %)"
        )

        # create dashtable and bar chart
        children=html.Div([
            html.Br(), html.Br(),
            # html.H3(filename),
            # html.Br(),
            html.Div([
                dbc.Row([
                    dbc.Card([dbc.CardBody([
                        html.H3(filename),
                        html.Br(),
                        dbc.Row([
                            dbc.Col(dbc.Card(dbc.CardBody([html.P("Positive"), html.H4(str(Positive))]), className='text-center m-4', color="success", inverse=True)),
                            dbc.Col(dbc.Card(dbc.CardBody([html.P("Negative"), html.H4(str(Negative))]), className='text-center m-4', color="danger", inverse=True)),
                            dbc.Col(dbc.Card(dbc.CardBody([html.P("Neutral"), html.H4(str(Neutral))]), className='text-center m-4', color="info", inverse=False))
                        ])
                    ])])
                ]),
                html.Br(), html.Br(),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardBody([
                        dcc.Graph(
                            id='pie-chart',
                            figure=piefig
                        )
                    ])], style={}), width=6),
                    dbc.Col(dbc.Card([dbc.CardBody([
                        dcc.Graph(
                            id='bar-chart',
                            figure=fig
                        )
                    ])], style={}), width=6),
                ]),
                html.Br(), html.Br(),
                dbc.Row([dbc.Card([dbc.CardBody([
                    html.H3('Top 5 Positive Statements'),
                    table1,
                    html.Br(), html.Br(),
                    html.H3('Top 5 Negative Statements'),
                    table2,
                    html.Br(), html.Br(),
                    html.H3('Top 5 Neutral Statements'),
                    table3
                ])])
                ])
            ]),
        ])
        return children

    else:
        return html.Div()


layout = html.Div([
    html.H4('Upload PDF file to do Sentiment Analysis', id="first-title", style={"cursor": "pointer", 'display': 'inline-block'}),
    dbc.Tooltip("Sentiment Analysis takes awhile depending on the size of the file. Please wait for it to load.", target="first-title", placement="top"),
    html.Br(),
    dcc.Upload(
        id='upload-pdf',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select PDF files')
        ]),
        style={
            'width': '90%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    html.Div(id='output-pdf', children=[])
])
